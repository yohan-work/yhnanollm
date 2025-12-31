from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from .vector_store import VectorStore

class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        """
        하이브리드 검색기 초기화 (Vector + BM25)
        
        Args:
            vector_store: 초기화된 VectorStore 인스턴스
        """
        self.vector_store = vector_store
        self.kiwi = Kiwi()
        self.bm25 = None
        self.chunks = []  # BM25용 청크 저장소 (메타데이터 포함)
        self.is_initialized = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Kiwi를 사용한 한국어 토큰화"""
        return [token.form for token in self.kiwi.tokenize(text)]

    def build_bm25(self, chunks: List[Dict[str, Any]]):
        """BM25 인덱스 생성
        
        Args:
            chunks: 텍스트와 메타데이터가 포함된 청크 리스트
        """
        print("BM25 인덱스 빌드 시작...")
        self.chunks = chunks
        
        tokenized_corpus = [self._tokenize(chunk['text']) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_initialized = True
        print(f"BM25 인덱스 빌드 완료: {len(chunks)}개 문서")

    def search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.0, alpha: float = 0.5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 수
            similarity_threshold: 벡터 검색 임계값 (사용 안 함 - 하이브리드 점수로 대체)
            alpha: 가중치 조정 (0.0=Vector only, 1.0=BM25 only, 0.5=Half/Half)
            
        Returns:
            (검색된 청크 리스트, 통계 정보)
        """
        if not self.is_initialized:
            # BM25가 준비되지 않았으면 벡터 검색만 수행
            print("경고: BM25가 초기화되지 않았습니다. 벡터 검색만 수행합니다.")
            results = self.vector_store.search(query, top_k)
            # 호환성을 위해 리스트 포맷으로 변환하지 않고 그대로 반환하면 RAGChain에서 에러남
            # RAGChain은 HybridRetriever가 (results_list, stats)를 반환할 것으로 기대함
            # 따라서 변환 로직이 필요함.
            
            # VectorStore returns dict with list of lists
            docs = results.get('documents', [[]])[0]
            metas = results.get('metadatas', [[]])[0]
            dists = results.get('distances', [[]])[0]
            
            formatted_results = []
            for doc, meta, dist in zip(docs, metas, dists):
                if dist <= (1.0 - similarity_threshold):
                    formatted_results.append({
                        'text': doc,
                        'metadata': meta,
                        'distance': dist
                    })
            
            stats = {
                'documents_found': len(formatted_results),
                'avg_distance': sum(d['distance'] for d in formatted_results) / len(formatted_results) if formatted_results else 0.0,
                'top_k': top_k,
                'method': 'vector_only (fallback)'
            }
            return formatted_results, stats

        # 1. 벡터 검색
        vector_results_dict = self.vector_store.search(query, top_k * 2) 
        
        # VectorStore.search returns a dict with lists inside lists (ChromaDB format)
        # e.g., {'documents': [['text1', 'text2']], 'metadatas': [[{'meta1'}, {'meta2'}]], 'distances': [[0.1, 0.2]]}
        
        vector_docs = vector_results_dict.get('documents', [[]])[0]
        vector_metas = vector_results_dict.get('metadatas', [[]])[0]
        vector_dists = vector_results_dict.get('distances', [[]])[0]
        
        # 2. BM25 검색
        tokenized_query = self._tokenize(query)
        # BM25 점수 계산 (모든 문서에 대해)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. 점수 결합 (RRF: Reciprocal Rank Fusion)
        rrf_score = {}
        k = 60
        
        # 벡터 랭킹 반영
        for rank, (doc_text, meta, dist) in enumerate(zip(vector_docs, vector_metas, vector_dists)):
            # 청크를 식별할 수 있는 고유 키
            chunk_id = meta.get('chunk_id')
            filename = meta.get('filename')
            doc_id = f"{chunk_id}_{filename}"
            
            if doc_id not in rrf_score:
                rrf_score[doc_id] = {
                    'score': 0, 
                    'doc': {
                        'text': doc_text,
                        'metadata': meta,
                        'distance': dist
                    }
                }
            
            rrf_score[doc_id]['score'] += 1 / (rank + 1 + k)

        # BM25 랭킹 반영 (Top N)
        # BM25 점수 기준으로 상위 N개 뽑기
        top_n_bm25 = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        for rank, idx in enumerate(top_n_bm25):
            chunk = self.chunks[idx]
            chunk_id = chunk['metadata'].get('chunk_id')
            filename = chunk['metadata'].get('filename')
            doc_id = f"{chunk_id}_{filename}"
            
            if doc_id not in rrf_score:
                # distance 정보는 BM25에서는 알 수 없으므로 0.0 처리
                chunk_result = chunk.copy()
                chunk_result['distance'] = 0.0 
                rrf_score[doc_id] = {'score': 0, 'doc': chunk_result}
            
            # BM25에 약간의 가중치(epsilon)를 더해 동점 시 우선순위 부여
            rrf_score[doc_id]['score'] += (1 / (rank + 1 + k)) * 1.5

        # 최종 정렬
        sorted_docs = sorted(rrf_score.values(), key=lambda x: x['score'], reverse=True)
        final_results = [item['doc'] for item in sorted_docs[:top_k]]
        
        stats = {
            'documents_found': len(final_results),
            'avg_distance': 0.0,
            'top_k': top_k,
            'method': 'hybrid (RRF)'
        }
        
        return final_results, stats

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """문서 추가 및 BM25 재빌드"""
        # 기존 청크에 추가
        self.chunks.extend(chunks)
        # BM25 전체 재빌드 (메모리 방식이라 어쩔 수 없음)
        self.build_bm25(self.chunks)
    
    def clear(self):
        """초기화"""
        self.chunks = []
        self.bm25 = None
        self.is_initialized = False
