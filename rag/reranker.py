"""
Reranker 모듈
Cross-Encoder를 사용하여 검색 결과의 순위를 재정렬
"""

from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
import torch

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Reranker 초기화
        
        Args:
            model_name: Cross-Encoder 모델 이름
        """
        self.model_name = model_name
        print(f"🔄 Reranker 모델 로딩 중: {model_name}...")
        
        # GPU 가속 확인
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        try:
            self.model = CrossEncoder(model_name, device=device)
            self.is_ready = True
            print("✅ Reranker 모델 준비 완료")
        except Exception as e:
            print(f"⚠️ Reranker 모델 로드 실패: {e}")
            self.model = None
            self.is_ready = False
            
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        문서 재순위화
        
        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트 (text 필드 필수)
            top_k: 반환할 상위 문서 수
            
        Returns:
            (재정렬된 문서 리스트, 점수 리스트)
        """
        if not self.is_ready or not documents:
            return documents[:top_k], []
        
        # 입력 쌍 생성 (쿼리, 문서)
        pairs = []
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get('text', '')
            else:
                text = str(doc)
            pairs.append([query, text])
            
        # 점수 계산
        try:
            scores = self.model.predict(pairs)
            
            # 점수와 문서를 함께 정렬
            scored_docs = []
            for i, score in enumerate(scores):
                doc_copy = documents[i].copy() if isinstance(documents[i], dict) else {'text': documents[i]}
                doc_copy['rerank_score'] = float(score)  # 점수 추가
                scored_docs.append((doc_copy, score))
            
            # 점수 내림차순 정렬
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Top-K 추출
            final_docs = [item[0] for item in scored_docs[:top_k]]
            final_scores = [item[1] for item in scored_docs[:top_k]]
            
            return final_docs, final_scores
            
        except Exception as e:
            print(f"❌ Reranking 중 오류 발생: {e}")
            return documents[:top_k], []
