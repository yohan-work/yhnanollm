"""
벡터 스토어 모듈
ChromaDB를 사용한 문서 임베딩 및 검색
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import hashlib


class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        벡터 스토어 초기화
        
        Args:
            persist_directory: ChromaDB 저장 경로
        """
        self.persist_directory = persist_directory
        
        # ChromaDB 클라이언트 생성
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "yhnanollm document embeddings"}
        )
        
        # 임베딩 모델 로드 (한국어 지원)
        print("🔄 임베딩 모델 로딩 중...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ 임베딩 모델 준비 완료")
    
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """
        문서 청크를 벡터 DB에 추가
        
        Args:
            chunks: 문서 청크 리스트
        """
        if not chunks:
            return
        
        print(f"{len(chunks)}개 청크 임베딩 중...")
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # 텍스트를 임베딩으로 변환
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # 고유 ID 생성 (파일명 + 청크 ID)
        ids = []
        for meta in metadatas:
            doc_id = f"{meta['filename']}_{meta['chunk_id']}"
            # 해시로 고유성 보장
            doc_hash = hashlib.md5(doc_id.encode()).hexdigest()[:8]
            ids.append(f"{doc_hash}_{meta['chunk_id']}")
        
        # ChromaDB에 추가
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"{len(chunks)}개 청크 저장 완료")
    
    def search(self, query: str, top_k: int = 3) -> Dict:
        """
        질문과 유사한 문서 검색
        
        Args:
            query: 검색 질문
            top_k: 반환할 문서 수
            
        Returns:
            검색 결과 (documents, metadatas, distances)
        """
        # 질문을 임베딩으로 변환
        query_embedding = self.embedder.encode([query])
        
        # 유사도 검색
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.collection.count())
        )
        
        return results
    
    def get_document_count(self) -> int:
        """
        저장된 문서 청크 수 반환
        """
        return self.collection.count()
    
    def clear(self) -> None:
        """
        모든 문서 삭제
        """
        # 컬렉션 삭제 후 재생성
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "yhnanollm document embeddings"}
        )
        print("벡터 DB 초기화 완료")

