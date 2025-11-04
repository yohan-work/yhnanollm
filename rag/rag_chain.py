"""
RAG 체인 모듈
검색 증강 생성 로직
"""

from typing import Optional
from .prompts import get_prompt_template, format_prompt


class RAGChain:
    def __init__(
        self, 
        vector_store, 
        llm_chat, 
        document_manager=None, 
        top_k: int = 3,
        prompt_template: str = "default",
        similarity_threshold: float = 0.0
    ):
        """
        RAG 체인 초기화
        
        Args:
            vector_store: VectorStore 인스턴스
            llm_chat: LocalLLMChat 인스턴스
            document_manager: DocumentManager 인스턴스 (선택)
            top_k: 검색할 문서 수
            prompt_template: 프롬프트 템플릿 이름
            similarity_threshold: 유사도 임계값 (0.0~1.0)
        """
        self.vector_store = vector_store
        self.llm_chat = llm_chat
        self.document_manager = document_manager
        self.top_k = top_k
        self.prompt_template = prompt_template
        self.similarity_threshold = similarity_threshold
    
    def format_prompt_with_context(self, question: str, context: str) -> str:
        """
        컨텍스트와 질문을 결합한 프롬프트 생성
        
        Args:
            question: 사용자 질문
            context: 검색된 문서 컨텍스트
            
        Returns:
            포맷된 프롬프트
        """
        # 동적 프롬프트 템플릿 사용
        return format_prompt(context, question, self.prompt_template)
    
    def search_documents(self, question: str) -> tuple[str, list, list]:
        """
        질문과 관련된 문서 검색
        
        Args:
            question: 사용자 질문
            
        Returns:
            (컨텍스트 문자열, 메타데이터 리스트, 유사도 점수 리스트)
        """
        # 벡터 DB에서 유사 문서 검색
        results = self.vector_store.search(question, top_k=self.top_k)
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            return None, [], []
        
        # 검색된 문서들을 하나의 컨텍스트로 결합
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # 유사도 임계값 필터링 (distance가 낮을수록 유사함)
        # ChromaDB는 코사인 거리를 사용 (0에 가까울수록 유사)
        filtered_docs = []
        filtered_metas = []
        filtered_distances = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            # 거리가 임계값보다 낮으면 (유사하면) 포함
            if dist <= (1.0 - self.similarity_threshold):
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_distances.append(dist)
        
        if not filtered_docs:
            return None, [], []
        
        # 컨텍스트 생성 (출처 포함)
        context_parts = []
        for idx, (doc, meta) in enumerate(zip(filtered_docs, filtered_metas), 1):
            source = meta.get('filename', 'Unknown') if meta else 'Unknown'
            context_parts.append(f"[출처 {idx}: {source}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        return context, filtered_metas, filtered_distances
    
    def answer(self, question: str, use_rag: bool = True) -> tuple[str, Optional[list], Optional[dict]]:
        """
        질문에 답변 생성
        
        Args:
            question: 사용자 질문
            use_rag: RAG 사용 여부
            
        Returns:
            (답변, 사용된 문서 메타데이터, 검색 통계)
        """
        if not use_rag or self.vector_store.get_document_count() == 0:
            # RAG 미사용 또는 문서 없음 - 기본 모드
            answer = self.llm_chat.chat(question)
            return answer, None, None
        
        # 관련 문서 검색
        context, metadatas, distances = self.search_documents(question)
        
        if context is None:
            # 관련 문서를 찾지 못함 - 기본 모드로 폴백
            answer = self.llm_chat.chat(question)
            return answer, None, {'reason': 'no_similar_documents'}
        
        # 검색 통계 생성
        search_stats = {
            'documents_found': len(metadatas),
            'avg_distance': sum(distances) / len(distances) if distances else 0,
            'prompt_template': self.prompt_template,
            'top_k': self.top_k
        }
        
        # 검색 통계 업데이트 (DocumentManager가 있는 경우)
        if self.document_manager and metadatas:
            # 사용된 문서의 검색 횟수 증가
            used_files = set()
            for meta in metadatas:
                filename = meta.get('filename')
                if filename and filename not in used_files:
                    self.document_manager.increment_search_count(filename)
                    used_files.add(filename)
        
        # 컨텍스트 포함 프롬프트 생성
        prompt = self.format_prompt_with_context(question, context)
        
        # LLM으로 답변 생성
        answer = self.llm_chat.chat(prompt)
        
        return answer, metadatas, search_stats
    
    def update_config(
        self, 
        top_k: Optional[int] = None,
        prompt_template: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        RAG 체인 설정 동적 업데이트
        
        Args:
            top_k: 검색할 문서 수
            prompt_template: 프롬프트 템플릿 이름
            similarity_threshold: 유사도 임계값
        """
        if top_k is not None:
            self.top_k = top_k
        if prompt_template is not None:
            self.prompt_template = prompt_template
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        
        print(f"✅ RAG 설정 업데이트:")
        print(f"   - top_k: {self.top_k}")
        print(f"   - prompt_template: {self.prompt_template}")
        print(f"   - similarity_threshold: {self.similarity_threshold}")

