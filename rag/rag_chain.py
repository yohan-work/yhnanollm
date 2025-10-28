"""
RAG 체인 모듈
검색 증강 생성 로직
"""

from typing import Optional


class RAGChain:
    def __init__(self, vector_store, llm_chat, document_manager=None, top_k: int = 3):
        """
        RAG 체인 초기화
        
        Args:
            vector_store: VectorStore 인스턴스
            llm_chat: LocalLLMChat 인스턴스
            document_manager: DocumentManager 인스턴스 (선택)
            top_k: 검색할 문서 수
        """
        self.vector_store = vector_store
        self.llm_chat = llm_chat
        self.document_manager = document_manager
        self.top_k = top_k
    
    def format_prompt_with_context(self, question: str, context: str) -> str:
        """
        컨텍스트와 질문을 결합한 프롬프트 생성
        
        Args:
            question: 사용자 질문
            context: 검색된 문서 컨텍스트
            
        Returns:
            포맷된 프롬프트
        """
        prompt = f"""다음 문서 내용을 참고해서 질문에 답변해주세요.

문서 내용:
{context}

질문: {question}

답변:"""
        return prompt
    
    def search_documents(self, question: str) -> tuple[str, list]:
        """
        질문과 관련된 문서 검색
        
        Args:
            question: 사용자 질문
            
        Returns:
            (컨텍스트 문자열, 메타데이터 리스트)
        """
        # 벡터 DB에서 유사 문서 검색
        results = self.vector_store.search(question, top_k=self.top_k)
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            return None, []
        
        # 검색된 문서들을 하나의 컨텍스트로 결합
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        # 컨텍스트 생성 (출처 포함)
        context_parts = []
        for idx, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            source = meta.get('filename', 'Unknown') if meta else 'Unknown'
            context_parts.append(f"[출처 {idx}: {source}]\n{doc}")
        
        context = "\n\n".join(context_parts)
        
        return context, metadatas
    
    def answer(self, question: str, use_rag: bool = True) -> tuple[str, Optional[list]]:
        """
        질문에 답변 생성
        
        Args:
            question: 사용자 질문
            use_rag: RAG 사용 여부
            
        Returns:
            (답변, 사용된 문서 메타데이터)
        """
        if not use_rag or self.vector_store.get_document_count() == 0:
            # RAG 미사용 또는 문서 없음 - 기본 모드
            answer = self.llm_chat.chat(question)
            return answer, None
        
        # 관련 문서 검색
        context, metadatas = self.search_documents(question)
        
        if context is None:
            # 관련 문서를 찾지 못함 - 기본 모드로 폴백
            answer = self.llm_chat.chat(question)
            return answer, None
        
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
        
        return answer, metadatas

