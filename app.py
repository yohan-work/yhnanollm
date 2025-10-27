#!/usr/bin/env python3
"""
yhnanollm v1.0.0-beta
"""

import gradio as gr
from pathlib import Path
import shutil
from chat import LocalLLMChat
from rag import DocumentProcessor, VectorStore, RAGChain


# 전역 변수
llm_chat = None
vector_store = None
rag_chain = None
doc_processor = None


def initialize_system():
    """시스템 초기화 (모델 + RAG)"""
    global llm_chat, vector_store, rag_chain, doc_processor
    
    # 1. LLM 모델 초기화
    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    adapter_path = "models/lora-adapter"
    
    if not Path(adapter_path).exists():
        print(f" 경고: 어댑터를 찾을 수 없습니다: {adapter_path}")
        print("   베이스 모델만 사용합니다.")
        adapter_path = None
    
    print("모델 초기화 중...")
    llm_chat = LocalLLMChat(
        model_path=model_path,
        adapter_path=adapter_path,
        max_tokens=150
    )
    llm_chat.load_model()
    print("모델 준비 완료!")
    
    # 2. RAG 시스템 초기화
    print("\nRAG 시스템 초기화 중...")
    doc_processor = DocumentProcessor(chunk_size=500)
    vector_store = VectorStore(persist_directory="chroma_db")
    rag_chain = RAGChain(vector_store, llm_chat, top_k=3)
    print("RAG 시스템 준비 완료!")


def upload_pdf(file):
    """PDF 파일 업로드 및 처리"""
    if file is None:
        return "파일을 선택해주세요.", ""
    
    try:
        # 파일 저장
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = Path(file.name)
        dest_path = upload_dir / file_path.name
        
        # 파일 복사
        shutil.copy(file.name, dest_path)
        
        # PDF 처리
        chunks = doc_processor.process_pdf(str(dest_path))
        
        # 벡터 DB에 저장
        vector_store.add_documents(chunks)
        
        doc_count = vector_store.get_document_count()
        
        return (
            f"업로드 완료: {file_path.name}\n"
            f"📊 청크 수: {len(chunks)}개\n"
            f"💾 총 문서: {doc_count}개 청크",
            f"현재 {doc_count}개 청크 저장됨"
        )
    
    except Exception as e:
        return f"❌ 업로드 실패: {str(e)}", ""


def chat_with_rag(message, history, use_rag):
    """RAG 기능이 포함된 채팅"""
    if not message.strip():
        return history, ""
    
    try:
        # RAG 모드에 따라 답변 생성
        answer, sources = rag_chain.answer(message, use_rag=use_rag)
        
        # 히스토리에 추가
        history.append((message, answer))
        
        return history, ""
    
    except Exception as e:
        error_msg = f"❌ 오류: {str(e)}"
        history.append((message, error_msg))
        return history, ""


def clear_db():
    """벡터 DB 초기화"""
    try:
        vector_store.clear()
        return "모든 문서가 삭제되었습니다.", "문서 없음"
    except Exception as e:
        return f"❌ 오류: {str(e)}", ""


def create_interface():
    """Gradio 인터페이스 생성"""
    
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    footer {
        display: none !important;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.Markdown("# 🤖 yhnanollm with RAG")
        gr.Markdown("로컬 LLM + 문서 기반 질의응답")
        
        with gr.Row():
            # 왼쪽: 채팅 영역
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="메시지",
                        placeholder="질문을 입력하세요...",
                        scale=4
                    )
                    send_btn = gr.Button("전송", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 대화 초기화")
                    rag_mode = gr.Checkbox(
                        label="RAG 모드 (문서 참고)",
                        value=False,
                        info="체크하면 업로드된 문서를 참고하여 답변합니다"
                    )
                
                gr.Examples(
                    examples=[
                        "안녕하세요?",
                        "React가 뭐야?",
                        "파이썬이란?",
                        "이 문서의 주요 내용은?",
                    ],
                    inputs=msg
                )
            
            # 오른쪽: RAG 컨트롤
            with gr.Column(scale=1):
                gr.Markdown("### 문서 관리")
                
                file_upload = gr.File(
                    label="PDF 업로드",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                upload_status = gr.Textbox(
                    label="업로드 상태",
                    interactive=False,
                    lines=4
                )
                
                doc_info = gr.Textbox(
                    label="문서 정보",
                    value="문서 없음",
                    interactive=False
                )
                
                clear_db_btn = gr.Button("모든 문서 삭제", variant="stop")
        
        # 이벤트 핸들러
        msg.submit(
            chat_with_rag,
            inputs=[msg, chatbot, rag_mode],
            outputs=[chatbot, msg]
        )
        
        send_btn.click(
            chat_with_rag,
            inputs=[msg, chatbot, rag_mode],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            lambda: [],
            outputs=chatbot
        )
        
        file_upload.change(
            upload_pdf,
            inputs=file_upload,
            outputs=[upload_status, doc_info]
        )
        
        clear_db_btn.click(
            clear_db,
            outputs=[upload_status, doc_info]
        )
    
    return interface


def main():
    """메인 함수"""
    # 시스템 초기화
    initialize_system()
    
    # 인터페이스 생성 및 실행
    interface = create_interface()
    
    print("\n" + "="*60)
    print("🚀 yhnanollm with RAG 시작!")
    print("="*60)
    print("브라우저에서 http://localhost:7860 을 열어주세요")
    print("PDF 업로드 후 RAG 모드를 활성화하여 사용하세요")
    print("종료하려면 Ctrl+C를 누르세요")
    print("="*60 + "\n")
    
    # 서버 실행
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
