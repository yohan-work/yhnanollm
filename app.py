#!/usr/bin/env python3
"""
yhnanollm v1.0.0-beta
"""

import gradio as gr
from pathlib import Path
import shutil
from chat import LocalLLMChat
from rag import DocumentProcessor, VectorStore, RAGChain, DocumentManager


# 전역 변수
llm_chat = None
vector_store = None
rag_chain = None
doc_processor = None
doc_manager = None


def initialize_system():
    """시스템 초기화 (모델 + RAG)"""
    global llm_chat, vector_store, rag_chain, doc_processor, doc_manager
    
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
    doc_manager = DocumentManager(metadata_path="doc_metadata.json")
    rag_chain = RAGChain(vector_store, llm_chat, doc_manager, top_k=3)
    print("RAG 시스템 준비 완료!")


def upload_pdf(file):
    """PDF 파일 업로드 및 처리"""
    if file is None:
        return "파일을 선택해주세요.", get_document_table(), get_doc_list()
    
    try:
        # 파일 저장
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = Path(file.name)
        dest_path = upload_dir / file_path.name
        
        # 파일 크기 확인
        file_size = Path(file.name).stat().st_size
        
        # 파일 복사
        shutil.copy(file.name, dest_path)
        
        # PDF 처리
        chunks = doc_processor.process_pdf(str(dest_path))
        
        # 벡터 DB에 저장
        vector_store.add_documents(chunks)
        
        # 메타데이터 저장
        doc_manager.add_document(
            filename=file_path.name,
            file_size=file_size,
            chunk_count=len(chunks)
        )
        
        doc_count = vector_store.get_document_count()
        
        status_msg = (
            f"✅ 업로드 완료: {file_path.name}\n"
            f"📊 청크 수: {len(chunks)}개\n"
            f"💾 총 문서: {doc_manager.get_document_count()}개"
        )
        
        return status_msg, get_document_table(), get_doc_list()
    
    except Exception as e:
        return f"❌ 업로드 실패: {str(e)}", get_document_table(), get_doc_list()


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


def get_document_table():
    """문서 목록을 DataFrame 형태로 반환"""
    docs = doc_manager.get_all_documents()
    
    if not docs:
        return [["문서 없음", "-", "-", "-", "-"]]
    
    # DataFrame 데이터 생성
    table_data = []
    for doc in docs:
        table_data.append([
            doc['filename'],
            doc['chunk_count'],
            doc['upload_time'],
            doc['file_size_kb'],
            doc['search_count']
        ])
    
    return table_data


def get_doc_list():
    """문서 목록을 Dropdown용 리스트로 반환"""
    filenames = doc_manager.get_filenames()
    return gr.update(choices=filenames, value=filenames[0] if filenames else None)


def refresh_document_list():
    """문서 목록 새로고침"""
    return get_document_table(), get_doc_list()


def delete_document(filename):
    """특정 문서 삭제"""
    if not filename:
        return "삭제할 문서를 선택해주세요.", get_document_table(), get_doc_list()
    
    try:
        # 벡터 DB에서 삭제
        deleted_chunks = vector_store.delete_document_by_filename(filename)
        
        # 메타데이터에서 삭제
        doc_manager.delete_document(filename)
        
        # 업로드 폴더에서 파일 삭제
        upload_path = Path("uploads") / filename
        if upload_path.exists():
            upload_path.unlink()
        
        status_msg = f"✅ 삭제 완료: {filename} ({deleted_chunks}개 청크)"
        return status_msg, get_document_table(), get_doc_list()
    
    except Exception as e:
        return f"❌ 삭제 실패: {str(e)}", get_document_table(), get_doc_list()


def clear_all_documents():
    """모든 문서 삭제"""
    try:
        # 벡터 DB 초기화
        vector_store.clear()
        
        # 메타데이터 초기화
        doc_manager.clear_all()
        
        # 업로드 폴더 파일 삭제
        upload_dir = Path("uploads")
        if upload_dir.exists():
            for file in upload_dir.iterdir():
                if file.is_file():
                    file.unlink()
        
        return "✅ 모든 문서가 삭제되었습니다.", get_document_table(), get_doc_list()
    
    except Exception as e:
        return f"❌ 오류: {str(e)}", get_document_table(), get_doc_list()


def create_interface():
    """Gradio 인터페이스 생성"""
    
    custom_css = """
    .gradio-container {
        min-width: 1200px !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    footer {
        display: none !important;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.Markdown("# yhnanollm")
        gr.Markdown("로컬 LLM + 문서 기반 질의응답 | 문서 관리 기능")
        
        with gr.Row():
            # 왼쪽: 채팅 영역
            with gr.Column(scale=2):
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
            
            # 오른쪽: 문서 관리 영역
            with gr.Column(scale=1):
                gr.Markdown("### 문서 관리")
                
                # 파일 업로드
                file_upload = gr.File(
                    label="PDF 업로드",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                upload_status = gr.Textbox(
                    label="상태",
                    interactive=False,
                    lines=3
                )
                
                # 문서 목록 테이블
                gr.Markdown("#### 📋 문서 목록")
                doc_table = gr.Dataframe(
                    headers=["파일명", "청크", "업로드 시간", "크기(KB)", "검색 횟수"],
                    value=get_document_table(),
                    interactive=False,
                    wrap=True,
                    max_height=250
                )
                
                refresh_btn = gr.Button("🔄 새로고침", size="sm")
                
                # 개별 문서 삭제
                gr.Markdown("#### 🗑️ 개별 삭제")
                with gr.Row():
                    doc_selector = gr.Dropdown(
                        label="문서 선택",
                        choices=doc_manager.get_filenames(),
                        scale=3
                    )
                    delete_btn = gr.Button("삭제", variant="stop", scale=1, size="sm")
                
                # 전체 삭제
                clear_all_btn = gr.Button("⚠️ 모든 문서 삭제", variant="stop")
                
                gr.Markdown("---")
                gr.Markdown("#### 💡 사용 팁")
                gr.Markdown("""
                1. PDF 파일 업로드
                2. RAG 모드 활성화
                3. 문서 내용 질문
                4. 검색 횟수로 활용도 확인
                """)
        
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
            outputs=[upload_status, doc_table, doc_selector]
        )
        
        refresh_btn.click(
            refresh_document_list,
            outputs=[doc_table, doc_selector]
        )
        
        delete_btn.click(
            delete_document,
            inputs=doc_selector,
            outputs=[upload_status, doc_table, doc_selector]
        )
        
        clear_all_btn.click(
            clear_all_documents,
            outputs=[upload_status, doc_table, doc_selector]
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
