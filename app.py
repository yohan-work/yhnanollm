#!/usr/bin/env python3
"""
yhnanollm v1.0.0-beta
"""

import gradio as gr
from pathlib import Path
import shutil
from chat import LocalLLMChat
from rag import DocumentProcessor, VectorStore, RAGChain, DocumentManager, HybridRetriever
from rag.reranker import Reranker  # [NEW] Reranker 임포트
from rag.config import RAGConfig, get_preset, list_presets
from rag.prompts import list_templates


# 전역 변수
llm_chat = None
vector_store = None
rag_chain = None
doc_processor = None
doc_manager = None
current_config = None
hybrid_retriever = None
reranker = None  # [NEW]


def initialize_system(config: RAGConfig = None):
    """시스템 초기화 (모델 + RAG)"""
    global llm_chat, vector_store, rag_chain, doc_processor, doc_manager, current_config, hybrid_retriever, reranker
    
    # 설정 로드 또는 기본값 사용
    if config is None:
        config = RAGConfig()
    
    current_config = config
    
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
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        repetition_penalty=1.1,
        top_p=0.9
    )
    llm_chat.load_model()
    print("모델 준비 완료!")
    
    # 2. RAG 시스템 초기화
    print("\nRAG 시스템 초기화 중...")
    doc_processor = DocumentProcessor(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    vector_store = VectorStore(
        persist_directory=config.persist_directory,
        embedding_model=config.embedding_model
    )
    doc_manager = DocumentManager(metadata_path=config.metadata_path)
    
    # Hybrid Retriever 초기화
    hybrid_retriever = HybridRetriever(vector_store)
    
    # 기존 문서가 있다면 BM25 인덱스 재빌드
    print("BM25 인덱스 복구 중...")
    try:
        filenames = doc_manager.get_filenames()
        if filenames:
            all_chunks = []
            print(f"  - {len(filenames)}개 문서 로드 중...")
            for filename in filenames:
                # VectorStore에서 문서 청크 가져오기
                doc_data = vector_store.get_documents_by_filename(filename)
                
                # 포맷 변환 (VectorStore 반환값 -> Chunk List)
                if doc_data['ids']:
                    for i in range(len(doc_data['ids'])):
                        chunk = {
                            'text': doc_data['documents'][i],
                            'metadata': doc_data['metadatas'][i]
                        }
                        all_chunks.append(chunk)
            
            if all_chunks:
                hybrid_retriever.add_documents(all_chunks)
                print(f"  ✅ {len(all_chunks)}개 청크로 BM25 인덱스 재빌드 완료")
            else:
                print("  ⚠️ 저장된 청크를 찾을 수 없습니다.")
        else:
            print("  ℹ️ 저장된 문서가 없습니다.")
            
    except Exception as e:
        print(f"  ❌ BM25 인덱스 복구 실패: {str(e)}")
        
    # Reranker 초기화 [NEW]
    if config.use_reranker:
        print("\nReranker 초기화 중...")
        reranker = Reranker(model_name=config.reranker_model)
    else:
        reranker = None
    
    rag_chain = RAGChain(
        retriever=hybrid_retriever,
        llm_chat=llm_chat,
        document_manager=doc_manager,
        top_k=config.top_k,
        prompt_template=config.prompt_template,
        similarity_threshold=config.similarity_threshold,
        reranker=reranker,
        top_k_retrieval=config.top_k_retrieval
    )
    print("RAG 시스템 준비 완료!")


def upload_file(file):
    """파일 업로드 및 처리"""
    if file is None:
        return "파일을 선택해주세요.", get_document_table(), get_doc_list()
    
    try:
        print(f"\n{'='*60}")
        print(f"문서 업로드 시작")
        print(f"{'='*60}")
        
        # 파일 저장
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = Path(file)
        dest_path = upload_dir / file_path.name
        
        print(f"파일 복사: {file_path.name}")
        
        # 파일 크기 확인
        file_size = file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f" 파일 크기: {file_size_mb:.2f} MB")
        
        # 파일 복사
        shutil.copy(file, dest_path)
        print(f"   ✓ 복사 완료")
        
        # 문서 처리
        print(f"\n  텍스트 추출 중...")
        chunks = doc_processor.process_document(str(dest_path))
        print(f"   ✓ 생성된 청크: {len(chunks)}개")
        
        # 벡터 DB에 저장
        print(f"\n 임베딩 생성 및 저장 중...")
        print(f"   ⏳ 잠시만 기다려주세요 (청크가 많으면 시간이 걸릴 수 있습니다)")
        vector_store.add_documents(chunks)
        print(f"   ✓ 벡터 DB 저장 완료")
        
        # Hybrid Retriever에 추가
        print(f"\n BM25 인덱싱 중...")
        hybrid_retriever.add_documents(chunks)
        print(f"   ✓ BM25 인덱싱 완료")
        
        # 메타데이터 저장
        print(f"\n 메타데이터 저장")
        doc_manager.add_document(
            filename=file_path.name,
            file_size=file_size,
            chunk_count=len(chunks)
        )
        print(f"   ✓ 메타데이터 저장 완료")
        
        status_msg = (
            f"업로드 완료: {file_path.name}\n"
            f"청크 수: {len(chunks)}개\n"
            f"파일 크기: {file_size_mb:.2f} MB\n"
            f"총 문서: {doc_manager.get_document_count()}개"
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 업로드 완료!")
        print(f"{'='*60}\n")
        
        return status_msg, get_document_table(), get_doc_list()
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"업로드 오류:")
        print(f"{'='*60}")
        print(error_detail)
        print(f"{'='*60}\n")
        return f"업로드 실패: {str(e)}", get_document_table(), get_doc_list()


def chat_with_rag(message, history, use_rag):
    """RAG 기능이 포함된 채팅"""
    if not message.strip():
        return history, "", ""
    
    try:
        print(f"\n{'='*60}")
        print(f"[채팅] 새 메시지: {message}")
        print(f"[채팅] RAG 모드: {use_rag}")
        print(f"{'='*60}")
        
        # RAG 모드에 따라 답변 생성
        answer, sources, stats = rag_chain.answer(message, use_rag=use_rag)
        
        print(f"[채팅] 답변 받음: {answer[:100]}...")
        
        # 히스토리에 추가
        history.append((message, answer))
        
        # 검색 통계 표시
        stats_text = ""
        if stats and use_rag:
            if stats.get('reranking_applied'):
                score_label = "평균 점수 (Rerank)"
                score_value = f"{stats.get('avg_distance', 0):.4f}"
            else:
                score_label = "평균 유사도"
                score_value = f"{1 - stats.get('avg_distance', 0):.2%}"
                
            stats_text = f"""검색 정보:
• 검색된 문서: {stats.get('documents_found', 0)}개
• {score_label}: {score_value}
• 프롬프트: {stats.get('prompt_template', 'N/A')}
• Top-K: {stats.get('top_k', 0)}"""
        
        print(f"[채팅] 완료\n")
        return history, "", stats_text
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"[채팅] ❌ 오류 발생:")
        print(f"{'='*60}")
        print(error_detail)
        print(f"{'='*60}\n")
        
        error_msg = f"❌ 오류: {str(e)}\n\n자세한 내용은 터미널 로그를 확인하세요."
        history.append((message, error_msg))
        return history, "", f"오류 발생: {str(e)}"


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
        
        # Hybrid Retriever 초기화
        if hybrid_retriever:
            hybrid_retriever.clear()
        
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


def update_rag_settings(chunk_size, chunk_overlap, top_k, prompt_template, similarity_threshold):
    """RAG 설정 업데이트"""
    global rag_chain, doc_processor, current_config
    
    try:
        # 설정 업데이트
        current_config.chunk_size = chunk_size
        current_config.chunk_overlap = chunk_overlap
        current_config.top_k = top_k
        current_config.prompt_template = prompt_template
        current_config.similarity_threshold = similarity_threshold
        
        # RAG 체인 설정 업데이트
        rag_chain.update_config(
            top_k=top_k,
            prompt_template=prompt_template,
            similarity_threshold=similarity_threshold
        )
        
        # 문서 프로세서 재생성 (새 청크 설정 적용)
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return f"""✅ 설정이 업데이트되었습니다:
• 청크 크기: {chunk_size}자
• 청크 오버랩: {chunk_overlap}자
• Top-K: {top_k}개
• 프롬프트: {prompt_template}
• 유사도 임계값: {similarity_threshold}

⚠️ 청크 크기 변경은 새로 업로드되는 문서에만 적용됩니다."""
    
    except Exception as e:
        return f"❌ 설정 업데이트 실패: {str(e)}"


def apply_preset(preset_name):
    """프리셋 설정 적용"""
    try:
        preset_config = get_preset(preset_name)
        
        return (
            preset_config.chunk_size,
            preset_config.chunk_overlap,
            preset_config.top_k,
            preset_config.prompt_template,
            preset_config.similarity_threshold,
            f"✅ '{preset_name}' 프리셋이 적용되었습니다."
        )
    
    except Exception as e:
        return None, None, None, None, None, f"❌ 프리셋 적용 실패: {str(e)}"


def save_current_config(config_name):
    """현재 설정 저장"""
    try:
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / f"{config_name}.json"
        current_config.save(str(config_path))
        
        return f"✅ 설정이 저장되었습니다: {config_path}"
    
    except Exception as e:
        return f"❌ 설정 저장 실패: {str(e)}"


def get_config_info():
    """현재 설정 정보 반환"""
    if current_config:
        return f"""📋 현재 RAG 설정:

**문서 처리**
• 청크 크기: {current_config.chunk_size}자
• 청크 오버랩: {current_config.chunk_overlap}자

**검색**
• Top-K: {current_config.top_k}개
• 유사도 임계값: {current_config.similarity_threshold}

**프롬프트**
• 템플릿: {current_config.prompt_template}

**LLM**
• Max Tokens: {current_config.max_tokens}
• Temperature: {current_config.temperature}"""
    return "설정 정보를 불러올 수 없습니다."


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
        gr.Markdown("# 🤖 yhnanollm")
        gr.Markdown("로컬 LLM + 문서 기반 질의응답 | 실시간 파라미터 조정")
        
        with gr.Tabs():
            # 탭 1: 채팅 및 문서 관리
            with gr.Tab("💬 채팅"):
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
                        gr.Markdown("### 📁 문서 관리")
                        
                        # 파일 업로드
                        file_upload = gr.File(
                            label="문서 업로드 (PDF, TXT, DOCX)",
                            file_types=[".pdf", ".txt", ".docx"],
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
            
            # 탭 2: RAG 설정
            with gr.Tab("⚙️ RAG 설정"):
                gr.Markdown("### 🎛️ 파라미터 튜닝")
                gr.Markdown("RAG 시스템의 동작을 실시간으로 조정할 수 있습니다.")
                
                with gr.Row():
                    # 왼쪽: 설정 조정
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📝 문서 처리")
                        
                        chunk_size_slider = gr.Slider(
                            minimum=200,
                            maximum=1000,
                            step=50,
                            value=500,
                            label="청크 크기 (문자 수)",
                            info="작을수록 정확하지만 맥락이 부족할 수 있습니다"
                        )
                        
                        chunk_overlap_slider = gr.Slider(
                            minimum=0,
                            maximum=200,
                            step=10,
                            value=50,
                            label="청크 오버랩 (문자 수)",
                            info="문장이 잘리는 것을 방지합니다"
                        )
                        
                        gr.Markdown("#### 🔍 검색")
                        
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=3,
                            label="검색 문서 개수 (Top-K)",
                            info="많을수록 정보가 풍부하지만 노이즈가 증가할 수 있습니다"
                        )
                        
                        similarity_threshold_slider = gr.Slider(
                            minimum=0.0,
                            maximum=0.9,
                            step=0.1,
                            value=0.0,
                            label="유사도 임계값",
                            info="이 값보다 낮은 유사도의 문서는 제외됩니다 (0=전체 포함)"
                        )
                        
                        gr.Markdown("#### 💬 프롬프트")
                        
                        prompt_dropdown = gr.Dropdown(
                            choices=list(list_templates().keys()),
                            value="default",
                            label="프롬프트 템플릿",
                            info="다양한 프롬프트 전략을 선택할 수 있습니다"
                        )
                        
                        # 프롬프트 설명
                        prompt_desc = gr.Markdown(list_templates()["default"])
                        
                        # 설정 적용 버튼
                        with gr.Row():
                            apply_settings_btn = gr.Button(
                                "✅ 설정 적용",
                                variant="primary",
                                scale=2
                            )
                            reset_btn = gr.Button("🔄 기본값으로 재설정", scale=1)
                        
                        settings_status = gr.Textbox(
                            label="상태",
                            interactive=False,
                            lines=8
                        )
                    
                    # 오른쪽: 프리셋 및 정보
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 프리셋")
                        gr.Markdown("사전 정의된 설정을 빠르게 적용할 수 있습니다.")
                        
                        preset_info = gr.Markdown("""
**사용 가능한 프리셋:**
- **default**: 균형잡힌 기본 설정
- **precise**: 정확도 우선 (작은 청크)
- **comprehensive**: 포괄적 검색 (큰 청크, 많은 문서)
- **fast**: 빠른 응답 (최소 설정)
                        """)
                        
                        preset_dropdown = gr.Dropdown(
                            choices=["default", "precise", "comprehensive", "fast"],
                            value="default",
                            label="프리셋 선택"
                        )
                        
                        apply_preset_btn = gr.Button("📥 프리셋 적용", variant="secondary")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 💾 설정 저장")
                        
                        config_name_input = gr.Textbox(
                            label="설정 이름",
                            placeholder="my_config",
                            value="my_config"
                        )
                        
                        save_config_btn = gr.Button("💾 현재 설정 저장")
                        
                        save_config_status = gr.Textbox(
                            label="저장 상태",
                            interactive=False,
                            lines=2
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### 📊 현재 설정 정보")
                        
                        config_info_display = gr.Markdown(get_config_info())
                        
                        refresh_config_btn = gr.Button("🔄 정보 새로고침", size="sm")
            
            # 탭 3: 프롬프트 템플릿 정보
            with gr.Tab("📚 프롬프트 템플릿"):
                gr.Markdown("### 사용 가능한 프롬프트 템플릿")
                
                templates_info = list_templates()
                for name, desc in templates_info.items():
                    with gr.Accordion(f"{name}", open=False):
                        gr.Markdown(f"**설명:** {desc}")
        
        # 검색 통계 표시
        search_stats = gr.Textbox(
            label="검색 통계",
            interactive=False,
            lines=5,
            visible=True
        )
        
        # 이벤트 핸들러
        msg.submit(
            chat_with_rag,
            inputs=[msg, chatbot, rag_mode],
            outputs=[chatbot, msg, search_stats]
        )
        
        send_btn.click(
            chat_with_rag,
            inputs=[msg, chatbot, rag_mode],
            outputs=[chatbot, msg, search_stats]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, search_stats]
        )
        
        file_upload.upload(
            upload_file,
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
        
        # 이벤트 핸들러 - 설정 탭
        apply_settings_btn.click(
            update_rag_settings,
            inputs=[
                chunk_size_slider,
                chunk_overlap_slider,
                top_k_slider,
                prompt_dropdown,
                similarity_threshold_slider
            ],
            outputs=settings_status
        )
        
        reset_btn.click(
            lambda: (500, 50, 3, "default", 0.0, "✅ 기본값으로 재설정되었습니다."),
            outputs=[
                chunk_size_slider,
                chunk_overlap_slider,
                top_k_slider,
                prompt_dropdown,
                similarity_threshold_slider,
                settings_status
            ]
        )
        
        apply_preset_btn.click(
            apply_preset,
            inputs=preset_dropdown,
            outputs=[
                chunk_size_slider,
                chunk_overlap_slider,
                top_k_slider,
                prompt_dropdown,
                similarity_threshold_slider,
                settings_status
            ]
        )
        
        save_config_btn.click(
            save_current_config,
            inputs=config_name_input,
            outputs=save_config_status
        )
        
        refresh_config_btn.click(
            get_config_info,
            outputs=config_info_display
        )
        
        # 프롬프트 선택 시 설명 업데이트
        prompt_dropdown.change(
            lambda x: list_templates().get(x, ""),
            inputs=prompt_dropdown,
            outputs=prompt_desc
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
    print("문서(PDF/TXT/DOCX) 업로드 후 RAG 모드를 활성화하여 사용하세요")
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
