#!/usr/bin/env python3
"""
yhnanollm v1.0.0-beta - Gradio 웹 인터페이스
브라우저에서 로컬 LLM과 대화할 수 있는 웹 UI
"""

import gradio as gr
from pathlib import Path
from chat import LocalLLMChat


# 전역 변수로 모델 관리
llm_chat = None


def initialize_model():
    """앱 시작시 모델 로드"""
    global llm_chat
    
    model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    adapter_path = "models/lora-adapter"
    
    # 어댑터 존재 확인
    if not Path(adapter_path).exists():
        print(f"⚠️  경고: 어댑터를 찾을 수 없습니다: {adapter_path}")
        print("   베이스 모델만 사용합니다.")
        adapter_path = None
    
    print("🔄 모델 초기화 중...")
    llm_chat = LocalLLMChat(
        model_path=model_path,
        adapter_path=adapter_path,
        max_tokens=100
    )
    llm_chat.load_model()
    print("✅ 모델 준비 완료!")


def chat_fn(message, history):
    """채팅 함수 - Gradio ChatInterface용"""
    if not message.strip():
        return ""
    
    try:
        # 응답 생성
        response = llm_chat.chat(message)
        return response
    
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"


def create_interface():
    """Gradio 인터페이스 생성"""
    
    # 커스텀 CSS
    custom_css = """
    .gradio-container {
        min-width: 1000px !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    footer {
        display: none !important;
    }
    """
    
    # ChatInterface 사용 (Gradio 4.0+)
    interface = gr.ChatInterface(
        fn=chat_fn,
        title="yhnanollm",
        description="""
        yhnanollm v1.0.0-beta
        """,
        examples=[
            "안녕하세요?",
            "React가 뭐야?",
            "파이썬이란?",
            "LoRA가 뭐야?",
            "함수란?",
        ],
        theme=gr.themes.Soft(),
        css=custom_css,
    )
    
    return interface


def main():
    """메인 함수"""
    # 모델 초기화
    initialize_model()
    
    # 인터페이스 생성 및 실행
    interface = create_interface()
    
    print("\n" + "="*60)
    print("🚀 웹 인터페이스 시작!")
    print("="*60)
    print("브라우저에서 http://localhost:7860 을 열어주세요")
    print("종료하려면 Ctrl+C를 누르세요")
    print("="*60 + "\n")
    
    # 서버 실행
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 외부 공유 비활성화
        show_error=True,
    )


if __name__ == "__main__":
    main()

