#!/usr/bin/env python3
"""
yhnanollm v1.0.0-beta - CLI 대화형 인터페이스
로컬 LLM과 대화할 수 있는 간단한 채팅 프로그램
"""

import argparse
import sys
from pathlib import Path
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors


class LocalLLMChat:
    def __init__(self, model_path, adapter_path=None, max_tokens=100, temperature=0.3, repetition_penalty=1.1, top_p=0.9):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.model = None
        self.tokenizer = None
        self.history = []
        
    def load_model(self):
        """모델 로드"""
        print("🔄 모델 로딩 중...")
        if self.adapter_path:
            print(f"   베이스 모델: {self.model_path}")
            print(f"   LoRA 어댑터: {self.adapter_path}")
            self.model, self.tokenizer = load(
                self.model_path,
                adapter_path=self.adapter_path
            )
        else:
            print(f"   모델: {self.model_path}")
            self.model, self.tokenizer = load(self.model_path)
        print("✅ 모델 로드 완료!\n")
    
    def format_prompt(self, user_input):
        """프롬프트 포맷팅"""
        return f"### Instruction:\n{user_input}\n\n반드시 순수 한국어로만 답변하세요. 다른 언어를 절대 사용하지 마세요.\n### Response:"
    
    def chat(self, user_input, skip_format=False):
        """사용자 입력에 대한 응답 생성
        
        Args:
            user_input: 사용자 입력 또는 이미 포맷된 프롬프트
            skip_format: True면 포맷팅을 건너뛰고 user_input을 그대로 사용
        """
        # RAG에서 이미 포맷된 프롬프트를 받은 경우 포맷팅 건너뛰기
        if skip_format:
            prompt = user_input
        else:
            prompt = self.format_prompt(user_input)
        
        try:
            # sampler와 logits_processors 생성
            sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
            logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty)
            
            # stream_generate 사용하여 응답 생성
            response_parts = []
            for response_obj in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                response_parts.append(response_obj.text)
            
            response = "".join(response_parts)
            
            # 히스토리에 저장
            self.history.append({"user": user_input, "assistant": response})
            
            return response
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ LLM 생성 오류:\n{error_detail}")
            return f"❌ 오류: {e}"
    
    def show_history(self):
        """대화 히스토리 출력"""
        if not self.history:
            print("아직 대화 히스토리가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("📜 대화 히스토리")
        print("="*60)
        for i, conv in enumerate(self.history, 1):
            print(f"\n[{i}] 사용자: {conv['user']}")
            print(f"    어시스턴트: {conv['assistant']}")
        print("="*60 + "\n")
    
    def clear_history(self):
        """히스토리 초기화"""
        self.history = []
        print("✅ 대화 히스토리가 초기화되었습니다.\n")
    
    def print_help(self):
        """도움말 출력"""
        print("\n" + "="*60)
        print("📖 사용 가능한 명령어")
        print("="*60)
        print("  exit, quit, q     : 프로그램 종료")
        print("  history, h        : 대화 히스토리 보기")
        print("  clear, c          : 히스토리 초기화")
        print("  help, ?           : 도움말 보기")
        print("="*60 + "\n")
    
    def run(self):
        """메인 대화 루프"""
        # 모델 로드
        self.load_model()
        
        # 환영 메시지
        print("="*60)
        print("🤖 yhnanollm v1.0.0-beta - 로컬 LLM 채팅")
        print("="*60)
        print("무엇이든 물어보세요! (종료: 'exit', 도움말: 'help')\n")
        
        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n💬 You: ").strip()
                
                # 빈 입력 처리
                if not user_input:
                    continue
                
                # 명령어 처리
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 안녕히 가세요!")
                    break
                
                elif user_input.lower() in ['history', 'h']:
                    self.show_history()
                    continue
                
                elif user_input.lower() in ['clear', 'c']:
                    self.clear_history()
                    continue
                
                elif user_input.lower() in ['help', '?']:
                    self.print_help()
                    continue
                
                # 응답 생성
                print("🤖 Bot: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 안녕히 가세요!")
                break
            
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="yhnanollm CLI 채팅 인터페이스"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="베이스 모델 경로 또는 HuggingFace repo"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="models/lora-adapter",
        help="LoRA 어댑터 경로 (선택사항)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="최대 생성 토큰 수"
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="어댑터 없이 베이스 모델만 사용"
    )
    
    args = parser.parse_args()
    
    # 어댑터 경로 확인
    adapter_path = None if args.no_adapter else args.adapter
    
    # 어댑터가 없으면 경고
    if adapter_path and not Path(adapter_path).exists():
        print(f"⚠️  경고: 어댑터 경로를 찾을 수 없습니다: {adapter_path}")
        print("   베이스 모델만 사용합니다.\n")
        adapter_path = None
    
    # 채팅 시작
    chat = LocalLLMChat(
        model_path=args.model,
        adapter_path=adapter_path,
        max_tokens=args.max_tokens
    )
    
    chat.run()


if __name__ == "__main__":
    main()

