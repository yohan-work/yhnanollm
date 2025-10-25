#!/usr/bin/env python3
"""
MLX LoRA Merge & Test Script for yhnanollm v1.0.0-beta
LoRA 어댑터를 베이스 모델과 병합하고 테스트합니다.
"""

import argparse
from pathlib import Path
from mlx_lm import load, generate


def test_model(model, tokenizer, prompt):
    """모델 테스트 및 생성"""
    print(f"\n💬 프롬프트: {prompt}")
    print("🤖 응답: ", end="", flush=True)
    
    # 응답 생성 (MLX-LM 최신 API 사용)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        verbose=False
    )
    
    print(response)
    return response


def main():
    parser = argparse.ArgumentParser(description="MLX LoRA Merge & Test")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit",
                        help="Base model path or HF repo (토큰 불필요한 공개 모델)")
    parser.add_argument("--adapter", type=str, default="models/lora-adapter",
                        help="LoRA adapter path")
    parser.add_argument("--output", type=str, default="models/merged-model",
                        help="Output directory for merged model")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip merging and only test")
    parser.add_argument("--prompt", type=str, default="안녕하세요?",
                        help="Test prompt")
    
    args = parser.parse_args()
    
    print(f"🔧 yhnanollm v1.0.0-beta LoRA Merge & Test")
    print(f"📦 베이스 모델: {args.model}")
    print(f"🎯 LoRA 어댑터: {args.adapter}")
    
    # LoRA 어댑터가 있는지 확인
    adapter_path = Path(args.adapter)
    if not adapter_path.exists() or not any(adapter_path.glob("*.safetensors")):
        print("\n⚠️  경고: LoRA 어댑터를 찾을 수 없습니다.")
        print(f"   {args.adapter} 경로에 학습된 어댑터가 있는지 확인하세요.")
        print("   먼저 scripts/finetune.py를 실행하여 학습을 진행하세요.")
        return
    
    if not args.test_only:
        # 모델 병합
        print(f"\n🔄 LoRA 어댑터 병합 중...")
        print(f"💾 출력 경로: {args.output}")
        
        # 출력 디렉토리 생성
        Path(args.output).mkdir(parents=True, exist_ok=True)
        
        # MLX-LM을 사용한 병합
        print("\n💡 MLX-LM CLI를 사용하여 병합:")
        print(f"   mlx_lm.fuse \\")
        print(f"     --model {args.model} \\")
        print(f"     --adapter-path {args.adapter} \\")
        print(f"     --save-path {args.output}")
        
        print("\n" + "="*60)
        print("⚠️  실제 병합을 위해 위 명령어를 터미널에서 실행하세요.")
        print("="*60)
    
    # 모델 테스트
    print(f"\n🧪 모델 테스트 시작...")
    
    try:
        # LoRA 어댑터와 함께 모델 로드
        print(f"📥 모델 로딩 중... (LoRA 포함)")
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter
        )
        print("✅ 모델 로딩 완료")
        
        # 테스트 프롬프트들
        test_prompts = [
            args.prompt,
            "MLX가 뭐야?",
            "파이썬이란?"
        ]
        
        print("\n" + "="*60)
        print("테스트 결과:")
        print("="*60)
        
        for prompt in test_prompts:
            test_model(model, tokenizer, prompt)
            print("-"*60)
        
        print("\n✨ 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("   모델과 어댑터 경로를 확인하세요.")


if __name__ == "__main__":
    main()

