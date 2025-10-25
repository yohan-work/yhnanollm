#!/usr/bin/env python3
"""
MLX LoRA Fine-tuning Script for yhnanollm v1.0.0-beta
Apple Silicon (M3 Pro) optimized
"""

import argparse
import json
from pathlib import Path
from mlx_lm import load, lora


def prepare_data(data_path):
    """Alpaca 형식 데이터를 MLX 형식으로 변환"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Alpaca 형식을 MLX가 요구하는 형식으로 변환
    formatted_data = []
    for item in data:
        text = f"### Instruction:\n{item['instruction']}\n"
        if item.get('input'):
            text += f"### Input:\n{item['input']}\n"
        text += f"### Response:\n{item['output']}"
        formatted_data.append({"text": text})
    
    return formatted_data


def main():
    parser = argparse.ArgumentParser(description="MLX LoRA Fine-tuning")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit",
                        help="Base model path or HF repo (토큰 불필요한 공개 모델)")
    parser.add_argument("--data", type=str, default="data/data-mini.json",
                        help="Training data path (Alpaca format)")
    parser.add_argument("--output", type=str, default="models/lora-adapter",
                        help="Output directory for LoRA weights")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    
    args = parser.parse_args()
    
    print(f"🚀 yhnanollm v1.0.0-beta Fine-tuning 시작")
    print(f"📦 모델: {args.model}")
    print(f"📊 데이터: {args.data}")
    print(f"💾 출력: {args.output}")
    
    # 출력 디렉토리 생성
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # 데이터 준비
    print("\n📝 데이터 로딩 중...")
    train_data = prepare_data(args.data)
    print(f"✅ {len(train_data)}개의 학습 샘플 준비 완료")
    
    # MLX LoRA 파인튜닝
    # 실제 학습을 위해서는 mlx-lm의 lora.py를 직접 호출하거나
    # 명령줄에서 mlx_lm.lora를 사용할 수 있습니다
    print("\n⚙️  LoRA 학습 파라미터:")
    print(f"  - Iterations: {args.iters}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - LoRA Rank: {args.lora_rank}")
    print(f"  - Batch Size: {args.batch_size}")
    
    # 데이터를 임시 파일로 저장 (MLX-LM이 요구하는 형식)
    temp_data_path = Path(args.output) / "train.jsonl"
    with open(temp_data_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n💡 MLX-LM CLI를 사용하여 학습을 시작합니다:")
    print(f"   mlx_lm.lora \\")
    print(f"     --model {args.model} \\")
    print(f"     --train \\")
    print(f"     --data {temp_data_path} \\")
    print(f"     --iters {args.iters} \\")
    print(f"     --learning-rate {args.learning_rate} \\")
    print(f"     --batch-size {args.batch_size} \\")
    print(f"     --adapter-path {args.output}")
    print(f"\n   # LoRA rank는 config로 설정하거나 기본값(8) 사용")
    
    print("\n" + "="*60)
    print("⚠️  실제 학습을 위해 위 명령어를 터미널에서 실행하세요.")
    print("="*60)


if __name__ == "__main__":
    main()

