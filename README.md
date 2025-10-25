# yhnanollm v1.0.0-beta

MLX 기반 한국어 파인튜닝
Apple Silicon (M3 Pro) 최적화된 TinyLlama 모델 LoRA 파인튜닝

## 프로젝트 개요

- **베이스 모델**: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- **방법론**: LoRA (Low-Rank Adaptation)
- **데이터 형식**: Alpaca JSON
- **목표**: 짧은 한국어 응답 파인튜닝 테스트

## 프로젝트 구조

```
yhnanollm/
├── data/
│   └── data-mini.json          # 샘플 한국어 학습 데이터 (5개 예시)
├── models/
│   ├── lora-adapter/           # LoRA 어댑터 가중치 (학습 후)
│   └── merged-model/           # 병합된 최종 모델 (선택사항)
├── scripts/
│   ├── finetune.py             # LoRA 파인튜닝 스크립트
│   └── merge_and_test.py       # 어댑터 병합 및 테스트 스크립트
├── requirements.txt            # Python 의존성
├── .gitignore
└── README.md
```

## 사용 방법

### Step 1: 데이터 준비

샘플 데이터가 이미 `data/data-mini.json`에 준비되어 있습니다.

**데이터 형식 (Alpaca JSON)**:

```json
[
  {
    "instruction": "질문 또는 명령",
    "input": "추가 입력 (선택사항)",
    "output": "원하는 응답"
  }
]
```

### Step 2: LoRA 파인튜닝

```bash
# 기본 설정으로 학습
python scripts/finetune.py

# 커스텀 설정으로 학습
python scripts/finetune.py \
  --model mlx-community/TinyLlama-1.1B-Chat-v1.0 \
  --data data/data-mini.json \
  --output models/lora-adapter \
  --iters 100 \
  --learning-rate 1e-5 \
  --lora-rank 8 \
  --batch-size 2
```

**실제 학습 실행 (MLX-LM CLI 사용, 토큰 불필요)**:

```bash
# 토큰 없이 공개 모델 다운로드 및 학습
mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train \
  --data models/lora-adapter/train.jsonl \
  --iters 100 \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --adapter-path models/lora-adapter
```

### Step 3: LoRA 어댑터 병합

```bash
# LoRA 어댑터를 베이스 모델과 병합
mlx_lm.fuse \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --adapter-path models/lora-adapter \
  --save-path models/merged-model
```

### Step 4: 모델 테스트

```bash
# LoRA 어댑터와 함께 테스트
python scripts/merge_and_test.py

# 커스텀 프롬프트로 테스트
python scripts/merge_and_test.py --prompt "파이썬이란?"

# 병합 건너뛰고 테스트만 실행
python scripts/merge_and_test.py --test-only
```

### Step 5: 간단한 추론 테스트

```bash
# 병합된 모델로 직접 추론
mlx_lm.generate \
  --model models/merged-model \
  --prompt "안녕하세요?" \
  --max-tokens 50

# LoRA 어댑터를 사용한 추론
mlx_lm.generate \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --adapter-path models/lora-adapter \
  --prompt "안녕하세요?" \
  --max-tokens 50
```

## 학습 파라미터

| 파라미터          | 기본값 | 설명                        |
| ----------------- | ------ | --------------------------- |
| `--iters`         | 100    | 학습 반복 횟수              |
| `--learning-rate` | 1e-5   | 학습률                      |
| `--lora-rank`     | 8      | LoRA 랭크 (낮을수록 가벼움) |
| `--batch-size`    | 2      | 배치 크기                   |

## 팁 & 트러블슈팅

### M3 Pro 최적화

- MLX는 Apple Silicon에 최적화되어 있어 GPU 가속을 자동으로 활용합니다
- 메모리가 부족하면 `--batch-size`를 1로 낮추세요
- LoRA rank를 4로 낮추면 메모리 사용량이 줄어듭니다

### 데이터 추가

더 많은 학습 데이터를 추가하려면 `data/data-mini.json`에 Alpaca 형식으로 추가하세요.

### 학습 시간

- 5개 샘플, 100 iterations: 약 2-5분 (M3 Pro 기준)
- 더 많은 데이터와 iterations를 사용할 경우 시간이 증가합니다
