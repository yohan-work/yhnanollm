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
│   └── data-mini.json          # 샘플 한국어 학습 데이터 (35개 예시)
├── models/
│   └── lora-adapter/           # LoRA 어댑터 가중치 (학습 후)
├── scripts/
│   ├── finetune.py             # LoRA 파인튜닝 스크립트
│   └── merge_and_test.py       # 어댑터 병합 및 테스트 스크립트
├── chat.py                     # 대화형 채팅 인터페이스
├── requirements.txt            # Python 의존성
├── .gitignore
└── README.md
```

## 환경 설정

```bash
# 1. 가상환경 생성
python3 -m venv venv

# 2. 가상환경 활성화
source venv/bin/activate

# 3. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt
```

## 빠른 시작 (CLI)

학습이 완료되었다면 바로 대화를 시작하세요!

```bash
# 가상환경 활성화
source venv/bin/activate

# 채팅 시작!
python chat.py
```

**사용 예시**:

```
💬 You: 안녕하세요?
🤖 Bot: 안녕하세요! 무엇을 도와드릴까요?

💬 You: React가 뭐야?
🤖 Bot: React는 사용자 인터페이스를 만들기 위한 JavaScript 라이브러리입니다.
```

## 📖 처음부터 학습하기

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
# 1단계: 데이터 준비
python scripts/finetune.py

# 2단계: 실제 학습 실행 (토큰 불필요!)
mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train \
  --data data \
  --iters 100 \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --adapter-path models/lora-adapter
```

학습은 약 2-5분 소요됩니다. 완료되면 `models/lora-adapter/`에 어댑터 파일이 생성됩니다.

### Step 3: 대화형 채팅 시작

```bash
# 학습된 모델과 대화하기
python chat.py

# 옵션 지정
python chat.py --max-tokens 150

# 베이스 모델만 사용 (어댑터 없이)
python chat.py --no-adapter
```

**채팅 명령어**:

- `exit`, `quit`, `q` - 종료
- `history`, `h` - 대화 히스토리 보기
- `clear`, `c` - 히스토리 초기화
- `help`, `?` - 도움말

### Step 4: LoRA 어댑터 병합 (선택사항)

```bash
# LoRA 어댑터를 베이스 모델과 병합
mlx_lm.fuse \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --adapter-path models/lora-adapter \
  --save-path models/merged-model
```

### Step 5: 스크립트로 테스트 (선택사항)

```bash
# LoRA 어댑터와 함께 테스트
python scripts/merge_and_test.py --test-only

# 커스텀 프롬프트로 테스트
python scripts/merge_and_test.py --test-only --prompt "파이썬이란?"
```

### Step 6: MLX CLI로 직접 사용 (고급)

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

- 35개 샘플, 200 iterations: 약 4-5분 (M3 Pro 기준)
- 더 많은 데이터와 iterations를 사용할 경우 시간이 증가합니다
