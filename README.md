# yhnanollm v1.0.0-beta

<img width="956" height="876" alt="스크린샷 2025-10-27 오전 10 29 52" src="https://github.com/user-attachments/assets/0b95ca6a-eca8-4d20-aa6d-82bdef39c009" />

MLX 기반 한국어 파인튜닝 + RAG (문서 기반 질의응답)
Apple Silicon (M3 Pro) 최적화된 로컬 LLM

## 프로젝트 개요

- **베이스 모델**: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- **방법론**: LoRA (Low-Rank Adaptation) + RAG (Retrieval-Augmented Generation)
- **데이터 형식**: Alpaca JSON
- **목표**: 짧은 한국어 응답 파인튜닝 + PDF 문서 기반 질의응답

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
├── rag/                        # RAG 모듈 (NEW!)
│   ├── __init__.py
│   ├── document_processor.py  # PDF 처리 및 청킹
│   ├── vector_store.py        # ChromaDB 벡터 DB
│   └── rag_chain.py           # RAG 로직
├── chat.py                     # CLI 대화형 인터페이스
├── app.py                      # Gradio 웹 인터페이스 with RAG
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

### 방법 1: Web Interface with RAG

```bash
# 가상환경 활성화
source venv/bin/activate

# 웹 UI 시작
python app.py
```

브라우저에서 **http://localhost:7860** 을 열면 채팅 UI가 나타납니다!

- **PDF 업로드**: 문서를 업로드하여 내용 기반 질의응답
- **RAG 모드**: 문서 참고 ON/OFF 토글
- **문서 관리**: 업로드된 문서 확인 및 삭제

**사용 방법**:

1. PDF 파일 업로드
2. "RAG 모드 (문서 참고)" 체크박스 활성화
3. 문서 내용에 대해 질문하기!

### 방법 2: CLI 터미널

```bash
# 가상환경 활성화
source venv/bin/activate

# CLI 채팅 시작
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

### RAG 기술 스택

- **ChromaDB**: 로컬 벡터 데이터베이스 (문서 저장)
- **sentence-transformers**: 텍스트 임베딩 모델
- **PyPDF2**: PDF 텍스트 추출
- **청킹**: 문서를 500자 단위로 분할하여 저장

### 문서 관리

- **업로드**: PDF 파일 선택하여 업로드
- **확인**: "문서 정보"에서 저장된 청크 수 확인
- **삭제**: "모든 문서 삭제" 버튼으로 DB 초기화

### 예상 사용 사례

1. **기술 문서 QA**: 제품 매뉴얼, API 문서
2. **보고서 요약**: 회사 보고서, 논문
3. **교육 자료**: 교과서, 강의 자료
4. **내부 지식베이스**: 사내 규정, 가이드
