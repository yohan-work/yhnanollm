"""
RAG 프롬프트 템플릿 모듈
다양한 프롬프트 전략 제공
"""

from typing import Dict, Callable


class PromptTemplate:
    """프롬프트 템플릿 클래스"""
    
    def __init__(self, name: str, template_fn: Callable[[str, str], str], description: str):
        """
        Args:
            name: 템플릿 이름
            template_fn: (context, question) -> prompt 함수
            description: 템플릿 설명
        """
        self.name = name
        self.template_fn = template_fn
        self.description = description
    
    def format(self, context: str, question: str) -> str:
        """프롬프트 생성"""
        return self.template_fn(context, question)


# ===== 프롬프트 템플릿 정의 =====

def default_template(context: str, question: str) -> str:
    """기본 프롬프트 (현재 사용 중)"""
    return f"""다음 문서 내용을 참고해서 질문에 답변해주세요.

문서 내용:
{context}

질문: {question}

답변:"""


def detailed_template(context: str, question: str) -> str:
    """상세형 프롬프트 - 더 구체적인 지시사항"""
    return f"""당신은 문서 기반 질의응답 전문가입니다. 제공된 문서 내용을 바탕으로 질문에 정확하게 답변해주세요.

[지시사항]
1. 문서에 명시된 내용만을 기반으로 답변하세요
2. 답변은 명확하고 간결하게 작성하세요
3. 문서에서 답을 찾을 수 없다면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요
4. 가능한 경우 출처(페이지, 섹션)를 명시하세요

[참고 문서]
{context}

[질문]
{question}

[답변]"""


def step_by_step_template(context: str, question: str) -> str:
    """단계별 추론형 프롬프트 - 복잡한 질문에 적합"""
    return f"""다음 문서를 읽고 질문에 단계별로 답변해주세요.

[문서 내용]
{context}

[질문]
{question}

[답변 방식]
1. 문서에서 관련된 정보를 먼저 파악하세요
2. 정보를 종합하여 답변을 구성하세요
3. 최종 답변을 명확하게 제시하세요

[답변]"""


def concise_template(context: str, question: str) -> str:
    """간결형 프롬프트 - 짧고 빠른 답변"""
    return f"""문서를 참고하여 질문에 간단명료하게 답변하세요.

문서: {context}

질문: {question}

답변(한 문장으로):"""


def source_aware_template(context: str, question: str) -> str:
    """출처 강조형 프롬프트 - 출처 표시 강화"""
    return f"""제공된 문서를 기반으로 질문에 답변하고, 답변의 출처를 명시해주세요.

[참고 문서]
{context}

[질문]
{question}

[답변 형식]
- 답변: [여기에 답변 작성]
- 출처: [참고한 문서의 출처 표시]

[답변]"""


def korean_optimized_template(context: str, question: str) -> str:
    """한국어 최적화 프롬프트"""
    return f"""아래 문서 내용을 바탕으로 질문에 정확하고 자연스러운 한국어로 답변해주세요.

【문서】
{context}

【질문】
{question}

【답변】"""


def educational_template(context: str, question: str) -> str:
    """교육용 프롬프트 - 설명형 답변"""
    return f"""학생에게 설명하듯이 문서 내용을 바탕으로 친절하게 답변해주세요.

[학습 자료]
{context}

[학생의 질문]
{question}

[선생님의 답변]"""


def analytical_template(context: str, question: str) -> str:
    """분석형 프롬프트 - 심층 분석"""
    return f"""문서 내용을 분석하여 질문에 대한 종합적인 답변을 제공해주세요.

[분석 대상 문서]
{context}

[분석 요청]
{question}

[분석 결과]
- 핵심 내용:
- 세부 설명:
- 결론:"""


# ===== 템플릿 레지스트리 =====

PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "default": PromptTemplate(
        name="default",
        template_fn=default_template,
        description="기본 프롬프트 - 균형잡힌 답변"
    ),
    "detailed": PromptTemplate(
        name="detailed",
        template_fn=detailed_template,
        description="상세형 - 명확한 지시사항과 구조화된 답변"
    ),
    "step_by_step": PromptTemplate(
        name="step_by_step",
        template_fn=step_by_step_template,
        description="단계별 추론형 - 복잡한 질문에 적합"
    ),
    "concise": PromptTemplate(
        name="concise",
        template_fn=concise_template,
        description="간결형 - 짧고 빠른 답변"
    ),
    "source_aware": PromptTemplate(
        name="source_aware",
        template_fn=source_aware_template,
        description="출처 강조형 - 출처 표시 강화"
    ),
    "korean_optimized": PromptTemplate(
        name="korean_optimized",
        template_fn=korean_optimized_template,
        description="한국어 최적화 - 자연스러운 한국어 답변"
    ),
    "educational": PromptTemplate(
        name="educational",
        template_fn=educational_template,
        description="교육용 - 친절한 설명형 답변"
    ),
    "analytical": PromptTemplate(
        name="analytical",
        template_fn=analytical_template,
        description="분석형 - 심층 분석 및 구조화"
    )
}


def get_prompt_template(name: str = "default") -> PromptTemplate:
    """
    프롬프트 템플릿 가져오기
    
    Args:
        name: 템플릿 이름
        
    Returns:
        PromptTemplate 인스턴스
    """
    if name not in PROMPT_TEMPLATES:
        print(f"⚠️ 알 수 없는 프롬프트 템플릿: {name}")
        print(f"   기본 템플릿을 사용합니다.")
        return PROMPT_TEMPLATES["default"]
    
    return PROMPT_TEMPLATES[name]


def list_templates() -> Dict[str, str]:
    """사용 가능한 모든 템플릿 목록"""
    return {
        name: template.description
        for name, template in PROMPT_TEMPLATES.items()
    }


def format_prompt(context: str, question: str, template_name: str = "default") -> str:
    """
    프롬프트 생성 헬퍼 함수
    
    Args:
        context: 문서 컨텍스트
        question: 사용자 질문
        template_name: 템플릿 이름
        
    Returns:
        포맷된 프롬프트
    """
    template = get_prompt_template(template_name)
    return template.format(context, question)

