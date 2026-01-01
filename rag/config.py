"""
RAG 시스템 설정 관리 모듈
모든 튜닝 가능한 파라미터를 중앙에서 관리
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class RAGConfig:
    """RAG 시스템 설정 클래스"""
    
    # 문서 처리 파라미터
    chunk_size: int = 500  # 청크 크기 (문자 수)
    chunk_overlap: int = 50  # 청크 간 중복 크기
    
    # 검색 파라미터
    top_k: int = 3  # 검색할 문서 수
    similarity_threshold: float = 0.0  # 유사도 임계값 (0.0~1.0)
    
    # 임베딩 모델
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    
    # 프롬프트 설정
    prompt_template: str = "default"  # default, detailed, step_by_step, concise
    
    # LLM 설정
    max_tokens: int = 150
    temperature: float = 0.3
    
    # 기타 설정
    persist_directory: str = "chroma_db"
    metadata_path: str = "doc_metadata.json"
    
    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환"""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'embedding_model': self.embedding_model,
            'prompt_template': self.prompt_template,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'persist_directory': self.persist_directory,
            'metadata_path': self.metadata_path
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RAGConfig':
        """딕셔너리로부터 설정 생성"""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """설정을 JSON 파일로 저장"""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ 설정 저장 완료: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RAGConfig':
        """JSON 파일에서 설정 로드"""
        config_path = Path(path)
        
        if not config_path.exists():
            print(f"⚠️ 설정 파일이 없습니다: {path}")
            print("   기본 설정을 사용합니다.")
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        print(f"✅ 설정 로드 완료: {path}")
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """설정 정보를 보기 좋게 출력"""
        return f"""RAGConfig(
  문서 처리:
    - chunk_size: {self.chunk_size}
    - chunk_overlap: {self.chunk_overlap}
  검색:
    - top_k: {self.top_k}
    - similarity_threshold: {self.similarity_threshold}
  모델:
    - embedding_model: {self.embedding_model}
    - prompt_template: {self.prompt_template}
  LLM:
    - max_tokens: {self.max_tokens}
    - temperature: {self.temperature}
)"""


# 사전 정의된 설정 프리셋
PRESETS = {
    "default": RAGConfig(
        chunk_size=500,
        chunk_overlap=50,
        top_k=3,
        prompt_template="default"
    ),
    "precise": RAGConfig(
        chunk_size=300,
        chunk_overlap=100,
        top_k=2,
        similarity_threshold=0.3,
        prompt_template="detailed"
    ),
    "comprehensive": RAGConfig(
        chunk_size=800,
        chunk_overlap=100,
        top_k=5,
        prompt_template="detailed"
    ),
    "fast": RAGConfig(
        chunk_size=400,
        chunk_overlap=30,
        top_k=1,
        prompt_template="concise"
    )
}


def get_preset(name: str) -> RAGConfig:
    """
    사전 정의된 설정 프리셋 가져오기
    
    Args:
        name: 프리셋 이름 (default, precise, comprehensive, fast)
        
    Returns:
        RAGConfig 인스턴스
    """
    if name not in PRESETS:
        print(f"⚠️ 알 수 없는 프리셋: {name}")
        print(f"   사용 가능한 프리셋: {list(PRESETS.keys())}")
        return PRESETS["default"]
    
    return PRESETS[name]


def list_presets() -> dict:
    """사용 가능한 모든 프리셋 반환"""
    return {
        name: {
            'chunk_size': config.chunk_size,
            'top_k': config.top_k,
            'prompt_template': config.prompt_template
        }
        for name, config in PRESETS.items()
    }

