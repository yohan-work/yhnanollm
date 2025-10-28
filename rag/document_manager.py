"""
문서 메타데이터 관리 모듈
업로드된 문서의 정보와 사용 통계를 관리
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class DocumentManager:
    def __init__(self, metadata_path: str = "doc_metadata.json"):
        """
        문서 메타데이터 관리자 초기화
        
        Args:
            metadata_path: 메타데이터 JSON 파일 경로
        """
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """메타데이터 파일 로드"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("메타데이터 파일 손상, 새로 생성합니다.")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """메타데이터 파일 저장"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def add_document(self, filename: str, file_size: int, chunk_count: int) -> None:
        """
        문서 메타데이터 추가
        
        Args:
            filename: 파일명
            file_size: 파일 크기 (바이트)
            chunk_count: 청크 수
        """
        self.metadata[filename] = {
            'filename': filename,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': file_size,
            'file_size_kb': round(file_size / 1024, 2),
            'chunk_count': chunk_count,
            'search_count': 0
        }
        self._save_metadata()
        print(f"문서 메타데이터 저장: {filename}")
    
    def get_document(self, filename: str) -> Optional[Dict]:
        """
        특정 문서 메타데이터 조회
        
        Args:
            filename: 파일명
            
        Returns:
            문서 메타데이터 또는 None
        """
        return self.metadata.get(filename)
    
    def get_all_documents(self) -> List[Dict]:
        """
        모든 문서 메타데이터 조회
        
        Returns:
            문서 메타데이터 리스트 (최신순)
        """
        docs = list(self.metadata.values())
        # 업로드 시간 역순 정렬 (최신순)
        docs.sort(key=lambda x: x['upload_time'], reverse=True)
        return docs
    
    def delete_document(self, filename: str) -> bool:
        """
        문서 메타데이터 삭제
        
        Args:
            filename: 파일명
            
        Returns:
            삭제 성공 여부
        """
        if filename in self.metadata:
            del self.metadata[filename]
            self._save_metadata()
            print(f"문서 메타데이터 삭제: {filename}")
            return True
        return False
    
    def increment_search_count(self, filename: str) -> None:
        """
        문서 검색 횟수 증가
        
        Args:
            filename: 파일명
        """
        if filename in self.metadata:
            self.metadata[filename]['search_count'] += 1
            self._save_metadata()
    
    def clear_all(self) -> None:
        """모든 메타데이터 삭제"""
        self.metadata = {}
        self._save_metadata()
        print("모든 문서 메타데이터 삭제 완료")
    
    def get_total_chunks(self) -> int:
        """전체 청크 수 반환"""
        return sum(doc['chunk_count'] for doc in self.metadata.values())
    
    def get_document_count(self) -> int:
        """문서 수 반환"""
        return len(self.metadata)
    
    def get_filenames(self) -> List[str]:
        """모든 파일명 리스트 반환"""
        return list(self.metadata.keys())

