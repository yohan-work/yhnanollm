"""
문서 처리 모듈
PDF 텍스트 추출 및 청킹
"""

from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        문서 처리기 초기화
        
        Args:
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 중복 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num}]\n{page_text}"
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"PDF 텍스트 추출 실패: {str(e)}")
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, str]]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            filename: 파일명 (메타데이터용)
            
        Returns:
            청크 리스트 (각 청크는 text와 metadata 포함)
        """
        chunks = []
        text_length = len(text)
        
        start = 0
        chunk_id = 0
        
        while start < text_length:
            # 청크 끝 위치 계산
            end = start + self.chunk_size
            
            # 마지막 청크가 아니면 문장 경계에서 자르기
            if end < text_length:
                # 가능한 문장 끝 찾기
                for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'filename': filename,
                        'chunk_id': chunk_id,
                        'start': start,
                        'end': end
                    }
                })
                chunk_id += 1
            
            # 다음 청크 시작 위치 (overlap 적용)
            start = end - self.chunk_overlap if end < text_length else text_length
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        PDF 처리 (추출 + 청킹)
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            처리된 청크 리스트
        """
        filename = Path(pdf_path).name
        
        print(f"PDF 처리 중: {filename}")
        
        # 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        print(f"텍스트 추출 완료: {len(text)} 문자")
        
        # 청킹
        chunks = self.chunk_text(text, filename)
        print(f"청킹 완료: {len(chunks)}개 청크")
        
        return chunks

