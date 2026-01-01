"""
문서 처리 모듈
PDF 텍스트 추출 및 청킹
"""

from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
import docx


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

    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        텍스트 파일에서 텍스트 추출
        
        Args:
            txt_path: 텍스트 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            # EUC-KR 등 다른 인코딩 시도
            try:
                with open(txt_path, 'r', encoding='euc-kr') as f:
                    return f.read().strip()
            except Exception as e:
                raise Exception(f"텍스트 파일 인코딩 오류: {str(e)}")
        except Exception as e:
            raise Exception(f"텍스트 파일 읽기 실패: {str(e)}")

    def extract_text_from_docx(self, docx_path: str) -> str:
        """
        Word 파일에서 텍스트 추출
        
        Args:
            docx_path: Word 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            doc = docx.Document(docx_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text).strip()
        except Exception as e:
            raise Exception(f"Word 파일 텍스트 추출 실패: {str(e)}")
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, str]]:
        """
        텍스트를 의미 단위로 청크 분할 (Recursive Character Splitter 로직)
        
        Args:
            text: 분할할 텍스트
            filename: 파일명 (메타데이터용)
            
        Returns:
            청크 리스트 (각 청크는 text와 metadata 포함)
        """
        chunks = []
        
        # 분할 구분자 우선순위 (문단 -> 문장 -> 단어 -> 문자)
        # 한국어 문맥을 고려한 구분자
        separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
        
        def _split_text(text: str, separators: List[str]) -> List[str]:
            """재귀적으로 텍스트 분할"""
            final_chunks = []
            
            # 더 이상 쪼갤 구분자가 없으면 글자수대로 강제 분할
            if not separators:
                return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            
            separator = separators[0]
            new_separators = separators[1:]
            
            # 현재 구분자로 분할
            splits = text.split(separator)
            
            current_chunk = []
            current_length = 0
            
            for split in splits:
                # 구분자가 공백이 아닌 경우 다시 붙여줘야 함 (문장 부호 등)
                if separator not in ["\n\n", "\n", " "]:
                    split_len = len(split) + len(separator) 
                    text_piece = split + separator
                else:
                    split_len = len(split) + (1 if separator == " " else 0) # 줄바꿈은 길이 계산에서 제외하거나 1로 취급
                    text_piece = split + (separator if separator != "\n\n" else "\n") # 문단 구분은 줄바꿈 하나로 축소
                
                # 매우 긴 단락은 재귀적으로 다시 쪼갬
                if len(text_piece) > self.chunk_size:
                    if current_chunk:
                        joined_chunk = "".join(current_chunk).strip()
                        if joined_chunk:
                            final_chunks.append(joined_chunk)
                        current_chunk = []
                        current_length = 0
                    
                    # 재귀 호출
                    final_chunks.extend(_split_text(split, new_separators))
                    continue
                
                # 현재 청크에 추가해도 크기를 넘지 않으면 추가
                if current_length + split_len <= self.chunk_size:
                    current_chunk.append(text_piece)
                    current_length += split_len
                else:
                    # 크기를 넘으면 현재까지를 저장하고 새로 시작
                    if current_chunk:
                        joined_chunk = "".join(current_chunk).strip()
                        if joined_chunk:
                            final_chunks.append(joined_chunk)
                    
                    # 오버랩 처리를 위해 이전 청크의 뒷부분을 가져오면 좋겠지만,
                    # 여기서는 단순화를 위해 현재 조각부터 다시 시작
                    current_chunk = [text_piece]
                    current_length = split_len
            
            # 남은 조각 처리
            if current_chunk:
                joined_chunk = "".join(current_chunk).strip()
                if joined_chunk:
                    final_chunks.append(joined_chunk)
            
            return final_chunks

        # 로직 실행
        text_chunks = _split_text(text, separators)
        
        # 결과 포맷팅
        for i, chunk_text in enumerate(text_chunks):
            # 너무 짧은 청크는 제외 (노이즈 제거)
            if len(chunk_text) < 10:
                continue
                
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'filename': filename,
                    'chunk_id': i,
                    'start': -1, # 복잡해지므로 생략
                    'end': -1
                }
            })
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        문서 처리 (추출 + 청킹)
        지원 형식: .pdf, .txt, .docx
        
        Args:
            file_path: 파일 경로
            
        Returns:
            처리된 청크 리스트
        """
        path = Path(file_path)
        filename = path.name
        extension = path.suffix.lower()
        
        print(f"문서 처리 중: {filename} ({extension})")
        
        # 텍스트 추출
        if extension == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif extension == '.txt':
            text = self.extract_text_from_txt(file_path)
        elif extension == '.docx':
            text = self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}")
            
        print(f"텍스트 추출 완료: {len(text)} 문자")
        
        # 청킹
        chunks = self.chunk_text(text, filename)
        print(f"청킹 완료: {len(chunks)}개 청크")
        
        return chunks

