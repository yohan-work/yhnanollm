#!/usr/bin/env python3
"""
RAG 파라미터 튜닝 및 실험 도구
다양한 파라미터 조합을 테스트하고 최적 설정을 찾습니다
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from itertools import product
import sys

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import LocalLLMChat
from rag import DocumentProcessor, VectorStore, RAGChain, DocumentManager
from rag.config import RAGConfig
from rag.prompts import list_templates


class RAGTuner:
    """RAG 파라미터 튜닝 클래스"""
    
    def __init__(
        self,
        model_path: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        adapter_path: str = "models/lora-adapter",
        test_data_path: str = None
    ):
        """
        Args:
            model_path: LLM 모델 경로
            adapter_path: LoRA 어댑터 경로
            test_data_path: 테스트 질문 JSON 파일 경로
        """
        self.model_path = model_path
        self.adapter_path = adapter_path if Path(adapter_path).exists() else None
        self.test_data_path = test_data_path
        
        # LLM 초기화
        print("🔄 LLM 모델 로딩 중...")
        self.llm_chat = LocalLLMChat(
            model_path=self.model_path,
            adapter_path=self.adapter_path,
            max_tokens=150
        )
        self.llm_chat.load_model()
        print("✅ LLM 모델 준비 완료")
        
        # 테스트 질문 로드
        self.test_questions = self._load_test_questions()
    
    def _load_test_questions(self):
        """테스트 질문 세트 로드"""
        if self.test_data_path and Path(self.test_data_path).exists():
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('questions', [])
        
        # 기본 테스트 질문
        return [
            {
                "question": "이 문서의 주요 내용은 무엇인가요?",
                "category": "summary"
            },
            {
                "question": "신청 방법은 어떻게 되나요?",
                "category": "process"
            },
            {
                "question": "신청 자격 요건이 무엇인가요?",
                "category": "requirements"
            },
            {
                "question": "지원 금액은 얼마인가요?",
                "category": "specific"
            },
            {
                "question": "문의처는 어디인가요?",
                "category": "contact"
            }
        ]
    
    def run_experiment(
        self,
        chunk_sizes: list = [300, 500, 800],
        chunk_overlaps: list = [50, 100],
        top_ks: list = [1, 3, 5],
        prompt_templates: list = ["default", "detailed", "korean_optimized"],
        output_dir: str = "experiments"
    ):
        """
        파라미터 조합별 실험 실행
        
        Args:
            chunk_sizes: 테스트할 청크 크기 리스트
            chunk_overlaps: 테스트할 청크 오버랩 리스트
            top_ks: 테스트할 top-k 값 리스트
            prompt_templates: 테스트할 프롬프트 템플릿 리스트
            output_dir: 결과 저장 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"tuning_results_{timestamp}.csv"
        
        # 결과 저장용 CSV 헤더
        headers = [
            'chunk_size', 'chunk_overlap', 'top_k', 'prompt_template',
            'avg_answer_length', 'documents_used', 'avg_distance',
            'total_time', 'config_score'
        ]
        
        results = []
        total_combinations = len(chunk_sizes) * len(chunk_overlaps) * len(top_ks) * len(prompt_templates)
        current = 0
        
        print(f"\n🔬 총 {total_combinations}개 조합 테스트 시작")
        print(f"📊 테스트 질문 수: {len(self.test_questions)}")
        print("="*60)
        
        # 모든 파라미터 조합 테스트
        for chunk_size, chunk_overlap, top_k, prompt_template in product(
            chunk_sizes, chunk_overlaps, top_ks, prompt_templates
        ):
            current += 1
            print(f"\n[{current}/{total_combinations}] 테스트 중...")
            print(f"  chunk_size={chunk_size}, overlap={chunk_overlap}, "
                  f"top_k={top_k}, prompt={prompt_template}")
            
            try:
                # RAG 시스템 초기화
                config = RAGConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    prompt_template=prompt_template
                )
                
                result = self._test_configuration(config)
                result.update({
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'top_k': top_k,
                    'prompt_template': prompt_template
                })
                
                results.append(result)
                
                print(f"  ✅ 완료 - 점수: {result['config_score']:.2f}")
            
            except Exception as e:
                print(f"  ❌ 오류: {str(e)}")
                continue
        
        # 결과를 CSV로 저장
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✅ 결과 저장 완료: {results_file}")
        
        # 최적 설정 분석 및 추천
        self._analyze_results(results, output_path / f"analysis_{timestamp}.json")
        
        return results
    
    def _test_configuration(self, config: RAGConfig):
        """
        특정 설정으로 테스트 실행
        
        Args:
            config: RAGConfig 인스턴스
            
        Returns:
            테스트 결과 딕셔너리
        """
        import time
        
        # 임시 벡터 DB 생성 (실제 문서가 이미 업로드되어 있다고 가정)
        vector_store = VectorStore(persist_directory=config.persist_directory)
        doc_manager = DocumentManager(metadata_path=config.metadata_path)
        
        # RAG 체인 생성
        rag_chain = RAGChain(
            vector_store=vector_store,
            llm_chat=self.llm_chat,
            document_manager=doc_manager,
            top_k=config.top_k,
            prompt_template=config.prompt_template,
            similarity_threshold=config.similarity_threshold
        )
        
        # 문서가 없으면 경고
        if vector_store.get_document_count() == 0:
            print("  ⚠️ 벡터 DB에 문서가 없습니다. 먼저 문서를 업로드하세요.")
            return {
                'avg_answer_length': 0,
                'documents_used': 0,
                'avg_distance': 0,
                'total_time': 0,
                'config_score': 0
            }
        
        # 모든 테스트 질문에 대해 실행
        answer_lengths = []
        documents_counts = []
        distances = []
        
        start_time = time.time()
        
        for test_item in self.test_questions:
            question = test_item['question']
            
            try:
                answer, metadatas, stats = rag_chain.answer(question, use_rag=True)
                
                answer_lengths.append(len(answer))
                
                if stats:
                    documents_counts.append(stats.get('documents_found', 0))
                    distances.append(stats.get('avg_distance', 0))
            
            except Exception as e:
                print(f"    질문 실패: {question[:30]}... - {str(e)}")
                continue
        
        total_time = time.time() - start_time
        
        # 평균 계산
        avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        avg_documents_used = sum(documents_counts) / len(documents_counts) if documents_counts else 0
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        # 설정 점수 계산 (간단한 휴리스틱)
        # 점수가 높을수록 좋음
        config_score = self._calculate_score(
            avg_answer_length,
            avg_documents_used,
            avg_distance,
            total_time
        )
        
        return {
            'avg_answer_length': round(avg_answer_length, 2),
            'documents_used': round(avg_documents_used, 2),
            'avg_distance': round(avg_distance, 4),
            'total_time': round(total_time, 2),
            'config_score': round(config_score, 2)
        }
    
    def _calculate_score(self, answer_length, documents_used, avg_distance, total_time):
        """
        설정 점수 계산
        
        휴리스틱:
        - 답변 길이가 적절한지 (너무 짧거나 길지 않은지)
        - 문서 활용도 (검색된 문서 수)
        - 검색 정확도 (평균 거리가 낮을수록 좋음)
        - 속도 (빠를수록 좋음)
        """
        # 답변 길이 점수 (100~500자가 이상적)
        length_score = 10.0
        if 100 <= answer_length <= 500:
            length_score = 10.0
        elif answer_length < 100:
            length_score = answer_length / 10
        else:
            length_score = max(0, 10 - (answer_length - 500) / 100)
        
        # 문서 활용 점수
        doc_score = min(10.0, documents_used * 3)
        
        # 검색 정확도 점수 (거리가 낮을수록 좋음)
        distance_score = max(0, 10 - avg_distance * 10)
        
        # 속도 점수 (5초 이내가 이상적)
        speed_score = max(0, 10 - total_time / 5)
        
        # 가중 평균
        total_score = (
            length_score * 0.3 +
            doc_score * 0.3 +
            distance_score * 0.3 +
            speed_score * 0.1
        )
        
        return total_score
    
    def _analyze_results(self, results, output_path):
        """결과 분석 및 최적 설정 추천"""
        if not results:
            print("⚠️ 분석할 결과가 없습니다.")
            return
        
        # 점수별 정렬
        sorted_results = sorted(results, key=lambda x: x['config_score'], reverse=True)
        
        # 상위 3개 설정
        top_3 = sorted_results[:3]
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_configurations': len(results),
            'top_configurations': top_3,
            'recommendations': self._generate_recommendations(top_3, results)
        }
        
        # JSON으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("📊 분석 결과")
        print("="*60)
        print(f"\n🏆 최고 점수 설정:")
        best = top_3[0]
        print(f"  - 점수: {best['config_score']:.2f}")
        print(f"  - chunk_size: {best['chunk_size']}")
        print(f"  - chunk_overlap: {best['chunk_overlap']}")
        print(f"  - top_k: {best['top_k']}")
        print(f"  - prompt_template: {best['prompt_template']}")
        
        print(f"\n💡 추천 사항:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
        
        print(f"\n📄 상세 분석: {output_path}")
    
    def _generate_recommendations(self, top_configs, all_results):
        """최적 설정 추천 생성"""
        recommendations = []
        
        if not top_configs:
            return ["결과가 부족하여 추천할 수 없습니다."]
        
        best = top_configs[0]
        
        # 청크 크기 추천
        recommendations.append(
            f"청크 크기는 {best['chunk_size']}자가 최적입니다."
        )
        
        # Top-K 추천
        recommendations.append(
            f"검색 문서 수는 {best['top_k']}개가 적합합니다."
        )
        
        # 프롬프트 템플릿 추천
        recommendations.append(
            f"'{best['prompt_template']}' 프롬프트 템플릿을 사용하세요."
        )
        
        # 속도 vs 품질 트레이드오프
        if best['total_time'] > 10:
            recommendations.append(
                "응답 시간이 다소 길 수 있습니다. 빠른 응답이 필요하면 top_k를 줄이세요."
            )
        
        return recommendations


def create_test_questions_template(output_path: str = "test_questions.json"):
    """테스트 질문 템플릿 생성"""
    template = {
        "questions": [
            {
                "question": "이 문서의 주요 내용은 무엇인가요?",
                "category": "summary",
                "expected_topics": ["문서 개요", "핵심 내용"]
            },
            {
                "question": "신청 방법은 어떻게 되나요?",
                "category": "process",
                "expected_topics": ["절차", "단계"]
            }
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 테스트 질문 템플릿 생성: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG 파라미터 튜닝 도구")
    parser.add_argument("--model", type=str, 
                       default="mlx-community/Llama-3.2-1B-Instruct-4bit",
                       help="LLM 모델 경로")
    parser.add_argument("--adapter", type=str, 
                       default="models/lora-adapter",
                       help="LoRA 어댑터 경로")
    parser.add_argument("--test-questions", type=str,
                       help="테스트 질문 JSON 파일 경로")
    parser.add_argument("--output", type=str, 
                       default="experiments",
                       help="결과 저장 디렉토리")
    parser.add_argument("--create-template", action="store_true",
                       help="테스트 질문 템플릿 생성")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 테스트 (파라미터 조합 축소)")
    
    args = parser.parse_args()
    
    if args.create_template:
        create_test_questions_template()
        return
    
    # 튜너 초기화
    tuner = RAGTuner(
        model_path=args.model,
        adapter_path=args.adapter,
        test_data_path=args.test_questions
    )
    
    # 실험 파라미터 설정
    if args.quick:
        # 빠른 테스트용 축소 파라미터
        chunk_sizes = [500]
        chunk_overlaps = [50]
        top_ks = [2, 3]
        prompt_templates = ["default", "korean_optimized"]
    else:
        # 전체 테스트
        chunk_sizes = [300, 500, 800]
        chunk_overlaps = [50, 100]
        top_ks = [1, 3, 5]
        prompt_templates = ["default", "detailed", "korean_optimized", "concise"]
    
    # 실험 실행
    results = tuner.run_experiment(
        chunk_sizes=chunk_sizes,
        chunk_overlaps=chunk_overlaps,
        top_ks=top_ks,
        prompt_templates=prompt_templates,
        output_dir=args.output
    )
    
    print("\n" + "="*60)
    print("✅ RAG 튜닝 완료!")
    print("="*60)


if __name__ == "__main__":
    main()

