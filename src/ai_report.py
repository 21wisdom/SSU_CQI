"""
Claude API 기반 AI 리포트 생성 모듈
스트리밍 방식으로 6섹션 리포트 생성
"""

import os
from typing import Generator


REPORT_SYSTEM_PROMPT = """당신은 대학 강의 품질 개선(CQI: Continuous Quality Improvement)을 전문으로 하는
교육학 전문가이자 데이터 분석가입니다. 강의평가 텍스트 분석 결과를 바탕으로
교수자와 교육 담당자에게 실질적으로 도움이 되는 심층 리포트를 작성합니다.

리포트는 다음 6개 섹션으로 구성하세요:

## 1. 종합 진단 (Executive Summary)
- 전반적인 학습자 반응 패턴 요약
- 주요 긍정/개선 키워드 해석
- 전체 강의 품질 진단 (강점/약점)

## 2. 학습자 페르소나 분석
- 키워드 패턴 기반 3~4개 학습자 유형 도출
- 각 유형의 특성, 요구사항, 학습 행동 기술

## 3. 진로·취업 연계성 분석
- 실무 역량 관련 키워드 분석
- 진로 연계 강화 방안 제안

## 4. 핵심역량 연계성 분석
- 창의융합·의사소통·문제해결·자기주도 역량과의 연계
- 역량별 강화 전략

## 5. 강의 개선 추천
- 교수학습 방법 개선안 (3가지 이상)
- 강의 내용 개선안 (3가지 이상)
- 우선순위와 실행 로드맵

## 6. 참고자료 및 최신 동향
- 관련 교육학 이론 및 연구 동향
- 유사 사례 및 벤치마킹 방향

각 섹션을 구체적이고 실행 가능한 내용으로 작성하세요."""


def build_analysis_summary(freq_df, tfidf_df, topics_df, meta: dict) -> str:
    """분석 결과를 텍스트 요약으로 변환"""
    lines = [f"=== 강의평가 분석 결과 요약 ==="]
    lines.append(f"총 응답 수: {meta.get('total_docs', 'N/A')}개")

    if freq_df is not None and not freq_df.empty:
        top_kw = freq_df.head(15)["키워드"].tolist()
        lines.append(f"\n【빈도 상위 키워드】: {', '.join(top_kw)}")

    if tfidf_df is not None and not tfidf_df.empty:
        top_tfidf = tfidf_df.head(10)["키워드"].tolist()
        lines.append(f"\n【TF-IDF 상위 키워드】: {', '.join(top_tfidf)}")

    if topics_df is not None and not topics_df.empty:
        lines.append("\n【토픽 모델링 결과】")
        for _, row in topics_df.iterrows():
            lines.append(f"  - {row['토픽 번호']}: {row['핵심 단어']}")

    if meta.get("subject"):
        lines.append(f"\n과목명: {meta['subject']}")

    return "\n".join(lines)


def generate_report_stream(
    freq_df, tfidf_df, topics_df, meta: dict, api_key: str
) -> Generator[str, None, None]:
    """
    Claude API 스트리밍으로 AI 리포트 생성
    Yields: 텍스트 청크
    """
    try:
        import anthropic
    except ImportError:
        yield "anthropic 패키지가 설치되어 있지 않습니다. pip install anthropic"
        return

    if not api_key:
        yield "API 키가 설정되지 않았습니다. .env 파일에 ANTHROPIC_API_KEY를 입력하세요."
        return

    client = anthropic.Anthropic(api_key=api_key)
    analysis_summary = build_analysis_summary(freq_df, tfidf_df, topics_df, meta)

    user_message = f"""다음 강의평가 분석 결과를 바탕으로 CQI 개선 리포트를 작성해주세요.

{analysis_summary}

위 분석 결과를 종합하여 6개 섹션으로 구성된 상세한 CQI 리포트를 작성해주세요.
각 섹션은 교수자가 바로 실행할 수 있는 구체적인 내용을 포함해야 합니다."""

    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=REPORT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\n오류가 발생했습니다: {str(e)}"
