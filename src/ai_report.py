"""
WISDOM Lab - AI 리포트 생성 모듈
문서유형(학술논문 / 정부보고서 / 연구보고서) × 어조 선택 지원
"""

import anthropic
from typing import Generator

# ── 문서 유형별 시스템 프롬프트 ───────────────────────────────
_SYSTEM_PROMPTS = {
    "academic": """당신은 교육학, 사회과학 분야의 학술논문 작성을 지원하는 AI 연구 어시스턴트입니다.
분석 결과를 바탕으로 학술논문 본문 수준의 텍스트를 작성합니다.
- 어조: 3인칭 객관적 서술 (예: ~로 나타났다, ~임을 확인하였다, ~분석하였다)
- 문체: 격식체, 학술 용어 사용, 통계 수치를 정확히 인용
- 구조: 서론적 맥락 → 분석결과 기술 → 해석 → 시사점 순서로 전개
- KCI·APA 스타일에 준하는 결과 기술 방식 적용""",

    "government": """당신은 정부부처 및 공공기관 보고서 작성을 지원하는 AI 어시스턴트입니다.
분석 결과를 정부보고서 형식으로 작성합니다.
- 어조: 개조식, 명사형 종결 (-함, -임, -됨, -필요함)
- 문체: 간결하고 명확한 서술, 불필요한 수식어 배제
- 구조: 핵심 사항을 먼저 제시(두괄식), 이후 세부 내용을 불릿으로 나열
- 숫자와 통계치를 전면에 제시하고 해석을 간략히 덧붙임""",

    "research": """당신은 연구기관 및 싱크탱크의 연구보고서 작성을 지원하는 AI 어시스턴트입니다.
분석 결과를 연구보고서 형식으로 작성합니다.
- 어조: 혼합형 — 본문은 서술체(-하였다, -나타났다), 소결·시사점은 개조식(-함) 혼용
- 문체: 학술논문보다 평이하고 정부보고서보다 풍부한 서술
- 구조: 연구배경 → 분석결과 → 주요 발견 → 정책·실무 시사점
- 독자: 정책 담당자, 연구자, 실무자를 함께 고려""",
}

_SECTION_TEMPLATES = {
    "academic": """## 1. 분석 개요 및 연구 배경
## 2. 텍스트 분석 결과
## 3. 정량 분석 결과
## 4. 종합 논의
## 5. 결론 및 제언""",

    "government": """## 1. 분석 개요
## 2. 주요 분석 결과
## 3. 정책적 시사점
## 4. 제언""",

    "research": """## 1. 연구 배경 및 목적
## 2. 분석 결과 요약
## 3. 주요 발견 및 해석
## 4. 시사점
## 5. 결론""",
}

_DOC_LABELS = {
    "academic": "학술논문",
    "government": "정부보고서",
    "research": "연구보고서",
}


def _build_data_summary(freq_df, tfidf_df, topics_df, quant_summary: str, meta: dict) -> str:
    parts = []
    subject = meta.get("subject", "")
    doc_label = _DOC_LABELS.get(meta.get("doc_type", "research"), "연구보고서")
    parts.append(
        f"[분석 개요]\n"
        f"- 문서 유형: {doc_label}\n"
        f"- 분석 주제: {subject or '미입력'}\n"
        f"- 총 문서 수: {meta.get('total_docs', 0)}건\n"
    )

    if freq_df is not None and not freq_df.empty:
        top5 = freq_df.head(5)
        kw_str = ", ".join(f"{r.iloc[0]}({r.iloc[1]}회)" for _, r in top5.iterrows())
        parts.append(f"[키워드 빈도 상위 5개]\n{kw_str}\n")

    if tfidf_df is not None and not tfidf_df.empty:
        top5t = tfidf_df.head(5)
        tf_str = ", ".join(f"{r.iloc[0]}({r.iloc[1]:.3f})" for _, r in top5t.iterrows())
        parts.append(f"[TF-IDF 상위 5개]\n{tf_str}\n")

    if topics_df is not None and not topics_df.empty:
        topic_lines = []
        for _, row in topics_df.iterrows():
            topic_lines.append(
                f"  - {row.get('토픽 번호','')}: "
                f"대표어={row.get('대표 키워드','')}, "
                f"키워드={row.get('핵심 단어','')}"
            )
        parts.append("[NMF 토픽 분석 결과]\n" + "\n".join(topic_lines) + "\n")

    if quant_summary:
        parts.append(f"[정량 분석 요약]\n{quant_summary}\n")

    return "\n".join(parts)


def generate_report_stream(
    freq_df,
    tfidf_df,
    topics_df,
    quant_summary: str,
    meta: dict,
    api_key: str,
    doc_type: str = "research",
) -> Generator[str, None, None]:
    """
    스트리밍 방식으로 AI 리포트를 생성합니다.
    doc_type: 'academic' | 'government' | 'research'
    """
    if not api_key:
        yield "API 키가 설정되지 않았습니다."
        return

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        yield f"API 클라이언트 초기화 오류: {e}"
        return

    system_prompt = _SYSTEM_PROMPTS.get(doc_type, _SYSTEM_PROMPTS["research"])
    sections = _SECTION_TEMPLATES.get(doc_type, _SECTION_TEMPLATES["research"])
    doc_label = _DOC_LABELS.get(doc_type, "연구보고서")
    data_summary = _build_data_summary(freq_df, tfidf_df, topics_df, quant_summary, {**meta, "doc_type": doc_type})

    user_message = (
        f"아래 분석 데이터를 바탕으로 **{doc_label}** 형식의 리포트를 작성해 주세요.\n\n"
        f"{data_summary}\n"
        f"[요청 리포트 구조]\n{sections}\n\n"
        "각 섹션을 충실히 작성하되, 분석 결과에서 확인된 수치와 키워드를 구체적으로 인용하세요. "
        "통계적으로 유의미한 결과는 반드시 강조하고, 연구·정책적 의미를 제시하세요."
    )

    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=3000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\n오류 발생: {str(e)}"
