"""
WISDOM Lab - AI 리포트 생성 모듈
분석 데이터 중심 동적 구성, 128k 출력 지원
"""

import anthropic
from typing import Generator

_SYSTEM_PROMPTS = {
    "academic": """당신은 교육학, 사회과학 분야의 학술논문 작성을 지원하는 AI 연구 어시스턴트입니다.

[작성 규칙]
- 어조: 3인칭 객관적 서술 (~로 나타났다, ~임을 확인하였다, ~분석하였다)
- 문체: 격식체, 학술 용어 사용, 통계 수치를 정확히 인용
- 구조: 서론적 맥락 → 분석결과 기술 → 해석 → 시사점
- KCI·APA 스타일에 준하는 결과 기술 방식 적용
- 분량: A4 4~5쪽 기준 (본문 5,000~7,000자)

[레이아웃 규칙 — 반드시 준수]
- 대제목: "Ⅰ. 제목" 형식 (로마 숫자)
- 중제목: "1. 제목" 형식 (아라비아 숫자)
- 소제목: "(1) 제목" 형식
- 표: 정량 데이터는 반드시 마크다운 표로 먼저 제시 후 해석 작성""",

    "government": """당신은 정부부처 및 공공기관 보고서 작성을 지원하는 AI 어시스턴트입니다.

[작성 규칙]
- 어조: 개조식, 명사형 종결 (-함, -임, -됨, -필요함)
- 문체: 간결하고 명확한 서술, 불필요한 수식어 배제
- 구조: 핵심 사항을 먼저 제시(두괄식), 이후 세부 내용 나열
- 숫자와 통계치를 전면에 제시하고 해석을 간략히 덧붙임
- 분량: A4 4~5쪽 기준 (본문 5,000~7,000자)

[레이아웃 규칙 — 반드시 준수]
- 대제목: "Ⅰ. 제목" 형식 (로마 숫자)
- 중제목: "1. 제목" 형식 (아라비아 숫자)
- 소제목: "(1) 제목" 형식
- 표: 정량 데이터는 반드시 마크다운 표로 먼저 제시 후 해석 작성""",

    "research": """당신은 연구기관 및 싱크탱크의 연구보고서 작성을 지원하는 AI 어시스턴트입니다.

[작성 규칙]
- 어조: 혼합형 — 본문은 서술체(-하였다, -나타났다), 소결·시사점은 개조식(-함) 혼용
- 문체: 학술논문보다 평이하고 정부보고서보다 풍부한 서술
- 구조: 연구배경 → 분석결과 → 주요 발견 → 정책·실무 시사점
- 독자: 정책 담당자, 연구자, 실무자를 함께 고려
- 분량: A4 4~5쪽 기준 (본문 5,000~7,000자)

[레이아웃 규칙 — 반드시 준수]
- 대제목: "Ⅰ. 제목" 형식 (로마 숫자)
- 중제목: "1. 제목" 형식 (아라비아 숫자)
- 소제목: "(1) 제목" 형식
- 표: 정량 데이터는 반드시 마크다운 표로 먼저 제시 후 해석 작성""",
}

_DOC_LABELS = {
    "academic": "학술논문",
    "government": "정부보고서",
    "research": "연구보고서",
}


def _build_section_template(doc_type: str, has_text: bool, has_quant: bool) -> str:
    """수행된 분석에 따라 동적으로 보고서 섹션 구조를 생성"""
    if doc_type == "academic":
        sections = ["Ⅰ. 연구 개요 및 배경\n1. 분석 목적\n2. 분석 대상 및 방법"]
        if has_text:
            sections.append(
                "Ⅱ. 텍스트 분석 결과\n"
                "1. 핵심 키워드 빈도 분석\n(1) 상위 키워드 분포\n(2) TF-IDF 기반 주요어 분석\n"
                "2. 토픽 모델링 결과"
            )
        if has_quant:
            n = "Ⅲ" if has_text else "Ⅱ"
            sections.append(
                f"{n}. 정량 분석 결과\n"
                "1. 기술통계 및 분포 특성\n"
                "2. 통계 검증 결과 및 해석"
            )
        last = ("Ⅳ" if has_text and has_quant else ("Ⅲ" if has_text or has_quant else "Ⅱ"))
        sections.append(
            f"{last}. 종합 논의 및 결론\n"
            "1. 주요 발견 요약\n2. 연구·교육적 시사점\n3. 제언"
        )

    elif doc_type == "government":
        sections = ["Ⅰ. 분석 개요\n1. 분석 목적 및 배경\n2. 분석 자료 현황"]
        body_n = "Ⅱ"
        subsections = []
        if has_text:
            subsections.append("1. 키워드 및 텍스트 분석\n(1) 핵심 키워드 현황\n(2) 주제 영역별 분석")
        if has_quant:
            idx = len(subsections) + 1
            subsections.append(f"{idx}. 정량 분석 결과")
        if subsections:
            sections.append(f"{body_n}. 주요 분석 결과\n" + "\n".join(subsections))
        sections.append(
            "Ⅲ. 정책적 시사점\n1. 현황 진단\n2. 개선 방향\n\nⅣ. 제언\n1. 단기 과제\n2. 중장기 과제"
        )

    else:  # research
        sections = ["Ⅰ. 연구 배경 및 목적\n1. 연구의 필요성\n2. 분석 범위 및 방법"]
        body_subsections = []
        if has_text:
            body_subsections.append("1. 텍스트 분석\n(1) 주요 키워드 분석\n(2) 토픽 분석")
        if has_quant:
            idx = len(body_subsections) + 1
            body_subsections.append(f"{idx}. 정량 분석")
        if body_subsections:
            sections.append("Ⅱ. 분석 결과\n" + "\n".join(body_subsections))
        sections.append(
            "Ⅲ. 주요 발견 및 해석\n1. 핵심 발견 사항\n2. 결과 해석\n\n"
            "Ⅳ. 시사점 및 결론\n1. 실무·정책 시사점\n2. 결론 및 향후 과제"
        )

    return "\n\n".join(sections)


def _build_data_summary(freq_df, tfidf_df, topics_df, quant_results: list, meta: dict) -> str:
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
        top10 = freq_df.head(10)
        rows = "\n".join(f"| {r.iloc[0]} | {r.iloc[1]}회 |" for _, r in top10.iterrows())
        parts.append(
            f"[키워드 빈도 상위 10개]\n"
            f"| 키워드 | 빈도 |\n|--------|------|\n{rows}\n"
        )

    if tfidf_df is not None and not tfidf_df.empty:
        top10t = tfidf_df.head(10)
        rows = "\n".join(f"| {r.iloc[0]} | {r.iloc[1]:.4f} |" for _, r in top10t.iterrows())
        parts.append(
            f"[TF-IDF 상위 10개]\n"
            f"| 키워드 | TF-IDF |\n|--------|--------|\n{rows}\n"
        )

    if topics_df is not None and not topics_df.empty:
        topic_lines = []
        for _, row in topics_df.iterrows():
            topic_lines.append(
                f"| {row.get('토픽 번호','')} | "
                f"{row.get('대표 키워드','')} | "
                f"{row.get('핵심 단어','')} |"
            )
        header = "| 토픽 번호 | 대표 키워드 | 핵심 단어 |\n|-----------|-------------|-----------|"
        parts.append(f"[NMF 토픽 분석 결과]\n{header}\n" + "\n".join(topic_lines) + "\n")

    if quant_results:
        parts.append("[정량 분석 결과 목록]")
        for i, r in enumerate(quant_results, 1):
            parts.append(f"--- {i}. {r.get('label', r.get('method', ''))} ---\n{r.get('summary_text', '')}\n")

    return "\n".join(parts)


def generate_report_stream(
    freq_df,
    tfidf_df,
    topics_df,
    quant_results: list,
    meta: dict,
    api_key: str,
    doc_type: str = "research",
) -> Generator[str, None, None]:
    """스트리밍 방식으로 AI 리포트를 생성합니다. quant_results는 정량 분석 결과 dict 목록."""
    if not api_key:
        yield "API 키가 설정되지 않았습니다."
        return

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        yield f"API 클라이언트 초기화 오류: {e}"
        return

    has_text = any([
        freq_df is not None and not freq_df.empty,
        tfidf_df is not None and not tfidf_df.empty,
        topics_df is not None and not topics_df.empty,
    ])
    has_quant = bool(quant_results)

    system_prompt = _SYSTEM_PROMPTS.get(doc_type, _SYSTEM_PROMPTS["research"])
    sections = _build_section_template(doc_type, has_text, has_quant)
    doc_label = _DOC_LABELS.get(doc_type, "연구보고서")
    data_summary = _build_data_summary(
        freq_df, tfidf_df, topics_df, quant_results, {**meta, "doc_type": doc_type}
    )

    quant_count = len(quant_results) if quant_results else 0
    quant_names = ", ".join(r["method"] for r in quant_results) if quant_results else ""
    analysis_note = ""
    if has_text and has_quant:
        analysis_note = f"텍스트 분석과 정량 분석({quant_count}개: {quant_names}) 결과를 모두 포함하여 작성하세요."
    elif has_text:
        analysis_note = "텍스트 분석 결과를 중심으로 보고서를 작성하세요. 정량 분석 결과가 없으므로 관련 섹션은 생략합니다."
    elif has_quant:
        analysis_note = f"정량 분석 결과({quant_count}개: {quant_names})를 중심으로 보고서를 작성하세요. 텍스트 분석 결과가 없으므로 관련 섹션은 생략합니다."

    user_message = (
        f"아래 분석 데이터를 바탕으로 **{doc_label}** 형식의 리포트를 작성해 주세요.\n\n"
        f"### 분석 데이터\n{data_summary}\n\n"
        f"### 요청 리포트 구조\n{sections}\n\n"
        "### 작성 지침 (반드시 준수)\n"
        f"1. **분석 범위**: {analysis_note}\n"
        "2. **분량**: 반드시 A4 4~5쪽 분량, 최소 6,000자 이상 작성하세요. "
        "각 섹션마다 최소 3~5문단 이상 충실히 서술하세요. 절대 짧게 요약하지 마세요.\n"
        "3. **제목 체계**: 대제목 'Ⅰ.', 중제목 '1.', 소제목 '(1)' 형식을 반드시 지키세요.\n"
        "4. **시작**: 리포트 맨 앞에 '# 제목' 줄을 추가하지 마세요. 반드시 'Ⅰ.'로 바로 시작하세요.\n"
        "5. **표**: 정량 데이터는 반드시 마크다운 표(| 구분 | 내용 |)로 먼저 제시하고 그 아래에 해석을 서술하세요.\n"
        "6. **연속 서술**: 문단 사이 빈 줄 없이 이어서 쓰세요.\n"
        "7. **구체성**: 분석 결과의 수치, 키워드, 토픽명을 반드시 구체적으로 인용하며 서술하세요.\n"
        "8. **완결**: 요청된 섹션 구조 전체를 빠짐없이 완결하세요. 절대 생략하거나 요약 처리하지 마세요.\n"
        "9. **해석 깊이**: 단순 나열이 아니라 결과의 의미, 시사점, 배경을 깊이 있게 해석하세요.\n"
        "10. **통합**: 여러 분석 결과가 있는 경우 서로 연계하여 통합적으로 해석하세요."
    )

    # 128k 확장 출력 시도, 실패 시 일반 출력
    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            betas=["output-128k-2025-02-19"],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception:
        try:
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=8000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"\n\n오류 발생: {str(e)}"
