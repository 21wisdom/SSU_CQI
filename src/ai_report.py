"""
WISDOM Lab - AI 리포트 생성 모듈
문서유형(학술논문 / 정부보고서 / 연구보고서) × 어조 선택 지원
출력 분량: A4 4쪽 이상 (최소 6,000자)
"""

import anthropic
from typing import Generator

# ── 문서 유형별 시스템 프롬프트 ───────────────────────────────
_SYSTEM_PROMPTS = {
    "academic": """당신은 교육학, 사회과학 분야의 학술논문 작성을 지원하는 AI 연구 어시스턴트입니다.
분석 결과를 바탕으로 학술논문 본문 수준의 텍스트를 작성합니다.

[작성 규칙]
- 어조: 3인칭 객관적 서술 (~로 나타났다, ~임을 확인하였다, ~분석하였다)
- 문체: 격식체, 학술 용어 사용, 통계 수치를 정확히 인용
- 구조: 서론적 맥락 → 분석결과 기술 → 해석 → 시사점
- KCI·APA 스타일에 준하는 결과 기술 방식 적용
- 분량: A4 5쪽 이상 (본문 6,000~8,000자)

[레이아웃 규칙 — 반드시 준수]
- 대제목: "Ⅰ. 제목" 형식 (로마 숫자)
- 중제목: "1. 제목" 형식 (아라비아 숫자)
- 소제목: "(1) 제목" 형식
- 본문: 항목 중심 서술 (개조식 + 설명 병행), 문단 사이 빈 줄 없음
- 표: 정량 데이터는 반드시 마크다운 표로 먼저 제시 후 해석 작성
  예시: | 구분 | 빈도 | 비율 |
        |------|------|------|
        | 키워드A | 25 | 18.5% |""",

    "government": """당신은 정부부처 및 공공기관 보고서 작성을 지원하는 AI 어시스턴트입니다.
분석 결과를 정부보고서 형식으로 작성합니다.

[작성 규칙]
- 어조: 개조식, 명사형 종결 (-함, -임, -됨, -필요함)
- 문체: 간결하고 명확한 서술, 불필요한 수식어 배제
- 구조: 핵심 사항을 먼저 제시(두괄식), 이후 세부 내용 나열
- 숫자와 통계치를 전면에 제시하고 해석을 간략히 덧붙임
- 분량: A4 5쪽 이상 (본문 6,000~8,000자)

[레이아웃 규칙 — 반드시 준수]
- 대제목: "Ⅰ. 제목" 형식 (로마 숫자)
- 중제목: "1. 제목" 형식 (아라비아 숫자)
- 소제목: "(1) 제목" 형식
- 본문: 항목 중심 서술 (개조식 + 설명 병행), 문단 사이 빈 줄 없음
- 표: 정량 데이터는 반드시 마크다운 표로 먼저 제시 후 해석 작성
  예시: | 구분 | 빈도 | 비율 |
        |------|------|------|
        | 키워드A | 25 | 18.5% |""",

    "research": """당신은 연구기관 및 싱크탱크의 연구보고서 작성을 지원하는 AI 어시스턴트입니다.
분석 결과를 연구보고서 형식으로 작성합니다.

[작성 규칙]
- 어조: 혼합형 — 본문은 서술체(-하였다, -나타났다), 소결·시사점은 개조식(-함) 혼용
- 문체: 학술논문보다 평이하고 정부보고서보다 풍부한 서술
- 구조: 연구배경 → 분석결과 → 주요 발견 → 정책·실무 시사점
- 독자: 정책 담당자, 연구자, 실무자를 함께 고려
- 분량: A4 5쪽 이상 (본문 6,000~8,000자)

[레이아웃 규칙 — 반드시 준수]
- 대제목: "Ⅰ. 제목" 형식 (로마 숫자)
- 중제목: "1. 제목" 형식 (아라비아 숫자)
- 소제목: "(1) 제목" 형식
- 본문: 항목 중심 서술 (개조식 + 설명 병행), 문단 사이 빈 줄 없음
- 표: 정량 데이터는 반드시 마크다운 표로 먼저 제시 후 해석 작성
  예시: | 구분 | 빈도 | 비율 |
        |------|------|------|
        | 키워드A | 25 | 18.5% |""",
}

_DOC_LABELS = {
    "academic": "학술논문",
    "government": "정부보고서",
    "research": "연구보고서",
}


def _build_section_template(doc_type: str, has_text: bool, has_quant: bool) -> str:
    """분석 데이터 유무에 따라 섹션 구조를 동적으로 생성"""
    if has_text and has_quant:
        if doc_type == "academic":
            return """\
Ⅰ. 연구 개요 및 배경
1. 분석 목적
2. 분석 대상 및 방법

Ⅱ. 텍스트 분석 결과
1. 핵심 키워드 빈도 분석
(1) 상위 키워드 분포
(2) TF-IDF 기반 주요어 분석
2. 토픽 모델링 결과

Ⅲ. 정량 분석 결과
1. 기술통계 및 주요 수치
2. 통계 검증 결과 해석

Ⅳ. 통합 논의
1. 텍스트·정량 결과 교차 해석
2. 주요 발견 요약

Ⅴ. 결론 및 시사점
1. 연구·교육적 시사점
2. 제언"""
        elif doc_type == "government":
            return """\
Ⅰ. 분석 개요
1. 분석 목적 및 배경
2. 분석 자료 현황

Ⅱ. 텍스트 분석 결과
1. 핵심 키워드 현황
2. 주제 영역별 분석

Ⅲ. 정량 분석 결과
1. 주요 수치 현황
2. 통계 검증 결과

Ⅳ. 종합 시사점
1. 현황 진단
2. 개선 방향

Ⅴ. 제언
1. 단기 과제
2. 중장기 과제"""
        else:
            return """\
Ⅰ. 연구 배경 및 목적
1. 연구의 필요성
2. 분석 범위 및 방법

Ⅱ. 텍스트 분석 결과
1. 주요 키워드 분석
2. 토픽 분석

Ⅲ. 정량 분석 결과
1. 기술통계 및 주요 지표
2. 통계 검증 결과

Ⅳ. 통합 논의 및 해석
1. 핵심 발견 사항
2. 결과 교차 해석

Ⅴ. 시사점 및 결론
1. 실무·정책 시사점
2. 결론 및 향후 과제"""

    elif has_quant and not has_text:
        if doc_type == "academic":
            return """\
Ⅰ. 연구 개요
1. 분석 목적
2. 분석 변수 및 방법

Ⅱ. 정량 분석 결과
1. 기술통계 요약
2. 주요 통계 검증 결과
(1) 분석별 결과 기술
(2) 통계적 유의성 해석

Ⅲ. 종합 논의
1. 주요 발견 요약
2. 결과 해석 및 함의

Ⅳ. 결론 및 시사점
1. 연구·교육적 시사점
2. 제언"""
        elif doc_type == "government":
            return """\
Ⅰ. 분석 개요
1. 분석 목적 및 배경
2. 분석 자료 현황

Ⅱ. 정량 분석 결과
1. 주요 수치 현황
2. 분석별 세부 결과
(1) 결과 기술
(2) 통계 해석

Ⅲ. 정책적 시사점
1. 현황 진단
2. 개선 방향

Ⅳ. 제언
1. 단기 과제
2. 중장기 과제"""
        else:
            return """\
Ⅰ. 연구 배경 및 목적
1. 연구의 필요성
2. 분석 범위 및 방법

Ⅱ. 정량 분석 결과
1. 기술통계 및 주요 지표
2. 분석별 세부 결과
(1) 결과 기술
(2) 통계적 유의성 해석

Ⅲ. 주요 발견 및 해석
1. 핵심 발견 사항
2. 결과 해석

Ⅳ. 시사점 및 결론
1. 실무·정책 시사점
2. 결론 및 향후 과제"""

    else:  # text only
        if doc_type == "academic":
            return """\
Ⅰ. 연구 개요 및 배경
1. 분석 목적
2. 분석 대상 및 방법

Ⅱ. 텍스트 분석 결과
1. 핵심 키워드 빈도 분석
(1) 상위 키워드 분포
(2) TF-IDF 기반 주요어 분석
2. 토픽 모델링 결과

Ⅲ. 종합 논의
1. 주요 발견 요약
2. 결과 해석 및 함의

Ⅳ. 결론 및 시사점
1. 연구·교육적 시사점
2. 제언"""
        elif doc_type == "government":
            return """\
Ⅰ. 분석 개요
1. 분석 목적 및 배경
2. 분석 자료 현황

Ⅱ. 텍스트 분석 결과
1. 핵심 키워드 현황
(1) 상위 키워드 분포
(2) 주제 영역별 분석
2. 주요 토픽 분석

Ⅲ. 정책적 시사점
1. 현황 진단
2. 개선 방향

Ⅳ. 제언
1. 단기 과제
2. 중장기 과제"""
        else:
            return """\
Ⅰ. 연구 배경 및 목적
1. 연구의 필요성
2. 분석 범위 및 방법

Ⅱ. 텍스트 분석 결과
1. 주요 키워드 분석
(1) 빈도 분석
(2) TF-IDF 분석
2. 토픽 분석

Ⅲ. 주요 발견 및 해석
1. 핵심 발견 사항
2. 결과 해석

Ⅳ. 시사점 및 결론
1. 실무·정책 시사점
2. 결론 및 향후 과제"""


def _build_data_summary(freq_df, tfidf_df, topics_df, quant_results: list, meta: dict) -> str:
    parts = []
    subject = meta.get("subject", "")
    doc_label = _DOC_LABELS.get(meta.get("doc_type", "research"), "연구보고서")
    has_text = freq_df is not None and not freq_df.empty
    has_quant = bool(quant_results)

    parts.append(
        f"[분석 개요]\n"
        f"- 문서 유형: {doc_label}\n"
        f"- 분석 주제: {subject or '미입력'}\n"
        f"- 총 문서 수: {meta.get('total_docs', 0)}건\n"
        f"- 텍스트 분석: {'수행' if has_text else '미수행'}\n"
        f"- 정량 분석: {len(quant_results)}개 수행\n"
    )

    if has_text:
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

    if has_quant:
        for entry in quant_results:
            label = entry.get("label", entry.get("method", ""))
            summary = entry.get("summary_text", "")
            result_df = entry.get("result_df")
            if summary:
                parts.append(f"[정량분석 - {label}]\n{summary}\n")
            elif result_df is not None:
                parts.append(f"[정량분석 - {label}]\n{result_df.to_string(index=False)}\n")

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
    """스트리밍 방식으로 AI 리포트를 생성합니다."""
    if not api_key:
        yield "API 키가 설정되지 않았습니다."
        return

    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        yield f"API 클라이언트 초기화 오류: {e}"
        return

    has_text = freq_df is not None and not freq_df.empty
    has_quant = bool(quant_results)

    system_prompt = _SYSTEM_PROMPTS.get(doc_type, _SYSTEM_PROMPTS["research"])
    sections = _build_section_template(doc_type, has_text, has_quant)
    doc_label = _DOC_LABELS.get(doc_type, "연구보고서")
    data_summary = _build_data_summary(
        freq_df, tfidf_df, topics_df, quant_results, {**meta, "doc_type": doc_type}
    )

    user_message = (
        f"아래 분석 데이터를 바탕으로 **{doc_label}** 형식의 리포트를 작성해 주세요.\n\n"
        f"### 분석 데이터\n{data_summary}\n\n"
        f"### 요청 리포트 구조\n{sections}\n\n"
        "### 작성 지침 (반드시 준수)\n"
        "1. **분량**: 반드시 A4 5쪽 이상, 최소 6,000자 이상 작성하세요. "
        "각 섹션마다 최소 3~5문단 이상 충실히 서술하세요. 절대 짧게 요약하지 마세요.\n"
        "2. **제목 체계**: 대제목 'Ⅰ.', 중제목 '1.', 소제목 '(1)' 형식을 반드시 지키세요.\n"
        "3. **시작**: 리포트 맨 앞에 '# 제목' 줄을 추가하지 마세요. 반드시 'Ⅰ.'로 바로 시작하세요.\n"
        "4. **표**: 정량 데이터는 반드시 마크다운 표(| 구분 | 내용 |)로 먼저 제시하고 그 아래에 해석을 서술하세요.\n"
        "5. **연속 서술**: 문단 사이 빈 줄 없이 이어서 쓰세요.\n"
        "6. **구체성**: 분석 결과의 수치, 키워드, 토픽명을 반드시 구체적으로 인용하며 서술하세요.\n"
        "7. **완결**: Ⅰ.부터 마지막 섹션까지 빠짐없이 완결하세요. 중간에 절대 생략하거나 요약 처리하지 마세요.\n"
        "8. **해석 깊이**: 단순 나열이 아니라 결과의 의미, 시사점, 배경을 깊이 있게 해석하세요.\n"
        "9. **데이터 중심**: 제공된 분석 데이터에 포함된 항목만 다루세요. "
        "수행하지 않은 분석(예: 텍스트 분석 미수행 시 키워드·토픽 섹션)은 포함하지 마세요."
    )

    def _do_stream(max_tokens: int, use_beta: bool):
        kwargs = dict(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        if use_beta:
            kwargs["betas"] = ["output-128k-2025-02-19"]
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    try:
        for text in _do_stream(16000, use_beta=True):
            yield text
    except Exception:
        try:
            for text in _do_stream(8000, use_beta=False):
                yield text
        except Exception as e2:
            yield f"\n\n오류 발생: {str(e2)}"
