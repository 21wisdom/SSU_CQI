"""
WISDOM Lab - 연구자 개인용 텍스트·정량 데이터 분석 플랫폼
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="WISDOM Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 WISDOM Lab")
    st.caption("연구자 개인용 텍스트·정량 분석 플랫폼")
    st.markdown("---")
    st.markdown("""
**지원 분석**
- 📂 텍스트 + 정량 데이터 업로드
- 🔤 키워드·TF-IDF·워드클라우드
- 🕸️ 공출현 네트워크
- 🧩 NMF 토픽 모델링
- 📊 기술통계·상관·T검정·ANOVA·회귀·로짓
- 🤖 AI 리포트 (학술논문 / 정부보고서 / 연구보고서)
    """)
    st.markdown("---")
    st.caption("v2.0.0 | WISDOM Lab © 2025")

# ── 세션 상태 초기화 ─────────────────────────────────────────
defaults = {
    "preprocessed": None,
    "freq_df": None,
    "tfidf_df": None,
    "topics_df": None,
    "nmf_model": None,
    "quant_df": None,       # 정량 데이터프레임
    "quant_summary": "",    # AI 리포트용 정량 결과 요약
    "text_df": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 탭 구성 ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 데이터 업로드",
    "🔤 텍스트 분석",
    "🕸️ 네트워크 분석",
    "🧩 토픽 모델링",
    "📊 정량 분석",
    "🤖 AI 리포트",
])

# ══════════════════════════════════════════════════════════════
# 탭 1: 데이터 업로드
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("📂 데이터 업로드")
    st.markdown("텍스트 데이터(정성)와 수치 데이터(정량)를 함께 또는 따로 업로드할 수 있습니다.")

    upload_mode = st.radio(
        "업로드 방식",
        ["단일 파일 (텍스트+수치 통합)", "파일 분리 (텍스트 / 수치 별도)"],
        horizontal=True,
    )

    # ── 단일 파일 모드 ────────────────────────────────────────
    if upload_mode == "단일 파일 (텍스트+수치 통합)":
        use_sample = st.checkbox("🔍 샘플 데이터로 시작하기")
        df = None

        if use_sample:
            sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_cqi_data.xlsx")
            if os.path.exists(sample_path):
                df = pd.read_excel(sample_path)
                st.success(f"샘플 데이터 로드 완료: {len(df)}행")
            else:
                st.error("샘플 파일을 찾을 수 없습니다.")
        else:
            uploaded = st.file_uploader(
                "Excel / CSV 업로드",
                type=["xlsx", "xls", "csv"],
                help="학술논문 목록, 설문 결과, 강의평가 등 어떤 데이터든 가능합니다.",
            )
            if uploaded:
                df = pd.read_excel(uploaded) if uploaded.name.endswith(("xlsx", "xls")) else pd.read_csv(uploaded)
                st.success(f"업로드 완료: {len(df)}행 × {len(df.columns)}열")

        if df is not None:
            st.subheader("📋 데이터 미리보기")
            st.dataframe(df.head(10), use_container_width=True)

            text_cols = df.select_dtypes(include="object").columns.tolist()
            num_cols = df.select_dtypes(include="number").columns.tolist()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔤 텍스트 분석 설정**")
                text_col = st.selectbox("분석할 텍스트 열", options=["(선택 안함)"] + text_cols)
                group_options = ["(선택 안함)"] + df.columns.tolist()
                group_col_raw = st.selectbox("그룹 기준 열 (선택사항)", options=group_options)
                group_col = None if group_col_raw == "(선택 안함)" else group_col_raw

            with col2:
                st.markdown("**📊 정량 분석 설정**")
                if num_cols:
                    st.info(f"수치형 열 {len(num_cols)}개 자동 감지: {', '.join(num_cols[:5])}{'...' if len(num_cols) > 5 else ''}")
                    st.session_state.quant_df = df[num_cols + ([] if group_col is None else [group_col])]
                else:
                    st.warning("수치형 열이 없습니다.")

            if st.button("🚀 데이터 전처리 시작", type="primary", use_container_width=True):
                st.session_state.text_df = df

                if text_col and text_col != "(선택 안함)":
                    with st.spinner("형태소 분석 중..."):
                        from src.preprocessor import preprocess_dataframe
                        result = preprocess_dataframe(df, text_col, group_col)
                        st.session_state.preprocessed = result
                    st.success(f"✅ 텍스트 분석 완료 | 유효 문서: {result['non_empty_docs']}개")

                if num_cols:
                    st.session_state.quant_df = df
                    st.success(f"✅ 정량 데이터 준비 완료 | 수치 열: {len(num_cols)}개")

    # ── 파일 분리 모드 ────────────────────────────────────────
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔤 텍스트 데이터 (논문 제목·초록·키워드·정성의견 등)**")
            text_file = st.file_uploader("텍스트 파일 업로드", type=["xlsx", "xls", "csv"], key="text_upload")
            if text_file:
                text_df = pd.read_excel(text_file) if text_file.name.endswith(("xlsx", "xls")) else pd.read_csv(text_file)
                st.dataframe(text_df.head(5), use_container_width=True)
                text_col2 = st.selectbox("분석할 텍스트 열", text_df.columns.tolist(), key="tc2")
                group_col2_raw = st.selectbox("그룹 열 (선택)", ["(없음)"] + text_df.columns.tolist(), key="gc2")
                group_col2 = None if group_col2_raw == "(없음)" else group_col2_raw
                if st.button("🔤 텍스트 전처리", type="primary"):
                    with st.spinner("형태소 분석 중..."):
                        from src.preprocessor import preprocess_dataframe
                        result = preprocess_dataframe(text_df, text_col2, group_col2)
                        st.session_state.preprocessed = result
                        st.session_state.text_df = text_df
                    st.success(f"✅ 완료 | 유효 문서: {result['non_empty_docs']}개")

        with col2:
            st.markdown("**📊 정량 데이터 (설문 척도·빈도·점수·수치 변수 등)**")
            quant_file = st.file_uploader("정량 파일 업로드", type=["xlsx", "xls", "csv"], key="quant_upload")
            if quant_file:
                quant_df = pd.read_excel(quant_file) if quant_file.name.endswith(("xlsx", "xls")) else pd.read_csv(quant_file)
                st.dataframe(quant_df.head(5), use_container_width=True)
                st.session_state.quant_df = quant_df
                st.success(f"✅ 정량 데이터 로드 완료: {len(quant_df)}행 × {len(quant_df.columns)}열")


# ══════════════════════════════════════════════════════════════
# 탭 2: 텍스트 분석
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("🔤 텍스트 분석")
    st.caption("논문 키워드, 초록, 설문 텍스트 등 정성 데이터의 키워드·TF-IDF·워드클라우드 분석")

    if st.session_state.preprocessed is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 텍스트 데이터를 전처리하세요.")
    else:
        result = st.session_state.preprocessed
        nouns_list = result["nouns"]

        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.slider("상위 키워드 수", 10, 50, 30)
            analysis_type = st.radio("분석 유형", ["빈도 분석", "TF-IDF", "워드클라우드"])

        with col2:
            from src.keyword_analyzer import frequency_analysis, tfidf_analysis, plot_frequency_bar, generate_wordcloud

            if analysis_type == "빈도 분석":
                freq_df = frequency_analysis(nouns_list, top_n)
                st.session_state.freq_df = freq_df
                st.subheader("📊 키워드 빈도 분석")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.dataframe(freq_df, use_container_width=True, height=400)
                with col_b:
                    chart_bytes = plot_frequency_bar(freq_df)
                    st.image(chart_bytes, use_container_width=True)
                # 다운로드
                st.download_button("📥 빈도표 CSV 다운로드", freq_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "keyword_freq.csv", "text/csv")

            elif analysis_type == "TF-IDF":
                tfidf_df = tfidf_analysis(nouns_list, top_n)
                st.session_state.tfidf_df = tfidf_df
                st.subheader("📈 TF-IDF 분석")
                if tfidf_df.empty:
                    st.warning("TF-IDF 분석을 위해 더 많은 문서가 필요합니다.")
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.dataframe(tfidf_df, use_container_width=True, height=400)
                    with col_b:
                        chart_bytes = plot_frequency_bar(
                            tfidf_df.rename(columns={"TF-IDF 점수": "빈도"}),
                            title="TF-IDF 상위 키워드"
                        )
                        st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 TF-IDF표 CSV 다운로드", tfidf_df.to_csv(index=False, encoding="utf-8-sig"),
                                       "tfidf.csv", "text/csv")

            else:
                st.subheader("☁️ 워드클라우드")
                wc_bytes = generate_wordcloud(nouns_list)
                if wc_bytes:
                    st.image(wc_bytes, use_container_width=True)
                    st.download_button("📥 워드클라우드 PNG 다운로드", wc_bytes, "wordcloud.png", "image/png")
                else:
                    st.warning("워드클라우드 생성에 실패했습니다.")


# ══════════════════════════════════════════════════════════════
# 탭 3: 네트워크 분석
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🕸️ 공출현 네트워크 분석")
    st.caption("키워드 간 동시 출현 관계를 시각화합니다.")

    if st.session_state.preprocessed is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 텍스트 데이터를 전처리하세요.")
    else:
        result = st.session_state.preprocessed
        nouns_list = result["nouns"]

        col1, col2 = st.columns([1, 3])
        with col1:
            window_size = st.slider("공출현 윈도우 크기", 2, 10, 5)
            min_count = st.slider("최소 공출현 횟수", 1, 10, 2)
            top_nodes = st.slider("표시 노드 수", 10, 80, 40)

        with col2:
            if st.button("🕸️ 네트워크 생성", type="primary"):
                from src.network_analyzer import (
                    build_cooccurrence_matrix, build_graph,
                    get_centrality_df, build_pyvis_html,
                )
                with st.spinner("네트워크 분석 중..."):
                    cooc = build_cooccurrence_matrix(nouns_list, window_size, min_count)
                    G = build_graph(cooc, top_nodes)
                    html = build_pyvis_html(G)
                    centrality_df = get_centrality_df(G)

                st.subheader("인터랙티브 네트워크 그래프")
                st.components.v1.html(html, height=520, scrolling=False)
                st.subheader("📋 중심성 지표")
                st.dataframe(centrality_df.head(20), use_container_width=True)
                st.download_button("📥 중심성 CSV 다운로드",
                                   centrality_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "centrality.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# 탭 4: NMF 토픽 모델링
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("🧩 NMF 토픽 모델링")
    st.caption("비음수 행렬 분해(NMF) 기반 토픽 분석 — TF-IDF 벡터를 분해해 잠재 주제를 추출합니다.")

    if st.session_state.preprocessed is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 텍스트 데이터를 전처리하세요.")
    else:
        result = st.session_state.preprocessed
        nouns_list = result["nouns"]

        col1, col2 = st.columns([1, 3])
        with col1:
            auto_optimize = st.checkbox("최적 토픽 수 자동 탐색", value=False)
            if not auto_optimize:
                num_topics = st.slider("토픽 수", 2, 10, 5)
            passes = st.slider("최대 반복 횟수", 50, 300, 200)

        with col2:
            if st.button("🧩 토픽 모델링 실행", type="primary"):
                from src.topic_modeler import train_lda, get_optimal_topics, get_topics_df, get_topic_word_weights

                if auto_optimize:
                    with st.spinner("최적 토픽 수 탐색 중..."):
                        scores = get_optimal_topics(nouns_list)
                    if scores:
                        best_k = max(scores, key=scores.get)
                        st.info(f"최적 토픽 수: {best_k}개 (재구성 적합도: {scores[best_k]:.4f})")
                        num_topics = best_k
                        st.line_chart({k: v for k, v in scores.items()})
                    else:
                        num_topics = 5

                with st.spinner(f"{num_topics}개 토픽 학습 중..."):
                    model, corpus, vectorizer = train_lda(nouns_list, num_topics, passes)
                    st.session_state.nmf_model = model
                    topics_df = get_topics_df(model)
                    st.session_state.topics_df = topics_df

                if model is None:
                    st.error("토픽 모델링 실패: 충분한 데이터가 필요합니다.")
                else:
                    st.success(f"✅ {num_topics}개 토픽 추출 완료! (NMF)")
                    st.subheader("📋 토픽 요약")
                    st.dataframe(topics_df, use_container_width=True)
                    st.download_button("📥 토픽표 CSV 다운로드",
                                       topics_df.to_csv(index=False, encoding="utf-8-sig"),
                                       "topics.csv", "text/csv")

                    st.subheader("📊 토픽별 상세 키워드")
                    cols = st.columns(min(num_topics, 3))
                    for i in range(num_topics):
                        with cols[i % 3]:
                            word_df = get_topic_word_weights(model, i)
                            st.markdown(f"**토픽 {i+1}** ({topics_df.iloc[i]['대표 키워드']})")
                            st.dataframe(word_df, use_container_width=True, height=200)


# ══════════════════════════════════════════════════════════════
# 탭 5: 정량 분석
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("📊 정량 분석")
    st.caption("기술통계·빈도·상관·T검정·ANOVA·단순회귀·로지스틱 회귀 분석 및 시각화")

    if st.session_state.quant_df is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 업로드하세요.")
    else:
        df_q = st.session_state.quant_df
        num_cols = df_q.select_dtypes(include="number").columns.tolist()
        all_cols = df_q.columns.tolist()

        analysis_method = st.selectbox(
            "분석 방법 선택",
            [
                "기술통계 (Descriptive Statistics)",
                "빈도분석 (Frequency Analysis)",
                "상관관계 분석 (Correlation)",
                "독립표본 T-검정 (Independent T-test)",
                "대응표본 T-검정 (Paired T-test)",
                "분산분석 ANOVA (One-way)",
                "단순선형회귀 (Simple Regression)",
                "로지스틱 회귀 (Logistic Regression)",
            ],
        )

        st.markdown("---")
        from src.quant_analyzer import (
            descriptive_stats, frequency_analysis_quant, correlation_analysis,
            ttest_independent, ttest_paired, anova_oneway,
            simple_regression, logistic_regression,
        )

        # ── 기술통계 ──────────────────────────────────────────
        if analysis_method.startswith("기술통계"):
            selected = st.multiselect("분석할 수치형 변수 선택", num_cols, default=num_cols[:min(5, len(num_cols))])
            if st.button("▶ 분석 실행", type="primary") and selected:
                with st.spinner("분석 중..."):
                    result_df, chart_bytes = descriptive_stats(df_q, selected)
                st.subheader("기술통계 결과")
                st.dataframe(result_df, use_container_width=True)
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 그래프 PNG 다운로드", chart_bytes, "desc_stat.png", "image/png")
                st.download_button("📥 결과표 CSV 다운로드",
                                   result_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "desc_stats.csv", "text/csv")
                st.session_state.quant_summary += f"\n[기술통계] 분석 변수: {', '.join(selected)}\n{result_df.to_string(index=False)}"

        # ── 빈도분석 ──────────────────────────────────────────
        elif analysis_method.startswith("빈도분석"):
            col_sel = st.selectbox("분석할 변수 선택", all_cols)
            if st.button("▶ 분석 실행", type="primary"):
                with st.spinner("분석 중..."):
                    result_df, chart_bytes = frequency_analysis_quant(df_q, col_sel)
                st.subheader(f"빈도분석: {col_sel}")
                st.dataframe(result_df, use_container_width=True)
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 그래프 PNG 다운로드", chart_bytes, "freq.png", "image/png")
                st.download_button("📥 결과표 CSV 다운로드",
                                   result_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "freq.csv", "text/csv")

        # ── 상관관계 ──────────────────────────────────────────
        elif analysis_method.startswith("상관관계"):
            corr_method = st.radio("상관 방법", ["pearson", "spearman"], horizontal=True)
            selected = st.multiselect("분석할 수치형 변수 선택 (2개 이상)", num_cols,
                                       default=num_cols[:min(6, len(num_cols))])
            if st.button("▶ 분석 실행", type="primary") and len(selected) >= 2:
                with st.spinner("분석 중..."):
                    corr_df, pval_df, chart_bytes = correlation_analysis(df_q, selected, corr_method)
                st.subheader("상관계수 행렬")
                st.dataframe(corr_df, use_container_width=True)
                st.subheader("p-value 행렬")
                st.dataframe(pval_df, use_container_width=True)
                st.caption("* p<.05  ** p<.01  *** p<.001")
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 히트맵 PNG 다운로드", chart_bytes, "corr_heatmap.png", "image/png")
                st.download_button("📥 상관행렬 CSV 다운로드",
                                   corr_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "correlation.csv", "text/csv")
                st.session_state.quant_summary += f"\n[상관분석({corr_method})] 변수: {', '.join(selected)}"

        # ── 독립표본 T-검정 ───────────────────────────────────
        elif analysis_method.startswith("독립표본"):
            val_col = st.selectbox("종속변수 (수치형)", num_cols)
            grp_col = st.selectbox("집단변수 (2개 집단)", all_cols)
            if st.button("▶ 분석 실행", type="primary"):
                with st.spinner("분석 중..."):
                    result_df, chart_bytes = ttest_independent(df_q, val_col, grp_col)
                st.subheader(f"독립표본 T-검정: {val_col} by {grp_col}")
                st.dataframe(result_df, use_container_width=True)
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 그래프 PNG 다운로드", chart_bytes, "ttest.png", "image/png")
                st.download_button("📥 결과표 CSV 다운로드",
                                   result_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "ttest.csv", "text/csv")
                st.session_state.quant_summary += f"\n[독립표본 T검정] {val_col} by {grp_col}:\n{result_df.to_string(index=False)}"

        # ── 대응표본 T-검정 ───────────────────────────────────
        elif analysis_method.startswith("대응표본"):
            col_a = st.selectbox("변수 A (사전/그룹1)", num_cols, index=0)
            col_b = st.selectbox("변수 B (사후/그룹2)", num_cols, index=min(1, len(num_cols)-1))
            if st.button("▶ 분석 실행", type="primary"):
                with st.spinner("분석 중..."):
                    result_df, chart_bytes = ttest_paired(df_q, col_a, col_b)
                st.subheader(f"대응표본 T-검정: {col_a} vs {col_b}")
                st.dataframe(result_df, use_container_width=True)
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 그래프 PNG 다운로드", chart_bytes, "paired_ttest.png", "image/png")
                st.download_button("📥 결과표 CSV 다운로드",
                                   result_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "paired_ttest.csv", "text/csv")
                st.session_state.quant_summary += f"\n[대응표본 T검정] {col_a} vs {col_b}:\n{result_df.to_string(index=False)}"

        # ── ANOVA ─────────────────────────────────────────────
        elif analysis_method.startswith("분산분석"):
            val_col = st.selectbox("종속변수 (수치형)", num_cols)
            grp_col = st.selectbox("집단변수 (3개 이상 집단 권장)", all_cols)
            if st.button("▶ 분석 실행", type="primary"):
                with st.spinner("ANOVA 분석 중..."):
                    result_df, tukey_df, chart_bytes = anova_oneway(df_q, val_col, grp_col)
                st.subheader(f"일원분산분석 (ANOVA): {val_col} by {grp_col}")
                st.dataframe(result_df, use_container_width=True)
                if not tukey_df.empty:
                    st.subheader("📋 Tukey HSD 사후검정")
                    st.dataframe(tukey_df, use_container_width=True)
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 그래프 PNG 다운로드", chart_bytes, "anova.png", "image/png")
                st.download_button("📥 결과표 CSV 다운로드",
                                   result_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "anova.csv", "text/csv")
                st.session_state.quant_summary += f"\n[ANOVA] {val_col} by {grp_col}:\n{result_df.to_string(index=False)}"

        # ── 단순회귀 ──────────────────────────────────────────
        elif analysis_method.startswith("단순선형"):
            y_col = st.selectbox("종속변수 Y", num_cols, index=0)
            x_col = st.selectbox("독립변수 X", num_cols, index=min(1, len(num_cols)-1))
            if st.button("▶ 분석 실행", type="primary"):
                with st.spinner("회귀분석 중..."):
                    result_df, chart_bytes = simple_regression(df_q, y_col, x_col)
                st.subheader(f"단순선형회귀: {y_col} ~ {x_col}")
                st.dataframe(result_df, use_container_width=True)
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 그래프 PNG 다운로드", chart_bytes, "regression.png", "image/png")
                st.download_button("📥 결과표 CSV 다운로드",
                                   result_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "regression.csv", "text/csv")
                st.session_state.quant_summary += f"\n[단순회귀] {y_col}~{x_col}:\n{result_df.to_string(index=False)}"

        # ── 로지스틱 회귀 ─────────────────────────────────────
        elif analysis_method.startswith("로지스틱"):
            st.caption("종속변수는 0/1 이진 변수여야 합니다.")
            y_col = st.selectbox("종속변수 Y (0/1 이진)", num_cols, index=0)
            x_cols = st.multiselect("독립변수 X (복수 선택 가능)",
                                     [c for c in num_cols if c != y_col],
                                     default=[c for c in num_cols if c != y_col][:min(3, len(num_cols)-1)])
            if st.button("▶ 분석 실행", type="primary") and x_cols:
                with st.spinner("로지스틱 회귀 분석 중..."):
                    result_df, chart_bytes = logistic_regression(df_q, y_col, x_cols)
                st.subheader(f"로지스틱 회귀: {y_col} ~ {' + '.join(x_cols)}")
                st.dataframe(result_df, use_container_width=True)
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button("📥 그래프 PNG 다운로드", chart_bytes, "logit.png", "image/png")
                st.download_button("📥 결과표 CSV 다운로드",
                                   result_df.to_csv(index=False, encoding="utf-8-sig"),
                                   "logistic.csv", "text/csv")
                st.session_state.quant_summary += f"\n[로지스틱 회귀] {y_col}~{'+'.join(x_cols)}:\n{result_df.to_string(index=False)}"


# ══════════════════════════════════════════════════════════════
# 탭 6: AI 리포트
# ══════════════════════════════════════════════════════════════
with tab6:
    st.header("🤖 AI 리포트 생성")
    st.markdown("분석 결과를 바탕으로 문서 유형과 어조를 선택해 AI 리포트를 생성합니다.")

    if st.session_state.preprocessed is None and st.session_state.quant_df is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 업로드하세요.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 문서 유형 선택")
            doc_type = st.radio(
                "생성할 문서 형식",
                options=["academic", "government", "research"],
                format_func=lambda x: {
                    "academic": "📚 학술논문 — 격식체, 수동태 서술 (~하였다, ~나타났다)",
                    "government": "🏛️ 정부보고서 — 개조식, 명사형 종결 (-함, -임)",
                    "research": "📋 연구보고서 — 서술체+개조식 혼합, 정책·실무 시사점",
                }[x],
                label_visibility="collapsed",
            )

        with col2:
            st.subheader("⚙️ 리포트 설정")
            subject_input = st.text_input("연구 주제 / 과목명", placeholder="예: 비대면 강의 효과성 분석")
            include_quant = st.checkbox("정량 분석 결과 포함", value=True)
            api_key = st.text_input(
                "Anthropic API 키",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                type="password",
                help="Streamlit Cloud Secrets에 ANTHROPIC_API_KEY를 등록하면 자동 입력됩니다.",
            )

        st.markdown("---")

        if st.button("🤖 AI 리포트 생성", type="primary", use_container_width=True):
            if not api_key:
                st.error("API 키를 입력해주세요.")
            else:
                from src.ai_report import generate_report_stream

                meta = {
                    "total_docs": st.session_state.preprocessed.get("total_docs", 0)
                    if st.session_state.preprocessed else 0,
                    "subject": subject_input,
                    "doc_type": doc_type,
                }
                quant_summary = st.session_state.quant_summary if include_quant else ""

                report_placeholder = st.empty()
                full_report = ""

                with st.spinner("AI 리포트 생성 중..."):
                    for chunk in generate_report_stream(
                        st.session_state.freq_df,
                        st.session_state.tfidf_df,
                        st.session_state.topics_df,
                        quant_summary,
                        meta,
                        api_key,
                        doc_type=doc_type,
                    ):
                        full_report += chunk
                        report_placeholder.markdown(full_report)

                doc_labels = {"academic": "학술논문", "government": "정부보고서", "research": "연구보고서"}
                st.download_button(
                    label="📥 리포트 TXT 다운로드",
                    data=full_report.encode("utf-8"),
                    file_name=f"WISDOM_Lab_{doc_labels[doc_type]}_{subject_input or '분석결과'}.txt",
                    mime="text/plain",
                )
