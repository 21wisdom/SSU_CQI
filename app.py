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
    st.caption("v2.1.0 | WISDOM Lab © 2025")

# ── 세션 상태 초기화 ─────────────────────────────────────────
defaults = {
    "preprocessed": None,
    "freq_df": None,
    "tfidf_df": None,
    "topics_df": None,
    "nmf_model": None,
    "quant_df": None,
    "quant_results": [],    # 정량 분석 결과 목록
    "text_df": None,
    "last_report": "",
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
                st.download_button("📥 빈도표 CSV 다운로드", freq_df.to_csv(index=False).encode("utf-8-sig"),
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
                    st.download_button("📥 TF-IDF표 CSV 다운로드", tfidf_df.to_csv(index=False).encode("utf-8-sig"),
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
                                   centrality_df.to_csv(index=False).encode("utf-8-sig"),
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
                                       topics_df.to_csv(index=False).encode("utf-8-sig"),
                                       "topics.csv", "text/csv")

                    st.subheader("📊 토픽별 상세 키워드")
                    cols = st.columns(min(num_topics, 3))
                    for i in range(num_topics):
                        with cols[i % 3]:
                            word_df = get_topic_word_weights(model, i)
                            st.markdown(f"**토픽 {i+1}** ({topics_df.iloc[i]['대표 키워드']})")
                            st.dataframe(word_df, use_container_width=True, height=200)


# ══════════════════════════════════════════════════════════════
# 탭 5: 정량 분석 (다중 선택)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("📊 정량 분석")
    st.caption("분석 방법을 복수 선택하여 한 번에 실행하고 결과를 AI 리포트에 통합할 수 있습니다.")

    if st.session_state.quant_df is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 업로드하세요.")
    else:
        df_q = st.session_state.quant_df
        num_cols = df_q.select_dtypes(include="number").columns.tolist()
        all_cols = df_q.columns.tolist()

        from src.quant_analyzer import (
            descriptive_stats, frequency_analysis_quant, correlation_analysis,
            ttest_independent, ttest_paired, anova_oneway,
            simple_regression, logistic_regression,
        )

        METHOD_LIST = [
            ("기술통계 (Descriptive Statistics)",   "📊"),
            ("빈도분석 (Frequency Analysis)",        "📋"),
            ("상관관계 분석 (Correlation)",           "🔗"),
            ("독립표본 T-검정 (Independent T-test)", "🔬"),
            ("대응표본 T-검정 (Paired T-test)",      "🔄"),
            ("분산분석 ANOVA (One-way)",              "📈"),
            ("단순선형회귀 (Simple Regression)",      "📉"),
            ("로지스틱 회귀 (Logistic Regression)",  "🎯"),
        ]

        st.markdown("**분석 방법 선택 (복수 선택 가능)**")
        chk_cols = st.columns(2)
        selected_methods = []
        for i, (name, icon) in enumerate(METHOD_LIST):
            if chk_cols[i % 2].checkbox(f"{icon} {name}", key=f"chk_{i}"):
                selected_methods.append(name)

        # 저장된 이전 결과 요약 표시
        if st.session_state.quant_results:
            stored = st.session_state.quant_results
            labels = ", ".join(r["method"] for r in stored)
            st.info(f"📋 저장된 분석 결과 {len(stored)}개: **{labels}** — AI 리포트 탭에서 통합 보고서를 생성하세요.")

        if selected_methods:
            st.markdown("---")
            st.markdown("#### ⚙️ 분석별 파라미터 설정")

            method_params = {}
            for method in selected_methods:
                mkey = method.replace(" ", "_").replace("(", "").replace(")", "")[:12]
                with st.expander(f"**{method}** 파라미터", expanded=True):
                    if method.startswith("기술통계"):
                        sel = st.multiselect("분석할 수치형 변수", num_cols,
                                             default=num_cols[:min(5, len(num_cols))],
                                             key=f"{mkey}_vars")
                        method_params[method] = {"selected": sel}

                    elif method.startswith("빈도분석"):
                        col = st.selectbox("분석할 변수", all_cols, key=f"{mkey}_col")
                        method_params[method] = {"col": col}

                    elif method.startswith("상관관계"):
                        cm = st.radio("상관 방법", ["pearson", "spearman"], horizontal=True, key=f"{mkey}_cm")
                        sel = st.multiselect("수치형 변수 (2개 이상)", num_cols,
                                             default=num_cols[:min(6, len(num_cols))],
                                             key=f"{mkey}_vars")
                        method_params[method] = {"corr_method": cm, "selected": sel}

                    elif method.startswith("독립표본"):
                        val = st.selectbox("종속변수 (수치형)", num_cols, key=f"{mkey}_val")
                        grp = st.selectbox("집단변수 (2개 집단)", all_cols, key=f"{mkey}_grp")
                        method_params[method] = {"val_col": val, "grp_col": grp}

                    elif method.startswith("대응표본"):
                        ca = st.selectbox("변수 A (사전/그룹1)", num_cols, index=0, key=f"{mkey}_ca")
                        cb = st.selectbox("변수 B (사후/그룹2)", num_cols,
                                          index=min(1, len(num_cols) - 1), key=f"{mkey}_cb")
                        method_params[method] = {"col_a": ca, "col_b": cb}

                    elif method.startswith("분산분석"):
                        val = st.selectbox("종속변수 (수치형)", num_cols, key=f"{mkey}_val")
                        grp = st.selectbox("집단변수 (3개 이상 권장)", all_cols, key=f"{mkey}_grp")
                        method_params[method] = {"val_col": val, "grp_col": grp}

                    elif method.startswith("단순선형"):
                        yc = st.selectbox("종속변수 Y", num_cols, index=0, key=f"{mkey}_y")
                        xc = st.selectbox("독립변수 X", num_cols,
                                          index=min(1, len(num_cols) - 1), key=f"{mkey}_x")
                        method_params[method] = {"y_col": yc, "x_col": xc}

                    elif method.startswith("로지스틱"):
                        st.caption("종속변수는 0/1 이진 변수여야 합니다.")
                        yc = st.selectbox("종속변수 Y (0/1 이진)", num_cols, index=0, key=f"{mkey}_y")
                        remaining = [c for c in num_cols if c != yc]
                        xcs = st.multiselect("독립변수 X (복수 선택)",
                                             remaining,
                                             default=remaining[:min(3, len(remaining))],
                                             key=f"{mkey}_x")
                        method_params[method] = {"y_col": yc, "x_cols": xcs}

            st.markdown("---")
            if st.button("▶ 선택한 모든 분석 실행", type="primary", use_container_width=True):
                new_results = []

                for method in selected_methods:
                    params = method_params.get(method, {})
                    try:
                        # ── 기술통계 ──────────────────────────────
                        if method.startswith("기술통계"):
                            sel = params.get("selected", [])
                            if not sel:
                                st.warning(f"{method}: 변수를 선택하세요.")
                                continue
                            with st.spinner("기술통계 분석 중..."):
                                r_df, chart = descriptive_stats(df_q, sel)
                            st.subheader("📊 기술통계 결과")
                            st.dataframe(r_df, use_container_width=True)
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 기술통계 그래프", chart,
                                                   "desc_stat.png", "image/png")
                            st.download_button("📥 기술통계 CSV",
                                               r_df.to_csv(index=False).encode("utf-8-sig"),
                                               "desc_stats.csv", "text/csv")
                            new_results.append({
                                "method": "기술통계",
                                "label": method,
                                "result_df": r_df,
                                "extra_df": None,
                                "chart_bytes": chart,
                                "summary_text": f"[기술통계] 분석 변수: {', '.join(sel)}\n{r_df.to_string(index=False)}",
                            })

                        # ── 빈도분석 ──────────────────────────────
                        elif method.startswith("빈도분석"):
                            col = params.get("col")
                            with st.spinner("빈도분석 중..."):
                                r_df, chart = frequency_analysis_quant(df_q, col)
                            st.subheader(f"📊 빈도분석: {col}")
                            st.dataframe(r_df, use_container_width=True)
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 빈도분석 그래프", chart,
                                                   "freq.png", "image/png")
                            st.download_button("📥 빈도분석 CSV",
                                               r_df.to_csv(index=False).encode("utf-8-sig"),
                                               "freq.csv", "text/csv")
                            new_results.append({
                                "method": "빈도분석",
                                "label": method,
                                "result_df": r_df,
                                "extra_df": None,
                                "chart_bytes": chart,
                                "summary_text": f"[빈도분석] 변수: {col}\n{r_df.to_string(index=False)}",
                            })

                        # ── 상관관계 ──────────────────────────────
                        elif method.startswith("상관관계"):
                            sel = params.get("selected", [])
                            cm = params.get("corr_method", "pearson")
                            if len(sel) < 2:
                                st.warning(f"{method}: 2개 이상의 변수를 선택하세요.")
                                continue
                            with st.spinner("상관관계 분석 중..."):
                                corr_df, pval_df, chart = correlation_analysis(df_q, sel, cm)
                            st.subheader(f"📊 상관계수 행렬 ({cm})")
                            st.dataframe(corr_df, use_container_width=True)
                            st.subheader("p-value 행렬")
                            st.dataframe(pval_df, use_container_width=True)
                            st.caption("* p<.05  ** p<.01  *** p<.001")
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 히트맵 PNG", chart,
                                                   "corr_heatmap.png", "image/png")
                            st.download_button("📥 상관행렬 CSV",
                                               corr_df.to_csv(index=False).encode("utf-8-sig"),
                                               "correlation.csv", "text/csv")
                            new_results.append({
                                "method": "상관관계",
                                "label": method,
                                "result_df": corr_df,
                                "extra_df": pval_df,
                                "chart_bytes": chart,
                                "summary_text": f"[상관분석({cm})] 변수: {', '.join(sel)}\n상관계수 행렬:\n{corr_df.to_string()}\np-value 행렬:\n{pval_df.to_string()}",
                            })

                        # ── 독립표본 T-검정 ───────────────────────
                        elif method.startswith("독립표본"):
                            val = params.get("val_col")
                            grp = params.get("grp_col")
                            with st.spinner("독립표본 T-검정 분석 중..."):
                                r_df, chart = ttest_independent(df_q, val, grp)
                            st.subheader(f"📊 독립표본 T-검정: {val} by {grp}")
                            st.dataframe(r_df, use_container_width=True)
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 T검정 그래프", chart,
                                                   "ttest.png", "image/png")
                            st.download_button("📥 T검정 CSV",
                                               r_df.to_csv(index=False).encode("utf-8-sig"),
                                               "ttest.csv", "text/csv")
                            new_results.append({
                                "method": "독립표본T검정",
                                "label": method,
                                "result_df": r_df,
                                "extra_df": None,
                                "chart_bytes": chart,
                                "summary_text": f"[독립표본 T검정] {val} by {grp}:\n{r_df.to_string(index=False)}",
                            })

                        # ── 대응표본 T-검정 ───────────────────────
                        elif method.startswith("대응표본"):
                            ca = params.get("col_a")
                            cb = params.get("col_b")
                            with st.spinner("대응표본 T-검정 분석 중..."):
                                r_df, chart = ttest_paired(df_q, ca, cb)
                            st.subheader(f"📊 대응표본 T-검정: {ca} vs {cb}")
                            st.dataframe(r_df, use_container_width=True)
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 대응T검정 그래프", chart,
                                                   "paired_ttest.png", "image/png")
                            st.download_button("📥 대응T검정 CSV",
                                               r_df.to_csv(index=False).encode("utf-8-sig"),
                                               "paired_ttest.csv", "text/csv")
                            new_results.append({
                                "method": "대응표본T검정",
                                "label": method,
                                "result_df": r_df,
                                "extra_df": None,
                                "chart_bytes": chart,
                                "summary_text": f"[대응표본 T검정] {ca} vs {cb}:\n{r_df.to_string(index=False)}",
                            })

                        # ── ANOVA ─────────────────────────────────
                        elif method.startswith("분산분석"):
                            val = params.get("val_col")
                            grp = params.get("grp_col")
                            with st.spinner("ANOVA 분석 중..."):
                                r_df, tukey_df, chart = anova_oneway(df_q, val, grp)
                            st.subheader(f"📊 일원분산분석 (ANOVA): {val} by {grp}")
                            st.dataframe(r_df, use_container_width=True)
                            if tukey_df is not None and not tukey_df.empty:
                                st.subheader("📋 Tukey HSD 사후검정")
                                st.dataframe(tukey_df, use_container_width=True)
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 ANOVA 그래프", chart,
                                                   "anova.png", "image/png")
                            st.download_button("📥 ANOVA CSV",
                                               r_df.to_csv(index=False).encode("utf-8-sig"),
                                               "anova.csv", "text/csv")
                            tukey_text = ""
                            if tukey_df is not None and not tukey_df.empty:
                                tukey_text = f"\nTukey HSD:\n{tukey_df.to_string(index=False)}"
                            new_results.append({
                                "method": "ANOVA",
                                "label": method,
                                "result_df": r_df,
                                "extra_df": tukey_df,
                                "chart_bytes": chart,
                                "summary_text": f"[ANOVA] {val} by {grp}:\n{r_df.to_string(index=False)}{tukey_text}",
                            })

                        # ── 단순회귀 ──────────────────────────────
                        elif method.startswith("단순선형"):
                            yc = params.get("y_col")
                            xc = params.get("x_col")
                            with st.spinner("회귀분석 중..."):
                                r_df, chart = simple_regression(df_q, yc, xc)
                            st.subheader(f"📊 단순선형회귀: {yc} ~ {xc}")
                            st.dataframe(r_df, use_container_width=True)
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 회귀 그래프", chart,
                                                   "regression.png", "image/png")
                            st.download_button("📥 회귀분석 CSV",
                                               r_df.to_csv(index=False).encode("utf-8-sig"),
                                               "regression.csv", "text/csv")
                            new_results.append({
                                "method": "단순회귀",
                                "label": method,
                                "result_df": r_df,
                                "extra_df": None,
                                "chart_bytes": chart,
                                "summary_text": f"[단순회귀] {yc}~{xc}:\n{r_df.to_string(index=False)}",
                            })

                        # ── 로지스틱 회귀 ─────────────────────────
                        elif method.startswith("로지스틱"):
                            yc = params.get("y_col")
                            xcs = params.get("x_cols", [])
                            if not xcs:
                                st.warning(f"{method}: 독립변수를 선택하세요.")
                                continue
                            with st.spinner("로지스틱 회귀 분석 중..."):
                                r_df, chart = logistic_regression(df_q, yc, xcs)
                            st.subheader(f"📊 로지스틱 회귀: {yc} ~ {' + '.join(xcs)}")
                            st.dataframe(r_df, use_container_width=True)
                            if chart:
                                st.image(chart, use_container_width=True)
                                st.download_button("📥 로짓 그래프", chart,
                                                   "logit.png", "image/png")
                            st.download_button("📥 로짓 CSV",
                                               r_df.to_csv(index=False).encode("utf-8-sig"),
                                               "logistic.csv", "text/csv")
                            new_results.append({
                                "method": "로지스틱회귀",
                                "label": method,
                                "result_df": r_df,
                                "extra_df": None,
                                "chart_bytes": chart,
                                "summary_text": f"[로지스틱 회귀] {yc}~{'+'.join(xcs)}:\n{r_df.to_string(index=False)}",
                            })

                    except Exception as e:
                        st.error(f"**{method}** 분석 오류: {e}")

                st.session_state.quant_results = new_results
                if new_results:
                    names = ", ".join(r["method"] for r in new_results)
                    st.success(f"✅ {len(new_results)}개 분석 완료: {names} — AI 리포트 탭에서 통합 보고서를 생성하세요.")


# ══════════════════════════════════════════════════════════════
# 탭 6: AI 리포트
# ══════════════════════════════════════════════════════════════
with tab6:
    st.header("🤖 AI 리포트 생성")
    st.markdown("분석 결과를 바탕으로 문서 유형을 선택해 AI 통합 보고서를 생성합니다.")

    has_text = st.session_state.preprocessed is not None
    has_quant = bool(st.session_state.quant_results)

    if not has_text and not has_quant:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 업로드하고 분석을 진행하세요.")
    else:
        # 분석 현황 표시
        with st.expander("📋 현재 분석 결과 현황", expanded=True):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if has_text:
                    items = []
                    if st.session_state.freq_df is not None:
                        items.append("키워드 빈도")
                    if st.session_state.tfidf_df is not None:
                        items.append("TF-IDF")
                    if st.session_state.topics_df is not None:
                        items.append("토픽 모델링")
                    st.success(f"✅ 텍스트 분석: {', '.join(items) if items else '전처리 완료'}")
                else:
                    st.warning("텍스트 분석 없음")
            with col_s2:
                if has_quant:
                    names = ", ".join(r["method"] for r in st.session_state.quant_results)
                    st.success(f"✅ 정량 분석 {len(st.session_state.quant_results)}개: {names}")
                else:
                    st.warning("정량 분석 없음")

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
            api_key = st.text_input(
                "Anthropic API 키",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                type="password",
                help="Streamlit Cloud Secrets에 ANTHROPIC_API_KEY를 등록하면 자동 입력됩니다.",
            )

        st.markdown("---")

        if st.button("🤖 AI 통합 리포트 생성", type="primary", use_container_width=True):
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

                report_placeholder = st.empty()
                full_report = ""

                with st.spinner("AI 리포트 생성 중... (분량에 따라 1~2분 소요)"):
                    for chunk in generate_report_stream(
                        st.session_state.freq_df,
                        st.session_state.tfidf_df,
                        st.session_state.topics_df,
                        st.session_state.quant_results,
                        meta,
                        api_key,
                        doc_type=doc_type,
                    ):
                        full_report += chunk
                        report_placeholder.markdown(full_report)

                st.session_state.last_report = full_report

                doc_labels = {"academic": "학술논문", "government": "정부보고서", "research": "연구보고서"}
                file_base = f"WISDOM_Lab_{doc_labels[doc_type]}_{subject_input or '분석결과'}"

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="📥 TXT 다운로드",
                        data=full_report.encode("utf-8"),
                        file_name=f"{file_base}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with dl_col2:
                    try:
                        from src.report_to_docx import markdown_to_docx_bytes
                        docx_bytes = markdown_to_docx_bytes(
                            md_text=full_report,
                            title=f"WISDOM Lab {doc_labels[doc_type]}",
                            subject=subject_input or "",
                            doc_type_label=doc_labels[doc_type],
                        )
                        st.download_button(
                            label="📄 Word(.docx) 다운로드",
                            data=docx_bytes,
                            file_name=f"{file_base}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Word 변환 오류: {e}")
