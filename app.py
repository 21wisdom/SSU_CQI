"""
SSU CQI 강의평가 분석 플랫폼
Streamlit 기반 5탭 구성 앱
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="SSU CQI 분석 플랫폼",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 SSU CQI 분석")
    st.markdown("---")
    st.markdown("""
    **숭실대학교 강의품질개선(CQI)**
    강의평가 데이터를 업로드하여
    키워드·네트워크·토픽 분석과
    AI 리포트를 생성하세요.
    """)
    st.markdown("---")
    st.caption("버전 1.0.0 | 2025")

# ── 세션 상태 초기화 ──────────────────────────────────────────
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = None
if "freq_df" not in st.session_state:
    st.session_state.freq_df = None
if "tfidf_df" not in st.session_state:
    st.session_state.tfidf_df = None
if "topics_df" not in st.session_state:
    st.session_state.topics_df = None
if "lda_model" not in st.session_state:
    st.session_state.lda_model = None

# ── 탭 구성 ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 데이터 업로드",
    "🔤 키워드 분석",
    "🕸️ 네트워크 분석",
    "🧩 토픽 모델링",
    "🤖 AI 리포트",
])

# ══════════════════════════════════════════════════════════════
# 탭 1: 데이터 업로드
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("📂 데이터 업로드 및 전처리")
    st.markdown("""
    강의평가 Excel 파일을 업로드하고, 분석할 텍스트 열과 그룹 열을 선택하세요.
    """)

    # 샘플 데이터 사용 옵션
    use_sample = st.checkbox("🔍 샘플 데이터로 시작하기", value=False)

    uploaded_file = None
    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_cqi_data.xlsx")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                uploaded_file = f
                df = pd.read_excel(sample_path)
            st.success(f"샘플 데이터 로드 완료: {len(df)}개 행")
        else:
            st.error("샘플 데이터 파일을 찾을 수 없습니다.")
    else:
        uploaded = st.file_uploader(
            "Excel 파일 업로드 (.xlsx, .xls)",
            type=["xlsx", "xls"],
            help="강의평가 데이터가 담긴 엑셀 파일을 업로드하세요."
        )
        if uploaded:
            df = pd.read_excel(uploaded)
            st.success(f"업로드 완료: {len(df)}개 행, {len(df.columns)}개 열")

    if "df" in dir() and df is not None:
        st.subheader("📋 데이터 미리보기")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox(
                "분석할 텍스트 열 선택 *",
                options=df.columns.tolist(),
                help="강의평가 자유의견이 담긴 열을 선택하세요."
            )
        with col2:
            group_options = ["(선택 안함)"] + df.columns.tolist()
            group_col_raw = st.selectbox(
                "그룹 기준 열 선택 (선택사항)",
                options=group_options,
                help="과목명, 학과 등 그룹으로 나눌 기준 열을 선택하세요."
            )
            group_col = None if group_col_raw == "(선택 안함)" else group_col_raw

        if st.button("🚀 형태소 분석 시작", type="primary", use_container_width=True):
            with st.spinner("형태소 분석 중... (처음 실행 시 시간이 걸릴 수 있습니다)"):
                from src.preprocessor import preprocess_dataframe
                result = preprocess_dataframe(df, text_col, group_col)
                st.session_state.preprocessed = result
                st.session_state.df = df
                st.session_state.text_col = text_col
                st.session_state.group_col = group_col

            st.success(f"""
            ✅ 분석 완료!
            - 총 문서: {result['total_docs']}개
            - 유효 문서: {result['non_empty_docs']}개
            - 추출된 명사 예시: {', '.join(result['nouns'][0][:10]) if result['nouns'] and result['nouns'][0] else '없음'}
            """)

# ══════════════════════════════════════════════════════════════
# 탭 2: 키워드 분석
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("🔤 키워드 분석")

    if st.session_state.preprocessed is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 불러오고 형태소 분석을 실행하세요.")
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
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.dataframe(freq_df, use_container_width=True, height=400)
                with col_b:
                    chart_bytes = plot_frequency_bar(freq_df)
                    st.image(chart_bytes, use_container_width=True)

            elif analysis_type == "TF-IDF":
                tfidf_df = tfidf_analysis(nouns_list, top_n)
                st.session_state.tfidf_df = tfidf_df
                st.subheader("📈 TF-IDF 분석")
                if tfidf_df.empty:
                    st.warning("TF-IDF 분석을 위해 충분한 문서가 필요합니다.")
                else:
                    st.dataframe(tfidf_df, use_container_width=True, height=400)
                    chart_bytes = plot_frequency_bar(
                        tfidf_df.rename(columns={"TF-IDF 점수": "빈도"}),
                        title="TF-IDF 상위 키워드"
                    )
                    st.image(chart_bytes, use_container_width=True)

            else:  # 워드클라우드
                st.subheader("☁️ 워드클라우드")
                wc_bytes = generate_wordcloud(nouns_list)
                if wc_bytes:
                    st.image(wc_bytes, use_container_width=True)
                else:
                    st.warning("워드클라우드 생성에 실패했습니다. 데이터를 확인하세요.")

# ══════════════════════════════════════════════════════════════
# 탭 3: 네트워크 분석
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🕸️ 공출현 네트워크 분석")

    if st.session_state.preprocessed is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 불러오고 형태소 분석을 실행하세요.")
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
                    get_centrality_df, build_pyvis_html
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

# ══════════════════════════════════════════════════════════════
# 탭 4: 토픽 모델링
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("🧩 LDA 토픽 모델링")

    if st.session_state.preprocessed is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 불러오고 형태소 분석을 실행하세요.")
    else:
        result = st.session_state.preprocessed
        nouns_list = result["nouns"]

        col1, col2 = st.columns([1, 3])
        with col1:
            auto_optimize = st.checkbox("최적 토픽 수 자동 탐색", value=False)
            if not auto_optimize:
                num_topics = st.slider("토픽 수", 2, 10, 5)
            passes = st.slider("학습 반복 횟수", 5, 30, 10)

        with col2:
            if st.button("🧩 토픽 모델링 실행", type="primary"):
                from src.topic_modeler import train_lda, get_optimal_topics, get_topics_df, get_topic_word_weights

                if auto_optimize:
                    with st.spinner("최적 토픽 수 탐색 중... (수 분 소요)"):
                        scores = get_optimal_topics(nouns_list)
                    if scores:
                        best_k = max(scores, key=scores.get)
                        st.info(f"최적 토픽 수: {best_k} (Coherence: {scores[best_k]})")
                        num_topics = best_k
                        st.line_chart(scores)
                    else:
                        num_topics = 5

                with st.spinner(f"{num_topics}개 토픽 학습 중..."):
                    model, corpus, dictionary = train_lda(nouns_list, num_topics, passes)
                    st.session_state.lda_model = model
                    topics_df = get_topics_df(model)
                    st.session_state.topics_df = topics_df

                if model is None:
                    st.error("토픽 모델링 실패: 충분한 데이터가 필요합니다.")
                else:
                    st.success(f"{num_topics}개 토픽 학습 완료!")
                    st.subheader("📋 토픽 요약")
                    st.dataframe(topics_df, use_container_width=True)

                    st.subheader("📊 토픽별 상세 키워드")
                    cols = st.columns(min(num_topics, 3))
                    for i in range(num_topics):
                        with cols[i % 3]:
                            word_df = get_topic_word_weights(model, i)
                            st.markdown(f"**토픽 {i+1}** ({topics_df.iloc[i]['대표 키워드']})")
                            st.dataframe(word_df, use_container_width=True, height=200)

# ══════════════════════════════════════════════════════════════
# 탭 5: AI 리포트
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("🤖 AI CQI 리포트 생성")
    st.markdown("Claude AI가 분석 결과를 종합하여 6섹션 CQI 개선 리포트를 생성합니다.")

    if st.session_state.preprocessed is None:
        st.info("먼저 [📂 데이터 업로드] 탭에서 데이터를 불러오고 형태소 분석을 실행하세요.")
    else:
        # API 키 입력
        api_key = st.text_input(
            "Anthropic API 키",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help=".env 파일에 ANTHROPIC_API_KEY를 설정하거나 여기에 직접 입력하세요."
        )

        subject_input = st.text_input("과목명 (선택사항)", placeholder="예: 교육학개론")

        meta = {
            "total_docs": st.session_state.preprocessed.get("total_docs", 0),
            "subject": subject_input,
        }

        if st.button("🤖 AI 리포트 생성", type="primary", use_container_width=True):
            if not api_key:
                st.error("API 키를 입력해주세요.")
            else:
                from src.ai_report import generate_report_stream
                report_placeholder = st.empty()
                full_report = ""

                with st.spinner("AI 리포트 생성 중..."):
                    for chunk in generate_report_stream(
                        st.session_state.freq_df,
                        st.session_state.tfidf_df,
                        st.session_state.topics_df,
                        meta,
                        api_key,
                    ):
                        full_report += chunk
                        report_placeholder.markdown(full_report)

                # 다운로드 버튼
                st.download_button(
                    label="📥 리포트 TXT 다운로드",
                    data=full_report.encode("utf-8"),
                    file_name=f"CQI_리포트_{subject_input or '분석결과'}.txt",
                    mime="text/plain",
                )
