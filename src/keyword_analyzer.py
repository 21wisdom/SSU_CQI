"""
WISDOM Lab - 키워드 분석 모듈
빈도 분석, TF-IDF, 워드클라우드 생성
"""

from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import os


# ── 한국어 폰트 설정 (Streamlit Cloud: NanumGothic, Windows: 맑은고딕) ──
_FONT_PROP = None

def _get_font_prop():
    """한국어 폰트 프로퍼티 (캐싱)"""
    global _FONT_PROP
    if _FONT_PROP is not None:
        return _FONT_PROP

    # 폰트 캐시 재빌드 (Streamlit Cloud에서 패키지 설치 후 필요)
    fm._load_fontmanager(try_read_cache=False)

    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Streamlit Cloud (Linux)
        "/usr/share/fonts/opentype/nanum/NanumGothic.otf",
        "C:/Windows/Fonts/malgun.ttf",                            # Windows 맑은고딕
        "C:/Windows/Fonts/gulim.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",                  # macOS
        "/Library/Fonts/NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            _FONT_PROP = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = _FONT_PROP.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return _FONT_PROP

    # 시스템 폰트에서 나눔/맑은고딕 탐색
    for f in fm.findSystemFonts():
        lower = f.lower()
        if any(kw in lower for kw in ["nanum", "malgun", "gothic", "gulim", "applegothic"]):
            _FONT_PROP = fm.FontProperties(fname=f)
            plt.rcParams["font.family"] = _FONT_PROP.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return _FONT_PROP

    return None


def _apply_font(ax, font_prop):
    """ax 전체 텍스트 요소에 폰트 일괄 적용"""
    if font_prop is None:
        return
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    if ax.xaxis.label:
        ax.xaxis.label.set_fontproperties(font_prop)
    if ax.yaxis.label:
        ax.yaxis.label.set_fontproperties(font_prop)
    if ax.title:
        ax.title.set_fontproperties(font_prop)


# ── 분석 함수 ────────────────────────────────────────────────

def frequency_analysis(nouns_list: list, top_n: int = 30) -> pd.DataFrame:
    """전체 명사 빈도 분석"""
    all_nouns = [noun for nouns in nouns_list for noun in nouns]
    counter = Counter(all_nouns)
    df = pd.DataFrame(counter.most_common(top_n), columns=["키워드", "빈도"])
    df["비율(%)"] = (df["빈도"] / df["빈도"].sum() * 100).round(2)
    return df


def tfidf_analysis(nouns_list: list, top_n: int = 20) -> pd.DataFrame:
    """TF-IDF 분석"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [" ".join(nouns) for nouns in nouns_list if nouns]
    if len(corpus) < 2:
        return pd.DataFrame(columns=["키워드", "TF-IDF 점수"])
    vectorizer = TfidfVectorizer(max_features=200)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.mean(axis=0).A1
    df = pd.DataFrame({"키워드": feature_names, "TF-IDF 점수": scores})
    df = df.sort_values("TF-IDF 점수", ascending=False).head(top_n).reset_index(drop=True)
    df["TF-IDF 점수"] = df["TF-IDF 점수"].round(4)
    return df


def plot_frequency_bar(freq_df: pd.DataFrame, title: str = "키워드 빈도") -> bytes:
    """빈도/TF-IDF 가로 막대그래프 → PNG bytes"""
    font_prop = _get_font_prop()
    top = freq_df.head(20).copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    keywords = top.iloc[:, 0].tolist()[::-1]   # 키워드 열 (첫 번째)
    values   = top.iloc[:, 1].tolist()[::-1]   # 값 열 (두 번째)

    bars = ax.barh(range(len(keywords)), values, color="#4C72B0", alpha=0.85)

    # y축 레이블을 키워드로 설정
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords, fontsize=11,
                        fontproperties=font_prop if font_prop else None)

    # 각 막대 끝에 값 표시
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}" if isinstance(val, float) else str(val),
                va="center", ha="left", fontsize=9)

    ax.set_xlabel("값", fontproperties=font_prop if font_prop else None)
    ax.set_title(title, fontsize=14, fontweight="bold",
                  fontproperties=font_prop if font_prop else None)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_wordcloud(nouns_list: list) -> bytes:
    """워드클라우드 이미지 → PNG bytes"""
    from wordcloud import WordCloud
    all_nouns = [noun for nouns in nouns_list for noun in nouns]
    freq = Counter(all_nouns)
    if not freq:
        return None

    font_prop = _get_font_prop()
    font_path = font_prop.get_file() if font_prop else None

    # 나눔고딕 경로 직접 탐색 (wordcloud는 fname 필요)
    if font_path is None:
        nanum_candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/nanum/NanumGothic.otf",
            "C:/Windows/Fonts/malgun.ttf",
        ]
        for p in nanum_candidates:
            if os.path.exists(p):
                font_path = p
                break

    wc_kwargs = dict(width=900, height=450, background_color="white",
                     max_words=120, colormap="Blues")
    if font_path:
        wc_kwargs["font_path"] = font_path

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
