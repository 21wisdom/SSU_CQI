"""
키워드 분석 모듈
빈도 분석, TF-IDF, 워드클라우드 생성
"""

from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import os


def get_korean_font():
    """한글 폰트 경로 반환"""
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            return path
    # 시스템 폰트에서 한글 지원 폰트 탐색
    for f in fm.findSystemFonts():
        if any(kw in f.lower() for kw in ["nanum", "malgun", "gothic", "gulim"]):
            return f
    return None


def get_font_prop():
    """matplotlib 한글 폰트 프로퍼티"""
    font_path = get_korean_font()
    if font_path:
        return fm.FontProperties(fname=font_path)
    return None


def frequency_analysis(nouns_list: list, top_n: int = 30) -> pd.DataFrame:
    """
    전체 명사 빈도 분석
    nouns_list: 각 문서의 명사 리스트 (리스트의 리스트)
    반환: 빈도 데이터프레임
    """
    all_nouns = [noun for nouns in nouns_list for noun in nouns]
    counter = Counter(all_nouns)
    df = pd.DataFrame(counter.most_common(top_n), columns=["키워드", "빈도"])
    df["비율(%)"] = (df["빈도"] / df["빈도"].sum() * 100).round(2)
    return df


def tfidf_analysis(nouns_list: list, top_n: int = 20) -> pd.DataFrame:
    """
    TF-IDF 분석
    반환: 키워드별 TF-IDF 점수 데이터프레임
    """
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
    """빈도 막대그래프 생성 → bytes 반환"""
    font_prop = get_font_prop()
    fig, ax = plt.subplots(figsize=(10, 6))
    top = freq_df.head(20)
    bars = ax.barh(top["키워드"][::-1], top["빈도"][::-1], color="#4C72B0")
    ax.set_xlabel("빈도", fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop, fontsize=14)
    if font_prop:
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def generate_wordcloud(nouns_list: list) -> bytes:
    """워드클라우드 이미지 생성 → bytes 반환"""
    from wordcloud import WordCloud

    all_nouns = [noun for nouns in nouns_list for noun in nouns]
    freq = Counter(all_nouns)
    if not freq:
        return None

    font_path = get_korean_font()
    wc_kwargs = dict(
        width=800, height=400,
        background_color="white",
        max_words=100,
        colormap="Blues",
    )
    if font_path:
        wc_kwargs["font_path"] = font_path

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
