"""
NMF 토픽 모델링 모듈
scikit-learn 기반 토픽 분석 (gensim 대체 - Python 3.14 호환)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def train_lda(nouns_list: list, num_topics: int = 5, passes: int = 10):
    """
    NMF 모델 학습 (LDA 인터페이스 호환)
    반환: (model_dict, corpus, vectorizer)
    """
    # 빈 문서 제거
    filtered = [" ".join(nouns) for nouns in nouns_list if nouns]
    if len(filtered) < 5:
        return None, None, None

    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, max_features=500)
    try:
        tfidf_matrix = vectorizer.fit_transform(filtered)
    except ValueError:
        return None, None, None

    if tfidf_matrix.shape[1] == 0:
        return None, None, None

    n_topics = min(num_topics, tfidf_matrix.shape[1])
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    nmf.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    model_dict = {
        "nmf": nmf,
        "feature_names": feature_names,
        "num_topics": n_topics,
    }
    return model_dict, tfidf_matrix, vectorizer


def get_optimal_topics(nouns_list: list, start: int = 2, limit: int = 8) -> dict:
    """
    재구성 오류 기반 최적 토픽 수 탐색
    반환: {num_topics: reconstruction_error}
    """
    filtered = [" ".join(nouns) for nouns in nouns_list if nouns]
    if len(filtered) < 10:
        return {}

    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, max_features=500)
    try:
        tfidf_matrix = vectorizer.fit_transform(filtered)
    except ValueError:
        return {}

    scores = {}
    for k in range(start, min(limit + 1, tfidf_matrix.shape[1] + 1)):
        nmf = NMF(n_components=k, random_state=42, max_iter=200)
        nmf.fit(tfidf_matrix)
        # 낮은 재구성 오류 = 더 나은 모델 (음수로 반환해서 max()와 호환)
        scores[k] = round(-nmf.reconstruction_err_, 4)
    return scores


def get_topics_df(model, num_words: int = 10) -> pd.DataFrame:
    """토픽별 키워드 데이터프레임"""
    if model is None:
        return pd.DataFrame()

    nmf = model["nmf"]
    feature_names = model["feature_names"]
    num_topics = model["num_topics"]

    rows = []
    for topic_id in range(num_topics):
        top_indices = nmf.components_[topic_id].argsort()[-num_words:][::-1]
        words = [feature_names[i] for i in top_indices]
        keyword_str = ", ".join(words)
        top_word = words[0] if words else f"토픽{topic_id + 1}"
        rows.append({
            "토픽 번호": f"토픽 {topic_id + 1}",
            "대표 키워드": top_word,
            "핵심 단어": keyword_str,
        })
    return pd.DataFrame(rows)


def get_topic_word_weights(model, topic_id: int, num_words: int = 15) -> pd.DataFrame:
    """특정 토픽의 단어-가중치 데이터프레임"""
    if model is None:
        return pd.DataFrame()

    nmf = model["nmf"]
    feature_names = model["feature_names"]
    top_indices = nmf.components_[topic_id].argsort()[-num_words:][::-1]
    words = [(feature_names[i], round(float(nmf.components_[topic_id][i]), 4))
             for i in top_indices]
    df = pd.DataFrame(words, columns=["단어", "가중치"])
    return df
