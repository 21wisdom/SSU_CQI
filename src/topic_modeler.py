"""
LDA 토픽 모델링 모듈
gensim 기반 토픽 분석 + Coherence 최적화
"""

import pandas as pd
import numpy as np


def train_lda(nouns_list: list, num_topics: int = 5, passes: int = 10):
    """
    LDA 모델 학습
    반환: (model, corpus, dictionary)
    """
    try:
        from gensim import corpora
        from gensim.models import LdaModel
    except ImportError:
        return None, None, None

    # 빈 문서 제거
    filtered = [nouns for nouns in nouns_list if nouns]
    if len(filtered) < 5:
        return None, None, None

    dictionary = corpora.Dictionary(filtered)
    # 너무 희귀하거나 너무 흔한 단어 필터
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in filtered]

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=42,
        alpha="auto",
        per_word_topics=True,
    )
    return model, corpus, dictionary


def get_optimal_topics(nouns_list: list, start: int = 2, limit: int = 8) -> dict:
    """
    Coherence Score로 최적 토픽 수 탐색
    반환: {num_topics: coherence_score}
    """
    try:
        from gensim import corpora
        from gensim.models import LdaModel, CoherenceModel
    except ImportError:
        return {}

    filtered = [n for n in nouns_list if n]
    if len(filtered) < 10:
        return {}

    dictionary = corpora.Dictionary(filtered)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(t) for t in filtered]

    scores = {}
    for k in range(start, limit + 1):
        model = LdaModel(corpus, id2word=dictionary, num_topics=k,
                         passes=5, random_state=42)
        cm = CoherenceModel(model=model, texts=filtered,
                            dictionary=dictionary, coherence="c_v")
        scores[k] = round(cm.get_coherence(), 4)
    return scores


def get_topics_df(model, num_words: int = 10) -> pd.DataFrame:
    """토픽별 키워드 데이터프레임"""
    if model is None:
        return pd.DataFrame()

    rows = []
    for topic_id in range(model.num_topics):
        words = model.show_topic(topic_id, topn=num_words)
        keyword_str = ", ".join([w for w, _ in words])
        top_word = words[0][0] if words else f"토픽{topic_id + 1}"
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
    words = model.show_topic(topic_id, topn=num_words)
    df = pd.DataFrame(words, columns=["단어", "가중치"])
    df["가중치"] = df["가중치"].round(4)
    return df
