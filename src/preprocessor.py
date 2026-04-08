"""
형태소 분석 및 전처리 모듈
kiwipiepy를 사용한 한국어 형태소 분석
"""

import re
import os
from pathlib import Path


def load_stopwords(stopword_path: str = None) -> set:
    """불용어 사전 로드"""
    if stopword_path is None:
        stopword_path = Path(__file__).parent.parent / "assets" / "stopwords_ko.txt"

    stopwords = set()
    try:
        with open(stopword_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    stopwords.add(line)
    except FileNotFoundError:
        pass
    return stopwords


def clean_text(text: str) -> str:
    """텍스트 기본 정제"""
    if not isinstance(text, str):
        return ""
    # HTML 태그 제거
    text = re.sub(r"<[^>]+>", " ", text)
    # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", " ", text)
    # 연속 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_nouns(texts: list, stopwords: set = None) -> list:
    """
    kiwipiepy를 사용한 명사 추출
    texts: 문자열 리스트
    반환: 각 텍스트의 명사 리스트
    """
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
    except ImportError:
        # kiwipiepy 미설치 시 간단한 공백 분리로 대체
        return [text.split() for text in texts]

    if stopwords is None:
        stopwords = load_stopwords()

    result = []
    for text in texts:
        text = clean_text(text)
        if not text:
            result.append([])
            continue
        tokens = kiwi.tokenize(text)
        nouns = [
            token.form
            for token in tokens
            if token.tag.startswith("NN")  # 명사 태그
            and len(token.form) > 1        # 1글자 제거
            and token.form not in stopwords
        ]
        result.append(nouns)
    return result


def preprocess_dataframe(df, text_col: str, group_col: str = None) -> dict:
    """
    데이터프레임 전처리 메인 함수
    반환: {
        'texts': 원본 텍스트 리스트,
        'nouns': 추출된 명사 리스트,
        'groups': 그룹 라벨 리스트 (group_col 있을 때)
    }
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    stopwords = load_stopwords()
    nouns = extract_nouns(texts, stopwords)

    result = {
        "texts": texts,
        "nouns": nouns,
        "total_docs": len(texts),
        "non_empty_docs": sum(1 for t in texts if t.strip()),
    }

    if group_col and group_col in df.columns:
        result["groups"] = df[group_col].fillna("미분류").astype(str).tolist()
        result["group_values"] = df[group_col].dropna().unique().tolist()

    return result
