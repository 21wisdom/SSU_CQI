"""
공출현 네트워크 분석 모듈
networkx + pyvis 기반 인터랙티브 그래프
"""

from collections import Counter
from itertools import combinations
import pandas as pd
import networkx as nx
import json
import os
import tempfile


def build_cooccurrence_matrix(nouns_list: list, window: int = 5, min_count: int = 2) -> dict:
    """
    공출현 행렬 생성
    window: 공출현 윈도우 크기
    min_count: 최소 공출현 횟수
    """
    cooc = Counter()
    for nouns in nouns_list:
        for i in range(len(nouns)):
            window_words = nouns[i + 1: i + window + 1]
            for w2 in window_words:
                if nouns[i] != w2:
                    pair = tuple(sorted([nouns[i], w2]))
                    cooc[pair] += 1

    # 최소 빈도 필터
    cooc = {k: v for k, v in cooc.items() if v >= min_count}
    return cooc


def build_graph(cooc_dict: dict, top_n: int = 50) -> nx.Graph:
    """networkx 그래프 생성"""
    G = nx.Graph()
    # 상위 N개 엣지만 사용
    top_edges = sorted(cooc_dict.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]
    for (w1, w2), weight in top_edges:
        G.add_edge(w1, w2, weight=weight)

    # 고립 노드 제거 후 상위 노드만
    nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = [n for n, _ in nodes_by_degree]
    G = G.subgraph(top_nodes).copy()
    return G


def get_centrality_df(G: nx.Graph) -> pd.DataFrame:
    """중심성 지표 데이터프레임"""
    if len(G.nodes) == 0:
        return pd.DataFrame()

    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, weight="weight")
    close_cent = nx.closeness_centrality(G)

    df = pd.DataFrame({
        "키워드": list(degree_cent.keys()),
        "연결 중심성": [round(v, 4) for v in degree_cent.values()],
        "매개 중심성": [round(between_cent.get(k, 0), 4) for k in degree_cent.keys()],
        "근접 중심성": [round(close_cent.get(k, 0), 4) for k in degree_cent.keys()],
    })
    df = df.sort_values("연결 중심성", ascending=False).reset_index(drop=True)
    return df


def build_pyvis_html(G: nx.Graph) -> str:
    """pyvis 인터랙티브 HTML 생성"""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis 패키지가 필요합니다: pip install pyvis</p>"

    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.set_options(json.dumps({
        "nodes": {"font": {"size": 14}},
        "edges": {"color": {"inherit": True}, "smooth": False},
        "physics": {"stabilization": {"iterations": 100}}
    }))

    # 노드 크기: 연결 중심성 기반
    degree_cent = nx.degree_centrality(G)
    max_cent = max(degree_cent.values()) if degree_cent else 1

    for node in G.nodes():
        size = 10 + 30 * (degree_cent.get(node, 0) / max_cent)
        net.add_node(node, label=node, size=size, title=f"{node}\n연결: {G.degree(node)}")

    for u, v, data in G.edges(data=True):
        width = 1 + data.get("weight", 1) * 0.5
        net.add_edge(u, v, width=min(width, 10), title=f"공출현: {data.get('weight', 1)}")

    # 임시 파일로 저장 후 읽기
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        tmp_path = f.name

    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(tmp_path)
    return html
