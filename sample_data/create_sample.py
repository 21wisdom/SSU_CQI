"""
샘플 CQI 데이터 생성기
80행, 5개 과목의 강의평가 샘플 데이터를 생성합니다.
"""

import pandas as pd
import random
import os

random.seed(42)

SUBJECTS = [
    "교육학개론",
    "교육과정이론",
    "교육평가방법론",
    "HRD이론과실제",
    "평생교육론"
]

POSITIVE_COMMENTS = [
    "교수님의 설명이 매우 명확하고 이해하기 쉬웠습니다. 실제 사례를 통한 수업 방식이 인상적이었어요.",
    "수업 내용이 실무와 잘 연계되어 있어서 취업 후에도 도움이 될 것 같습니다.",
    "팀 프로젝트를 통해 협업 능력과 의사소통 능력을 기를 수 있었습니다.",
    "강의 자료가 체계적으로 구성되어 있어 복습하기 편했습니다.",
    "교수님이 학생들의 질문에 성실하게 답변해 주셔서 좋았습니다.",
    "이론과 실습이 균형 있게 구성되어 있어 학습 효과가 높았습니다.",
    "창의적인 사고를 기를 수 있는 수업이었습니다. 다양한 관점을 배웠어요.",
    "수업 분위기가 자유롭고 토론이 활발하게 이루어졌습니다.",
    "교수님이 학생 개개인의 수준을 고려한 맞춤형 피드백을 제공해 주셨습니다.",
    "최신 교육 트렌드와 연계한 내용이 포함되어 있어 유익했습니다.",
]

IMPROVEMENT_COMMENTS = [
    "수업 속도가 조금 빨라서 내용을 따라가기 어려울 때가 있었습니다.",
    "실습 시간이 더 많았으면 좋겠습니다. 이론 위주 수업이 아쉬웠어요.",
    "과제 분량이 많아서 다른 과목과 병행하기 힘들었습니다.",
    "강의실 환경이 수업에 적합하지 않아 불편했습니다.",
    "교재 내용과 강의 내용이 일치하지 않는 부분이 있었습니다.",
    "평가 기준이 더 명확했으면 좋겠습니다.",
    "온라인 자료 접근성이 개선되면 좋겠습니다.",
    "그룹 활동 시 역할 분배가 불균형했습니다.",
]

MIXED_COMMENTS = [
    "전반적으로 만족스러운 수업이었지만 일부 내용은 더 깊이 다루었으면 합니다.",
    "교수님의 열정은 대단하지만 수업 구성을 좀 더 체계화할 필요가 있습니다.",
    "흥미로운 주제를 다루었으나 시험 범위가 불명확했습니다.",
    "실무 경험 공유는 좋았지만 이론적 기반도 충실히 다루어 주었으면 합니다.",
    "수업 참여도는 높았지만 개인 역량 향상을 위한 기회가 더 있었으면 합니다.",
]

def generate_sample_data():
    rows = []
    for i in range(80):
        subject = random.choice(SUBJECTS)
        comment_pool = POSITIVE_COMMENTS * 3 + IMPROVEMENT_COMMENTS + MIXED_COMMENTS * 2
        comment = random.choice(comment_pool)
        satisfaction = random.choices([3, 4, 5], weights=[10, 30, 60])[0]
        rows.append({
            "학번": f"2024{i+1:04d}",
            "과목명": subject,
            "강의만족도": satisfaction,
            "자유의견": comment,
            "학년": random.choice([1, 2, 3, 4]),
            "수강학기": "2025-1",
        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(os.path.dirname(__file__), "sample_cqi_data.xlsx")
    df.to_excel(output_path, index=False)
    print(f"샘플 데이터 생성 완료: {output_path}")
    print(f"총 {len(df)}개 행, {df['과목명'].nunique()}개 과목")
    return df


if __name__ == "__main__":
    generate_sample_data()
