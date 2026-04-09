"""
WISDOM Lab - 정량 분석 모듈
기술통계 / 빈도분석 / 상관관계 / T-검정 / ANOVA / 단순회귀 / 로짓분석
결과표 + 시각화(bytes) 반환
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── 한국어 폰트 설정 ───────────────────────────────────────────
def _set_korean_font():
    candidates = [
        "NanumGothic", "NanumBarunGothic", "Apple SD Gothic Neo",
        "Malgun Gothic", "Noto Sans KR", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════
# 1. 기술통계 (Descriptive Statistics)
# ══════════════════════════════════════════════════════════════
def descriptive_stats(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, bytes]:
    """선택한 수치형 열의 기술통계량 + 박스플롯"""
    sub = df[columns].select_dtypes(include="number")
    if sub.empty:
        return pd.DataFrame(), b""

    desc = sub.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc.index.name = "변수"
    desc.columns = ["N", "평균", "표준편차", "최솟값", "Q1(25%)", "중앙값", "Q3(75%)", "최댓값"]

    # 왜도·첨도 추가
    desc["왜도"] = sub.skew().round(3)
    desc["첨도"] = sub.kurt().round(3)
    desc = desc.round(3)
    desc["N"] = desc["N"].astype(int)

    # 박스플롯
    _set_korean_font()
    fig, ax = plt.subplots(figsize=(max(6, len(sub.columns) * 1.5), 5))
    sub.boxplot(ax=ax, grid=False, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax.set_title("기술통계 박스플롯", fontsize=13, fontweight="bold")
    ax.set_ylabel("값")
    fig.tight_layout()

    return desc.reset_index(), _fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════
# 2. 빈도분석 (Frequency Analysis)
# ══════════════════════════════════════════════════════════════
def frequency_analysis_quant(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, bytes]:
    """범주형/명목형 변수 빈도표 + 막대그래프"""
    freq = df[column].value_counts().reset_index()
    freq.columns = ["값", "빈도"]
    freq["비율(%)"] = (freq["빈도"] / freq["빈도"].sum() * 100).round(2)
    freq["누적비율(%)"] = freq["비율(%)"].cumsum().round(2)

    _set_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 막대그래프
    axes[0].bar(freq["값"].astype(str), freq["빈도"], color="#4C72B0", alpha=0.8)
    axes[0].set_title(f"{column} 빈도 분포", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("빈도")
    axes[0].tick_params(axis="x", rotation=45)

    # 파이차트
    axes[1].pie(freq["빈도"], labels=freq["값"].astype(str),
                autopct="%1.1f%%", startangle=90,
                colors=sns.color_palette("Blues_d", len(freq)))
    axes[1].set_title(f"{column} 비율", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return freq, _fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════
# 3. 상관관계 분석 (Correlation)
# ══════════════════════════════════════════════════════════════
def correlation_analysis(df: pd.DataFrame, columns: list,
                          method: str = "pearson") -> tuple[pd.DataFrame, pd.DataFrame, bytes]:
    """상관계수 행렬 + p-value 행렬 + 히트맵"""
    sub = df[columns].select_dtypes(include="number").dropna()

    corr_matrix = sub.corr(method=method).round(3)

    # p-value 계산
    n = len(sub)
    cols = corr_matrix.columns.tolist()
    pval_data = {}
    for c in cols:
        pval_data[c] = {}
        for r in cols:
            if c == r:
                pval_data[c][r] = np.nan
            else:
                if method == "pearson":
                    _, p = stats.pearsonr(sub[c], sub[r])
                else:
                    _, p = stats.spearmanr(sub[c], sub[r])
                pval_data[c][r] = round(p, 4)
    pval_df = pd.DataFrame(pval_data)

    # 히트맵
    _set_korean_font()
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.9 + 2), max(5, len(cols) * 0.9)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, mask=mask,
                linewidths=0.5, annot_kws={"size": 10})
    ax.set_title(f"상관관계 히트맵 ({method.capitalize()})", fontsize=13, fontweight="bold")
    fig.tight_layout()

    return corr_matrix.reset_index().rename(columns={"index": "변수"}), pval_df, _fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════
# 4. T-검정 (Independent / Paired)
# ══════════════════════════════════════════════════════════════
def cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    """Cohen's d 효과크기"""
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.std() ** 2 + (n2 - 1) * g2.std() ** 2) / (n1 + n2 - 2))
    return round((g1.mean() - g2.mean()) / pooled_std, 4) if pooled_std != 0 else 0.0


def ttest_independent(df: pd.DataFrame, value_col: str,
                      group_col: str) -> tuple[pd.DataFrame, bytes]:
    """독립표본 T-검정"""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return pd.DataFrame({"오류": ["그룹이 정확히 2개여야 합니다."]}), b""

    g1 = df[df[group_col] == groups[0]][value_col].dropna()
    g2 = df[df[group_col] == groups[1]][value_col].dropna()

    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test
    d = cohens_d(g1, g2)

    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

    result = pd.DataFrame({
        "항목": ["집단1 평균", "집단2 평균", "t-통계량", "자유도(근사)", "p-value", "유의성", "Cohen's d", "효과크기 해석"],
        "값": [
            f"{groups[0]}: {g1.mean():.3f} (n={len(g1)})",
            f"{groups[1]}: {g2.mean():.3f} (n={len(g2)})",
            round(t_stat, 4), round(len(g1) + len(g2) - 2, 1),
            round(p_val, 4), sig, abs(d),
            "소(|d|<0.2)" if abs(d) < 0.2 else ("중(|d|<0.5)" if abs(d) < 0.5 else ("대(|d|<0.8)" if abs(d) < 0.8 else "매우 큰"))
        ]
    })

    # 박스플롯
    _set_korean_font()
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_df = pd.DataFrame({
        str(groups[0]): g1.values[:max(len(g1), len(g2))],
        str(groups[1]): g2.values[:max(len(g1), len(g2))],
    })
    plot_df_melt = pd.DataFrame({"값": pd.concat([g1, g2]).values,
                                  "집단": [str(groups[0])] * len(g1) + [str(groups[1])] * len(g2)})
    sns.boxplot(data=plot_df_melt, x="집단", y="값", ax=ax,
                palette=["#4C72B0", "#DD8452"], width=0.5)
    ax.set_title(f"독립표본 T-검정\n{value_col} (p={p_val:.4f}{sig})", fontsize=12, fontweight="bold")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()

    return result, _fig_to_bytes(fig)


def ttest_paired(df: pd.DataFrame, col1: str, col2: str) -> tuple[pd.DataFrame, bytes]:
    """대응표본 T-검정"""
    paired = df[[col1, col2]].dropna()
    g1, g2 = paired[col1], paired[col2]

    t_stat, p_val = stats.ttest_rel(g1, g2)
    d = cohens_d(g1, g2)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

    result = pd.DataFrame({
        "항목": [f"{col1} 평균", f"{col2} 평균", "평균 차이", "t-통계량", "자유도", "p-value", "유의성", "Cohen's d"],
        "값": [
            round(g1.mean(), 3), round(g2.mean(), 3),
            round(g1.mean() - g2.mean(), 3),
            round(t_stat, 4), len(paired) - 1,
            round(p_val, 4), sig, abs(d)
        ]
    })

    _set_korean_font()
    fig, ax = plt.subplots(figsize=(7, 5))
    diff = g1 - g2
    ax.hist(diff, bins=15, color="#4C72B0", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="차이=0")
    ax.axvline(diff.mean(), color="orange", linestyle="-", linewidth=1.5, label=f"평균차이={diff.mean():.3f}")
    ax.set_title(f"대응표본 T-검정: {col1} − {col2}\n(p={p_val:.4f}{sig})", fontsize=12, fontweight="bold")
    ax.set_xlabel("차이값")
    ax.legend()
    fig.tight_layout()

    return result, _fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════
# 5. 일원분산분석 ANOVA
# ══════════════════════════════════════════════════════════════
def anova_oneway(df: pd.DataFrame, value_col: str,
                 group_col: str) -> tuple[pd.DataFrame, pd.DataFrame, bytes]:
    """일원배치 분산분석 + 사후검정(Tukey HSD)"""
    groups = {g: df[df[group_col] == g][value_col].dropna().values
              for g in df[group_col].dropna().unique()}
    if len(groups) < 2:
        return pd.DataFrame({"오류": ["그룹이 2개 이상 필요합니다."]}), pd.DataFrame(), b""

    f_stat, p_val = stats.f_oneway(*groups.values())
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

    # 그룹별 기술통계
    group_stats = pd.DataFrame([
        {"집단": g, "N": len(v), "평균": round(np.mean(v), 3), "표준편차": round(np.std(v, ddof=1), 3)}
        for g, v in groups.items()
    ])

    anova_result = pd.DataFrame({
        "항목": ["F-통계량", "p-value", "유의성", "집단 수"],
        "값": [round(f_stat, 4), round(p_val, 4), sig, len(groups)]
    })

    # Tukey HSD (statsmodels)
    tukey_df = pd.DataFrame()
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(
            df[[group_col, value_col]].dropna()[value_col],
            df[[group_col, value_col]].dropna()[group_col]
        )
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    except Exception:
        pass

    # 박스플롯
    _set_korean_font()
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 5))
    plot_data = [v for v in groups.values()]
    bp = ax.boxplot(plot_data, labels=list(groups.keys()), patch_artist=True, notch=False)
    colors = sns.color_palette("Blues_d", len(groups))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"ANOVA: {value_col} by {group_col}\n(F={f_stat:.3f}, p={p_val:.4f}{sig})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    return pd.concat([anova_result, pd.DataFrame([{"항목": "---집단별 통계---", "값": ""}]),
                      group_stats.rename(columns={"집단": "항목"})], ignore_index=True), \
           tukey_df, _fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════
# 6. 단순선형회귀 (Simple Linear Regression)
# ══════════════════════════════════════════════════════════════
def simple_regression(df: pd.DataFrame, y_col: str,
                       x_col: str) -> tuple[pd.DataFrame, bytes]:
    """단순선형회귀분석"""
    import statsmodels.api as sm
    data = df[[y_col, x_col]].dropna()
    X = sm.add_constant(data[x_col])
    y = data[y_col]
    model = sm.OLS(y, X).fit()

    result = pd.DataFrame({
        "항목": ["R²", "수정 R²", "F-통계량", "p(F)", "관측수",
                "절편(β₀)", "계수(β₁)", "t-통계량(β₁)", "p(β₁)", "유의성"],
        "값": [
            round(model.rsquared, 4),
            round(model.rsquared_adj, 4),
            round(model.fvalue, 4),
            round(model.f_pvalue, 4),
            int(model.nobs),
            round(model.params.iloc[0], 4),
            round(model.params.iloc[1], 4),
            round(model.tvalues.iloc[1], 4),
            round(model.pvalues.iloc[1], 4),
            "***" if model.pvalues.iloc[1] < 0.001 else
            ("**" if model.pvalues.iloc[1] < 0.01 else
             ("*" if model.pvalues.iloc[1] < 0.05 else "n.s."))
        ]
    })

    # 산점도 + 회귀선
    _set_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 산점도
    axes[0].scatter(data[x_col], y, alpha=0.5, color="#4C72B0", s=30)
    x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
    axes[0].plot(x_range, model.params.iloc[0] + model.params.iloc[1] * x_range,
                 color="red", linewidth=2, label=f"y={model.params.iloc[0]:.3f}+{model.params.iloc[1]:.3f}x")
    axes[0].set_title(f"회귀분석: {y_col} ~ {x_col}\n(R²={model.rsquared:.3f})",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].legend()

    # 잔차 플롯
    residuals = model.resid
    axes[1].scatter(model.fittedvalues, residuals, alpha=0.5, color="#DD8452", s=30)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("잔차 플롯", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("예측값")
    axes[1].set_ylabel("잔차")

    fig.tight_layout()
    return result, _fig_to_bytes(fig)


# ══════════════════════════════════════════════════════════════
# 7. 로지스틱 회귀 (Logistic Regression)
# ══════════════════════════════════════════════════════════════
def logistic_regression(df: pd.DataFrame, y_col: str,
                         x_cols: list) -> tuple[pd.DataFrame, bytes]:
    """이항 로지스틱 회귀분석"""
    import statsmodels.api as sm
    data = df[[y_col] + x_cols].dropna()
    X = sm.add_constant(data[x_cols])
    y = data[y_col]

    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        return pd.DataFrame({"오류": [str(e)]}), b""

    # Odds Ratio
    params = model.params
    conf = model.conf_int()
    odds = np.exp(params)
    odds_ci_lo = np.exp(conf.iloc[:, 0])
    odds_ci_hi = np.exp(conf.iloc[:, 1])
    pvals = model.pvalues

    result_rows = []
    for var in params.index:
        sig = "***" if pvals[var] < 0.001 else ("**" if pvals[var] < 0.01 else ("*" if pvals[var] < 0.05 else "n.s."))
        result_rows.append({
            "변수": var,
            "계수(β)": round(params[var], 4),
            "Odds Ratio": round(odds[var], 4),
            "OR 95% CI 하": round(odds_ci_lo[var], 4),
            "OR 95% CI 상": round(odds_ci_hi[var], 4),
            "p-value": round(pvals[var], 4),
            "유의성": sig,
        })

    summary_rows = [
        {"변수": "── 모델 적합도 ──", "계수(β)": "", "Odds Ratio": "",
         "OR 95% CI 하": "", "OR 95% CI 상": "", "p-value": "", "유의성": ""},
        {"변수": "Log-Likelihood", "계수(β)": round(model.llf, 3), "Odds Ratio": "",
         "OR 95% CI 하": "", "OR 95% CI 상": "", "p-value": "", "유의성": ""},
        {"변수": "McFadden R²", "계수(β)": round(1 - model.llf / model.llnull, 4), "Odds Ratio": "",
         "OR 95% CI 하": "", "OR 95% CI 상": "", "p-value": "", "유의성": ""},
        {"변수": "AIC", "계수(β)": round(model.aic, 2), "Odds Ratio": "",
         "OR 95% CI 하": "", "OR 95% CI 상": "", "p-value": round(model.llr_pvalue, 4), "유의성": ""},
    ]
    result_df = pd.DataFrame(result_rows + summary_rows)

    # Odds Ratio 시각화
    _set_korean_font()
    x_vars = [v for v in params.index if v != "const"]
    if x_vars:
        fig, ax = plt.subplots(figsize=(8, max(4, len(x_vars) * 0.7 + 1)))
        y_pos = range(len(x_vars))
        or_vals = [odds[v] for v in x_vars]
        ci_lo = [odds[v] - odds_ci_lo[v] for v in x_vars]
        ci_hi = [odds_ci_hi[v] - odds[v] for v in x_vars]
        colors = ["#4C72B0" if o >= 1 else "#DD8452" for o in or_vals]

        ax.barh(y_pos, or_vals, xerr=[ci_lo, ci_hi], color=colors, alpha=0.7,
                capsize=4, align="center")
        ax.axvline(1, color="red", linestyle="--", linewidth=1.5, label="OR=1 (기준)")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(x_vars)
        ax.set_xlabel("Odds Ratio (95% CI)")
        ax.set_title("로지스틱 회귀: Odds Ratio", fontsize=12, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        return result_df, _fig_to_bytes(fig)

    return result_df, b""
