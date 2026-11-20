#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích thống kê và trực quan hóa dữ liệu thực nghiệm dự án Mergesort CUTOFF.
Đọc dữ liệu từ:
- results/timing_raw.csv
- results/ops_count.csv
- results/doubling_test.csv
Xuất kết quả ra:
- figures/fig1_cutoff_curve_random.png
- figures/fig2_speedup_by_distribution.png
- figures/fig3_cost_model_vs_time.png
- figures/fig4_doubling_loglog.png
- figures/fig5_variance_boxplot.png
- results/summary_report_tables.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ==================== CẤU HÌNH ĐƯỜNG DẪN & HẰNG SỐ ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

TIMING_CSV = RESULTS_DIR / "timing_raw.csv"
OPS_CSV = RESULTS_DIR / "ops_count.csv"
DOUBLING_CSV = RESULTS_DIR / "doubling_test.csv"
REPORT_MD = RESULTS_DIR / "summary_report_tables.md"

ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261120
DPI = 300

DIST_ORDER = ["RANDOM", "PARTIALLY_SORTED", "REVERSED", "FEW_UNIQUE"]
DIST_LABELS_VI = {
    "RANDOM": "Ngẫu nhiên (Random)",
    "PARTIALLY_SORTED": "Gần có thứ tự (Partially-Sorted)",
    "REVERSED": "Giảm dần (Reversed)",
    "FEW_UNIQUE": "Ít khóa phân biệt (Few-Unique)",
}

# Bảng màu Okabe-Ito thân thiện với in ấn học thuật và người mù màu
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "dark": "#222222",
    "gray": "#666666",
}

DIST_COLORS = {
    "RANDOM": PALETTE["blue"],
    "PARTIALLY_SORTED": PALETTE["green"],
    "REVERSED": PALETTE["red"],
    "FEW_UNIQUE": PALETTE["purple"],
}

DIST_MARKERS = {
    "RANDOM": "o",
    "PARTIALLY_SORTED": "s",
    "REVERSED": "^",
    "FEW_UNIQUE": "D",
}


def configure_matplotlib() -> None:
    """Cấu hình giao diện Matplotlib chuẩn học thuật, hỗ trợ đầy đủ tiếng Việt trên Windows."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Tahoma"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CCCCCC",
            "grid.color": "#D9D9D9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


# ==================== HÀM THỐNG KÊ CỐT LÕI ====================


def bootstrap_median_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Tính khoảng tin cậy 95% cho Median bằng phương pháp Percentile Bootstrap (tất định)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(arr[idx], axis=1)
    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    return (float(np.percentile(boot_medians, lo_pct)), float(np.percentile(boot_medians, hi_pct)))


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Kiểm định cặp đôi Wilcoxon signed-rank hai phía, xử lý an toàn trường hợp hiệu bằng 0.
    Trả về (statistic, p_value).
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(diff) == 0 or np.allclose(diff, 0.0):
        return (0.0, 1.0)
    try:
        res = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return (float(res.statistic), float(res.pvalue))
    except ValueError:
        return (0.0, 1.0)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Hiệu chỉnh đa kiểm định Holm-Bonferroni cho danh sách p-value."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    adjusted = [np.nan] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        if np.isnan(p):
            adjusted[orig_idx] = np.nan
            continue
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def align_paired_trials(sub_df: pd.DataFrame, cutoff_a: int, cutoff_b: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ghép cặp chính xác theo số thứ tự trial giữa hai mức CUTOFF trong cùng (distribution, n)."""
    a = sub_df[sub_df["cutoff"] == cutoff_a][["trial", "time_ms"]].rename(columns={"time_ms": "t_a"})
    b = sub_df[sub_df["cutoff"] == cutoff_b][["trial", "time_ms"]].rename(columns={"time_ms": "t_b"})
    merged = pd.merge(a, b, on="trial", how="inner").sort_values("trial")
    return merged["t_a"].to_numpy(dtype=float), merged["t_b"].to_numpy(dtype=float)


# ==================== NẠP DỮ LIỆU & PHÂN TÍCH ====================


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (TIMING_CSV, OPS_CSV, DOUBLING_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {p}")

    df_timing = pd.read_csv(TIMING_CSV)
    df_ops = pd.read_csv(OPS_CSV)
    df_doubling = pd.read_csv(DOUBLING_CSV)

    for df in (df_timing, df_ops, df_doubling):
        df.columns = [c.strip() for c in df.columns]
        if "distribution" in df.columns:
            df["distribution"] = df["distribution"].astype(str).str.strip()

    for col in ("n", "cutoff", "trial"):
        if col in df_timing.columns:
            df_timing[col] = df_timing[col].astype(int)
        if col in df_doubling.columns:
            df_doubling[col] = df_doubling[col].astype(int)
    df_timing["time_ms"] = df_timing["time_ms"].astype(float)
    df_doubling["time_ms"] = df_doubling["time_ms"].astype(float)

    df_ops["n"] = df_ops["n"].astype(int)
    df_ops["cutoff"] = df_ops["cutoff"].astype(int)
    df_ops["compares"] = df_ops["compares"].astype(np.int64)
    df_ops["theoretical_nlogn"] = df_ops["theoretical_nlogn"].astype(float)

    return df_timing, df_ops, df_doubling


def compute_descriptive_stats(df_timing: pd.DataFrame) -> pd.DataFrame:
    """Tính đầy đủ thống kê mô tả, 95% CI, % speedup so với CUTOFF = 1 cho từng (dist, n, cutoff)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    grouped = df_timing.groupby(["distribution", "n", "cutoff"], sort=True)
    for (dist, n, cutoff), grp in grouped:
        vals = grp["time_ms"].to_numpy(dtype=float)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ci_lo, ci_hi = bootstrap_median_ci(vals, rng)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(
            {
                "distribution": dist,
                "n": int(n),
                "cutoff": int(cutoff),
                "trials": int(len(vals)),
                "median_ms": float(med),
                "mean_ms": mean_val,
                "std_ms": std_val,
                "q1_ms": float(q1),
                "q3_ms": float(q3),
                "iqr_ms": float(q3 - q1),
                "min_ms": float(np.min(vals)),
                "max_ms": float(np.max(vals)),
                "ci95_lo_ms": ci_lo,
                "ci95_hi_ms": ci_hi,
            }
        )

    summary = pd.DataFrame(rows)

    # Tính % speedup so với baseline CUTOFF = 1 và chênh lệch so với CUTOFF tối ưu trong từng (dist, n)
    speedups = np.full(len(summary), np.nan)
    diff_vs_opt = np.full(len(summary), np.nan)
    is_opt = np.zeros(len(summary), dtype=bool)

    for (dist, n), grp in summary.groupby(["distribution", "n"]):
        base_rows = grp[grp["cutoff"] == 1]
        base_med = float(base_rows["median_ms"].iloc[0]) if not base_rows.empty else np.nan
        opt_idx = grp["median_ms"].idxmin()
        opt_med = float(summary.loc[opt_idx, "median_ms"])
        is_opt[opt_idx] = True

        for idx, row in grp.iterrows():
            if not np.isnan(base_med) and base_med > 0:
                speedups[idx] = (base_med - row["median_ms"]) / base_med * 100.0
            if opt_med > 0:
                diff_vs_opt[idx] = (row["median_ms"] - opt_med) / opt_med * 100.0

    summary["speedup_vs_1_pct"] = speedups
    summary["slower_than_opt_pct"] = diff_vs_opt
    summary["is_opt"] = is_opt

    # Sắp xếp chuẩn theo thứ tự phân phối
    dist_rank = {d: i for i, d in enumerate(DIST_ORDER)}
    summary["_d_rank"] = summary["distribution"].map(lambda x: dist_rank.get(x, 99))
    summary = summary.sort_values(["_d_rank", "n", "cutoff"]).drop(columns=["_d_rank"]).reset_index(drop=True)
    return summary


def detect_optimal_plateaus_and_tests(
    df_timing: pd.DataFrame, summary: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Nhận diện vùng phẳng tối ưu (Optimal Plateau) cho từng (distribution, n):
       Tập hợp các CUTOFF không có khác biệt ý nghĩa thống kê với CUTOFF_opt (Wilcoxon p >= 0.05).
    2. Kiểm định giả thuyết cặp đôi Wilcoxon:
       CUTOFF_opt vs 1, CUTOFF_opt vs 7, CUTOFF_opt vs 10, và 7 vs 10.
    """
    overview_rows = []
    test_rows = []

    dist_rank = {d: i for i, d in enumerate(DIST_ORDER)}
    groups = sorted(
        summary.groupby(["distribution", "n"]),
        key=lambda kv: (dist_rank.get(kv[0][0], 99), kv[0][1]),
    )

    for (dist, n), grp in groups:
        sub_raw = df_timing[(df_timing["distribution"] == dist) & (df_timing["n"] == n)]
        opt_row = grp.loc[grp["median_ms"].idxmin()]
        opt_c = int(opt_row["cutoff"])
        opt_med = float(opt_row["median_ms"])

        def get_med(c: int) -> float:
            r = grp[grp["cutoff"] == c]
            return float(r["median_ms"].iloc[0]) if not r.empty else np.nan

        med_1 = get_med(1)
        med_7 = get_med(7)
        med_10 = get_med(10)

        speedup_opt_vs_1 = (med_1 - opt_med) / med_1 * 100.0 if med_1 > 0 else np.nan
        diff_opt_vs_7 = (med_7 - opt_med) / med_7 * 100.0 if med_7 > 0 else np.nan
        diff_opt_vs_10 = (med_10 - opt_med) / med_10 * 100.0 if med_10 > 0 else np.nan

        # Tìm Optimal Plateau (Wilcoxon p >= 0.05 so với CUTOFF_opt) và nhóm trong biên <= 2%
        stat_plateau = []
        practical_2pct = []
        for c in sorted(grp["cutoff"].unique()):
            c_int = int(c)
            c_med = get_med(c_int)
            if (c_med - opt_med) / opt_med * 100.0 <= 2.0:
                practical_2pct.append(c_int)
            if c_int == opt_c:
                stat_plateau.append(c_int)
            else:
                x, y = align_paired_trials(sub_raw, opt_c, c_int)
                _, pval = safe_wilcoxon(x, y)
                if pval >= ALPHA:
                    stat_plateau.append(c_int)

        overview_rows.append(
            {
                "distribution": dist,
                "n": int(n),
                "opt_cutoff": opt_c,
                "opt_median_ms": opt_med,
                "baseline_1_ms": med_1,
                "cutoff_7_ms": med_7,
                "cutoff_10_ms": med_10,
                "speedup_vs_1_pct": speedup_opt_vs_1,
                "gain_vs_7_pct": diff_opt_vs_7,
                "gain_vs_10_pct": diff_opt_vs_10,
                "stat_plateau": stat_plateau,
                "practical_2pct_plateau": practical_2pct,
            }
        )

        # Kiểm định cặp đôi theo yêu cầu phản chứng (Falsifiability)
        comparisons = [
            ("CUTOFF_opt vs 1 (Baseline)", opt_c, 1),
            ("CUTOFF_opt vs 7 (Giáo trình cũ)", opt_c, 7),
            ("CUTOFF_opt vs 10 (Giáo trình mới)", opt_c, 10),
            ("CUTOFF 7 vs 10 (Đối chiếu giáo trình)", 7, 10),
        ]

        group_pvals = []
        group_entries = []
        for label, c_a, c_b in comparisons:
            med_a = get_med(c_a)
            med_b = get_med(c_b)
            delta_ms = med_a - med_b
            delta_pct = (med_b - med_a) / med_b * 100.0 if (med_b and not np.isnan(med_b)) else np.nan

            if c_a == c_b:
                stat_w, pval = np.nan, np.nan
            else:
                x, y = align_paired_trials(sub_raw, c_a, c_b)
                stat_w, pval = safe_wilcoxon(x, y)

            group_pvals.append(pval)
            group_entries.append(
                {
                    "distribution": dist,
                    "n": int(n),
                    "comparison": label,
                    "cutoff_a": c_a,
                    "cutoff_b": c_b,
                    "median_a_ms": med_a,
                    "median_b_ms": med_b,
                    "diff_a_minus_b_ms": delta_ms,
                    "speedup_a_over_b_pct": delta_pct,
                    "wilcoxon_stat": stat_w,
                    "p_value": pval,
                }
            )

        # Hiệu chỉnh Holm-Bonferroni cho 3 phép so sánh chính liên quan đến opt (và riêng 7 vs 10 hiển thị minh bạch)
        p_holm = holm_bonferroni(group_pvals[:3]) + [group_pvals[3]]
        for entry, ph in zip(group_entries, p_holm):
            entry["p_holm"] = ph
            if entry["cutoff_a"] == entry["cutoff_b"]:
                entry["conclusion"] = "Trùng mốc đối chiếu"
            elif entry["p_value"] < ALPHA:
                direction = "nhanh hơn" if entry["diff_a_minus_b_ms"] < 0 else "chậm hơn"
                entry["conclusion"] = f"Có ý nghĩa thống kê (p < 0.05; {entry['cutoff_a']} {direction} {entry['cutoff_b']})"
            else:
                entry["conclusion"] = "Không có ý nghĩa thống kê (p ≥ 0.05)"
            test_rows.append(entry)

    return pd.DataFrame(overview_rows), pd.DataFrame(test_rows)


def compute_multi_n_random_aggregate(df_timing: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa thời gian theo T(N) / (N * log2(N)) (đơn vị: ns / (N lg N)) trên toàn bộ các mốc N của RANDOM
    nhằm triệt tiêu hiệu ứng bậc thang của một giá trị N đơn lẻ.
    """
    sub = df_timing[df_timing["distribution"] == "RANDOM"].copy()
    n_list = sorted(sub["n"].unique())
    if len(n_list) <= 1:
        return pd.DataFrame()

    med_by_nc = (
        sub.groupby(["n", "cutoff"], as_index=False)["time_ms"]
        .median()
        .rename(columns={"time_ms": "median_ms"})
    )
    med_by_nc["ns_per_nlogn"] = (med_by_nc["median_ms"] * 1e6) / (
        med_by_nc["n"] * np.log2(med_by_nc["n"])
    )

    agg = (
        med_by_nc.groupby("cutoff", as_index=False)
        .agg(
            mean_ns_per_nlogn=("ns_per_nlogn", "mean"),
            std_ns_per_nlogn=("ns_per_nlogn", "std"),
            n_count=("n", "count"),
        )
        .sort_values("cutoff")
        .reset_index(drop=True)
    )
    base_val = float(agg.loc[agg["cutoff"] == 1, "mean_ns_per_nlogn"].iloc[0])
    agg["avg_speedup_vs_1_pct"] = (base_val - agg["mean_ns_per_nlogn"]) / base_val * 100.0
    return agg


def analyze_doubling_test(df_doubling: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lập bảng Doubling Test chuẩn giáo trình (N, Median T(N), Ratio, log2(Ratio))
    và khớp hồi quy Log-Log: log2(T(N)) = b * log2(N) + c => T(N) = a * N^b.
    """
    table_rows = []
    fit_rows = []

    dist_rank = {d: i for i, d in enumerate(DIST_ORDER)}
    groups = sorted(
        df_doubling.groupby(["distribution", "cutoff"]),
        key=lambda kv: (dist_rank.get(kv[0][0], 99), kv[0][1]),
    )

    for (dist, cutoff), grp in groups:
        agg = (
            grp.groupby("n", as_index=False)["time_ms"]
            .agg(
                median_ms="median",
                q1_ms=lambda s: np.percentile(s, 25),
                q3_ms=lambda s: np.percentile(s, 75),
                trials="count",
            )
            .sort_values("n")
            .reset_index(drop=True)
        )
        agg["iqr_ms"] = agg["q3_ms"] - agg["q1_ms"]

        prev_n = None
        prev_t = None
        for _, r in agg.iterrows():
            n_val = int(r["n"])
            t_val = float(r["median_ms"])
            if prev_n is not None and n_val == 2 * prev_n and prev_t > 0:
                ratio = t_val / prev_t
                log2_ratio = float(np.log2(ratio))
            else:
                ratio = np.nan
                log2_ratio = np.nan

            table_rows.append(
                {
                    "distribution": dist,
                    "cutoff": int(cutoff),
                    "n": n_val,
                    "trials": int(r["trials"]),
                    "median_ms": t_val,
                    "iqr_ms": float(r["iqr_ms"]),
                    "ratio": ratio,
                    "log2_ratio": log2_ratio,
                }
            )
            prev_n = n_val
            prev_t = t_val

        # Hồi quy tuyến tính Log-Log: log2(T_ms) = b * log2(N) + c
        log_n = np.log2(agg["n"].to_numpy(dtype=float))
        log_t_ms = np.log2(agg["median_ms"].to_numpy(dtype=float))
        reg = stats.linregress(log_n, log_t_ms)
        b_exp = float(reg.slope)
        c_int = float(reg.intercept)
        a_ms = float(2.0 ** c_int)
        a_sec = a_ms / 1000.0
        r_squared = float(reg.rvalue ** 2)

        fit_rows.append(
            {
                "distribution": dist,
                "cutoff": int(cutoff),
                "b_exponent": b_exp,
                "c_intercept_log2_ms": c_int,
                "a_coeff_ms": a_ms,
                "a_coeff_sec": a_sec,
                "r_squared": r_squared,
                "median_at_max_n_ms": float(agg["median_ms"].iloc[-1]),
                "max_n": int(agg["n"].iloc[-1]),
            }
        )

    df_table = pd.DataFrame(table_rows)
    df_fit = pd.DataFrame(fit_rows)

    # Tính % giảm hệ số dẫn đầu a (và giảm thời gian tại max N) so với CUTOFF = 1
    a_red = np.full(len(df_fit), np.nan)
    t_red = np.full(len(df_fit), np.nan)
    for dist, grp in df_fit.groupby("distribution"):
        base = grp[grp["cutoff"] == 1]
        if base.empty:
            continue
        base_a = float(base["a_coeff_sec"].iloc[0])
        base_t = float(base["median_at_max_n_ms"].iloc[0])
        for idx, row in grp.iterrows():
            a_red[idx] = (base_a - row["a_coeff_sec"]) / base_a * 100.0
            t_red[idx] = (base_t - row["median_at_max_n_ms"]) / base_t * 100.0

    df_fit["a_reduction_vs_1_pct"] = a_red
    df_fit["time_reduction_max_n_pct"] = t_red
    return df_table, df_fit


# ==================== VẼ 5 BIỂU ĐỒ CHẤT LƯỢNG CAO (300 DPI) ====================


def setup_cutoff_log2_axis(ax: plt.Axes, cutoffs: List[int]) -> None:
    """Đặt trục X theo thang log2 để vùng quét tinh 6..16 rõ ràng, không dính chữ."""
    ax.set_xscale("log", base=2)
    preferred_ticks = [1, 2, 4, 7, 10, 16, 24, 32, 48, 64, 96, 128]
    ticks = [t for t in preferred_ticks if t in cutoffs]
    if not ticks:
        ticks = sorted(cutoffs)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(min(cutoffs) * 0.85, max(cutoffs) * 1.18)


def plot_fig1_cutoff_curve_random(
    summary: pd.DataFrame, overview: pd.DataFrame
) -> None:
    """
    fig1_cutoff_curve_random.png:
    Đường cong thời gian (Median + dải IQR) theo CUTOFF trên RANDOM ở kích thước N lớn nhất.
    Chú thích chống chồng nhãn cho CUTOFF = 7, CUTOFF = 10 và điểm/vùng tối ưu.
    """
    rand_ns = summary[summary["distribution"] == "RANDOM"]["n"].unique()
    max_n = int(np.max(rand_ns))
    sub = summary[(summary["distribution"] == "RANDOM") & (summary["n"] == max_n)].sort_values("cutoff")
    ov = overview[(overview["distribution"] == "RANDOM") & (overview["n"] == max_n)].iloc[0]

    opt_c = int(ov["opt_cutoff"])
    plateau = list(ov["stat_plateau"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    x = sub["cutoff"].to_numpy()
    y_med = sub["median_ms"].to_numpy()
    y_q1 = sub["q1_ms"].to_numpy()
    y_q3 = sub["q3_ms"].to_numpy()

    # Tô vùng phẳng tối ưu (Optimal Plateau)
    if len(plateau) >= 2:
        p_min, p_max = min(plateau), max(plateau)
        ax.axvspan(
            p_min * 0.92,
            p_max * 1.08,
            color=PALETTE["green"],
            alpha=0.11,
            zorder=1,
            label=f"Vùng tối ưu thực tế (Wilcoxon p ≥ 0.05): CUTOFF thuộc {plateau}",
        )

    # Đường baseline CUTOFF = 1
    base_med = float(sub.loc[sub["cutoff"] == 1, "median_ms"].iloc[0])
    ax.axhline(
        base_med,
        color=PALETTE["gray"],
        linestyle=":",
        linewidth=1.2,
        zorder=2,
        label=f"Baseline CUTOFF = 1 ({base_med:.1f} ms)",
    )

    # Dải IQR và đường Median
    ax.fill_between(
        x, y_q1, y_q3, color=PALETTE["sky"], alpha=0.28, zorder=2, label="Khoảng tứ phân vị (IQR: Q1–Q3)"
    )
    ax.plot(
        x,
        y_med,
        color=PALETTE["blue"],
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        zorder=3,
        label="Trung vị thời gian chạy (Median)",
    )

    # Đánh dấu 3 mốc quan trọng (gộp nhãn nếu opt trùng 7 hoặc 10)
    highlight_specs = {}
    highlight_specs[7] = {
        "color": PALETTE["orange"],
        "marker": "s",
        "size": 85,
        "tag": "CUTOFF = 7 (Giáo trình cũ)",
    }
    highlight_specs[10] = {
        "color": PALETTE["purple"],
        "marker": "D",
        "size": 85,
        "tag": "CUTOFF = 10 (Giáo trình mới)",
    }
    if opt_c in highlight_specs:
        highlight_specs[opt_c]["tag"] += "\n(Tối ưu trên i5-1035G1)"
        highlight_specs[opt_c]["color"] = PALETTE["green"]
        highlight_specs[opt_c]["marker"] = "*"
        highlight_specs[opt_c]["size"] = 180
    else:
        highlight_specs[opt_c] = {
            "color": PALETTE["green"],
            "marker": "*",
            "size": 190,
            "tag": f"CUTOFF = {opt_c} (Tối ưu i5-1035G1)",
        }

    y_min_plot = float(np.min(y_q1))
    y_max_plot = float(np.max(y_q3))
    y_span = max(y_max_plot - y_min_plot, 1.0)
    ax.set_ylim(y_min_plot - 0.12 * y_span, y_max_plot + 0.34 * y_span)

    # Đặt các khung chú thích trong vùng trống phía trên đường cong (dưới legend) để không chạm trục hoành hay dải IQR
    ordered_keys = sorted(highlight_specs.keys())
    offset_presets = {
        3: [(-16, 105), (0, 128), (28, 86)],
        2: [(-10, 105), (24, 92)],
    }
    offsets = offset_presets[len(ordered_keys)]

    for key, (dx, dy) in zip(ordered_keys, offsets):
        spec = highlight_specs[key]
        row = sub[sub["cutoff"] == key].iloc[0]
        px = float(row["cutoff"])
        py = float(row["median_ms"])
        sp = float(row["speedup_vs_1_pct"])

        ax.scatter(
            [px],
            [py],
            color=spec["color"],
            marker=spec["marker"],
            s=spec["size"],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        va = "bottom" if dy > 0 else "top"
        ha = "center" if dx == 0 else ("right" if dx < 0 else "left")
        label_txt = f"{spec['tag']}\n{py:.1f} ms ({sp:+.2f}%)"
        ax.annotate(
            label_txt,
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8.8,
            fontweight="bold",
            color=PALETTE["dark"],
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="white",
                edgecolor=spec["color"],
                linewidth=1.3,
                alpha=0.95,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=spec["color"],
                lw=1.3,
                connectionstyle="arc3,rad=0.12",
            ),
            zorder=6,
        )

    setup_cutoff_log2_axis(ax, sorted(sub["cutoff"].tolist()))
    ax.set_xlabel("Ngưỡng CUTOFF (thang log₂)")
    ax.set_ylabel("Thời gian thực thi (ms)")
    ax.set_title(
        f"Đường cong thời gian chạy Mergesort theo CUTOFF — Dữ liệu RANDOM (N = {max_n:,})\n"
        f"HP Pavilion i5-1035G1 | OpenJDK 26.0.1 | {int(sub['trials'].iloc[0])} lượt lặp xen kẽ ngẫu nhiên"
    )
    ax.grid(True, which="major")
    ax.legend(loc="upper left", framealpha=0.94)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_cutoff_curve_random.png")
    plt.close(fig)


def plot_fig2_speedup_by_distribution(
    summary: pd.DataFrame, primary_n: int
) -> None:
    """
    fig2_speedup_by_distribution.png:
    So sánh % Speedup so với CUTOFF = 1 trên cả 4 phân phối tại primary_n.
    """
    sub = summary[summary["n"] == primary_n].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    ax.axhline(0.0, color=PALETTE["dark"], linestyle="--", linewidth=1.1, label="Mốc đối chứng CUTOFF = 1 (0%)")
    ax.axvline(7, color=PALETTE["orange"], linestyle=":", linewidth=1.1, alpha=0.85)
    ax.axvline(10, color=PALETTE["purple"], linestyle=":", linewidth=1.1, alpha=0.85)

    for dist in DIST_ORDER:
        dsub = sub[sub["distribution"] == dist].sort_values("cutoff")
        if dsub.empty:
            continue
        opt_row = dsub.loc[dsub["speedup_vs_1_pct"].idxmax()]
        opt_c = int(opt_row["cutoff"])
        opt_sp = float(opt_row["speedup_vs_1_pct"])

        ax.plot(
            dsub["cutoff"],
            dsub["speedup_vs_1_pct"],
            color=DIST_COLORS[dist],
            marker=DIST_MARKERS[dist],
            linewidth=2.0,
            markersize=5.2,
            label=f"{DIST_LABELS_VI[dist]}  [Tối ưu: C={opt_c}, {opt_sp:+.1f}%]",
        )
        ax.scatter(
            [opt_c],
            [opt_sp],
            color=DIST_COLORS[dist],
            marker="*",
            s=190,
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )

    # Ghi nhãn dọc tinh tế cho mốc 7 và 10 ở mép dưới
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo - 0.08 * (y_hi - y_lo), y_hi + 0.12 * (y_hi - y_lo))
    y_text = ax.get_ylim()[0] + 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(7 * 0.94, y_text, "C=7", color=PALETTE["orange"], fontsize=8.5, fontweight="bold", ha="right")
    ax.text(10 * 1.06, y_text, "C=10", color=PALETTE["purple"], fontsize=8.5, fontweight="bold", ha="left")

    setup_cutoff_log2_axis(ax, sorted(sub["cutoff"].unique().tolist()))
    ax.set_xlabel("Ngưỡng CUTOFF (thang log₂)")
    ax.set_ylabel("Mức cải thiện tốc độ so với CUTOFF = 1 (%)")
    ax.set_title(
        f"Dịch chuyển điểm CUTOFF tối ưu theo 4 phân phối dữ liệu đầu vào (N = {primary_n:,})\n"
        "Giá trị dương (+): nhanh hơn CUTOFF = 1 | Giá trị âm (−): chậm hơn CUTOFF = 1"
    )
    ax.grid(True, which="major")
    ax.legend(loc="lower left", framealpha=0.94)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_speedup_by_distribution.png")
    plt.close(fig)


def plot_fig3_cost_model_vs_time(
    summary: pd.DataFrame, df_ops: pd.DataFrame
) -> None:
    """
    fig3_cost_model_vs_time.png:
    2 subplot đồng bộ trục X (thang log2) đối chiếu Số phép so sánh (Cost Model)
    và Thời gian chạy thực tế (Wall-clock Time) trên RANDOM tại N của ops_count.csv.
    """
    ops_sub = df_ops[df_ops["distribution"] == "RANDOM"].sort_values("cutoff")
    ops_n = int(ops_sub["n"].iloc[0])
    time_sub = summary[(summary["distribution"] == "RANDOM") & (summary["n"] == ops_n)].sort_values("cutoff")
    merged = pd.merge(ops_sub, time_sub, on=["distribution", "n", "cutoff"], how="inner")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.08]})

    x = merged["cutoff"].to_numpy()
    comp_m = merged["compares"].to_numpy() / 1e6
    theo_m = float(merged["theoretical_nlogn"].iloc[0]) / 1e6

    opt_time_row = merged.loc[merged["median_ms"].idxmin()]
    opt_time_c = int(opt_time_row["cutoff"])
    min_comp_row = merged.loc[merged["compares"].idxmin()]
    min_comp_c = int(min_comp_row["cutoff"])

    # --- Subplot trên: Cost Model (# compares) ---
    ax1.axhline(
        theo_m,
        color=PALETTE["gray"],
        linestyle="--",
        linewidth=1.2,
        label=f"Cận trên lý thuyết N·lg(N) = {theo_m:.2f} triệu",
    )
    ax1.plot(
        x,
        comp_m,
        color=PALETTE["red"],
        linewidth=2.2,
        marker="s",
        markersize=5.2,
        label="Số phép so sánh thực tế (compareTo)",
    )
    ax1.axvline(opt_time_c, color=PALETTE["green"], linestyle="-.", linewidth=1.3, alpha=0.85)
    ax1.scatter(
        [min_comp_c],
        [float(min_comp_row["compares"]) / 1e6],
        color=PALETTE["red"],
        marker="o",
        s=95,
        edgecolor="black",
        zorder=5,
    )
    ax1.annotate(
        f"Số phép so sánh nhỏ nhất\nC = {min_comp_c} ({float(min_comp_row['compares'])/1e6:.2f}M)",
        xy=(min_comp_c, float(min_comp_row["compares"]) / 1e6),
        xytext=(28, 22),
        textcoords="offset points",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["red"]),
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=1.2),
    )

    # Ghi chú vùng phân kỳ bậc hai ở CUTOFF lớn nhất
    last_row = merged.iloc[-1]
    ax1.annotate(
        f"Chi phí bậc hai Insertion Sort bùng nổ\nC = {int(last_row['cutoff'])}: {float(last_row['compares'])/1e6:.2f}M phép",
        xy=(int(last_row["cutoff"]), float(last_row["compares"]) / 1e6),
        xytext=(-175, -36),
        textcoords="offset points",
        fontsize=8.6,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["red"]),
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=1.2),
    )

    ax1.set_ylabel("Số phép so sánh (triệu)")
    ax1.set_title(
        f"Đối chiếu Mô hình Chi phí (Số phép so sánh) vs Thời gian chạy thực tế — RANDOM (N = {ops_n:,})"
    )
    ax1.grid(True, which="major")
    ax1.legend(loc="upper left")

    # --- Subplot dưới: Thời gian chạy thực tế (Median + IQR) ---
    ax2.fill_between(
        x,
        merged["q1_ms"],
        merged["q3_ms"],
        color=PALETTE["sky"],
        alpha=0.28,
        label="Khoảng tứ phân vị (IQR: Q1–Q3)",
    )
    ax2.plot(
        x,
        merged["median_ms"],
        color=PALETTE["blue"],
        linewidth=2.2,
        marker="o",
        markersize=5.2,
        label="Thời gian thực thi trung vị (Median ms)",
    )
    ax2.axvline(
        opt_time_c,
        color=PALETTE["green"],
        linestyle="-.",
        linewidth=1.3,
        alpha=0.85,
        label=f"Tối ưu thời gian thực: CUTOFF = {opt_time_c} ({float(opt_time_row['median_ms']):.1f} ms)",
    )
    ax2.scatter(
        [opt_time_c],
        [float(opt_time_row["median_ms"])],
        color=PALETTE["green"],
        marker="*",
        s=190,
        edgecolor="black",
        zorder=6,
    )
    ax2.annotate(
        f"Tối ưu thời gian thực: C = {opt_time_c}\n"
        f"{float(opt_time_row['median_ms']):.1f} ms | {float(opt_time_row['compares'])/1e6:.2f}M phép so sánh",
        xy=(opt_time_c, float(opt_time_row["median_ms"])),
        xytext=(36, 30),
        textcoords="offset points",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["green"]),
        arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.2),
    )

    setup_cutoff_log2_axis(ax2, sorted(merged["cutoff"].tolist()))
    ax2.set_xlabel("Ngưỡng CUTOFF (thang log₂)")
    ax2.set_ylabel("Thời gian thực thi (ms)")
    ax2.grid(True, which="major")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_cost_model_vs_time.png")
    plt.close(fig)


def plot_fig4_doubling_loglog(
    df_doubling_table: pd.DataFrame, df_doubling_fit: pd.DataFrame
) -> None:
    """
    fig4_doubling_loglog.png:
    2 khung hình cạnh nhau cho phân phối RANDOM:
    - Trái: Đồ thị Log-Log (log2 N vs log2 T(N)) kèm đường hồi quy T(N) = a·N^b cho CUTOFF = 1 và CUTOFF tối ưu.
    - Phải: Thời gian chuẩn hóa T(N)/(N·log2 N) (ns/đơn vị) theo N, tách bạch rõ sự giảm hằng số hệ thống a.
    """
    dist = "RANDOM"
    tab = df_doubling_table[df_doubling_table["distribution"] == dist].copy()
    fits = df_doubling_fit[df_doubling_fit["distribution"] == dist].copy()

    # Chọn mốc tối ưu trong doubling_test.csv (không phải 1, có thời gian nhỏ nhất tại max N)
    non_base = fits[fits["cutoff"] != 1]
    opt_fit = non_base.loc[non_base["median_at_max_n_ms"].idxmin()]
    opt_c = int(opt_fit["cutoff"])
    base_fit = fits[fits["cutoff"] == 1].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # --- Panel trái: Log-Log Regression (CUTOFF = 1 vs CUTOFF tối ưu) ---
    for cutoff, fit_row, color, marker, title_lbl in [
        (1, base_fit, PALETTE["red"], "o", "CUTOFF = 1 (Baseline)"),
        (opt_c, opt_fit, PALETTE["blue"], "s", f"CUTOFF = {opt_c} (Tối ưu)"),
    ]:
        sub = tab[tab["cutoff"] == cutoff].sort_values("n")
        lx = np.log2(sub["n"].to_numpy(dtype=float))
        ly = np.log2(sub["median_ms"].to_numpy(dtype=float))
        b = float(fit_row["b_exponent"])
        c_int = float(fit_row["c_intercept_log2_ms"])
        a_sec = float(fit_row["a_coeff_sec"])
        r2 = float(fit_row["r_squared"])

        ax1.scatter(lx, ly, color=color, marker=marker, s=55, zorder=4)
        lx_line = np.linspace(lx.min(), lx.max(), 100)
        ly_line = b * lx_line + c_int
        eq_label = (
            f"{title_lbl}\n"
            f"  T(N) = {a_sec:.2e}·N^{b:.3f} (s) | R² = {r2:.5f}"
        )
        ax1.plot(lx_line, ly_line, color=color, linewidth=2.1, label=eq_label, zorder=3)

    ax1.set_xlabel("log₂(N)")
    ax1.set_ylabel("log₂(T(N))  [T tính theo ms]")
    ax1.set_title("Hồi quy Log-Log kiểm chứng Quy tắc Hàm mũ (RANDOM)")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.8)

    # --- Panel phải: Chuẩn hóa T(N)/(N·log2 N) cho tất cả mốc trong Doubling Test ---
    style_map = {
        1: (PALETTE["red"], "o", "CUTOFF = 1 (Baseline)"),
        7: (PALETTE["orange"], "^", "CUTOFF = 7 (Giáo trình cũ)"),
        10: (PALETTE["purple"], "D", "CUTOFF = 10 (Giáo trình mới)"),
    }
    for cutoff in sorted(fits["cutoff"].unique()):
        c_int = int(cutoff)
        sub = tab[tab["cutoff"] == c_int].sort_values("n")
        ns_norm = (sub["median_ms"] * 1e6) / (sub["n"] * np.log2(sub["n"]))
        if c_int == opt_c and c_int not in style_map:
            color, marker, lbl = PALETTE["green"], "*", f"CUTOFF = {c_int} (Tối ưu)"
            ms = 9.0
        elif c_int == opt_c:
            base_lbl = style_map[c_int][2]
            lbl = base_lbl.replace(")", ", Tối ưu)") if ")" in base_lbl else f"{base_lbl} (Tối ưu)"
            color, marker = PALETTE["green"], "*"
            ms = 9.0
        else:
            color, marker, lbl = style_map.get(c_int, (PALETTE["blue"], "s", f"CUTOFF = {c_int}"))
            ms = 6.0

        ax2.plot(
            np.log2(sub["n"]),
            ns_norm,
            color=color,
            marker=marker,
            markersize=ms,
            linewidth=2.0,
            label=lbl,
        )

    ax2.set_xlabel("log₂(N)  [từ 2¹⁶ đến 2²³]")
    ax2.set_ylabel("Thời gian chuẩn hóa  T(N) / (N·lg N)  [ns]")
    ax2.set_title("Tách bạch hằng số hệ thống a trên các mốc CUTOFF (RANDOM)")
    ax2.grid(True)
    ax2.legend(loc="best", fontsize=8.8)

    fig.suptitle(
        "Doubling Test: CUTOFF tối ưu hằng số dẫn đầu a và giữ nguyên bậc tăng trưởng Θ(N lg N)",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_doubling_loglog.png")
    plt.close(fig)


def plot_fig5_variance_boxplot(
    df_timing: pd.DataFrame, summary: pd.DataFrame, primary_n: int
) -> None:
    """
    fig5_variance_boxplot.png:
    Boxplot kèm điểm dữ liệu từng lượt đo cho các mốc tiêu biểu trên RANDOM tại primary_n.
    """
    sub_sum = summary[(summary["distribution"] == "RANDOM") & (summary["n"] == primary_n)]
    sub_raw = df_timing[(df_timing["distribution"] == "RANDOM") & (df_timing["n"] == primary_n)]
    opt_c = int(sub_sum.loc[sub_sum["median_ms"].idxmin(), "cutoff"])

    # Chọn danh sách mốc tiêu biểu không trùng lặp, giữ nguyên thứ tự tăng dần
    candidates = [1, 7, 10, opt_c, 64, 128]
    available = set(sub_sum["cutoff"].astype(int).tolist())
    selected = sorted({c for c in candidates if c in available})
    if len(selected) < 6 and 24 in available and 24 not in selected:
        selected = sorted(selected + [24])

    data_groups = [
        sub_raw[sub_raw["cutoff"] == c]["time_ms"].to_numpy(dtype=float) for c in selected
    ]

    labels = []
    box_colors = []
    for c in selected:
        if c == 1:
            labels.append("CUTOFF = 1\n(Baseline)")
            box_colors.append("#E8B4B8")
        elif c == opt_c:
            suffix = " & Cũ" if c == 7 else (" & Mới" if c == 10 else "")
            labels.append(f"CUTOFF = {c}\n(Tối ưu{suffix})")
            box_colors.append("#A8E6CF")
        elif c == 7:
            labels.append("CUTOFF = 7\n(Giáo trình cũ)")
            box_colors.append("#FFD3B6")
        elif c == 10:
            labels.append("CUTOFF = 10\n(Giáo trình mới)")
            box_colors.append("#D7BDE2")
        else:
            labels.append(f"CUTOFF = {c}\n(Mốc lớn)")
            box_colors.append("#D6EAF8")

    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    bp = ax.boxplot(
        data_groups,
        patch_artist=True,
        widths=0.52,
        showfliers=True,
        medianprops=dict(color=PALETTE["dark"], linewidth=2.0),
        whiskerprops=dict(color="#444444", linewidth=1.2),
        capprops=dict(color="#444444", linewidth=1.2),
        flierprops=dict(marker="x", markeredgecolor=PALETTE["red"], markersize=6),
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.1)

    # Chồng các điểm đo thô (jitter tất định) để thấy rõ độ tụ tập dữ liệu
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_global_max = max(float(np.max(g)) for g in data_groups)
    y_global_min = min(float(np.min(g)) for g in data_groups)
    y_span = max(y_global_max - y_global_min, 1.0)

    for idx, vals in enumerate(data_groups, start=1):
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            color=PALETTE["blue"],
            alpha=0.55,
            s=22,
            zorder=4,
        )
        med = float(np.median(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        top_y = float(np.max(vals))
        ax.text(
            idx,
            top_y + 0.03 * y_span,
            f"Med: {med:.1f}\nIQR: {iqr:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color=PALETTE["dark"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        )

    ax.set_ylim(y_global_min - 0.08 * y_span, y_global_max + 0.20 * y_span)
    ax.set_xticks(range(1, len(selected) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Thời gian thực thi từng lượt đo (ms)")
    ax.set_title(
        f"Phân bố thời gian chạy qua {len(data_groups[0])} lượt lặp tại các mốc CUTOFF tiêu biểu — RANDOM (N = {primary_n:,})\n"
        "Đánh giá độ ổn định phép đo và mức độ nhiễu hệ thống trên laptop HP Pavilion i5-1035G1"
    )
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_variance_boxplot.png")
    plt.close(fig)


# ==================== XUẤT BÁO CÁO MARKDOWN ====================


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def write_markdown_report(
    summary: pd.DataFrame,
    overview: pd.DataFrame,
    tests: pd.DataFrame,
    multi_n_agg: pd.DataFrame,
    df_ops: pd.DataFrame,
    df_doubling_table: pd.DataFrame,
    df_doubling_fit: pd.DataFrame,
    primary_n: int,
) -> None:
    lines: List[str] = []
    lines.append("# BẢNG TỔNG HỢP SỐ LIỆU PHÂN TÍCH THỰC NGHIỆM MERGESORT CUTOFF\n")
    lines.append(
        "- **Phương pháp ước lượng chính:** Trung vị (`Median`) và Khoảng tứ phân vị (`IQR = Q3 - Q1`) kèm khoảng tin cậy 95% (Percentile Bootstrap 10,000 mẫu lặp, `seed = 20261120`).\n"
    )
    lines.append(
        "- **Kiểm định giả thuyết:** Wilcoxon signed-rank test cặp đôi theo lượt đo (`α = 0.05`, kèm hiệu chỉnh Holm-Bonferroni).\n"
    )

    # --- 1. TỔNG QUAN ĐIỂM TỐI ƯU & VÙNG PHẲNG TỐI ƯU ---
    lines.append("\n## 1. Tổng quan điểm CUTOFF tối ưu và Vùng tối ưu thực tế (Optimal Plateau)\n")
    lines.append(
        "| Phân phối | N | CUTOFF tối ưu | Median tối ưu (ms) | Median C=1 (ms) | Speedup vs C=1 (%) | Chênh lệch vs C=7 (%) | Chênh lệch vs C=10 (%) | Vùng tối ưu thống kê (Wilcoxon p ≥ 0.05) | Biên thực tế (≤ 2% so với tối ưu) |"
    )
    lines.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |"
    )
    for _, r in overview.iterrows():
        lines.append(
            f"| `{r['distribution']}` | {int(r['n']):,} | **{int(r['opt_cutoff'])}** | "
            f"{r['opt_median_ms']:.2f} | {r['baseline_1_ms']:.2f} | {r['speedup_vs_1_pct']:+.2f}% | "
            f"{r['gain_vs_7_pct']:+.2f}% | {r['gain_vs_10_pct']:+.2f}% | "
            f"`{r['stat_plateau']}` | `{r['practical_2pct_plateau']}` |"
        )

    # --- 2. KIỂM ĐỊNH WILCOXON ---
    lines.append("\n## 2. Kết quả kiểm định giả thuyết cặp đôi (Wilcoxon Signed-Rank Test, α = 0.05)\n")
    lines.append(
        "| Phân phối | N | Cặp so sánh | Mốc A | Mốc B | Median A (ms) | Median B (ms) | Chênh lệch (B→A %) | p-value | p-value (Holm) | Kết luận |"
    )
    lines.append(
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"
    )
    for _, r in tests.iterrows():
        lines.append(
            f"| `{r['distribution']}` | {int(r['n']):,} | {r['comparison']} | "
            f"{int(r['cutoff_a'])} | {int(r['cutoff_b'])} | "
            f"{r['median_a_ms']:.2f} | {r['median_b_ms']:.2f} | {r['speedup_a_over_b_pct']:+.2f}% | "
            f"{fmt_p(r['p_value'])} | {fmt_p(r['p_holm'])} | {r['conclusion']} |"
        )

    # --- 3. TỔNG HỢP MULTI-N RANDOM ---
    if not multi_n_agg.empty:
        lines.append("\n## 3. Chuẩn hóa đa kích thước N trên RANDOM (Triệt tiêu hiệu ứng bậc thang)\n")
        lines.append(
            "Thời gian được chuẩn hóa về `T(N) / (N · log₂ N)` (đơn vị: ns cho mỗi đơn vị `N lg N`) và lấy trung bình qua tất cả các mốc `N` của `RANDOM`.\n"
        )
        lines.append("| CUTOFF | Trung bình T/(N·lg N) (ns) | Độ lệch chuẩn giữa các N (ns) | Số mốc N | % Cải thiện trung bình vs C=1 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        best_multi_idx = multi_n_agg["mean_ns_per_nlogn"].idxmin()
        for idx, r in multi_n_agg.iterrows():
            c_str = f"**{int(r['cutoff'])} ★**" if idx == best_multi_idx else f"{int(r['cutoff'])}"
            lines.append(
                f"| {c_str} | {r['mean_ns_per_nlogn']:.3f} | {r['std_ns_per_nlogn']:.3f} | "
                f"{int(r['n_count'])} | {r['avg_speedup_vs_1_pct']:+.2f}% |"
            )

    # --- 4. MÔ HÌNH CHI PHÍ VS THỜI GIAN THỰC ---
    lines.append("\n## 4. Đối chiếu Mô hình Chi phí (Số phép so sánh) với Thời gian thực\n")
    ops_n = int(df_ops["n"].iloc[0])
    ops_merged = pd.merge(
        df_ops,
        summary[summary["n"] == ops_n][["distribution", "n", "cutoff", "median_ms", "speedup_vs_1_pct"]],
        on=["distribution", "n", "cutoff"],
        how="left",
    )
    for dist in DIST_ORDER:
        dsub = ops_merged[ops_merged["distribution"] == dist].sort_values("cutoff")
        if dsub.empty:
            continue
        base_comp = float(dsub.loc[dsub["cutoff"] == 1, "compares"].iloc[0])
        lines.append(f"\n### 4.{DIST_ORDER.index(dist)+1}. Phân phối `{dist}` (N = {ops_n:,})\n")
        lines.append("| CUTOFF | Số phép so sánh | Tỉ lệ / (N·lg N) | % Thay đổi phép so sánh vs C=1 | Median thời gian (ms) | % Speedup vs C=1 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in dsub.iterrows():
            ratio_nlogn = r["compares"] / r["theoretical_nlogn"]
            delta_comp = (r["compares"] - base_comp) / base_comp * 100.0
            lines.append(
                f"| {int(r['cutoff'])} | {int(r['compares']):,} | {ratio_nlogn:.4f} | "
                f"{delta_comp:+.2f}% | {r['median_ms']:.2f} | {r['speedup_vs_1_pct']:+.2f}% |"
            )

    # --- 5. DOUBLING TEST & QUY TẮC HÀM MŨ ---
    lines.append("\n## 5. Doubling Test & Kiểm chứng Quy tắc Hàm mũ $T(N) = a \\cdot N^b$\n")
    lines.append("### 5.1. Tổng hợp hồi quy Log-Log ($\\log_2 T(N) = b \\cdot \\log_2 N + c$)\n")
    lines.append(
        "| Phân phối | CUTOFF | Số mũ thực nghiệm $b$ | Hệ số $a$ (ms) | Hệ số $a$ (giây) | $R^2$ | Median tại max N (ms) | % Giảm thời gian tại max N vs C=1 |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in df_doubling_fit.iterrows():
        lines.append(
            f"| `{r['distribution']}` | {int(r['cutoff'])} | {r['b_exponent']:.4f} | "
            f"{r['a_coeff_ms']:.4e} | {r['a_coeff_sec']:.4e} | {r['r_squared']:.6f} | "
            f"{r['median_at_max_n_ms']:.2f} | {r['time_reduction_max_n_pct']:+.2f}% |"
        )

    lines.append("\n### 5.2. Bảng chi tiết Doubling Test theo chuẩn giáo trình\n")
    for (dist, cutoff), grp in df_doubling_table.groupby(["distribution", "cutoff"], sort=False):
        lines.append(f"\n#### `{dist}` — `CUTOFF = {int(cutoff)}`\n")
        lines.append("| N | Số lượt | Median T(N) (ms) | IQR (ms) | Ratio T(N)/T(N/2) | log₂(Ratio) |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in grp.iterrows():
            ratio_s = "—" if pd.isna(r["ratio"]) else f"{r['ratio']:.3f}"
            log2_s = "—" if pd.isna(r["log2_ratio"]) else f"{r['log2_ratio']:.3f}"
            lines.append(
                f"| {int(r['n']):,} | {int(r['trials'])} | {r['median_ms']:.2f} | "
                f"{r['iqr_ms']:.2f} | {ratio_s} | {log2_s} |"
            )

    # --- 6. THỐNG KÊ MÔ TẢ CHI TIẾT ---
    lines.append("\n## 6. Bảng thống kê mô tả đầy đủ theo từng nhóm `distribution × N × cutoff`\n")
    for (dist, n), grp in summary.groupby(["distribution", "n"], sort=False):
        lines.append(f"\n### `{dist}` — N = {int(n):,} ({int(grp['trials'].iloc[0])} lượt/mốc)\n")
        lines.append(
            "| CUTOFF | Median (ms) | Mean (ms) | SD (ms) | IQR (ms) | Min (ms) | Max (ms) | 95% CI Median (ms) | Speedup vs C=1 (%) |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |")
        for _, r in grp.iterrows():
            c_txt = f"**{int(r['cutoff'])} ★**" if r["is_opt"] else f"{int(r['cutoff'])}"
            ci_txt = f"[{r['ci95_lo_ms']:.2f}, {r['ci95_hi_ms']:.2f}]"
            lines.append(
                f"| {c_txt} | {r['median_ms']:.2f} | {r['mean_ms']:.2f} | {r['std_ms']:.2f} | "
                f"{r['iqr_ms']:.2f} | {r['min_ms']:.2f} | {r['max_ms']:.2f} | {ci_txt} | "
                f"{r['speedup_vs_1_pct']:+.2f}% |"
            )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ==================== HÀM CHÍNH ====================


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    print("=== BAT DAU PHAN TICH DU LIEU THUC NGHIEM MERGESORT CUTOFF ===")
    df_timing, df_ops, df_doubling = load_datasets()

    # Xác định primary_n (kích thước N chung xuất hiện ở đủ cả 4 phân phối)
    n_counts = df_timing.groupby("n")["distribution"].nunique()
    primary_n = int(n_counts[n_counts == n_counts.max()].index.max())
    print(f"Da nap du lieu thanh cong | primary_n = {primary_n:,}")

    summary = compute_descriptive_stats(df_timing)
    overview, tests = detect_optimal_plateaus_and_tests(df_timing, summary)
    multi_n_agg = compute_multi_n_random_aggregate(df_timing)
    df_doubling_table, df_doubling_fit = analyze_doubling_test(df_doubling)

    print("[1/6] Ve fig1_cutoff_curve_random.png...")
    plot_fig1_cutoff_curve_random(summary, overview)

    print("[2/6] Ve fig2_speedup_by_distribution.png...")
    plot_fig2_speedup_by_distribution(summary, primary_n)

    print("[3/6] Ve fig3_cost_model_vs_time.png...")
    plot_fig3_cost_model_vs_time(summary, df_ops)

    print("[4/6] Ve fig4_doubling_loglog.png...")
    plot_fig4_doubling_loglog(df_doubling_table, df_doubling_fit)

    print("[5/6] Ve fig5_variance_boxplot.png...")
    plot_fig5_variance_boxplot(df_timing, summary, primary_n)

    print("[6/6] Xuat bao cao bang so lieu results/summary_report_tables.md...")
    write_markdown_report(
        summary=summary,
        overview=overview,
        tests=tests,
        multi_n_agg=multi_n_agg,
        df_ops=df_ops,
        df_doubling_table=df_doubling_table,
        df_doubling_fit=df_doubling_fit,
        primary_n=primary_n,
    )

    # In tóm tắt nhanh ra console
    print("\n=== TOM TAT NHANH KET QUA ===")
    for _, r in overview.iterrows():
        print(
            f"  {r['distribution']:16s} | N={int(r['n']):9,d} | "
            f"CUTOFF_opt={int(r['opt_cutoff']):3d} ({r['opt_median_ms']:6.2f} ms, "
            f"{r['speedup_vs_1_pct']:+5.2f}% vs C=1) | Plateau={r['stat_plateau']}"
        )
    print(f"\nHoan tat! Bieu do tai: {FIGURES_DIR}")
    print(f"Bang bao cao tai     : {REPORT_MD}")


if __name__ == "__main__":
    main()