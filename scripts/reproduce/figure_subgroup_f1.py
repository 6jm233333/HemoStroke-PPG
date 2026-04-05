
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# allow "python scripts/reproduce/figure_subgroup_f1.py" from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.subgroup_statistics import (  # noqa: E402
    build_significance_annotations,
    run_multi_group_test,
    run_two_group_test,
    summarize_subgroup_f1,
)


PANEL_ALIASES = ["panel", "domain", "category", "subgroup_type", "panel_name"]
SUBGROUP_ALIASES = ["subgroup", "group", "group_name", "level", "label"]
FOLD_ALIASES = ["fold", "cv_fold", "split", "fold_id"]
F1_ALIASES = ["f1", "f1_score", "macro_f1", "binary_f1"]


DEFAULT_PANEL_ORDER = ["Clinical Risk", "Race", "Age", "Gender"]
DEFAULT_SUBGROUP_ORDER = {
    "Clinical Risk": ["Low Risk", "Medium Risk", "High Risk"],
    "Race": ["White", "Asian", "Black"],
    "Age": ["Elderly (≥65)", "Non-Elderly"],
    "Gender": ["Male", "Female"],
}
DEFAULT_PANEL_COLORS = {
    "Clinical Risk": "#E66789",
    "Race": "#D5D05E",
    "Age": "#4E9AE8",
    "Gender": "#2EC4B6",
}


def _find_first_existing(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    colset = {str(c).strip(): c for c in columns}
    for a in aliases:
        if a in colset:
            return colset[a]
    return None


def _collect_csv_files(path_like: str) -> List[Path]:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.csv"))


def _normalize_panel_name(x: str) -> str:
    s = str(x).strip().lower()

    mapping = {
        "clinical risk": "Clinical Risk",
        "risk": "Clinical Risk",
        "comorbidity": "Clinical Risk",
        "race": "Race",
        "ethnicity": "Race",
        "age": "Age",
        "gender": "Gender",
        "sex": "Gender",
    }
    return mapping.get(s, str(x).strip())


def _normalize_subgroup_name(panel: str, subgroup: str) -> str:
    s = str(subgroup).strip()
    p = panel

    if p == "Clinical Risk":
        lut = {
            "low": "Low Risk",
            "low risk": "Low Risk",
            "medium": "Medium Risk",
            "medium risk": "Medium Risk",
            "mid risk": "Medium Risk",
            "high": "High Risk",
            "high risk": "High Risk",
        }
        return lut.get(s.lower(), s)

    if p == "Race":
        lut = {
            "white": "White",
            "asian": "Asian",
            "black": "Black",
            "black or african american": "Black",
        }
        return lut.get(s.lower(), s)

    if p == "Age":
        s_low = s.lower()
        if "elder" in s_low or "≥65" in s or ">=65" in s:
            return "Elderly (≥65)"
        if "non" in s_low or "<65" in s or "under" in s_low:
            return "Non-Elderly"
        return s

    if p == "Gender":
        lut = {
            "male": "Male",
            "female": "Female",
            "man": "Male",
            "woman": "Female",
        }
        return lut.get(s.lower(), s)

    return s


def _load_fold_level_f1(path_like: str) -> pd.DataFrame:
    frames = []
    files = _collect_csv_files(path_like)

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        if df.empty:
            continue

        df.columns = [str(c).strip() for c in df.columns]

        panel_col = _find_first_existing(df.columns, PANEL_ALIASES)
        subgroup_col = _find_first_existing(df.columns, SUBGROUP_ALIASES)
        fold_col = _find_first_existing(df.columns, FOLD_ALIASES)
        f1_col = _find_first_existing(df.columns, F1_ALIASES)

        if subgroup_col is None or f1_col is None:
            continue

        keep = [subgroup_col, f1_col]
        rename = {subgroup_col: "subgroup", f1_col: "f1"}

        if panel_col is not None:
            keep.append(panel_col)
            rename[panel_col] = "panel"
        if fold_col is not None:
            keep.append(fold_col)
            rename[fold_col] = "fold"

        part = df[keep].rename(columns=rename).copy()

        if "panel" not in part.columns:
            stem = fp.stem.lower()
            inferred_panel = None
            for key in ["risk", "race", "age", "gender", "sex"]:
                if key in stem:
                    inferred_panel = _normalize_panel_name(key)
                    break
            if inferred_panel is None:
                raise ValueError(
                    f"Could not infer panel for '{fp.name}'. "
                    "Please include a panel column or put panel name in the filename."
                )
            part["panel"] = inferred_panel

        if "fold" not in part.columns:
            m = re.search(r"(?:fold|cv|split)[_\-]?(\d+)", fp.stem.lower())
            part["fold"] = int(m.group(1)) if m else 0

        part["panel"] = part["panel"].map(_normalize_panel_name)
        part["subgroup"] = [
            _normalize_subgroup_name(panel, subgroup)
            for panel, subgroup in zip(part["panel"], part["subgroup"])
        ]
        part["f1"] = pd.to_numeric(part["f1"], errors="coerce")
        part["fold"] = pd.to_numeric(part["fold"], errors="coerce")
        part = part.dropna(subset=["f1", "fold"])
        part["fold"] = part["fold"].astype(int)

        frames.append(part)

    if not frames:
        raise ValueError(f"No usable fold-level subgroup F1 CSVs found under: {path_like}")

    return pd.concat(frames, ignore_index=True)


def _plot_sig_bracket(ax, x1: float, x2: float, y: float, text: str, h: float = 0.004) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c="black")
    ax.text((x1 + x2) / 2.0, y + h + 0.002, text, ha="center", va="bottom", fontsize=11)


def _panel_test_table(panel_df: pd.DataFrame, panel_name: str) -> pd.DataFrame:
    unique_groups = list(pd.unique(panel_df["subgroup"]))
    if len(unique_groups) < 2:
        return pd.DataFrame()

    if len(unique_groups) == 2:
        pairwise = run_two_group_test(panel_df, group_col="subgroup", value_col="f1")
        pairwise["panel"] = panel_name
        return pairwise

    res = run_multi_group_test(panel_df, group_col="subgroup", value_col="f1", correction="bonferroni")
    pairwise = res["pairwise"].copy()
    pairwise["panel"] = panel_name
    pairwise["global_test"] = res["global"]["test"].iloc[0]
    pairwise["global_p_raw"] = res["global"]["p_raw"].iloc[0]
    pairwise["global_p_adj"] = res["global"]["p_adj"].iloc[0]
    return pairwise


def plot_subgroup_figure(
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    axes = axes.flatten()
    panel_letters = ["a", "b", "c", "d"]

    for i, panel_name in enumerate(DEFAULT_PANEL_ORDER):
        ax = axes[i]
        panel_summary = summary_df[summary_df["panel"] == panel_name].copy()

        subgroup_order = DEFAULT_SUBGROUP_ORDER.get(panel_name, panel_summary["subgroup"].tolist())
        panel_summary["subgroup"] = pd.Categorical(panel_summary["subgroup"], categories=subgroup_order, ordered=True)
        panel_summary = panel_summary.sort_values("subgroup").reset_index(drop=True)

        x = np.arange(len(panel_summary))
        y = panel_summary["f1_mean"].to_numpy(dtype=float)
        err = panel_summary["f1_std"].to_numpy(dtype=float)

        color = DEFAULT_PANEL_COLORS.get(panel_name, "#5B8FF9")
        bars = ax.bar(
            x,
            y,
            yerr=err,
            capsize=4,
            width=0.62,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
            alpha=0.95,
        )

        group_avg = float(np.mean(y))
        ax.axhline(group_avg, linestyle="--", linewidth=1.2, color="gray", alpha=0.9)

        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.004, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

        panel_pairs = pairwise_df[pairwise_df["panel"] == panel_name].copy()
        annotations = build_significance_annotations(
            summary_df=panel_summary.assign(panel=panel_name),
            pairwise_df=panel_pairs,
            panel=panel_name,
            subgroup_order=subgroup_order,
            mean_col="f1_mean",
            err_col="f1_std",
            alpha=0.05,
            start_pad=0.012,
            step=0.028,
            text_pad=0.004,
            only_significant=True,
        )
        for ann in annotations:
            _plot_sig_bracket(ax, ann["x1"], ann["x2"], ann["y"], ann["text"])

        y_top = max(np.max(y + err), group_avg) + 0.03
        if annotations:
            y_top = max(y_top, max(a["text_y"] for a in annotations) + 0.02)

        ax.set_ylim(max(0.82, np.min(y - err) - 0.03), min(1.02, y_top))
        ax.set_xticks(x)
        ax.set_xticklabels(subgroup_order, rotation=0)
        ax.set_title(panel_name, fontsize=12, pad=10)

        if i in (0, 2):
            ax.set_ylabel("F1-Score")
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.18, 1.04, panel_letters[i], transform=ax.transAxes, fontsize=12, fontweight="bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#999999", ec="black", hatch="///", label="Subgroup F1"),
        plt.Line2D([0], [0], color="gray", lw=1.2, linestyle="--", label="Group Average"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, y=1.04, fontsize=13)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Figure 6 subgroup robustness plot.")
    parser.add_argument("--input", required=True, help="Fold-level subgroup F1 CSV or directory of CSVs.")
    parser.add_argument("--out-dir", default="outputs/figures/figure6_subgroup")
    parser.add_argument("--title", default="Subgroup Robustness Analysis (ResNet-1D)")
    args = parser.parse_args()

    raw_df = _load_fold_level_f1(args.input)

    summary_df = summarize_subgroup_f1(
        raw_df,
        panel_col="panel",
        subgroup_col="subgroup",
        fold_col="fold",
        f1_col="f1",
        panel_order=DEFAULT_PANEL_ORDER,
        subgroup_order_map=DEFAULT_SUBGROUP_ORDER,
    )

    pair_tables = []
    for panel_name in DEFAULT_PANEL_ORDER:
        panel_df = raw_df[raw_df["panel"] == panel_name].copy()
        if panel_df.empty:
            continue
        pair_tables.append(_panel_test_table(panel_df, panel_name))

    pairwise_df = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "subgroup_summary.csv"
    pvalue_csv = out_dir / "subgroup_pvalues.csv"

    summary_df.to_csv(summary_csv, index=False)
    if not pairwise_df.empty:
        pairwise_df.to_csv(pvalue_csv, index=False)
    else:
        pd.DataFrame().to_csv(pvalue_csv, index=False)

    plot_subgroup_figure(
        summary_df=summary_df,
        pairwise_df=pairwise_df,
        output_png=out_dir / "figure_subgroup_f1.png",
        output_pdf=out_dir / "figure_subgroup_f1.pdf",
        title=args.title,
    )

    print(f"[OK] Figure saved to: {out_dir / 'figure_subgroup_f1.png'}")
    print(f"[OK] Figure saved to: {out_dir / 'figure_subgroup_f1.pdf'}")
    print(f"[OK] Summary saved to: {summary_csv}")
    print(f"[OK] P-values saved to: {pvalue_csv}")


if __name__ == "__main__":
    main()
