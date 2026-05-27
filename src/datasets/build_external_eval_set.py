
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


EXCLUDE_PREFIXES = ("Summary_",)
EXCLUDE_NAMES = {
    "processing_summary.csv",
    "processing_summary_groups.csv",
    "cleaning_summary.csv",
    "engineer_summary.csv",
    "selection_summary.csv",
    "external_eval_summary.csv",
}

DEFAULT_METADATA_COLS = [
    "CSN",
    "Wave_Type",
    "WAVE_PATH",
    "Source_File",
    "Stroke_Occurred_Here",
    "Wave_Start",
    "Wave_End",
    "Actual_Stroke_Time",
    "Is_Stroke_Subject",
    "Beat_Idx",
    "Label",
    "Absolute_Time",
    "Time_Rel_Min",
    "MRN",
    "Visit_no",
    "Age",
    "Gender",
    "Race",
    "Ethnicity",
    "Dx_ICD10",
    "Dx_name",
    "Segment_Type",
    "Keep_Reason",
    "Gap_Hours",
]

RAW_BASES = [
    "T_pi", "T_sys", "T_dia", "T_sp", "IPR", "Tsys_Tdia", "Tsp_Tpi",
    "A_on", "A_sp", "A_off", "Pulse_Amplitude", "SI",
    "T_u", "Tu_Tpi", "T_v", "T_a", "Ta_Tpi", "T_b", "Tb_Tpi", "T_c", "Tc_Tpi",
    "T_d", "Td_Tpi", "T_e", "Te_Tpi", "T_f", "Tf_Tpi",
    "T_p1", "Tp1_Tpi", "T_p2", "Tp2_Tpi", "Tu_Ta_Tpi",
    "CV_T_pi", "CV_T_sys", "CV_Pulse_Amplitude",
]

COMPOSITE_FEATURES = ["NVI", "DSI", "NCI", "VSSI"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def is_feature_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    for prefix in EXCLUDE_PREFIXES:
        if path.name.startswith(prefix):
            return False
    return True


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )
        df[numeric_cols] = df[numeric_cols].bfill().ffill().fillna(0)

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    if non_numeric_cols:
        df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill().fillna("Unknown")

    return df


def engineer_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6
    cols = set(df.columns)

    if {"T_f", "T_p2", "IPR", "CV_T_pi"}.issubset(cols):
        df["NVI"] = (df["T_f"] + df["T_p2"] + df["IPR"]) / (df["CV_T_pi"] + eps)

    if {"T_f", "T_p2", "T_pi"}.issubset(cols):
        df["DSI"] = (df["T_f"] + df["T_p2"]) / (df["T_pi"] + eps)

    if {"T_f", "CV_T_pi"}.issubset(cols):
        df["NCI"] = df["T_f"] / ((df["CV_T_pi"] ** 2) + eps)

    if {"IPR", "T_dia"}.issubset(cols):
        df["VSSI"] = (df["IPR"] ** 2) / (df["T_dia"] + eps)

    return df


def engineer_kinematic_features(
    df: pd.DataFrame,
    baseline_fraction: float = 0.10,
    min_baseline_points: int = 5,
) -> pd.DataFrame:
    df = df.copy()

    if "Time_Rel_Min" in df.columns:
        df = df.sort_values("Time_Rel_Min", kind="mergesort")
    elif "Beat_Idx" in df.columns:
        df = df.sort_values("Beat_Idx", kind="mergesort")

    target_cols = [c for c in RAW_BASES + COMPOSITE_FEATURES if c in df.columns]
    if not target_cols:
        return df

    baseline_len = max(min_baseline_points, int(len(df) * baseline_fraction))
    new_features: dict[str, pd.Series] = {}

    for col in target_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().all():
            continue

        baseline = s.iloc[:baseline_len].median()
        if pd.isna(baseline) or abs(baseline) < 1e-12:
            baseline = 1e-6

        rel_series = (s - baseline) / baseline
        vel_series = rel_series.diff().fillna(0)
        acc_series = vel_series.diff().fillna(0)

        new_features[f"{col}_Rel"] = rel_series
        new_features[f"{col}_Vel"] = vel_series
        new_features[f"{col}_Accel"] = acc_series

    if new_features:
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    df = df.replace([np.inf, -np.inf], 0)
    return df


def resolve_selected_features(
    selected_features_json: str | None,
    config: dict[str, Any],
) -> list[str]:
    # 1) explicit JSON file takes priority
    if selected_features_json:
        with open(selected_features_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Selected_Features.json must contain a JSON list.")
        return [str(x) for x in data]

    # 2) fallback to training config fixed list
    fs_cfg = config.get("feature_selection", {})
    fixed_list = fs_cfg.get("fixed_feature_list", [])
    if fixed_list:
        return [str(x) for x in fixed_list]

    raise ValueError(
        "No selected feature list found. Provide --selected-features-json or "
        "set feature_selection.fixed_feature_list in training.yaml."
    )


def resolve_metadata_columns(config: dict[str, Any]) -> list[str]:
    fs_cfg = config.get("feature_selection", {})
    by_dataset = fs_cfg.get("metadata_columns", {})
    if "mcmed" in by_dataset:
        return list(by_dataset["mcmed"])
    return DEFAULT_METADATA_COLS.copy()


def process_one_file(
    file_path: Path,
    output_dir: Path,
    selected_features: list[str],
    metadata_cols: list[str],
    engineer_features_flag: bool,
    baseline_fraction: float,
    min_baseline_points: int,
) -> dict[str, Any]:
    df = pd.read_csv(file_path, low_memory=False)
    rows_in = len(df)

    if "Source_File" not in df.columns:
        df["Source_File"] = file_path.name

    df = fill_missing_values(df)

    if engineer_features_flag:
        df = engineer_composite_features(df)
        df = engineer_kinematic_features(
            df,
            baseline_fraction=baseline_fraction,
            min_baseline_points=min_baseline_points,
        )

    existing_metadata = [c for c in metadata_cols if c in df.columns]
    existing_selected = [c for c in selected_features if c in df.columns]
    missing_selected = [c for c in selected_features if c not in df.columns]

    final_columns = list(dict.fromkeys(existing_metadata + selected_features))
    out_df = df.reindex(columns=final_columns, fill_value=0)

    output_path = output_dir / file_path.name
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "file": file_path.name,
        "rows_in": int(rows_in),
        "rows_out": int(len(out_df)),
        "metadata_cols": int(len(existing_metadata)),
        "selected_feature_cols_expected": int(len(selected_features)),
        "selected_feature_cols_present": int(len(existing_selected)),
        "selected_feature_cols_missing": int(len(missing_selected)),
        "missing_features": missing_selected,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build external evaluation feature tables by applying a fixed selected feature set to MC-MED."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configs/training.yaml")
    parser.add_argument("--input-dir", type=str, required=True, help="MC-MED cleaned or engineered feature directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for packaged external-eval files")
    parser.add_argument(
        "--selected-features-json",
        type=str,
        default=None,
        help="Path to Selected_Features.json learned from internal data (recommended)",
    )
    parser.add_argument(
        "--skip-engineering",
        action="store_true",
        help="Skip composite/kinematic feature engineering if input-dir is already engineered",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fx_cfg = cfg.get("feature_engineering", {})
    baseline_fraction = float(fx_cfg.get("baseline_fraction", 0.10))
    min_baseline_points = int(fx_cfg.get("min_baseline_points", 5))

    selected_features = resolve_selected_features(
        selected_features_json=args.selected_features_json,
        config=cfg,
    )
    metadata_cols = resolve_metadata_columns(cfg)

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and is_feature_csv(p)])
    if not files:
        raise RuntimeError(f"No feature CSV files found under: {input_dir}")

    logs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"[build_external_eval_set] input_dir={input_dir}")
    print(f"[build_external_eval_set] output_dir={output_dir}")
    print(f"[build_external_eval_set] files={len(files)}")
    print(f"[build_external_eval_set] selected_features={len(selected_features)}")
    print(f"[build_external_eval_set] skip_engineering={args.skip_engineering}")

    for p in tqdm(files, desc="Packaging external eval set"):
        try:
            info = process_one_file(
                file_path=p,
                output_dir=output_dir,
                selected_features=selected_features,
                metadata_cols=metadata_cols,
                engineer_features_flag=not args.skip_engineering,
                baseline_fraction=baseline_fraction,
                min_baseline_points=min_baseline_points,
            )
            logs.append(info)
        except Exception as e:
            errors.append({"file": p.name, "error": f"{type(e).__name__}: {e}"})

    summary_df = pd.DataFrame(logs)
    summary_path = output_dir / "external_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    error_path = output_dir / "external_eval_errors.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    feature_manifest = {
        "selected_features": selected_features,
        "metadata_columns": metadata_cols,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "skip_engineering": bool(args.skip_engineering),
    }
    manifest_path = output_dir / "external_eval_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, ensure_ascii=False, indent=2)

    print("[build_external_eval_set] done.")
    print(f"  success_files={len(logs)}")
    print(f"  failed_files={len(errors)}")
    print(f"  summary={summary_path}")
    print(f"  errors={error_path}")
    print(f"  manifest={manifest_path}")


if __name__ == "__main__":
    main()
