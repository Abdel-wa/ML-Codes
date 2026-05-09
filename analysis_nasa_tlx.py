#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from scipy import stats
import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_EXCEL_PATH = Path(r"C:\Users\ouazoul\Desktop\Abdelwahab\NASA-TLX (réponses).xlsx")

COL_ID = "ID participant"
COL_TIME = "Horodatage"

# NASA-TLX columns detected from the Excel file (exact strings).
COL_EXIGENCE_MENTALE = (
    "Exigeance mentale : À quel point cette tâche a-t-elle été mentalement exigeante ? "
    "Échelle 0 (Pas du tout) → 10 (Très)"
)
COL_TEMPORALITE = "Temporalité: Le rythme de la tache t’a semblé? (0=> Trés lent 10=> Rapide)"
COL_EFFORT = (
    "À quel point as-tu dû faire des efforts pour atteindre ton niveau de performance ? "
    "(0= Petit effort  10 -Grand effort)"
)
COL_PERFORMANCE = (
    "Dans quelle mesure as-tu réussi à faire ce qui t’était demandé ? (0 = Parfait  10 =Echec)"
)
COL_FRUSTRATION = (
    "FRUSTRATION : À quel point la tâche vous a-t-elle frustrée ou stressée ? "
    "Échelle 0 → 10 ( 0 (Pas du tout) - 10 (Trés)"
)

DIMENSION_COLS = {
    "Exigence mentale": COL_EXIGENCE_MENTALE,
    "Temporalité": COL_TEMPORALITE,
    "Effort": COL_EFFORT,
    "Performance (0=Parfait, 10=Echec)": COL_PERFORMANCE,
    "Frustration": COL_FRUSTRATION,
}


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_").lower()
    return ascii_text or "metric"


def clean_participant_id(series: pd.Series) -> pd.Series:
    series = series.astype("string")
    series = series.str.replace("\u00a0", " ", regex=False).str.strip()
    return series


def find_sheet_with_column(path: Path, column: str) -> str:
    xl = pd.ExcelFile(path)
    for name in xl.sheet_names:
        df_head = pd.read_excel(path, sheet_name=name, nrows=0)
        if column in df_head.columns:
            return name
    raise ValueError(f"No sheet contains the required column: {column!r}")


def add_optional_physical_col(df: pd.DataFrame, dimension_cols: dict[str, str]) -> dict[str, str]:
    physical_candidates = [c for c in df.columns if "physique" in str(c).lower()]
    if not physical_candidates:
        return dimension_cols
    if len(physical_candidates) > 1:
        raise ValueError(f"Multiple 'physique' columns found: {physical_candidates}")
    updated = dict(dimension_cols)
    updated["Exigence physique"] = physical_candidates[0]
    return updated


def assign_phases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([COL_ID, COL_TIME])
    counts = df.groupby(COL_ID)[COL_ID].transform("size")
    if counts.max() > 2:
        too_many = df.loc[counts > 2, COL_ID].unique().tolist()
        raise AssertionError(
            "Some participants have more than 2 rows. "
            f"Examples: {too_many[:10]}"
        )

    order = df.groupby(COL_ID).cumcount()
    df = df.copy()
    df["Phase"] = np.where(order == 0, "Relaxation", "Stress")
    df.loc[counts == 1, "Phase"] = "Relaxation"
    return df

def plot_metric(
        df: pd.DataFrame,
        metric_col: str,
        metric_label: str,
        figures_dir: Path,
) -> None:
    phases = ["Relaxation", "Stress"]
    values = [df.loc[df["Phase"] == phase, metric_col].dropna() for phase in phases]

    fig, ax = plt.subplots(figsize=(7, 5))

    # --- 1. CALCUL STATISTIQUE (p-value) ---
    paired_ids = df.groupby(COL_ID)["Phase"].nunique()
    paired_ids = paired_ids[paired_ids == 2].index
    paired = df[df[COL_ID].isin(paired_ids)]
    pivot = paired.pivot(index=COL_ID, columns="Phase", values=metric_col).dropna()

    p_val_text = ""
    if len(pivot) >= 5:  # Le test nécessite un minimum de données
        # TEST DE SHAPIRO-WILK ---
        # On teste la normalité de la DIFFÉRENCE entre Stress et Relaxation
        diff = pivot["Stress"] - pivot["Relaxation"]
        _, p_shapiro = stats.shapiro(diff)
        print(f"[{metric_label}] Shapiro-Wilk p-value: {p_shapiro:.4f}")
        # ----------------------------------------
        # Test de Wilcoxon pour échantillons appariés
        _, p_val = stats.wilcoxon(pivot["Relaxation"], pivot["Stress"])
        if p_val < 0.001:
            p_val_text = "p < 0.001"
        else:
            p_val_text = f"p = {p_val:.4f}"




        # --- TEST DE FISHER (Homogénéité des variances) ---
        v1, v2 = np.var(values[0], ddof=1), np.var(values[1], ddof=1)
        f_stat = v1 / v2
        df1, df2 = len(values[0]) - 1, len(values[1]) - 1
        p_fisher = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        print(f"[{metric_label}] Fisher p-value (variances): {p_fisher:.4f}")

    # --- 2. TRACÉ DES BOXPLOTS (SANS LES TRAITS) ---
    box = ax.boxplot(
        values,
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.5},
    )

    box_colors = ["#90caf9", "#ffcc80"]  # Bleu pour Relax, Orange pour Stress
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # --- 3. POINTS INDIVIDUELS (Jitter) ---
    rng = np.random.default_rng(42)
    for idx, phase in enumerate(phases, start=1):
        y = df.loc[df["Phase"] == phase, metric_col].dropna().to_numpy()
        x = np.full(len(y), idx, dtype=float) + rng.normal(0, 0.04, size=len(y))
        ax.scatter(x, y, s=30, alpha=0.8, color=box_colors[idx - 1], edgecolor="#333333", linewidth=0.5)

    # --- 4. AFFICHAGE DE LA P-VALUE ---
    if p_val_text:
        # On place la p-value en haut du graphique
        y_max = 10.5
        ax.text(1.5, y_max, p_val_text, ha='center', va='bottom', fontsize=20, fontweight='bold', color='#d62728')
        # Optionnel : petite ligne horizontale pour le style
        ax.plot([1, 2], [y_max - 0.2, y_max - 0.2], color="black", lw=0.8, alpha=0.5)

    # Réglages des axes
    ax.set_ylim(-0.5, 12)  # On laisse de la place en haut pour la p-value
    ax.set_xticks([1, 2])
    ax.set_xticklabels(phases, fontsize=20)
    ax.set_ylabel("Score (0-10)", fontsize=20)
    ax.set_title(f"{metric_label}", fontsize=20, pad=20)
    ax.grid(axis="y", linestyle='--', alpha=0.3)

    figures_dir.mkdir(parents=True, exist_ok=True)
    filename = f"boxplot_{slugify(metric_label)}.png"
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=200)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="NASA-TLX Relaxation vs Stress analysis.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to the Excel file (.xlsx).",
    )
    args = parser.parse_args()

    excel_path = args.excel
    if not excel_path.exists():
        local_matches = list(Path.cwd().glob("*.xlsx"))
        if len(local_matches) == 1:
            excel_path = local_matches[0]
        else:
            raise FileNotFoundError(
                f"Excel file not found: {excel_path}. "
                f"Found {len(local_matches)} .xlsx files in cwd."
            )

    sheet_name = find_sheet_with_column(excel_path, COL_ID)
    print(f"Using Excel: {excel_path}")
    print(f"Using sheet: {sheet_name}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    assert COL_ID in df.columns, f"Missing required column: {COL_ID!r}"
    assert COL_TIME in df.columns, f"Missing required column: {COL_TIME!r}"

    df[COL_ID] = clean_participant_id(df[COL_ID])
    missing_id = df[COL_ID].isna() | (df[COL_ID] == "")
    if missing_id.any():
        print(f"WARNING: Dropping {missing_id.sum()} rows with missing participant ID.")
        df = df.loc[~missing_id].copy()

    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce", utc=True)
    missing_time = df[COL_TIME].isna()
    if missing_time.any():
        print(f"WARNING: Dropping {missing_time.sum()} rows with invalid Horodatage.")
        df = df.loc[~missing_time].copy()

    dimension_cols = add_optional_physical_col(df, DIMENSION_COLS)
    required_cols = list(dimension_cols.values())
    missing_cols = [c for c in required_cols if c not in df.columns]
    assert not missing_cols, f"Missing expected NASA-TLX columns: {missing_cols}"

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = assign_phases(df)

    df["TLX_Global"] = df[required_cols].mean(axis=1, skipna=True)

    print(f"Rows after cleaning: {len(df)}")
    print(f"Participants: {df[COL_ID].nunique()}")
    paired_ids = df.groupby(COL_ID)["Phase"].nunique()
    paired_ids = paired_ids[paired_ids == 2].index
    print(f"Paired participants (Relax+Stress): {len(paired_ids)}")

    rename_map = {v: k for k, v in dimension_cols.items()}
    long_df = df.rename(columns=rename_map)
    long_df = long_df[[COL_ID, "Phase"] + list(dimension_cols.keys()) + ["TLX_Global"]]
    long_df = long_df.rename(columns={COL_ID: "Participant"})

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    long_path = outputs_dir / "nasa_tlx_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8")

    paired_df = long_df[long_df["Participant"].isin(paired_ids)]
    metrics = list(dimension_cols.keys()) + ["TLX_Global"]
    wide = paired_df.pivot(index="Participant", columns="Phase", values=metrics)
    wide.columns = [f"{metric} [{phase}]" for metric, phase in wide.columns]
    for metric in metrics:
        wide[f"{metric} [Stress-Relax]"] = (
            wide[f"{metric} [Stress]"] - wide[f"{metric} [Relaxation]"]
        )
    wide = wide.reset_index()

    wide_path = outputs_dir / "nasa_tlx_paired_wide.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8")

    figures_dir = Path("figures")
    for label, col in dimension_cols.items():
        plot_metric(df, col, label, figures_dir)
    plot_metric(df, "TLX_Global", "TLX Global", figures_dir)



    print(f"Saved long table: {long_path}")
    print(f"Saved paired wide table: {wide_path}")
    print(f"Saved figures to: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()


