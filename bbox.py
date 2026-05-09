import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# --- CONFIGURATION ---
COL_RELAX = "Score STAI-S- Relaxation"
COL_STRESS = "Score STAI-S- Stress"

def plot_stai_csv(csv_path, figures_dir):
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', encoding='latin1')
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return

    if COL_RELAX not in df.columns or COL_STRESS not in df.columns:
        print(f"Erreur : Colonnes introuvables. Colonnes détectées : {list(df.columns)}")
        return

    # 1. Nettoyage des données (IMPORTANT : Création des variables d'abord)
    paired_data = df[[COL_RELAX, COL_STRESS]].dropna()
    relax_vals = paired_data[COL_RELAX]
    stress_vals = paired_data[COL_STRESS]

    if len(paired_data) < 3: # Shapiro nécessite au moins 3 points
        print("Erreur : Pas assez de données appariées pour effectuer les tests.")
        return

    # --- 2. TEST DE SHAPIRO-WILK ---
    diff_stai = stress_vals - relax_vals
    stat_sha, p_sha = stats.shapiro(diff_stai)

    print("-" * 35)
    print(f"ANALYSE DE NORMALITÉ (STAI)")
    print(f"p-value Shapiro-Wilk : {p_sha:.4f}")
    if p_sha < 0.05:
        print("Résultat : Distribution NON-NORMALE (p < 0.05)")
    else:
        print("Résultat : Distribution NORMALE (p >= 0.05)")
    print("-" * 35)

    # 3. Statistique (Wilcoxon)
    _, p_val = stats.wilcoxon(relax_vals, stress_vals)
    p_label = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"

    # 4. Graphique
    fig, ax = plt.subplots(figsize=(6, 7))
    colors = ['#90caf9', '#ffcc80']

    # Boxplot
    box = ax.boxplot([relax_vals, stress_vals], patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Points (Jitter)
    rng = np.random.default_rng(42)
    for i, vals in enumerate([relax_vals, stress_vals], start=1):
        x = np.full(len(vals), i) + rng.normal(0, 0.04, size=len(vals))
        ax.scatter(x, vals, color=colors[i - 1], edgecolor='black', s=45, alpha=0.8, zorder=3)

    # 5. Affichage p-value (Wilcoxon) sur le graphe
    y_max = max(relax_vals.max(), stress_vals.max())
    y_line = y_max + 2
    ax.plot([1, 1, 2, 2], [y_line, y_line + 1, y_line + 1, y_line], color="black", lw=1.2)
    ax.text(1.5, y_line + 1.5, p_label, ha='center', fontweight='bold', color='red', fontsize=20)

    # Esthétique
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Relaxation", "Stress"], fontweight='bold', fontsize=20)
    ax.set_ylabel("Score STAI-S", fontsize=20)
    ax.set_title("STAI\n(Relaxation vs Stress)", fontsize=20, pad=20)
    ax.set_ylim(bottom=min(relax_vals.min(), stress_vals.min()) - 5, top=y_line + 10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "comparaison_stai_final.png"
    plt.savefig(output_path, dpi=200)
    print(f"Graphique sauvegardé ici : {output_path}")
    plt.show()

if __name__ == "__main__":
    fichier_cible = r"C:\Users\ouazoul\Desktop\DATASET_EDA\Nouveau dossier\Subject_infos.csv"
    dossier_sortie = Path(r"C:\Users\ouazoul\Desktop\DATASET_EDA\Nouveau dossier\figures")
    plot_stai_csv(fichier_cible, dossier_sortie)