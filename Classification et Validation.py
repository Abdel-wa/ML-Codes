import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# =========================
# 1) CONFIGURATION
# =========================
DATA_PATH = r"C:\Users\ouazoul\Desktop\DATASET_EDA\Resultats_Dataset_EDA_XJ.csv"

# Liste des mots-clés pour identifier les features à utiliser
FEATURE_KEYWORDS = ["phasic", "tonic", "eda", "Spectral", "RMS", "auc"]

# =========================
# 2) CHARGEMENT & AUTO-DÉTECTION
# =========================
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()  # Nettoie les espaces

# Détection de colonne
col_participant = "Participant"
col_label = "Prediction"  # On utilise la sortie de notre modèle comme cible
# Si vous n'avez pas de colonne artifact dans ce fichier, on la définit par défaut
col_artifact = None

print(f"--- Configuration détectée ---")
print(f"Participant : {col_participant}")
print(f"Target      : {col_label}")
# Détection intelligente des colonnes clés
#col_participant = [c for c in df.columns if "partici" in c.lower()][0]
#col_label = [c for c in df.columns if "label" in c.lower() or "binary" in c.lower()][0]
#col_artifact = [c for c in df.columns if "artifact" in c.lower()][0]

#print(f"--- Configuration détectée ---")
#print(f"Participant : {col_participant}")
#print(f"Target      : {col_label}")
#print(f"Artifacts   : {col_artifact}")

# =========================
# 3) PRÉPARATION DES DONNÉES
# =========================
# Filtrage des lignes avec des artefacts (on ne garde que le 0)
df_clean = df[df[col_artifact] == 0].copy()
df_clean = df.copy() # On garde tout car cvxEDA a déja nettoyé via optimisation
# Sélection automatique des colonnes de données (Features)
# On prend tout ce qui contient nos mots-clés MAIS qui n'est pas le label ou l'ID
X_cols = [c for c in df.columns if any(k in c for k in FEATURE_KEYWORDS)
          and c not in [col_label, col_participant, col_artifact]]

X = df_clean[X_cols].copy()
y = df_clean[col_label].values
groups = df_clean[col_participant].astype(str).values
print(df_clean.columns)

print(f"Features utilisées ({len(X_cols)}) : {X_cols}")


## =========================
# 4) MODÈLE, VALIDATION & MATRICE DE CONFUSION
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# 1. Définition du pipeline
##Au lieu de devoir appeler manuellement chaque fonction sur X_train, puis sur X_test, tu n'utilises que deux commandes :
#pipe.fit() : Lance toute la chaîne d'apprentissage.
#pipe.predict() : Lance toute la chaîne de prédiction.
pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), ##étape d'imputation (SimpleImputer) qui remplace les valeurs manquantes par la médiane
    ("rf", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# 2. Configuration de la validation croisée
# (On s'assure que 'groups' est bien défini, sinon on le crée depuis df_clean)
if 'groups' not in locals():
    groups = df_clean[col_participant].astype(str).values

n_participants = len(np.unique(groups))
gkf = GroupKFold(n_splits=min(5, n_participants))

# 3. Initialisation des listes pour stocker TOUS les résultats
y_true_global = []
y_pred_global = []
results = []

print(f"\n--- Début de l'entraînement par Fold sur {n_participants} participants ---")

# 4. Boucle d'entraînement
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    # Séparation Train / Test
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Entraînement
    pipe.fit(X_train, y_train)

    # Prédiction
    y_pred = pipe.predict(X_test)

    # Stockage pour le bilan global
    y_true_global.extend(y_test)
    y_pred_global.extend(y_pred)

    # Score du fold
    score = balanced_accuracy_score(y_test, y_pred)
    results.append(score)
    print(f"Fold {fold} | Score : {score:.3f}")

# =========================
# 5) BILAN & GRAPHIQUE
# =========================
print("\n" + "=" * 30)
print(f"SCORE FINAL MOYEN : {np.mean(results):.3f} ± {np.std(results):.3f}")
print("=" * 30)

# --- A. TABLEAU D'IMPORTANCE DES VARIABLES ---
if hasattr(pipe.named_steps["rf"], "feature_importances_"):
    importance = pipe.named_steps["rf"].feature_importances_
    feat_imp = pd.Series(importance, index=X_cols).sort_values(ascending=False)
    print("\nTop 5 des variables explicatives :")
    print(feat_imp.head(5))

# --- B. GRAPHIQUE DE LA MATRICE DE CONFUSION GLOBALE ---
# On force les labels [0, 1] pour éviter l'erreur "Single label"
cm_global = confusion_matrix(y_true_global, y_pred_global, labels=[0, 1])

# Textes pour le graphique
group_names = ['Vrai Négatif\n(Repos bien prédit)', 'Faux Positif\n(Fausse alerte)',
               'Faux Négatif\n(Stress raté)', 'Vrai Positif\n(Stress bien prédit)']
group_counts = ["{0:0.0f}".format(value) for value in cm_global.flatten()]

if np.sum(cm_global) > 0:
    group_percentages = ["{0:.2%}".format(value) for value in cm_global.flatten()/np.sum(cm_global)]
else:
    group_percentages = ["0%" for _ in range(4)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

# Affichage
plt.figure(figsize=(8, 6))
sns.heatmap(cm_global, annot=labels, fmt='', cmap='Blues', cbar=False,
            xticklabels=['Relaxation (0)', 'Stress (1)'],
            yticklabels=['Relaxation (0)', 'Stress (1)'])

plt.title('Matrice de Confusion Globale (Tous participants)', fontsize=14)
plt.xlabel('Prédiction du Modèle', fontsize=12)
plt.ylabel('Réalité (Ground Truth)', fontsize=12)
plt.show()