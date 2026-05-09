import pandas as pd
import numpy as np
from cvxEDA import cvxEDA
import xgboost as xgb
import os
import glob

# 1. CONFIGURATION
MODEL_JSON_PATH = r"C:\Users\ouazoul\Desktop\DATASET_EDA\SA_Detection.json"
DATASET_PATH = r"C:\Users\ouazoul\Desktop\DATASET_EDA\Participants"


def extract_features(y, sr=4):
    y = np.asarray(y, dtype=float)
    [r, p, t, l, d, e, obj] = cvxEDA(y, 1. / sr)
    feats = {}

    # 1. Statistiques de base (27 features)
    for name, sig in zip(['phasic', 'tonic', 'eda'], [r, t, y]):
        sig = np.asarray(sig, dtype=float)
        feats[f'{name}_mean'] = np.mean(sig)
        feats[f'{name}_std'] = np.std(sig)
        feats[f'{name}_min'] = np.min(sig)
        feats[f'{name}_max'] = np.max(sig)
        feats[f'{name}_skew'] = pd.Series(sig).skew()
        feats[f'{name}_kurt'] = pd.Series(sig).kurt()
        q25, q50, q75 = np.percentile(sig, [25, 50, 75])
        feats[f'{name}_q25'], feats[f'{name}_q50'], feats[f'{name}_q75'] = q25, q50, q75

    # 2. Features dynamiques (4 features)
    feats['phasic_auc'] = np.trapezoid(r)
    feats['phasic_rms'] = np.sqrt(np.mean(np.square(r)))
    feats['tonic_slope'] = np.polyfit(np.arange(len(t)), t, 1)[0]

    fft_vals = np.abs(np.fft.rfft(r))
    freqs = np.fft.rfftfreq(len(r), 1. / sr)
    feats['spectral_centroid'] = np.sum(freqs * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) != 0 else 0

    # 3. AJOUT DES 6 MANQUANTES : IQR et Entropie (Pour arriver à 37)
    for sig, prefix in zip([r, t, y], ['phasic', 'tonic', 'eda']):
        # Interquartile Range
        feats[f'{prefix}_iqr'] = np.percentile(sig, 75) - np.percentile(sig, 25)
        # Entropie (Complexité du signal)
        sig_norm = np.square(sig / (np.sum(sig) + 1e-10))
        feats[f'{prefix}_entropy'] = -np.sum(sig_norm * np.log(sig_norm + 1e-10))

    return feats


if __name__ == "__main__":
    model = xgb.XGBClassifier()
    model.load_model(MODEL_JSON_PATH)
    expected_cols = model.get_booster().feature_names
    all_results = []

    # 2. RECHERCHE RÉCURSIVE (Cherche dans tous les sous-dossiers ID01, ID02...)
    # On cherche les fichiers qui finissent par _EDA.csv comme sur votre image
    fichiers = glob.glob(os.path.join(DATASET_PATH, "**", "*_EDA.csv"), recursive=True)
    print(f"Fichiers trouvés : {len(fichiers)}")

    for file_path in fichiers:
        nom_participant = os.path.basename(os.path.dirname(file_path))
        print(f"\n>>> Analyse de : {nom_participant}")

        try:
            df_p = pd.read_csv(file_path)  # Utilisation de read_csv car vos fichiers sont des CSV
            # On cherche la colonne 'value' ou 'EDA' (vérifiez le nom dans votre CSV)
            col_name = 'value' if 'value' in df_p.columns else 'EDA'
            signal = pd.to_numeric(df_p[col_name], errors='coerce').dropna().values

            # 3. DÉCOUPAGE EN FENÊTRES (Pour avoir plusieurs lignes par participant)
            fs = 4
            win_size = 5 * 60 * fs  # Fenêtre de 5 min
            step = 1 * 60 * fs  # On avance de 1 min à chaque fois

            for start in range(0, len(signal) - win_size + 1, step):
                y_window = signal[start:start + win_size]
                feats = extract_features(y_window)

                df_input = pd.DataFrame([feats]).reindex(columns=expected_cols, fill_value=0)
                final_pred = model.predict(df_input)[0]

                res_row = {
                    "Participant": nom_participant,
                    "Time_Min": (start / fs) / 60,
                    "Prediction": final_pred
                }
                res_row.update(feats)
                all_results.append(res_row)

            print(f"OK : {nom_participant} traité.")

        except Exception as e:
            print(f"Erreur sur {file_path} : {e}")

    # 4. SAUVEGARDE
    if all_results:
        df_final = pd.DataFrame(all_results)
        df_final.to_csv("Resultats_Dataset_EDA_XJ.csv", index=False)
        print(f"\nSUCCÈS : {len(df_final)} lignes générées !")