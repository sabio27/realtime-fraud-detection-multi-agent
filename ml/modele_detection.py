"""Entraîne un XGBoost sur le jeu de transactions simulées.

Ce modèle n'est pas utilisé par la simulation. J'ai d'abord voulu remplacer
les règles du détecteur par du machine learning, mais en simulation les
modèles laissaient passer beaucoup plus de fraudes que les règles. Le
script reste là comme trace de cet essai.

À lancer depuis la racine du projet :
    python -m ml.modele_detection
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score)
from xgboost import XGBClassifier

RACINE       = Path(__file__).resolve().parent.parent
DATASET_PATH = RACINE / "data" / "dataset_simulation.csv"
MODEL_PATH   = RACINE / "models" / "modele_fraude.pkl"
SEUIL_PATH   = RACINE / "models" / "seuil_fraude.pkl"

FEATURES = [
    "f1_montant_explosif", "f2_fragmentation", "f3_usurpation",
    "f4_rapidite", "f5_localisation", "f6_compte_dormant",
    "f7_micro", "f8_horaire", "f9_round_trip", "f10_mule",
    "montant", "heure", "frequence"
]


def entrainer_modele(afficher_details=True):
    if not os.path.exists(DATASET_PATH):
        print("Pas de dataset, génération en cours...")
        from ml.generer_dataset import generer_dataset
        generer_dataset(afficher_details=False)

    df = pd.read_csv(DATASET_PATH)

    if afficher_details:
        nb_fraudes  = df["est_fraude"].sum()
        nb_normales = (df["est_fraude"] == 0).sum()
        print(f"{len(df):,} transactions : {nb_normales:,} normales, "
              f"{nb_fraudes:,} fraudes, {len(FEATURES)} variables")

    X = df[FEATURES]
    y = df["est_fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Rééquilibrage des classes par le poids des fraudes
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    modele = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 6,
        learning_rate    = 0.1,
        scale_pos_weight = ratio,
        eval_metric      = "aucpr",
        n_jobs           = -1,
        random_state     = 42
    )
    modele.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Choix du seuil : meilleur F1 parmi ceux qui gardent un rappel >= 85 %
    y_proba = modele.predict_proba(X_test)[:, 1]

    if afficher_details:
        print(f"\n{'Seuil':<8} {'Precision':<12} {'Recall':<10} "
              f"{'F1':<10} {'FP':<8} {'FN'}")

    meilleur_seuil = 0.5
    meilleur_f1    = 0.0

    for seuil in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_t = (y_proba >= seuil).astype(int)
        p  = precision_score(y_test, y_pred_t, zero_division=0)
        r  = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        mc = confusion_matrix(y_test, y_pred_t)

        if afficher_details:
            print(f"{seuil:<8} {p:<12.2%} {r:<10.2%} "
                  f"{f1:<10.2%} {mc[0][1]:<8} {mc[1][0]}")

        if f1 > meilleur_f1 and r >= 0.85:
            meilleur_f1    = f1
            meilleur_seuil = seuil

    y_pred = (y_proba >= meilleur_seuil).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba)
    rapport   = classification_report(y_test, y_pred, target_names=["Normal", "Fraude"])
    matrice   = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(modele, MODEL_PATH)
    joblib.dump(meilleur_seuil, SEUIL_PATH)

    if afficher_details:
        print(f"\nSeuil retenu : {meilleur_seuil}\n")
        print(rapport)
        print(f"AUC-ROC : {auc_score:.4f}")
        print(f"Modèle sauvegardé : {MODEL_PATH}")

    return {
        "modele"    : modele,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "auc"       : auc_score,
        "matrice"   : matrice,
        "rapport"   : rapport,
        "fpr"       : fpr,
        "tpr"       : tpr,
        "seuil"     : meilleur_seuil,
        "features"  : FEATURES
    }


def charger_modele():
    """Charge le modèle sauvegardé, ou l'entraîne s'il n'existe pas encore."""
    if not os.path.exists(MODEL_PATH):
        resultats = entrainer_modele(afficher_details=False)
        return resultats["modele"], resultats["seuil"], FEATURES

    return joblib.load(MODEL_PATH), joblib.load(SEUIL_PATH), FEATURES


if __name__ == "__main__":
    resultats = entrainer_modele(afficher_details=True)
    print(f"\nPrecision : {resultats['precision']:.2%}")
    print(f"Recall    : {resultats['recall']:.2%}")
    print(f"F1-Score  : {resultats['f1']:.2%}")
