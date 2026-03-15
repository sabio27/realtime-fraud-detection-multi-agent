# =============================================================================
# modele_detection.py
# Rôle : Entraîne XGBoost sur notre dataset simulé (49 950 transactions)
#        avec 13 features métier mappées sur les 10 stratégies de fraude.
#
# Avantage vs Kaggle :
# → Features identiques à notre simulation → cohérence totale
# → Ratio 2:1 bien équilibré → pas besoin de SMOTE
# → XGBoost apprend NOS patterns de fraude
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

# =============================================================================
# CONSTANTES
# =============================================================================

DATASET_PATH = "dataset_simulation.csv"
MODEL_PATH   = "modele_fraude.pkl"
SEUIL_PATH   = "seuil_fraude.pkl"

# Features utilisées pour l'entraînement
# Exactement les mêmes que dans generer_dataset.py
FEATURES = [
    "f1_montant_explosif",
    "f2_fragmentation",
    "f3_usurpation",
    "f4_rapidite",
    "f5_localisation",
    "f6_compte_dormant",
    "f7_micro",
    "f8_horaire",
    "f9_round_trip",
    "f10_mule",
    "montant",
    "heure",
    "frequence"
]

# =============================================================================
# FONCTION PRINCIPALE D'ENTRAINEMENT
# =============================================================================

def entrainer_modele(afficher_details=True):
    """
    Charge le dataset simulé, entraîne XGBoost et sauvegarde le modèle.

    Pipeline :
    1. Chargement du dataset simulé
    2. Split Train/Test 80/20
    3. Entraînement XGBoost avec scale_pos_weight
    4. Recherche du seuil optimal
    5. Evaluation et sauvegarde
    """

    # -------------------------------------------------------------------------
    # ETAPE 1 — Chargement du dataset simulé
    # -------------------------------------------------------------------------
    if afficher_details:
        print("="*55)
        print("ETAPE 1 — Chargement du dataset simulé")
        print("="*55)

    if not os.path.exists(DATASET_PATH):
        print("Dataset non trouvé → génération en cours...")
        from generer_dataset import generer_dataset
        generer_dataset(afficher_details=False)

    df = pd.read_csv(DATASET_PATH)

    if afficher_details:
        nb_fraudes  = df["est_fraude"].sum()
        nb_normales = (df["est_fraude"] == 0).sum()
        print(f"Total          : {len(df):,} transactions")
        print(f"Normales       : {nb_normales:,} ({nb_normales/len(df)*100:.1f}%)")
        print(f"Fraudes        : {nb_fraudes:,} ({nb_fraudes/len(df)*100:.1f}%)")
        print(f"Ratio          : {nb_normales//nb_fraudes}:1")
        print(f"Features       : {len(FEATURES)}")

    # -------------------------------------------------------------------------
    # ETAPE 2 — Split Train/Test
    # -------------------------------------------------------------------------
    if afficher_details:
        print("\n" + "="*55)
        print("ETAPE 2 — Split Train/Test (80/20)")
        print("="*55)

    X = df[FEATURES]
    y = df["est_fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y
    )

    if afficher_details:
        print(f"Train : {X_train.shape[0]:,} transactions")
        print(f"Test  : {X_test.shape[0]:,} transactions")

    # -------------------------------------------------------------------------
    # ETAPE 3 — Entraînement XGBoost
    # -------------------------------------------------------------------------
    if afficher_details:
        print("\n" + "="*55)
        print("ETAPE 3 — Entraînement XGBoost")
        print("="*55)
        print("Entraînement en cours...")

    # Calcul du ratio pour scale_pos_weight
    nb_normales_train = (y_train == 0).sum()
    nb_fraudes_train  = (y_train == 1).sum()
    ratio             = nb_normales_train / nb_fraudes_train

    if afficher_details:
        print(f"scale_pos_weight = {ratio:.2f}")

    modele = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 6,
        learning_rate    = 0.1,
        scale_pos_weight = ratio,
        eval_metric      = "aucpr",
        n_jobs           = -1,
        random_state     = 42
    )

    modele.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False
    )

    if afficher_details:
        print("Entraînement terminé !")

    # -------------------------------------------------------------------------
    # ETAPE 4 — Recherche du seuil optimal
    # -------------------------------------------------------------------------
    if afficher_details:
        print("\n" + "="*55)
        print("ETAPE 4 — Recherche seuil optimal")
        print("="*55)

    y_proba = modele.predict_proba(X_test)[:, 1]

    if afficher_details:
        print(f"\n{'Seuil':<8} {'Precision':<12} {'Recall':<10} "
              f"{'F1':<10} {'FP':<8} {'FN'}")
        print("-"*55)

    meilleur_seuil = 0.5
    meilleur_f1    = 0.0

    for seuil in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_t = (y_proba >= seuil).astype(int)
        p  = precision_score(y_test, y_pred_t, zero_division=0)
        r  = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        mc = confusion_matrix(y_test, y_pred_t)
        fp = mc[0][1]
        fn = mc[1][0]

        if afficher_details:
            print(f"{seuil:<8} {p:<12.2%} {r:<10.2%} "
                  f"{f1:<10.2%} {fp:<8} {fn}")

        if f1 > meilleur_f1 and r >= 0.85:
            meilleur_f1    = f1
            meilleur_seuil = seuil

    if afficher_details:
        print(f"\nSeuil optimal : {meilleur_seuil}")

    # Prédictions finales
    SEUIL_DECISION = meilleur_seuil
    y_pred         = (y_proba >= SEUIL_DECISION).astype(int)

    # -------------------------------------------------------------------------
    # ETAPE 5 — Métriques finales
    # -------------------------------------------------------------------------
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    auc_score  = roc_auc_score(y_test, y_proba)
    rapport    = classification_report(
                    y_test, y_pred,
                    target_names=["Normal", "Fraude"]
                 )
    matrice    = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # -------------------------------------------------------------------------
    # ETAPE 6 — Sauvegarde
    # -------------------------------------------------------------------------
    joblib.dump(modele,        MODEL_PATH)
    joblib.dump(SEUIL_DECISION, SEUIL_PATH)

    if afficher_details:
        print("\n" + "="*55)
        print("RESULTATS FINAUX")
        print("="*55)
        print(rapport)
        print(f"AUC-ROC   : {auc_score:.4f}")
        print(f"Seuil     : {SEUIL_DECISION}")
        print(f"\nModele sauvegarde : {MODEL_PATH}")

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
        "seuil"     : SEUIL_DECISION,
        "features"  : FEATURES
    }


def charger_modele():
    """
    Charge le modèle depuis le fichier .pkl.
    Si absent → entraîne d'abord.
    """

    if not os.path.exists(MODEL_PATH):
        print("Modele non trouvé → entraînement en cours...")
        resultats = entrainer_modele(afficher_details=False)
        return resultats["modele"], resultats["seuil"], FEATURES

    modele = joblib.load(MODEL_PATH)
    seuil  = joblib.load(SEUIL_PATH)
    return modele, seuil, FEATURES


# =============================================================================
# EXECUTION DIRECTE
# =============================================================================
if __name__ == "__main__":
    resultats = entrainer_modele(afficher_details=True)
    print(f"\nPrecision : {resultats['precision']:.2%}")
    print(f"Recall    : {resultats['recall']:.2%}")
    print(f"F1-Score  : {resultats['f1']:.2%}")
    print(f"AUC-ROC   : {resultats['auc']:.4f}")