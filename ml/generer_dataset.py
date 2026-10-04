"""Génère un jeu de transactions étiquetées à partir des agents de la simulation.

Environ 50 000 transactions (100 clients, 50 fraudeurs, 333 rounds), avec
13 variables : une par stratégie de fraude, plus le montant, l'heure et la
fréquence bruts. Il sert à entraîner le modèle de modele_detection.py.

À lancer depuis la racine du projet :
    python -m ml.generer_dataset
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

from agents.client import AgentClient
from agents.fraudeur import AgentFraudeur

NB_CLIENTS   = 100
NB_FRAUDEURS = 50
NB_ROUNDS    = 333

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_simulation.csv"


def calculer_features(transaction, historique_client):
    """Calcule les 13 variables d'une transaction à partir de l'historique du client."""
    montant   = transaction["montant"]
    heure     = transaction["heure"]
    ville     = transaction["ville"]
    frequence = transaction["frequence"]

    montants_hist = [t["montant"]   for t in historique_client[-20:]]
    villes_hist   = [t["ville"]     for t in historique_client[-10:]]
    freq_hist     = [t["frequence"] for t in historique_client[-10:]]

    moyenne_montant = np.mean(montants_hist) if montants_hist else montant
    freq_moy        = np.mean(freq_hist)     if freq_hist     else frequence

    # Montant rapporté à la moyenne du client
    f1_montant_explosif = montant / max(moyenne_montant, 1)

    # Nombre de petits montants (< 10 000) dans l'historique récent
    f2_fragmentation = sum(1 for m in montants_hist if m < 10_000)

    # Écart relatif à la moyenne : très faible si le profil a été copié
    f3_usurpation = abs(montant - moyenne_montant) / max(moyenne_montant, 1)

    f4_rapidite = frequence

    # 1 si la ville n'apparaît pas dans les 10 dernières transactions
    f5_localisation = 0 if ville in villes_hist else 1

    # Baisse de fréquence par rapport à l'habitude
    f6_compte_dormant = max(0, freq_moy - frequence)

    # Part des montants sous 5 000 FCFA dans l'historique
    nb_micro = sum(1 for m in montants_hist if m < 5_000)
    f7_micro = nb_micro / max(len(montants_hist), 1)

    f8_horaire = 1 if (heure < 5 or heure > 22) else 0

    # Montant presque rond (multiple de 10 000), fréquent dans le round tripping
    f9_round_trip = 1 if montant % 10_000 < 500 else 0

    f10_mule = 1 if (50_000 < montant < 300_000 and 2 < frequence < 8) else 0

    return {
        "f1_montant_explosif" : round(f1_montant_explosif, 4),
        "f2_fragmentation"    : int(f2_fragmentation),
        "f3_usurpation"       : round(f3_usurpation, 4),
        "f4_rapidite"         : int(f4_rapidite),
        "f5_localisation"     : int(f5_localisation),
        "f6_compte_dormant"   : round(f6_compte_dormant, 4),
        "f7_micro"            : round(f7_micro, 4),
        "f8_horaire"          : int(f8_horaire),
        "f9_round_trip"       : int(f9_round_trip),
        "f10_mule"            : int(f10_mule),
        "montant"             : int(montant),
        "heure"               : int(heure),
        "frequence"           : int(frequence),
        "est_fraude"          : int(transaction["est_fraude"])
    }


def generer_dataset(afficher_details=True):
    if afficher_details:
        print(f"Génération : {NB_CLIENTS} clients, {NB_FRAUDEURS} fraudeurs, "
              f"{NB_ROUNDS} rounds "
              f"(~{(NB_CLIENTS + NB_FRAUDEURS) * NB_ROUNDS:,} transactions)")

    clients   = [AgentClient(i)   for i in range(NB_CLIENTS)]
    fraudeurs = [AgentFraudeur(i) for i in range(NB_FRAUDEURS)]

    # 50 dernières transactions de chaque client
    historiques = {i: [] for i in range(NB_CLIENTS)}
    lignes = []

    for round_num in range(NB_ROUNDS):
        if afficher_details and round_num % 50 == 0:
            print(f"  round {round_num}/{NB_ROUNDS} : {len(lignes):,} transactions")

        for client in clients:
            t = client.generer_transaction()
            t["est_fraude"] = False
            features = calculer_features(t, historiques[client.client_id])

            historiques[client.client_id].append(t)
            if len(historiques[client.client_id]) > 50:
                historiques[client.client_id].pop(0)

            lignes.append(features)

        for fraudeur in fraudeurs:
            profils = [c.get_profil() for c in clients]
            t = fraudeur.generer_transaction(profils)
            t["est_fraude"] = True

            # La fraude est évaluée face à l'historique d'un client tiré au hasard
            client_cible_id = random.randint(0, NB_CLIENTS - 1)
            lignes.append(calculer_features(t, historiques[client_cible_id]))

        for fraudeur in fraudeurs:
            fraudeur.enregistrer_round()

    df = pd.DataFrame(lignes)

    if afficher_details:
        nb_fraudes  = df["est_fraude"].sum()
        nb_normales = (df["est_fraude"] == 0).sum()
        print(f"\n{len(df):,} transactions : {nb_normales:,} normales, "
              f"{nb_fraudes:,} fraudes (ratio {nb_normales // nb_fraudes}:1)")
        print(f"Sauvegarde : {DATASET_PATH}")

    DATASET_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    return df


if __name__ == "__main__":
    df = generer_dataset(afficher_details=True)
    print(df.head(5).to_string())
