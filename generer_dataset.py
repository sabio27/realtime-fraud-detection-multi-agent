# =============================================================================
# generer_dataset.py
# Rôle : Génère un dataset de 50 000 transactions simulées labellisées
#        avec 13 features métier mappées sur les 10 stratégies de fraude.
#
# Feature Engineering :
#   13 features construites manuellement correspondent aux
#   10 stratégies de fraude de nos agents fraudeurs.
#   Le modèle XGBoost apprend à reconnaître chaque stratégie.
#
# Approche :
#   Phase 2 d'un vrai système bancaire :
#   "Entraînement sur données accumulées en production"
# =============================================================================

import pandas as pd
import numpy as np
import random
from agent_client   import AgentClient
from agent_fraudeur import AgentFraudeur

# =============================================================================
# PARAMETRES DE GENERATION
# =============================================================================

NB_CLIENTS   = 100   # Clients normaux
NB_FRAUDEURS = 50    # Agents fraudeurs
NB_ROUNDS    = 333   # Rounds → ~50 000 transactions total
DATASET_PATH = "dataset_simulation.csv"

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calculer_features(transaction, historique_client):
    """
    Calcule 13 features métier pour une transaction.
    Chaque feature correspond à une stratégie de fraude connue.

    Mapping stratégies → features :
      montant_explosif  → f1_montant_explosif
      fragmentation     → f2_fragmentation
      usurpation        → f3_usurpation
      rapidite          → f4_rapidite
      localisation      → f5_localisation
      compte_dormant    → f6_compte_dormant
      micro_transactions→ f7_micro
      horaires_suspects → f8_horaire
      round_tripping    → f9_round_trip
      mule_account      → f10_mule
      + 3 features brutes : montant, heure, frequence

    :param transaction       : transaction actuelle (dict)
    :param historique_client : historique des transactions du client (list)
    :return                  : dictionnaire de 13 features
    """

    montant   = transaction["montant"]
    heure     = transaction["heure"]
    ville     = transaction["ville"]
    frequence = transaction["frequence"]

    # Historique des montants et villes du client (20 dernières transactions)
    montants_hist = [t["montant"]   for t in historique_client[-20:]]
    villes_hist   = [t["ville"]     for t in historique_client[-10:]]
    freq_hist     = [t["frequence"] for t in historique_client[-10:]]

    moyenne_montant = np.mean(montants_hist) if montants_hist else montant
    freq_moy        = np.mean(freq_hist)     if freq_hist     else frequence

    # ── Feature 1 : montant_explosif ─────────────────────────────────────────
    # Ratio montant actuel / moyenne historique
    # Valeur élevée = montant anormalement grand par rapport aux habitudes
    # Cible stratégie : montant_explosif, round_tripping
    f1_montant_explosif = montant / max(moyenne_montant, 1)

    # ── Feature 2 : fragmentation ────────────────────────────────────────────
    # Nombre de petits montants récents (< 10 000 FCFA)
    # Valeur élevée = découpage d'un gros montant en petits paiements
    # Cible stratégie : fragmentation
    f2_fragmentation = sum(1 for m in montants_hist if m < 10_000)

    # ── Feature 3 : usurpation ───────────────────────────────────────────────
    # Ecart relatif entre montant actuel et moyenne historique
    # Valeur FAIBLE = profil copié (trop similaire au client légitime)
    # Valeur élevée = comportement différent de l'historique
    # Cible stratégie : usurpation
    f3_usurpation = abs(montant - moyenne_montant) / max(moyenne_montant, 1)

    # ── Feature 4 : rapidite ─────────────────────────────────────────────────
    # Fréquence de transactions dans la dernière période
    # Valeur élevée = beaucoup de transactions rapprochées
    # Cible stratégie : rapidite, fragmentation
    f4_rapidite = frequence

    # ── Feature 5 : localisation ─────────────────────────────────────────────
    # Ville inhabituelle pour ce client (1 = ville jamais vue)
    # Valeur 1 = transaction depuis une ville inconnue
    # Cible stratégie : localisation
    ville_connue   = 1 if ville in villes_hist else 0
    f5_localisation = 1 - ville_connue

    # ── Feature 6 : compte_dormant ───────────────────────────────────────────
    # Baisse soudaine de fréquence par rapport à l'historique
    # Valeur élevée = compte qui était actif et est devenu inactif
    # Cible stratégie : compte_dormant
    f6_compte_dormant = max(0, freq_moy - frequence)

    # ── Feature 7 : micro_transactions ───────────────────────────────────────
    # Ratio de micro-montants (< 5 000 FCFA) dans l'historique récent
    # Valeur élevée = accumulation de très petits montants
    # Cible stratégie : micro_transactions
    nb_micro = sum(1 for m in montants_hist if m < 5_000)
    f7_micro = nb_micro / max(len(montants_hist), 1)

    # ── Feature 8 : horaires_suspects ────────────────────────────────────────
    # Transaction nocturne (avant 5h ou après 22h)
    # Valeur 1 = heure nocturne suspecte
    # Cible stratégie : horaires_suspects
    f8_horaire = 1 if (heure < 5 or heure > 22) else 0

    # ── Feature 9 : round_tripping ───────────────────────────────────────────
    # Montant proche d'un multiple de 10 000 FCFA
    # Les round trips utilisent souvent des montants ronds
    # Valeur 1 = montant rond suspect
    # Cible stratégie : round_tripping
    montant_mod   = montant % 10_000
    f9_round_trip = 1 if montant_mod < 500 else 0

    # ── Feature 10 : mule_account ────────────────────────────────────────────
    # Montant intermédiaire + fréquence modérée
    # Caractéristique des comptes servant de relais
    # Valeur 1 = profil de compte mule détecté
    # Cible stratégie : mule_account
    f10_mule = 1 if (50_000 < montant < 300_000
                     and 2 < frequence < 8) else 0

    # ── Features brutes (3 features supplémentaires) ─────────────────────────
    # Montant, heure et fréquence bruts comme features complémentaires
    # Permettent au modèle d'apprendre des patterns directs

    return {
        # Features métier (10 stratégies)
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
        # Features brutes complémentaires
        "montant"             : int(montant),
        "heure"               : int(heure),
        "frequence"           : int(frequence),
        # Label
        "est_fraude"          : int(transaction["est_fraude"])
    }


# =============================================================================
# GENERATION DU DATASET
# =============================================================================

def generer_dataset(afficher_details=True):
    """
    Génère un dataset de ~50 000 transactions simulées labellisées.

    Processus :
    1. Créer 100 clients et 50 fraudeurs
    2. Simuler 333 rounds de transactions
    3. Calculer les 13 features pour chaque transaction
    4. Sauvegarder en CSV

    :return: Tuple (DataFrame, mapping ville→code)
    """

    if afficher_details:
        print("="*55)
        print("GENERATION DU DATASET DE SIMULATION")
        print("="*55)
        print(f"Clients   : {NB_CLIENTS}")
        print(f"Fraudeurs : {NB_FRAUDEURS}")
        print(f"Rounds    : {NB_ROUNDS}")
        print(f"Total estimé : ~{(NB_CLIENTS + NB_FRAUDEURS) * NB_ROUNDS:,} transactions")

    # Création des agents
    clients   = [AgentClient(i)   for i in range(NB_CLIENTS)]
    fraudeurs = [AgentFraudeur(i) for i in range(NB_FRAUDEURS)]

    # Historique des transactions par client
    # Clé : client_id, Valeur : liste de transactions
    historiques = {i: [] for i in range(NB_CLIENTS)}

    # Collecte de toutes les transactions
    toutes_transactions = []

    for round_num in range(NB_ROUNDS):

        if afficher_details and round_num % 50 == 0:
            print(f"Round {round_num}/{NB_ROUNDS} — "
                  f"{len(toutes_transactions):,} transactions générées...")

        # ── Transactions normales des clients ─────────────────────────────
        for client in clients:
            t             = client.generer_transaction()
            t["est_fraude"] = False
            features      = calculer_features(t, historiques[client.client_id])

            # Mise à jour de l'historique du client
            historiques[client.client_id].append(t)
            if len(historiques[client.client_id]) > 50:
                historiques[client.client_id].pop(0)

            toutes_transactions.append(features)

        # ── Transactions frauduleuses ──────────────────────────────────────
        for fraudeur in fraudeurs:
            profils = [c.get_profil() for c in clients]
            t       = fraudeur.generer_transaction(profils)
            t["est_fraude"] = True

            # Le fraudeur usurpe un client aléatoire → son historique
            client_cible_id = random.randint(0, NB_CLIENTS - 1)
            features        = calculer_features(
                                t,
                                historiques[client_cible_id]
                              )
            toutes_transactions.append(features)

        # Les fraudeurs enregistrent le round pour le Q-Learning
        for fraudeur in fraudeurs:
            fraudeur.enregistrer_round()

    # Conversion en DataFrame
    df = pd.DataFrame(toutes_transactions)

    # Statistiques finales
    nb_fraudes  = df["est_fraude"].sum()
    nb_normales = (df["est_fraude"] == 0).sum()

    if afficher_details:
        print(f"\n{'='*55}")
        print(f"DATASET GENERE")
        print(f"{'='*55}")
        print(f"Total transactions : {len(df):,}")
        print(f"Normales           : {nb_normales:,} "
              f"({nb_normales/len(df)*100:.1f}%)")
        print(f"Fraudes            : {nb_fraudes:,} "
              f"({nb_fraudes/len(df)*100:.1f}%)")
        print(f"Ratio              : {nb_normales//nb_fraudes}:1")
        print(f"\nFeatures ({len(df.columns)-1}) :")
        for col in df.columns:
            if col != "est_fraude":
                print(f"  → {col}")
        print(f"\nSauvegarde : {DATASET_PATH}")

    # Sauvegarde CSV
    df.to_csv(DATASET_PATH, index=False)

    return df


# =============================================================================
# EXECUTION DIRECTE
# =============================================================================

if __name__ == "__main__":
    df = generer_dataset(afficher_details=True)
    print(f"\nApercu :")
    print(df.head(5).to_string())