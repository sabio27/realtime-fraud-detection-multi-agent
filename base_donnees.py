# =============================================================================
# base_donnees.py
# Rôle : Gère la persistance des données entre les simulations.
#        Utilise SQLite (inclus dans Python — aucune installation requise).
#
# Avantages de cette approche :
# → Mémoire persistante entre simulations
# → Partagée entre tous les processus (résout le problème multiprocessing)
# → Les détecteurs s'améliorent au fil des simulations
# → Concept réel : "threat intelligence" bancaire
#
# Tables :
# ┌─────────────────────────────────────────────────────┐
# │ clients_profils    : ville habituelle par client    │
# │ historique_fraudes : transactions frauduleuses vues │
# │ stats_simulations  : bilan de chaque simulation     │
# └─────────────────────────────────────────────────────┘
# =============================================================================

import sqlite3   # Inclus dans Python — pas d'installation requise
import os
from datetime import datetime

# Nom du fichier de base de données (créé automatiquement si absent)
DB_PATH = "base_fraude.db"


# =============================================================================
# INITIALISATION DE LA BASE
# =============================================================================

def initialiser_base():
    """
    Crée la base de données et les tables si elles n'existent pas encore.
    Appelée au démarrage de chaque simulation.

    Tables créées :
    → clients_profils    : mémorise la ville habituelle de chaque client
    → historique_fraudes : garde trace de toutes les fraudes détectées
    → stats_simulations  : bilan global de chaque simulation lancée
    """

    # Connexion à la base (crée le fichier .db si absent)
    connexion = sqlite3.connect(DB_PATH)
    curseur   = connexion.cursor()

    # ── TABLE 1 : Profils clients ─────────────────────────────────────────
    # Mémorise la ville habituelle de chaque client entre les simulations
    # Un client qui change de ville = signal suspect (usurpation, localisation)
    curseur.execute("""
        CREATE TABLE IF NOT EXISTS clients_profils (
            client_id         INTEGER PRIMARY KEY,
            ville_habituelle  TEXT    NOT NULL,
            montant_moyen     REAL    DEFAULT 0,
            nb_transactions   INTEGER DEFAULT 0,
            derniere_maj      TEXT    NOT NULL
        )
    """)

    # ── TABLE 2 : Historique des fraudes détectées ────────────────────────
    # Garde trace de chaque fraude détectée pour enrichir la connaissance
    # Utile pour analyser les patterns et améliorer les règles
    curseur.execute("""
        CREATE TABLE IF NOT EXISTS historique_fraudes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            simulation_id  INTEGER NOT NULL,
            round_num      INTEGER NOT NULL,
            montant        REAL    NOT NULL,
            frequence      INTEGER NOT NULL,
            heure          INTEGER NOT NULL,
            ville          TEXT    NOT NULL,
            strategie      TEXT    NOT NULL,
            detectee        INTEGER NOT NULL  -- 1 = détectée, 0 = ratée
        )
    """)

    # ── TABLE 3 : Statistiques des simulations ────────────────────────────
    # Bilan de chaque simulation pour mesurer la progression
    # Permet de voir si le taux de détection augmente avec le temps
    curseur.execute("""
        CREATE TABLE IF NOT EXISTS stats_simulations (
            simulation_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            date_heure         TEXT    NOT NULL,
            nb_clients         INTEGER NOT NULL,
            nb_fraudeurs       INTEGER NOT NULL,
            nb_detecteurs      INTEGER NOT NULL,
            nb_rounds          INTEGER NOT NULL,
            total_transactions INTEGER NOT NULL,
            fraudes_detectees  INTEGER NOT NULL,
            fraudes_ratees     INTEGER NOT NULL,
            faux_positifs      INTEGER NOT NULL,
            taux_detection     REAL    NOT NULL
        )
    """)

    # Validation et fermeture propre de la connexion
    connexion.commit()
    connexion.close()


# =============================================================================
# GESTION DES PROFILS CLIENTS
# =============================================================================

def charger_villes_clients():
    """
    Charge depuis la base toutes les villes habituelles connues.
    Appelée au démarrage de chaque simulation pour initialiser
    les détecteurs avec la mémoire des simulations précédentes.

    :return: Dictionnaire {client_id: ville_habituelle}
             Exemple : {0: "Abidjan", 1: "Dakar", 5: "Lagos"}
    """

    if not os.path.exists(DB_PATH):
        # Base non encore créée → aucun historique disponible
        return {}

    connexion = sqlite3.connect(DB_PATH)
    curseur   = connexion.cursor()

    # Récupération de tous les profils clients stockés
    curseur.execute("SELECT client_id, ville_habituelle FROM clients_profils")
    lignes = curseur.fetchall()

    connexion.close()

    # Conversion en dictionnaire pour accès rapide par client_id
    return {ligne[0]: ligne[1] for ligne in lignes}


def sauvegarder_villes_clients(villes_clients):
    """
    Sauvegarde ou met à jour les villes habituelles des clients.
    Appelée à la fin de chaque simulation pour enrichir la mémoire.

    Comportement :
    → Si le client est nouveau → insertion
    → Si le client existe déjà → mise à jour de la ville

    :param villes_clients: Dictionnaire {client_id: ville_habituelle}
    """

    if not villes_clients:
        return  # Rien à sauvegarder

    connexion = sqlite3.connect(DB_PATH)
    curseur   = connexion.cursor()

    maintenant = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for client_id, ville in villes_clients.items():

        # INSERT OR REPLACE : insère si nouveau, remplace si existant
        curseur.execute("""
            INSERT OR REPLACE INTO clients_profils
                (client_id, ville_habituelle, nb_transactions, derniere_maj)
            VALUES (?, ?, 
                COALESCE(
                    (SELECT nb_transactions FROM clients_profils
                     WHERE client_id = ?), 0
                ) + 1,
                ?)
        """, (client_id, ville, client_id, maintenant))

    connexion.commit()
    connexion.close()


# =============================================================================
# GESTION DE L'HISTORIQUE DES FRAUDES
# =============================================================================

def sauvegarder_transactions(simulation_id, round_num, transactions, decisions):
    """
    Sauvegarde toutes les transactions frauduleuses du round dans la base.
    Seules les fraudes (détectées OU ratées) sont stockées.
    Les transactions normales ne sont pas sauvegardées (trop volumineuses).

    :param simulation_id : Identifiant de la simulation en cours
    :param round_num     : Numéro du round actuel
    :param transactions  : Liste de toutes les transactions du round
    :param decisions     : Liste des décisions (True=bloqué, False=autorisé)
    """

    connexion = sqlite3.connect(DB_PATH)
    curseur   = connexion.cursor()

    for transaction, decision in zip(transactions, decisions):

        # On ne sauvegarde que les transactions frauduleuses
        if not transaction["est_fraude"]:
            continue

        curseur.execute("""
            INSERT INTO historique_fraudes
                (simulation_id, round_num, montant, frequence,
                 heure, ville, strategie, detectee)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            simulation_id,
            round_num,
            transaction["montant"],
            transaction["frequence"],
            transaction["heure"],
            transaction["ville"],
            transaction.get("strategie", "inconnue"),
            1 if decision else 0   # 1=détectée, 0=ratée
        ))

    connexion.commit()
    connexion.close()


# =============================================================================
# GESTION DES STATISTIQUES DE SIMULATION
# =============================================================================

def sauvegarder_bilan(bilan, nb_clients, nb_fraudeurs,
                      nb_detecteurs, nb_rounds):
    """
    Sauvegarde le bilan global d'une simulation terminée.
    Retourne l'identifiant de la simulation créée.

    :param bilan        : Dictionnaire retourné par get_bilan_global()
    :param nb_clients   : Nombre de clients dans la simulation
    :param nb_fraudeurs : Nombre de fraudeurs dans la simulation
    :param nb_detecteurs: Nombre de détecteurs dans la simulation
    :param nb_rounds    : Nombre de rounds effectués
    :return             : simulation_id créé (entier)
    """

    connexion = sqlite3.connect(DB_PATH)
    curseur   = connexion.cursor()

    maintenant = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    curseur.execute("""
        INSERT INTO stats_simulations
            (date_heure, nb_clients, nb_fraudeurs, nb_detecteurs,
             nb_rounds, total_transactions, fraudes_detectees,
             fraudes_ratees, faux_positifs, taux_detection)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        maintenant,
        nb_clients,
        nb_fraudeurs,
        nb_detecteurs,
        nb_rounds,
        bilan["total_transactions"],
        bilan["total_fraudes_detectees"],
        bilan["total_fraudes_ratees"],
        bilan["total_faux_positifs"],
        bilan["taux_detection_global"]
    ))

    # Récupération de l'ID auto-généré
    simulation_id = curseur.lastrowid

    connexion.commit()
    connexion.close()

    return simulation_id


def charger_historique_simulations():
    """
    Charge le bilan de toutes les simulations passées.
    Utilisé par l'interface Streamlit pour afficher la progression.

    :return: Liste de dictionnaires, une entrée par simulation
    """

    if not os.path.exists(DB_PATH):
        return []

    connexion = sqlite3.connect(DB_PATH)
    curseur   = connexion.cursor()

    curseur.execute("""
        SELECT simulation_id, date_heure, nb_clients, nb_fraudeurs,
               nb_detecteurs, nb_rounds, total_transactions,
               fraudes_detectees, fraudes_ratees, faux_positifs,
               taux_detection
        FROM stats_simulations
        ORDER BY simulation_id ASC
    """)

    lignes = curseur.fetchall()
    connexion.close()

    # Conversion en liste de dictionnaires pour faciliter l'affichage
    return [
        {
            "simulation_id"     : l[0],
            "date_heure"        : l[1],
            "nb_clients"        : l[2],
            "nb_fraudeurs"      : l[3],
            "nb_detecteurs"     : l[4],
            "nb_rounds"         : l[5],
            "total_transactions": l[6],
            "fraudes_detectees" : l[7],
            "fraudes_ratees"    : l[8],
            "faux_positifs"     : l[9],
            "taux_detection"    : l[10]
        }
        for l in lignes
    ]


def get_derniere_simulation_id():
    """
    Retourne l'identifiant de la dernière simulation enregistrée.
    Retourne 0 si aucune simulation n'a encore été sauvegardée.

    :return: Entier (simulation_id)
    """

    if not os.path.exists(DB_PATH):
        return 0

    connexion = sqlite3.connect(DB_PATH)
    curseur   = connexion.cursor()

    curseur.execute("""
        SELECT MAX(simulation_id) FROM stats_simulations
    """)

    resultat = curseur.fetchone()[0]
    connexion.close()

    return resultat if resultat is not None else 0


# =============================================================================
# EXECUTION DIRECTE — Test de la base
# =============================================================================

if __name__ == "__main__":

    print("Initialisation de la base de données...")
    initialiser_base()
    print(f"Base créée : {DB_PATH}")

    # Test d'écriture et lecture
    villes_test = {0: "Abidjan", 1: "Dakar", 2: "Lagos"}
    sauvegarder_villes_clients(villes_test)

    villes_chargees = charger_villes_clients()
    print(f"Villes chargées : {villes_chargees}")

    historique = charger_historique_simulations()
    print(f"Simulations passées : {len(historique)}")
    print("Base de données opérationnelle !")