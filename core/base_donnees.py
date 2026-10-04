"""Persistance SQLite entre les simulations.

Trois tables :
    clients_profils    : ville habituelle de chaque client
    historique_fraudes : chaque transaction frauduleuse, détectée ou non
    stats_simulations  : bilan de chaque simulation

La table des villes permet aux détecteurs de repartir avec ce qu'ils ont
appris la fois précédente. Elle sert aussi de mémoire commune aux
processus, qui ne partagent pas leur RAM.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "base_fraude.db"


def _connexion():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialiser_base():
    """Crée les tables si elles n'existent pas encore."""
    connexion = _connexion()
    curseur   = connexion.cursor()

    curseur.execute("""
        CREATE TABLE IF NOT EXISTS clients_profils (
            client_id         INTEGER PRIMARY KEY,
            ville_habituelle  TEXT    NOT NULL,
            montant_moyen     REAL    DEFAULT 0,
            nb_transactions   INTEGER DEFAULT 0,
            derniere_maj      TEXT    NOT NULL
        )
    """)

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
            detectee       INTEGER NOT NULL  -- 1 = détectée, 0 = ratée
        )
    """)

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

    connexion.commit()
    connexion.close()


def charger_villes_clients():
    """Renvoie {client_id: ville_habituelle}, vide si la base n'existe pas."""
    if not os.path.exists(DB_PATH):
        return {}

    connexion = _connexion()
    curseur   = connexion.cursor()
    curseur.execute("SELECT client_id, ville_habituelle FROM clients_profils")
    lignes = curseur.fetchall()
    connexion.close()

    return {client_id: ville for client_id, ville in lignes}


def sauvegarder_villes_clients(villes_clients):
    """Insère les nouveaux clients et met à jour ceux déjà connus."""
    if not villes_clients:
        return

    connexion = _connexion()
    curseur   = connexion.cursor()
    maintenant = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for client_id, ville in villes_clients.items():
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


def sauvegarder_transactions(simulation_id, round_num, transactions, decisions):
    """Enregistre les fraudes du round. Les transactions normales ne sont pas gardées."""
    connexion = _connexion()
    curseur   = connexion.cursor()

    for transaction, decision in zip(transactions, decisions):
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
            1 if decision else 0
        ))

    connexion.commit()
    connexion.close()


def sauvegarder_bilan(bilan, nb_clients, nb_fraudeurs, nb_detecteurs, nb_rounds):
    """Enregistre le bilan d'une simulation et renvoie son identifiant."""
    connexion = _connexion()
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

    simulation_id = curseur.lastrowid
    connexion.commit()
    connexion.close()
    return simulation_id


def charger_historique_simulations():
    """Bilans de toutes les simulations passées, de la plus ancienne à la plus récente."""
    if not os.path.exists(DB_PATH):
        return []

    connexion = _connexion()
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

    colonnes = ["simulation_id", "date_heure", "nb_clients", "nb_fraudeurs",
                "nb_detecteurs", "nb_rounds", "total_transactions",
                "fraudes_detectees", "fraudes_ratees", "faux_positifs",
                "taux_detection"]
    return [dict(zip(colonnes, ligne)) for ligne in lignes]


def get_derniere_simulation_id():
    """Identifiant de la dernière simulation enregistrée, 0 s'il n'y en a pas."""
    if not os.path.exists(DB_PATH):
        return 0

    connexion = _connexion()
    curseur   = connexion.cursor()
    curseur.execute("SELECT MAX(simulation_id) FROM stats_simulations")
    resultat = curseur.fetchone()[0]
    connexion.close()

    return resultat if resultat is not None else 0
