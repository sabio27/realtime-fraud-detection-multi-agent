"""Moteur de la simulation.

À chaque round, les clients et les fraudeurs génèrent leurs transactions,
qui sont réparties entre plusieurs processus détecteurs. On utilise
multiprocessing plutôt que des threads, pour échapper au GIL et occuper
réellement plusieurs cœurs.

Côté mémoire, chaque processus travaille sur sa propre copie des données
(lecture concurrente) et seul le processus principal rassemble les
résultats (écriture exclusive). C'est le modèle CREW du cours.
"""

import multiprocessing
import random
import time
from collections import Counter

from agents.client import AgentClient
from agents.fraudeur import AgentFraudeur
from agents.detecteur import AgentDetecteur
from agents.superviseur import AgentSuperviseur
from core.base_donnees import (initialiser_base, sauvegarder_bilan,
                               sauvegarder_villes_clients,
                               sauvegarder_transactions,
                               get_derniere_simulation_id)


def analyser_portion(args):
    """Travail d'un processus fils : analyser sa part des transactions.

    La fonction est au niveau du module parce que, sous Windows,
    multiprocessing démarre les processus en mode "spawn" et ne peut
    appeler que des fonctions importables. Pour la même raison, on lui
    passe des données simples (listes, dictionnaires) plutôt que des objets
    agents : le processus recrée son propre détecteur.
    """
    portion, detecteur_id = args

    detecteur = AgentDetecteur(detecteur_id=detecteur_id)

    resultats = []
    for transaction in portion:
        decision = detecteur.analyser(transaction)
        detecteur.enregistrer_resultat(transaction, decision)
        resultats.append((transaction, decision))

    # On renvoie aussi les compteurs, pour mettre à jour le détecteur
    # du processus principal, qui n'a rien vu passer.
    return {
        "detecteur_id"        : detecteur_id,
        "resultats"           : resultats,
        "nb_analyses"         : detecteur.nb_analyses,
        "nb_fraudes_detectees": detecteur.nb_fraudes_detectees,
        "nb_fraudes_ratees"   : detecteur.nb_fraudes_ratees,
        "nb_faux_positifs"    : detecteur.nb_faux_positifs,
        "villes_clients"      : detecteur.villes_clients
    }


class Simulation:

    def __init__(self, nb_clients, nb_fraudeurs, nb_detecteurs, nb_rounds):
        self.nb_rounds     = nb_rounds
        self.nb_detecteurs = nb_detecteurs

        # Seulement les cœurs physiques : avec l'hyperthreading, deux
        # processus sur le même cœur se gênent plus qu'ils ne s'aident.
        coeurs_physiques  = max(1, multiprocessing.cpu_count() // 2)
        self.nb_processus = min(nb_detecteurs, coeurs_physiques)

        self.clients    = [AgentClient(i)    for i in range(nb_clients)]
        self.fraudeurs  = [AgentFraudeur(i)  for i in range(nb_fraudeurs)]
        self.detecteurs = [AgentDetecteur(i) for i in range(nb_detecteurs)]

        self.superviseur = AgentSuperviseur(
            fraudeurs  = self.fraudeurs,
            detecteurs = self.detecteurs,
            clients    = self.clients
        )

        self.resultats = []

        # Pool créé une seule fois pour toute la simulation : relancer des
        # processus à chaque round coûterait plus cher que l'analyse.
        self.pool = multiprocessing.Pool(processes=self.nb_processus)

    def synchroniser_detecteur(self, detecteur, retour_processus):
        """Reporte sur le détecteur principal ce que son processus a calculé."""
        detecteur.nb_analyses          += retour_processus["nb_analyses"]
        detecteur.nb_fraudes_detectees += retour_processus["nb_fraudes_detectees"]
        detecteur.nb_fraudes_ratees    += retour_processus["nb_fraudes_ratees"]
        detecteur.nb_faux_positifs     += retour_processus["nb_faux_positifs"]

        villes = retour_processus.get("villes_clients", {})
        detecteur.villes_clients.update(villes)
        sauvegarder_villes_clients(villes)

    def generer_transactions(self):
        transactions    = []
        profils_clients = self.superviseur.collecter_profils_clients()

        for client in self.clients:
            transactions.append(client.generer_transaction())
        for fraudeur in self.fraudeurs:
            transactions.append(fraudeur.generer_transaction(profils_clients))

        random.shuffle(transactions)

        # Repérage des id clients en double, fait ici et pas dans les
        # détecteurs : une fois le round découpé, chaque processus ne voit
        # qu'une partie des transactions et raterait les doublons.
        # Seule la transaction frauduleuse est marquée, le vrai client non.
        compteur = Counter(
            t["client_id"] for t in transactions
            if t.get("client_id") is not None
        )
        for t in transactions:
            cid = t.get("client_id")
            t["id_duplique"] = (
                cid is not None and compteur[cid] > 1 and t["est_fraude"]
            )

        return transactions

    def distribuer_transactions(self, transactions):
        """Découpe le round en parts égales, la dernière prend le reste."""
        nb_total       = len(transactions)
        taille_portion = nb_total // self.nb_detecteurs
        portions       = []

        for i in range(self.nb_detecteurs):
            debut = i * taille_portion
            fin   = nb_total if i == self.nb_detecteurs - 1 else debut + taille_portion
            portions.append(transactions[debut:fin])

        return portions

    def executer_round_parallele(self, transactions):
        """Analyse le round en parallèle et renvoie le temps mis, en secondes."""
        portions = self.distribuer_transactions(transactions)

        # Pour l'animation : on note quel détecteur traite chaque transaction
        for i in range(self.nb_detecteurs):
            for t in portions[i]:
                t["detecteur_assign"] = i

        args_processus = [(portions[i], i) for i in range(self.nb_detecteurs)]

        self.resultats = []
        debut = time.time()
        retours = self.pool.map(analyser_portion, args_processus)
        fin = time.time()

        for retour in retours:
            i = retour["detecteur_id"]
            self.synchroniser_detecteur(self.detecteurs[i], retour)
            self.resultats.extend(retour["resultats"])

        return fin - debut

    def executer_round_sequentiel(self, transactions):
        """Référence pour le speedup : tout le round dans un seul processus."""
        self.resultats = []
        debut = time.time()

        with multiprocessing.Pool(processes=1) as pool_seq:
            retours = pool_seq.map(analyser_portion, [(transactions, 99)])

        fin = time.time()

        for retour in retours:
            self.resultats.extend(retour["resultats"])

        return fin - debut

    def lancer(self, callback=None):
        """Joue tous les rounds et renvoie le bilan global.

        callback(metriques) est appelé à la fin de chaque round
        (l'interface s'en sert pour avancer la barre de progression).
        """
        initialiser_base()
        self.simulation_id = get_derniere_simulation_id() + 1

        # Un premier round en séquentiel, qui sert de référence
        transactions_ref = self.generer_transactions()
        temps_seq = self.executer_round_sequentiel(transactions_ref)
        self.superviseur.temps_sequentiel = temps_seq

        for round_num in range(self.nb_rounds):
            transactions = self.generer_transactions()

            temps_par = self.executer_round_parallele(transactions)
            self.superviseur.temps_parallele = temps_par

            txs = [r[0] for r in self.resultats]
            dec = [r[1] for r in self.resultats]

            metriques = self.superviseur.calculer_metriques_round(txs, dec)
            metriques["temps_execution"] = temps_par
            metriques["speedup"] = temps_seq / temps_par if temps_par > 0 else 1.0

            self.superviseur.declencher_apprentissage(txs, dec)
            self.superviseur.enregistrer_round(metriques)

            sauvegarder_transactions(self.simulation_id, round_num, txs, dec)

            if callback:
                callback(metriques)

        bilan = self.superviseur.get_bilan_global()

        sauvegarder_bilan(
            bilan,
            nb_clients    = len(self.clients),
            nb_fraudeurs  = len(self.fraudeurs),
            nb_detecteurs = len(self.detecteurs),
            nb_rounds     = self.nb_rounds
        )

        # Les villes apprises serviront à la prochaine simulation
        for detecteur in self.detecteurs:
            sauvegarder_villes_clients(detecteur.villes_clients)

        self.pool.close()
        self.pool.join()

        return bilan
