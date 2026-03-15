# =============================================================================
# simulation.py
# Rôle : Moteur central de la simulation multi-agents parallèle.
#        Utilise multiprocessing pour contourner le GIL
#        de Python et obtenir un VRAI parallélisme sur plusieurs cœurs.
#
# Architecture mémoire :
#        → Chaque processus reçoit SA propre copie des données
#        → Pas de mémoire partagée entre processus (sécurité)
#        → Le Pool est libéré automatiquement via "with" (context manager)
#        → On utilise les cœurs PHYSIQUES uniquement (cpu_count // 2)
#
# Modèle PRAM utilisé : CREW
#        → Concurrent Read  : plusieurs processus lisent leurs données
#        → Exclusive Write  : le processus principal collecte les résultats
# =============================================================================

import multiprocessing   # pour parallélisme — contourne le GIL de Python
import time              # Pour mesurer les temps d'exécution (speedup)
import random            # Pour mélanger les transactions aléatoirement

from agent_client      import AgentClient
from agent_fraudeur    import AgentFraudeur
from agent_detecteur   import AgentDetecteur
from agent_superviseur import AgentSuperviseur
from base_donnees      import (initialiser_base, sauvegarder_bilan,
                               sauvegarder_villes_clients,
                               sauvegarder_transactions,
                               get_derniere_simulation_id)

# =============================================================================
# FONCTION PURE DE DETECTION — exécutée dans chaque processus fils
# =============================================================================
# IMPORTANT : Cette fonction DOIT être définie au niveau MODULE (hors classe)
# car sur Windows, multiprocessing utilise "spawn" pour créer les processus.
# Spawn recrée l'environnement Python from scratch → seules les fonctions
# définies au niveau module sont accessibles par les processus fils.
#
# Elle reçoit uniquement des données sérialisables (pas d'objets complexes)
# pour éviter les erreurs de pickle entre processus.
# =============================================================================

def analyser_portion(args):
    """
    Fonction pure exécutée dans un processus fils indépendant.
    Analyse une portion de transactions avec les seuils fournis.

    Pourquoi une fonction pure ?
    → multiprocessing.Pool.map() ne peut passer que des données sérialisables
    → Les objets Python complexes (AgentDetecteur) ne sont pas transmissibles
    → On passe uniquement les DONNEES (seuils + transactions)
    → Le processus recrée un détecteur LOCAL à partir de ces données

    Cycle mémoire de cette fonction :
    → Allocation  : à l'appel (processus fils créé par spawn)
    → Utilisation : analyse des transactions
    → Libération  : automatique à la fin (processus fils se termine)

    :param args: Tuple contenant (seuils du détecteur, portion de transactions,
                                  id du détecteur)
    :return    : Dictionnaire avec les résultats et statistiques
    """

    # Décompactage des arguments reçus du processus principal
    # On reçoit des données simples (dict, list) — pas d'objets complexes
    seuils, portion, detecteur_id = args

    # --- Recréation d'un détecteur LOCAL dans ce processus -------------------
    # Chaque processus fils crée son propre détecteur avec les seuils reçus
    # Ce détecteur est LOCAL → pas partagé → pas de race condition possible
    from agent_detecteur import AgentDetecteur
    detecteur = AgentDetecteur(detecteur_id=detecteur_id)

    # Application des seuils transmis par le processus principal
    # (les seuils ont évolué pendant les rounds précédents)
    #detecteur.seuil_montant    = seuils["seuil_montant"]
    #detecteur.seuil_frequence  = seuils["seuil_frequence"]
    #detecteur.seuil_heure_nuit = seuils["seuil_heure_nuit"]
    #detecteur.seuil_micro      = seuils["seuil_micro"]
    #detecteur.seuil_round_trip = seuils["seuil_round_trip"]
    #detecteur.seuil_mule       = seuils["seuil_mule"]

    # --- Analyse de chaque transaction de la portion -------------------------
    resultats_locaux = []

    for transaction in portion:

        # Analyse selon les règles adaptatives du détecteur
        decision = detecteur.analyser(transaction)

        # Enregistrement local du résultat
        detecteur.enregistrer_resultat(transaction, decision)

        # Stockage du résultat sous forme de tuple sérialisable
        # (dictionnaire + booléen) → transmissible entre processus
        resultats_locaux.append((transaction, decision))

    # --- Retour des résultats ET des statistiques au processus principal -----
    # On retourne aussi les statistiques pour mettre à jour le vrai détecteur
    return {
        "detecteur_id"      : detecteur_id,
        "resultats"         : resultats_locaux,

        # Statistiques à synchroniser avec le vrai détecteur
        "nb_analyses"       : detecteur.nb_analyses,
        "nb_fraudes_detectees": detecteur.nb_fraudes_detectees,
        "nb_fraudes_ratees" : detecteur.nb_fraudes_ratees,
        "nb_faux_positifs"  : detecteur.nb_faux_positifs,

        # Seuils après adaptation locale (pour synchronisation)
        "seuils_finaux": {},
        "villes_clients"      : detecteur.villes_clients
    }


def analyser_sequentiel(args):
    """
    Fonction pure pour le mode séquentiel (référence pour le speedup).
    Utilisée avec un seul processus pour mesurer le temps de base.

    :param args: Tuple (seuils, toutes les transactions, id détecteur)
    :return    : Dictionnaire avec les résultats
    """

    # Même logique que analyser_portion mais pour un seul détecteur
    seuils, transactions, detecteur_id = args

    from agent_detecteur import AgentDetecteur
    detecteur = AgentDetecteur(detecteur_id=detecteur_id)

    #detecteur.seuil_montant    = seuils["seuil_montant"]
    #detecteur.seuil_frequence  = seuils["seuil_frequence"]
    #detecteur.seuil_heure_nuit = seuils["seuil_heure_nuit"]
    #detecteur.seuil_micro      = seuils["seuil_micro"]
    #detecteur.seuil_round_trip = seuils["seuil_round_trip"]
    #detecteur.seuil_mule       = seuils["seuil_mule"]

    resultats = []
    for transaction in transactions:
        decision = detecteur.analyser(transaction)
        detecteur.enregistrer_resultat(transaction, decision)
        resultats.append((transaction, decision))

    return {
        "detecteur_id"        : detecteur_id,
        "resultats"           : resultats,
        "nb_analyses"         : detecteur.nb_analyses,
        "nb_fraudes_detectees": detecteur.nb_fraudes_detectees,
        "nb_fraudes_ratees"   : detecteur.nb_fraudes_ratees,
        "nb_faux_positifs"    : detecteur.nb_faux_positifs,
        "seuils_finaux": {},
        "villes_clients"      : detecteur.villes_clients
    }


# =============================================================================
# CLASSE SIMULATION
# =============================================================================

class Simulation:
    """
    Moteur de simulation multi-agents avec vrai parallélisme.
    Utilise multiprocessing.Pool pour distribuer les transactions
    entre plusieurs processus s'exécutant sur des cœurs physiques distincts.
    """

    def __init__(self, nb_clients, nb_fraudeurs, nb_detecteurs, nb_rounds):
        """
        Initialise la simulation.

        :param nb_clients   : Nombre d'agents clients normaux
        :param nb_fraudeurs : Nombre d'agents fraudeurs
        :param nb_detecteurs: Nombre de détecteurs (processus parallèles)
        :param nb_rounds    : Nombre de rounds à simuler
        """

        self.nb_rounds     = nb_rounds
        self.nb_detecteurs = nb_detecteurs

        # --- Calcul du nombre optimal de processus ---------------------------
        # On utilise les cœurs PHYSIQUES uniquement pour éviter la contention
        # du cache et l'overhead de l'hyperthreading
        # cpu_count() // 2 = cœurs physiques (les cœurs logiques sont // 1)
        coeurs_physiques   = multiprocessing.cpu_count() // 2
        # On ne dépasse pas le nombre de détecteurs demandés
        # ni le nombre de cœurs physiques disponibles
        self.nb_processus  = min(nb_detecteurs, coeurs_physiques)

        # --- Création des agents ---------------------------------------------
        self.clients   = [AgentClient(i)   for i in range(nb_clients)]
        self.fraudeurs = [AgentFraudeur(i) for i in range(nb_fraudeurs)]
        self.detecteurs= [AgentDetecteur(i) for i in range(nb_detecteurs)]

        # --- Création du superviseur central ---------------------------------
        self.superviseur = AgentSuperviseur(
            fraudeurs  = self.fraudeurs,
            detecteurs = self.detecteurs,
            clients    = self.clients
        )

        # Stockage des résultats du round en cours
        self.resultats = []

        # Pool persistant — créé une seule fois
        # Evite l'overhead de spawn à chaque round
        self.pool = multiprocessing.Pool(processes=self.nb_processus)

    def get_seuils_detecteur(self, detecteur):
        """
        Retourne les seuils du détecteur pour affichage.
        Adapté aux règles calibrées (pas de seuils variables).
        """
        return {
            "seuil_montant"      : 1_000_000,
            "seuil_frequence"    : 20,
            "seuil_heure_nuit"   : 22,
            "seuil_micro"        : 1_000,
            "seuil_round_trip"   : 500_000,
            "seuil_mule"         : 150_000,
            "seuil_score_global" : 30
        }

    def synchroniser_detecteur(self, detecteur, retour_processus):
        """
        Synchronise les statistiques du détecteur principal avec
        les résultats retournés par son processus fils.

        Pourquoi cette synchronisation ?
        → Le processus fils a travaillé sur une COPIE du détecteur
        → Les statistiques (nb_analyses, etc.) ont été mises à jour
          dans la copie locale du processus fils
        → On doit reporter ces statistiques sur le vrai détecteur
          pour que l'interface Streamlit affiche les bonnes valeurs

        :param detecteur         : Le vrai AgentDetecteur à mettre à jour
        :param retour_processus  : Dictionnaire retourné par analyser_portion()
        """

        # Mise à jour des statistiques cumulées
        detecteur.nb_analyses          += retour_processus["nb_analyses"]
        detecteur.nb_fraudes_detectees += retour_processus["nb_fraudes_detectees"]
        detecteur.nb_fraudes_ratees    += retour_processus["nb_fraudes_ratees"]
        detecteur.nb_faux_positifs     += retour_processus["nb_faux_positifs"]

        # Mise à jour des seuils adaptés par le processus fils
        #seuils = retour_processus["seuils_finaux"]
        #detecteur.seuil_montant    = seuils["seuil_montant"]
        #detecteur.seuil_frequence  = seuils["seuil_frequence"]
        #detecteur.seuil_heure_nuit = seuils["seuil_heure_nuit"]
        #detecteur.seuil_micro      = seuils["seuil_micro"]
        #detecteur.seuil_round_trip = seuils["seuil_round_trip"]
        #detecteur.seuil_mule       = seuils["seuil_mule"]

        # Fusion des villes mémorisées par le processus fils
        # → Met à jour le vrai détecteur ET la base immédiatement
        villes = retour_processus.get("villes_clients", {})
        detecteur.villes_clients.update(villes)
        sauvegarder_villes_clients(villes)  # NOUVEAU

    def generer_transactions(self):
        """
        Génère toutes les transactions d'un round.
        Mélange transactions normales et frauduleuses aléatoirement.

        :return: Liste mélangée de toutes les transactions
        """

        transactions    = []
        profils_clients = self.superviseur.collecter_profils_clients()

        # Transactions normales des clients
        for client in self.clients:
            transactions.append(client.generer_transaction())

        # Transactions frauduleuses
        for fraudeur in self.fraudeurs:
            transactions.append(
                fraudeur.generer_transaction(profils_clients)
            )

        # Mélange pour simuler un flux bancaire réaliste
        random.shuffle(transactions)

        # ── Marquage centralisé des IDs dupliqués ─────────────────────
        # Effectué AVANT la distribution aux processus
        # → Chaque processus reçoit le flag id_duplique déjà calculé
        # → Résout le problème de comptage distribué
        #
        # Principe : si un client_id apparaît plus d'une fois
        # dans le même round → quelqu'un usurpe cet ID
        from collections import Counter

        # Compte les occurrences de chaque client_id dans ce round
        compteur = Counter(
            t["client_id"] for t in transactions
            if t.get("client_id") is not None
        )

        # Marque chaque transaction dont l'ID est dupliqué
        for t in transactions:
            cid = t.get("client_id")
            if cid is not None and compteur[cid] > 1:
                # Cet ID apparaît plusieurs fois → usurpation probable
                # Seul le fraudeur (est_fraude=True) est marqué
                # Le vrai client garde id_duplique=False
                if t["est_fraude"]:
                    t["id_duplique"] = True
                else:
                    t["id_duplique"] = False
            else:
                t["id_duplique"] = False

        return transactions

    def distribuer_transactions(self, transactions):
        """
        Distribue équitablement les transactions entre les détecteurs.
        Chaque détecteur reçoit une portion approximativement égale.

        :param transactions: Liste complète des transactions
        :return            : Liste de portions (une par détecteur)
        """

        nb_total      = len(transactions)
        taille_portion = nb_total // self.nb_detecteurs
        portions      = []

        for i in range(self.nb_detecteurs):
            debut = i * taille_portion
            # Le dernier détecteur prend le reste (division non entière)
            fin   = nb_total if i == self.nb_detecteurs - 1 else debut + taille_portion
            portions.append(transactions[debut:fin])

        return portions

    def executer_round_parallele(self, transactions):
        """
        Exécute un round en VRAI parallélisme avec multiprocessing.Pool.

        Cycle mémoire complet :
        1. ALLOCATION   : Pool crée nb_processus processus fils
                          Chaque fils reçoit sa portion de données
        2. EXECUTION    : Chaque processus analyse sa portion (vrai parallélisme)
        3. RETOUR       : Les résultats sont retournés au processus principal
        4. LIBERATION   : Le "with" libère automatiquement le Pool et ses processus

        :param transactions: Toutes les transactions du round
        :return            : Temps d'exécution en secondes
        """

        # Distribution équitable des transactions
        portions = self.distribuer_transactions(transactions)

        # Construction des arguments pour chaque processus fils
        # Chaque élément = (seuils_du_detecteur, sa_portion, son_id)
        # IMPORTANT : uniquement des données sérialisables (dict, list, int)

        # Marque chaque transaction avec l'ID du détecteur qui la traitera
        # Permet à l'animation de montrer le vrai parallélisme

        for i in range(self.nb_detecteurs):
            for t in portions[i]:
                t["detecteur_assign"] = i

        args_processus = [
            (
                self.get_seuils_detecteur(self.detecteurs[i]),  # Seuils sérialisés
                portions[i],                                     # Portion de transactions
                i                                                # ID du détecteur
            )
            for i in range(self.nb_detecteurs)
        ]

        # Vide les résultats du round précédent
        self.resultats = []

        # Mesure du temps de début
        debut = time.time()

        # --- Vrai parallélisme avec multiprocessing.Pool ---------------------
        # "with" garantit la libération automatique de toute la mémoire
        # allouée par le Pool à la fin du bloc (même en cas d'erreur)
        #with multiprocessing.Pool(processes=self.nb_processus) as pool:

            # pool.map() :
            # → Distribue args_processus entre les processus du Pool
            # → Chaque processus exécute analyser_portion() avec ses args
            # → BLOQUE jusqu'à ce que TOUS les processus aient terminé
            # → Retourne la liste des résultats dans le même ordre
        retours = self.pool.map(analyser_portion, args_processus)

        # Mesure du temps de fin (après libération du Pool)
        fin = time.time()

        # --- Synchronisation des résultats -----------------------------------
        # On reporte les statistiques de chaque processus fils
        # sur le vrai détecteur correspondant
        for retour in retours:
            i = retour["detecteur_id"]
            self.synchroniser_detecteur(self.detecteurs[i], retour)
            # Collecte de tous les résultats dans la liste principale
            self.resultats.extend(retour["resultats"])

        return fin - debut

    def executer_round_sequentiel(self, transactions):
        """
        Exécute un round en mode séquentiel (1 seul processus).
        Sert de référence pour calculer le speedup.

        Utilise aussi multiprocessing pour un seul processus
        afin que la comparaison soit équitable (même overhead de spawn).

        :param transactions: Toutes les transactions du round
        :return            : Temps d'exécution en secondes
        """

        # Seuils du premier détecteur comme référence
        seuils_ref = self.get_seuils_detecteur(self.detecteurs[0])

        # Arguments pour un seul processus séquentiel
        args = [(seuils_ref, transactions, 99)]

        self.resultats = []
        debut = time.time()

        # Pool d'UN SEUL processus pour mesure de référence équitable
        # (même overhead de création de processus que le mode parallèle)
        with multiprocessing.Pool(processes=1) as pool_seq:
            retours = pool_seq.map(analyser_sequentiel, args)

        fin = time.time()

        # Collecte des résultats séquentiels
        for retour in retours:
            self.resultats.extend(retour["resultats"])

        return fin - debut

    def lancer(self, callback=None):
        """
        Lance la simulation complète sur tous les rounds.

        Ordre d'exécution :
        1. Mesure séquentielle (référence speedup)
        2. Boucle sur nb_rounds en mode parallèle
        3. Apprentissage des agents après chaque round
        4. Retour du bilan global

        :param callback: Fonction appelée à chaque round (mise à jour Streamlit)
        :return        : Bilan global de la simulation
        """

        # Initialisation de la base de données si première utilisation
        initialiser_base()

        # Identifiant de cette simulation (dernier_id + 1)
        self.simulation_id = get_derniere_simulation_id() + 1

        # --- Mesure de référence séquentielle --------------------------------
        transactions_ref  = self.generer_transactions()
        temps_seq         = self.executer_round_sequentiel(transactions_ref)
        self.superviseur.temps_sequentiel = temps_seq

        # --- Boucle principale des rounds parallèles -------------------------
        for round_num in range(self.nb_rounds):

            # Génération des transactions du round
            transactions = self.generer_transactions()

            # Exécution parallèle sur les cœurs physiques
            temps_par = self.executer_round_parallele(transactions)
            self.superviseur.temps_parallele = temps_par

            # Extraction des transactions et décisions
            txs = [r[0] for r in self.resultats]
            dec = [r[1] for r in self.resultats]

            # Calcul des métriques du round
            metriques = self.superviseur.calculer_metriques_round(txs, dec)
            metriques["temps_execution"] = temps_par
            metriques["speedup"]         = (
                temps_seq / temps_par if temps_par > 0 else 1.0
            )

            # Apprentissage de tous les agents
            self.superviseur.declencher_apprentissage(txs, dec)
            self.superviseur.enregistrer_round(metriques)

            # Sauvegarde des transactions frauduleuses dans la base
            txs_round = [r[0] for r in self.resultats]
            dec_round = [r[1] for r in self.resultats]
            sauvegarder_transactions(
                self.simulation_id, round_num,
                txs_round, dec_round
            )

            # Mise à jour de l'interface Streamlit
            if callback:
                callback(metriques)

        # ── FIN DE SIMULATION ─────────────────────────────────────────────
        bilan = self.superviseur.get_bilan_global()

        # Sauvegarde du bilan global dans la base
        sauvegarder_bilan(
            bilan,
            nb_clients    = len(self.clients),
            nb_fraudeurs  = len(self.fraudeurs),
            nb_detecteurs = len(self.detecteurs),
            nb_rounds     = self.nb_rounds
        )

        # Sauvegarde des villes mémorisées par les détecteurs
        # → Disponibles pour la prochaine simulation
        for detecteur in self.detecteurs:
            sauvegarder_villes_clients(detecteur.villes_clients)

        # Fermeture propre du pool persistant
        self.pool.close()
        self.pool.join()

        return bilan