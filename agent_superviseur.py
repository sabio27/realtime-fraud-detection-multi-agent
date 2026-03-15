# =============================================================================
# agent_superviseur.py
# Rôle : Coordonne tous les agents de la simulation (clients, fraudeurs,
#        détecteurs). Il orchestre chaque round, collecte les résultats,
#        calcule les métriques globales et déclenche l'apprentissage
#        de chaque agent à la fin de chaque round.
#        C'est le "chef d'orchestre" du système multi-agents.
# =============================================================================

class AgentSuperviseur:
    """
    Le superviseur central qui :
    - Orchestre le déroulement de chaque round
    - Collecte et centralise toutes les statistiques
    - Déclenche l'apprentissage des fraudeurs et l'adaptation des détecteurs
    - Calcule le speedup parallèle vs séquentiel
    - Fournit toutes les données nécessaires à l'interface Streamlit
    """

    def __init__(self, fraudeurs, detecteurs, clients):
        """
        Initialise le superviseur avec les trois types d'agents.

        :param fraudeurs  : Liste des agents fraudeurs
        :param detecteurs : Liste des agents détecteurs
        :param clients    : Liste des agents clients normaux
        """

        # --- Références vers tous les agents de la simulation ----------------
        # Le superviseur connaît tous les agents pour les coordonner
        self.fraudeurs  = fraudeurs   # Liste des AgentFraudeur
        self.detecteurs = detecteurs  # Liste des AgentDetecteur
        self.clients    = clients     # Liste des AgentClient

        # --- Compteur de rounds ----------------------------------------------
        # Numéro du round en cours (commence à 0, incrémenté à chaque round)
        self.round_actuel = 0

        # --- Historique global par round -------------------------------------
        # Chaque entrée correspond à un round et contient toutes les métriques
        # Utilisé pour tracer tous les graphiques sur l'interface
        self.historique = []

        # --- Statistiques globales cumulées ----------------------------------
        # Nombre total de transactions analysées depuis le début
        self.total_transactions = 0

        # Nombre total de fraudes correctement détectées
        self.total_fraudes_detectees = 0

        # Nombre total de fraudes manquées (passées sans détection)
        self.total_fraudes_ratees = 0

        # Nombre total de faux positifs (clients normaux bloqués)
        self.total_faux_positifs = 0

        # --- Données de speedup ----------------------------------------------
        # Temps d'exécution en mode séquentiel (1 seul détecteur)
        # Calculé lors de la première simulation séquentielle
        self.temps_sequentiel = None

        # Temps d'exécution en mode parallèle (N détecteurs simultanés)
        self.temps_parallele = None

    def collecter_profils_clients(self):
        """
        Collecte les profils de tous les clients normaux.
        Ces profils sont transmis aux fraudeurs pour la stratégie d'usurpation.
        Un fraudeur peut ainsi copier exactement le comportement d'un vrai client.

        :return: Liste des profils clients (dictionnaires)
        """

        # Appelle get_profil() sur chaque client et retourne la liste complète
        return [client.get_profil() for client in self.clients]

    def calculer_metriques_round(self, transactions, resultats):
        """
        Calcule toutes les métriques pour un round donné.
        Compare les décisions des détecteurs avec la réalité de chaque transaction.

        :param transactions : Liste de toutes les transactions du round
        :param resultats    : Liste des décisions (True=bloqué, False=autorisé)
        :return             : Dictionnaire contenant toutes les métriques du round
        """

        # Compteurs locaux pour ce round spécifique
        nb_fraudes_detectees = 0  # Fraudes correctement bloquées
        nb_fraudes_ratees    = 0  # Fraudes passées sans détection
        nb_faux_positifs     = 0  # Clients normaux bloqués à tort
        nb_vrais_negatifs    = 0  # Clients normaux correctement autorisés

        # Parcours de chaque transaction et de sa décision associée
        for transaction, decision in zip(transactions, resultats):

            est_fraude = transaction["est_fraude"]  # Vérité terrain

            if est_fraude and decision:
                # Fraude détectée correctement → bonne détection
                nb_fraudes_detectees += 1

            elif est_fraude and not decision:
                # Fraude manquée → échec du détecteur
                nb_fraudes_ratees += 1

            elif not est_fraude and decision:
                # Client normal bloqué → faux positif
                nb_faux_positifs += 1

                # On notifie le client qu'il a été bloqué injustement
                # pour mettre à jour ses statistiques de blocage
                client_id = transaction.get("client_id")
                if client_id is not None:
                    self.clients[client_id].enregistrer_blocage()

            else:
                # Transaction normale autorisée → fonctionnement correct
                nb_vrais_negatifs += 1

        # Nombre total de fraudes dans ce round
        total_fraudes = nb_fraudes_detectees + nb_fraudes_ratees

        # Nombre total de transactions normales dans ce round
        total_normales = nb_faux_positifs + nb_vrais_negatifs

        # --- Calcul des taux -------------------------------------------------

        # Taux de détection = fraudes détectées / total fraudes
        # Mesure l'efficacité globale du système de détection
        taux_detection = (
            nb_fraudes_detectees / total_fraudes
            if total_fraudes > 0 else 0.0
        )

        # Taux de faux positifs = clients bloqués / total clients normaux
        # Mesure l'impact négatif sur les clients innocents
        taux_faux_positifs = (
            nb_faux_positifs / total_normales
            if total_normales > 0 else 0.0
        )

        # --- Mise à jour des statistiques globales cumulées ------------------
        self.total_transactions      += len(transactions)
        self.total_fraudes_detectees += nb_fraudes_detectees
        self.total_fraudes_ratees    += nb_fraudes_ratees
        self.total_faux_positifs     += nb_faux_positifs

        # --- Construction du dictionnaire de métriques du round --------------
        metriques = {
            "round"               : self.round_actuel,
            "nb_transactions"     : len(transactions),
            "nb_fraudes_detectees": nb_fraudes_detectees,
            "nb_fraudes_ratees"   : nb_fraudes_ratees,
            "nb_faux_positifs"    : nb_faux_positifs,
            "nb_vrais_negatifs"   : nb_vrais_negatifs,
            "taux_detection"      : taux_detection,
            "taux_faux_positifs"  : taux_faux_positifs,

            # Stratégies utilisées par les fraudeurs ce round
            # Utile pour le graphique d'évolution des stratégies
            "strategies_utilisees": [
                f.strategie_actuelle for f in self.fraudeurs
            ],

            # Taux de réussite de chaque fraudeur ce round
            "taux_reussite_fraudeurs": [
                f.get_taux_reussite() for f in self.fraudeurs
            ],

            # Taux d'échec de chaque détecteur ce round
            "taux_echec_detecteurs": [
                d.get_taux_echec() for d in self.detecteurs
            ],

            # Seuils actuels de chaque détecteur (pour suivre leur adaptation)
            "seuils_detecteurs": [
                {
                    "id"             : d.detecteur_id,
                    "seuil_montant"  : 1_000_000,
                    "seuil_frequence": 20
                }
                for d in self.detecteurs
            ]
        }

        return metriques

    def declencher_apprentissage(self, transactions, resultats):
        """
        Déclenche l'apprentissage de tous les agents après un round.
        Appelée par simulation.py à la fin de chaque round.

        Pour les fraudeurs : met à jour les scores Q selon les résultats
        Pour les détecteurs : adapte les seuils selon les performances

        :param transactions : Liste des transactions du round
        :param resultats    : Liste des décisions des détecteurs
        """

        # --- Apprentissage des fraudeurs -------------------------------------
        for transaction, decision in zip(transactions, resultats):

            # On ne traite que les transactions frauduleuses
            if transaction["est_fraude"]:

                # Retrouve le fraudeur qui a généré cette transaction
                fraudeur_id = transaction["fraudeur_id"]
                fraudeur    = self.fraudeurs[fraudeur_id]

                # Déclenche l'apprentissage Q-Learning du fraudeur
                # decision = True signifie que la fraude a été détectée
                fraudeur.apprendre(detectee=decision)

        # --- Enregistrement des rounds pour l'historique ---------------------
        # Chaque agent enregistre son taux du round pour les courbes
        for fraudeur in self.fraudeurs:
            fraudeur.enregistrer_round()

        for detecteur in self.detecteurs:
            detecteur.enregistrer_round()

        # --- Adaptation des seuils des détecteurs ----------------------------
        # Chaque détecteur ajuste ses seuils selon ses performances du round
        for detecteur in self.detecteurs:
            detecteur.adapter_seuils()

    def enregistrer_round(self, metriques):
        """
        Enregistre les métriques du round dans l'historique global.
        Incrémente le compteur de rounds.

        :param metriques: Dictionnaire des métriques calculées pour ce round
        """

        # Ajoute les métriques du round à l'historique
        self.historique.append(metriques)

        # Passe au round suivant
        self.round_actuel += 1

    def get_bilan_global(self):
        """
        Retourne le bilan complet de toute la simulation.
        Appelée à la fin de la simulation pour afficher le résumé final
        sur l'interface Streamlit.

        :return: Dictionnaire contenant toutes les statistiques globales
        """

        # Calcul du taux de détection global sur toute la simulation
        total_fraudes_global = (
            self.total_fraudes_detectees + self.total_fraudes_ratees
        )
        taux_detection_global = (
            self.total_fraudes_detectees / total_fraudes_global
            if total_fraudes_global > 0 else 0.0
        )

        # Calcul du speedup si les deux temps sont disponibles
        # Speedup = temps séquentiel / temps parallèle
        speedup = (
            self.temps_sequentiel / self.temps_parallele
            if self.temps_sequentiel and self.temps_parallele else None
        )

        return {
            "total_transactions"     : self.total_transactions,
            "total_fraudes_detectees": self.total_fraudes_detectees,
            "total_fraudes_ratees"   : self.total_fraudes_ratees,
            "total_faux_positifs"    : self.total_faux_positifs,
            "taux_detection_global"  : taux_detection_global,
            "nb_rounds"              : self.round_actuel,
            "speedup"                : speedup,
            "temps_sequentiel"       : self.temps_sequentiel,
            "temps_parallele"        : self.temps_parallele
        }