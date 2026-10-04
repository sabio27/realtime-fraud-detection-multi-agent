"""Superviseur : fait le bilan de chaque round et déclenche l'apprentissage."""


class AgentSuperviseur:

    def __init__(self, fraudeurs, detecteurs, clients):
        self.fraudeurs  = fraudeurs
        self.detecteurs = detecteurs
        self.clients    = clients

        self.round_actuel = 0
        self.historique = []   # métriques de chaque round, pour les graphiques

        # Cumuls sur toute la simulation
        self.total_transactions = 0
        self.total_fraudes_detectees = 0
        self.total_fraudes_ratees = 0
        self.total_faux_positifs = 0

        # Renseignés par simulation.py pour le calcul du speedup
        self.temps_sequentiel = None
        self.temps_parallele = None

    def collecter_profils_clients(self):
        """Profils des clients, transmis aux fraudeurs pour l'usurpation."""
        return [client.get_profil() for client in self.clients]

    def calculer_metriques_round(self, transactions, resultats):
        nb_fraudes_detectees = 0
        nb_fraudes_ratees    = 0
        nb_faux_positifs     = 0
        nb_vrais_negatifs    = 0

        for transaction, decision in zip(transactions, resultats):
            est_fraude = transaction["est_fraude"]

            if est_fraude and decision:
                nb_fraudes_detectees += 1
            elif est_fraude and not decision:
                nb_fraudes_ratees += 1
            elif not est_fraude and decision:
                nb_faux_positifs += 1
                client_id = transaction.get("client_id")
                if client_id is not None:
                    self.clients[client_id].enregistrer_blocage()
            else:
                nb_vrais_negatifs += 1

        total_fraudes  = nb_fraudes_detectees + nb_fraudes_ratees
        total_normales = nb_faux_positifs + nb_vrais_negatifs

        taux_detection = (
            nb_fraudes_detectees / total_fraudes if total_fraudes > 0 else 0.0
        )
        taux_faux_positifs = (
            nb_faux_positifs / total_normales if total_normales > 0 else 0.0
        )

        self.total_transactions      += len(transactions)
        self.total_fraudes_detectees += nb_fraudes_detectees
        self.total_fraudes_ratees    += nb_fraudes_ratees
        self.total_faux_positifs     += nb_faux_positifs

        return {
            "round"                  : self.round_actuel,
            "nb_transactions"        : len(transactions),
            "nb_fraudes_detectees"   : nb_fraudes_detectees,
            "nb_fraudes_ratees"      : nb_fraudes_ratees,
            "nb_faux_positifs"       : nb_faux_positifs,
            "nb_vrais_negatifs"      : nb_vrais_negatifs,
            "taux_detection"         : taux_detection,
            "taux_faux_positifs"     : taux_faux_positifs,
            "strategies_utilisees"   : [f.strategie_actuelle for f in self.fraudeurs],
            "taux_reussite_fraudeurs": [f.get_taux_reussite() for f in self.fraudeurs],
            "taux_echec_detecteurs"  : [d.get_taux_echec() for d in self.detecteurs],
        }

    def declencher_apprentissage(self, transactions, resultats):
        """Chaque fraudeur apprend du sort de sa transaction (bloquée ou non)."""
        for transaction, decision in zip(transactions, resultats):
            if transaction["est_fraude"]:
                fraudeur = self.fraudeurs[transaction["fraudeur_id"]]
                fraudeur.apprendre(detectee=decision)

        for fraudeur in self.fraudeurs:
            fraudeur.enregistrer_round()
        for detecteur in self.detecteurs:
            detecteur.enregistrer_round()

    def enregistrer_round(self, metriques):
        self.historique.append(metriques)
        self.round_actuel += 1

    def get_bilan_global(self):
        total_fraudes = self.total_fraudes_detectees + self.total_fraudes_ratees
        taux_detection_global = (
            self.total_fraudes_detectees / total_fraudes if total_fraudes > 0 else 0.0
        )
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
