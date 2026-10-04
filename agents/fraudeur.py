"""Fraudeur : choisit une stratégie de fraude et apprend laquelle passe le mieux.

L'apprentissage est un Q-learning simplifié, sans état : chaque stratégie a
un score, mis à jour selon que la fraude est passée ou non. Le choix suit
une politique epsilon-greedy (on explore 20 % du temps).
"""

import random

from agents.client import VILLES


class AgentFraudeur:

    # Les 10 stratégies, inspirées de cas de fraude réels
    STRATEGIES = [
        "montant_explosif",    # un seul virement énorme
        "fragmentation",       # un gros montant découpé en petits paiements
        "usurpation",          # copie du profil d'un vrai client
        "rapidite",            # rafale de transactions
        "localisation",        # ville incohérente
        "compte_dormant",      # compte endormi qui se réveille d'un coup
        "micro_transactions",  # beaucoup de très petits montants
        "horaires_suspects",   # virements en pleine nuit
        "round_tripping",      # l'argent tourne entre comptes intermédiaires
        "mule_account"         # un tiers sert de relais
    ]

    def __init__(self, fraudeur_id):
        self.fraudeur_id = fraudeur_id

        self.q_scores = {s: 0.0 for s in self.STRATEGIES}
        self.epsilon = 0.2   # part d'exploration
        self.alpha = 0.1     # vitesse d'apprentissage

        self.strategie_actuelle = random.choice(self.STRATEGIES)

        self.nb_transactions = 0
        self.nb_reussites = 0
        self.nb_echecs = 0
        self.historique_reussite = []  # taux de réussite à chaque round
        self.actif = True

    def choisir_strategie(self):
        """Epsilon-greedy : au hasard 20 % du temps, sinon la meilleure connue."""
        if random.random() < self.epsilon:
            self.strategie_actuelle = random.choice(self.STRATEGIES)
        else:
            self.strategie_actuelle = max(self.q_scores, key=self.q_scores.get)
        return self.strategie_actuelle

    def generer_transaction(self, clients_normaux):
        """Construit une transaction frauduleuse selon la stratégie choisie.

        clients_normaux : profils des vrais clients, utilisés pour l'usurpation.
        """
        strategie = self.choisir_strategie()

        ville_base = random.choice(VILLES)
        heure = random.randint(0, 23)
        id_usurpe = None

        if strategie == "montant_explosif":
            montant   = random.randint(1_000_000, 5_000_000)
            frequence = 1
            ville     = ville_base
            heure     = random.randint(8, 18)  # heure de bureau pour paraître normal

        elif strategie == "fragmentation":
            montant   = random.randint(10_000, 50_000)
            frequence = random.randint(8, 15)
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "usurpation":
            if clients_normaux:
                cible     = random.choice(clients_normaux)
                montant   = cible["montant_moyen"]
                frequence = cible["frequence"]
                ville     = cible["ville"]
                heure     = cible["heure_habituelle"]
                # Le fraudeur reprend l'id du client : c'est ce qui permet de
                # le repérer quand le vrai client transige au même round.
                id_usurpe = cible["client_id"]
            else:
                montant   = random.randint(20_000, 100_000)
                frequence = 2
                ville     = ville_base
                heure     = random.randint(8, 18)

        elif strategie == "rapidite":
            montant   = random.randint(5_000, 30_000)
            frequence = random.randint(20, 50)
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "localisation":
            montant   = random.randint(50_000, 300_000)
            frequence = random.randint(3, 7)
            ville     = random.choice([v for v in VILLES if v != ville_base])
            heure     = random.randint(8, 18)

        elif strategie == "compte_dormant":
            montant   = random.randint(100_000, 800_000)
            frequence = random.randint(5, 12)
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "micro_transactions":
            montant   = random.randint(100, 999)
            frequence = random.randint(50, 200)
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "horaires_suspects":
            montant   = random.randint(200_000, 1_000_000)
            frequence = random.randint(2, 6)
            ville     = ville_base
            heure     = random.randint(0, 4)

        elif strategie == "round_tripping":
            montant   = random.randint(500_000, 2_000_000)
            frequence = random.randint(4, 8)
            ville     = random.choice(VILLES)
            heure     = random.randint(8, 18)

        elif strategie == "mule_account":
            montant   = random.randint(150_000, 600_000)
            frequence = random.randint(3, 8)
            ville     = random.choice(VILLES)
            heure     = random.randint(8, 18)

        self.nb_transactions += 1

        # Sans usurpation, on donne un id client négatif pour ne jamais
        # tomber sur celui d'un vrai client.
        if id_usurpe is not None:
            client_id = id_usurpe
        else:
            client_id = -(self.fraudeur_id + 1)

        return {
            "id"          : f"F{self.fraudeur_id}_{self.nb_transactions}",
            "fraudeur_id" : self.fraudeur_id,
            "client_id"   : client_id,
            "montant"     : montant,
            "frequence"   : frequence,
            "ville"       : ville,
            "heure"       : heure,
            "strategie"   : strategie,
            "est_fraude"  : True
        }

    def apprendre(self, detectee):
        """Q(s) <- Q(s) + alpha * (r - Q(s)), avec r = +1 si la fraude passe, -1 sinon."""
        reward = -1 if detectee else +1
        q = self.q_scores[self.strategie_actuelle]
        self.q_scores[self.strategie_actuelle] = q + self.alpha * (reward - q)

        if detectee:
            self.nb_echecs += 1
        else:
            self.nb_reussites += 1

    def get_taux_reussite(self):
        if self.nb_transactions == 0:
            return 0.0
        return self.nb_reussites / self.nb_transactions

    def enregistrer_round(self):
        self.historique_reussite.append(self.get_taux_reussite())
