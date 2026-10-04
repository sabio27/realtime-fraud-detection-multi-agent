"""Client normal : génère des transactions légitimes.

Les clients servent de bruit de fond. Le détecteur doit laisser passer
leurs transactions tout en bloquant celles des fraudeurs, et c'est grâce
à eux qu'on mesure les faux positifs.
"""

import random

VILLES = ["Abidjan", "Dakar", "Lagos", "Accra", "Bamako",
          "Cotonou", "Lome", "Ouagadougou", "Niamey", "Conakry"]


class AgentClient:

    def __init__(self, client_id):
        self.client_id = client_id

        # Graine fixée par l'id : un même client garde le même profil d'une
        # simulation à l'autre, sinon la mémoire des villes en base ne
        # servirait à rien.
        rng = random.Random(client_id * 42)

        # Profil stable du client (les fraudeurs le copient pour l'usurpation)
        self.ville = rng.choice(VILLES)
        self.montant_moyen = rng.randint(10_000, 200_000)  # FCFA
        self.frequence = rng.randint(1, 5)                 # transactions par round
        self.heure_habituelle = rng.randint(8, 20)

        self.nb_transactions = 0
        self.nb_bloque = 0  # nombre de fois bloqué à tort

    def generer_transaction(self):
        """Transaction conforme au profil, avec de petites variations."""
        variation = random.randint(
            -int(self.montant_moyen * 0.2),
             int(self.montant_moyen * 0.2)
        )
        montant = max(1000, self.montant_moyen + variation)
        heure = max(0, min(23, self.heure_habituelle + random.randint(-1, 1)))
        frequence = max(1, self.frequence + random.randint(-1, 1))

        self.nb_transactions += 1

        return {
            "id"         : f"C{self.client_id}_{self.nb_transactions}",
            "client_id"  : self.client_id,
            "montant"    : montant,
            "frequence"  : frequence,
            "ville"      : self.ville,
            "heure"      : heure,
            "strategie"  : "normal",
            "est_fraude" : False
        }

    def get_profil(self):
        return {
            "client_id"       : self.client_id,
            "ville"           : self.ville,
            "montant_moyen"   : self.montant_moyen,
            "frequence"       : self.frequence,
            "heure_habituelle": self.heure_habituelle
        }

    def enregistrer_blocage(self):
        self.nb_bloque += 1

    def get_taux_blocage(self):
        if self.nb_transactions == 0:
            return 0.0
        return self.nb_bloque / self.nb_transactions
