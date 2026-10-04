"""Détecteur : attribue un score de risque à chaque transaction.

Le score (plafonné à 100) additionne les points de dix règles métier.
Une transaction est bloquée dès que le score atteint 30.

Les seuils sont placés juste au-delà de ce qu'un client normal peut
produire dans la simulation :
    montant   : 8 000 à 240 000 FCFA
    fréquence : 1 à 6 transactions par round
    heure     : 7 h à 21 h

J'ai testé des modèles de machine learning à la place de ces règles
(voir le dossier ml/), mais ils laissaient passer beaucoup plus de fraudes.
"""

import math

from core.base_donnees import charger_villes_clients

SEUIL_BLOCAGE = 30


class AgentDetecteur:

    def __init__(self, detecteur_id):
        self.detecteur_id = detecteur_id

        # Ville habituelle de chaque client, rechargée depuis la base pour
        # profiter de ce que les simulations précédentes ont appris.
        self.villes_clients = charger_villes_clients()

        self.nb_analyses = 0
        self.nb_fraudes_detectees = 0   # vrais positifs
        self.nb_fraudes_ratees = 0      # faux négatifs
        self.nb_faux_positifs = 0       # clients bloqués à tort
        self.historique_echec = []      # taux de fraudes ratées à chaque round

    def calculer_score_risque(self, transaction):
        montant   = transaction["montant"]
        frequence = transaction["frequence"]
        heure     = transaction["heure"]
        ville     = transaction["ville"]
        client_id = transaction.get("client_id")

        score = 0

        # Le même id client apparaît deux fois dans le round : un fraudeur
        # se fait passer pour un client actif au même moment. Le marquage est
        # fait dans simulation.py, avant le découpage entre processus, car
        # aucun processus ne voit seul toutes les transactions du round.
        if transaction.get("id_duplique", False):
            score += 50

        # Montant explosif (un client ne dépasse jamais 240 000)
        if montant >= 1_000_000:
            score += 60
        elif montant >= 500_000:
            score += 40

        # Rafale de transactions (un client en fait au plus 6)
        if frequence >= 20:
            score += 60
        elif frequence >= 10:
            score += 40
        elif frequence >= 7:
            score += 20

        # Micro-transactions : moins de 1 000 FCFA, mais très souvent
        if montant < 1_000 and frequence > 6:
            score += 60

        # Nuit (un client ne transige pas avant 7 h)
        if heure <= 4:
            score += 50
        elif heure <= 6:
            score += 25

        # Round tripping : gros montant qui circule plusieurs fois
        if montant >= 500_000 and frequence >= 4:
            score += 50

        # Fragmentation : petits montants très fréquents
        if frequence >= 8 and montant <= 50_000:
            score += 40

        # Compte dormant réveillé. On exige 8 transactions et plus : un client
        # peut très bien avoir 200 000 FCFA et 5 ou 6 transactions.
        if 100_000 <= montant <= 800_000 and frequence >= 8:
            score += 35

        # Compte mule : montant au-dessus de ce qu'un client envoie
        if montant > 240_000 and frequence >= 3:
            score += 25

        # Heure tardive : signal faible, utile en combinaison
        if heure >= 22:
            score += 15

        # Changement de ville (localisation, usurpation). La première fois
        # qu'on voit un client, on retient sa ville sans le pénaliser.
        if client_id is not None:
            if client_id in self.villes_clients:
                if ville != self.villes_clients[client_id]:
                    score += 50
            else:
                self.villes_clients[client_id] = ville

        return min(score, 100)

    def analyser(self, transaction):
        """Renvoie True si la transaction doit être bloquée."""
        # Charge CPU volontaire, pour simuler le coût d'un vrai contrôle
        # (signature, accès base...). Sans elle, l'analyse est si rapide
        # que le parallélisme ne peut rien montrer.
        _ = sum(math.sqrt(i) for i in range(1_000_000))

        self.nb_analyses += 1
        return self.calculer_score_risque(transaction) >= SEUIL_BLOCAGE

    def enregistrer_resultat(self, transaction, decision):
        est_fraude = transaction["est_fraude"]
        if est_fraude and decision:
            self.nb_fraudes_detectees += 1
        elif est_fraude and not decision:
            self.nb_fraudes_ratees += 1
        elif not est_fraude and decision:
            self.nb_faux_positifs += 1

    def get_taux_echec(self):
        """Part des fraudes qui sont passées."""
        total = self.nb_fraudes_detectees + self.nb_fraudes_ratees
        if total == 0:
            return 0.0
        return self.nb_fraudes_ratees / total

    def enregistrer_round(self):
        self.historique_echec.append(self.get_taux_echec())
