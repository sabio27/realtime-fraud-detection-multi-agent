# =============================================================================
# agent_fraudeur.py
# Rôle : Définit le comportement d'un agent fraudeur dans la simulation.
#        Chaque fraudeur génère des transactions frauduleuses ET apprend
#        progressivement à contourner la détection grâce au Q-Learning.
#        10 stratégies de fraude inspirées des cas réels en entreprise.
# =============================================================================

import random   # Pour les choix aléatoires lors de l'exploration des stratégies
import numpy as np  # Pour les calculs mathématiques sur les scores Q

class AgentFraudeur:
    """
    Un agent fraudeur autonome qui :
    - Possède 10 stratégies de fraude inspirées du monde réel
    - Apprend quelle stratégie fonctionne le mieux (Q-Learning simplifié)
    - Alterne entre exploitation (meilleure stratégie connue)
      et exploration (essayer de nouvelles stratégies)
    - Mesure son propre taux de réussite par round
    """

    # -------------------------------------------------------------------------
    # Les 10 stratégies de fraude disponibles
    # Inspirées des vraies techniques utilisées dans la fraude financière
    # -------------------------------------------------------------------------
    STRATEGIES = [
        "montant_explosif",    # Virement énorme et inhabituel
        "fragmentation",       # Découper un gros montant en petites transactions
        "usurpation",          # Copier le profil exact d'un client normal
        "rapidite",            # Envoyer beaucoup de transactions très rapidement
        "localisation",        # Transactions depuis des villes géographiquement impossibles
        "compte_dormant",      # Compte inactif longtemps, soudainement très actif
        "micro_transactions",  # Des milliers de très petits prélèvements invisibles
        "horaires_suspects",   # Transactions effectuées en pleine nuit
        "round_tripping",      # Envoyer et récupérer le même argent via des intermédiaires
        "mule_account"         # Utiliser un tiers innocent comme relais financier
    ]

    def __init__(self, fraudeur_id):
        """
        Initialise un agent fraudeur avec ses attributs de base.

        :param fraudeur_id: Identifiant unique du fraudeur (ex: 0, 1, 2...)
        """

        # Identifiant unique pour distinguer chaque fraudeur dans la simulation
        self.fraudeur_id = fraudeur_id

        # --- Paramètres Q-Learning -------------------------------------------
        # Table des scores Q : un score par stratégie, tous initialisés à 0.0
        # Score élevé = stratégie qui a bien fonctionné dans le passé
        # Score faible = stratégie souvent détectée
        self.q_scores = {s: 0.0 for s in self.STRATEGIES}

        # Epsilon : taux d'exploration (20% du temps on teste une stratégie aléatoire)
        # Cela évite que le fraudeur se spécialise trop vite sur une seule stratégie
        # et passe à côté de meilleures options non encore découvertes
        self.epsilon = 0.2

        # Alpha : taux d'apprentissage (vitesse de mise à jour des scores Q)
        # 0.1 = apprentissage progressif et stable, évite les oscillations brutales
        self.alpha = 0.1

        # Stratégie active au moment présent
        # Choisie aléatoirement au départ car aucune connaissance initiale
        self.strategie_actuelle = random.choice(self.STRATEGIES)

        # --- Statistiques individuelles du fraudeur --------------------------
        # Nombre total de transactions frauduleuses générées depuis le début
        self.nb_transactions = 0

        # Nombre de transactions qui ont réussi à passer sans être détectées
        self.nb_reussites = 0

        # Nombre de transactions qui ont été bloquées par un détecteur
        self.nb_echecs = 0

        # Historique des taux de réussite enregistrés à chaque round
        # Utilisé pour tracer la courbe d'apprentissage sur l'interface
        self.historique_reussite = []

        # Indicateur d'activité : True si le fraudeur est encore actif
        self.actif = True

    def choisir_strategie(self):
        """
        Choisit la prochaine stratégie selon le principe Epsilon-Greedy.

        Principe Epsilon-Greedy (coeur du Q-Learning) :
        - EXPLORATION (epsilon = 20%) : choisir une stratégie au hasard
          → Permet de découvrir de nouvelles stratégies potentiellement meilleures
        - EXPLOITATION (1 - epsilon = 80%) : choisir la meilleure stratégie connue
          → Utilise la connaissance accumulée pour maximiser les réussites

        :return: La stratégie choisie (string)
        """

        # Tirage d'un nombre aléatoire entre 0.0 et 1.0
        if random.random() < self.epsilon:
            # EXPLORATION : stratégie aléatoire parmi les 10 disponibles
            self.strategie_actuelle = random.choice(self.STRATEGIES)
        else:
            # EXPLOITATION : stratégie avec le score Q maximum
            # Si plusieurs stratégies ont le même score, max() prend la première
            self.strategie_actuelle = max(self.q_scores, key=self.q_scores.get)

        return self.strategie_actuelle

    def generer_transaction(self, clients_normaux):
        """
        Génère une transaction frauduleuse selon la stratégie active.
        Chaque stratégie produit un profil de transaction spécifique
        conçu pour tromper les règles de détection des détecteurs.

        :param clients_normaux: Liste des profils clients normaux
                                (nécessaire pour la stratégie usurpation)
        :return: Dictionnaire représentant la transaction frauduleuse
        """

        # Sélection de la stratégie pour cette transaction
        strategie = self.choisir_strategie()

        # Liste de villes africaines utilisées dans la simulation
        villes = ["Abidjan", "Dakar", "Lagos", "Accra", "Bamako",
                  "Cotonou", "Lome", "Ouagadougou", "Niamey", "Conakry"]

        # Ville de base du fraudeur (fixée aléatoirement)
        ville_base = random.choice(villes)

        # Heure simulée de la transaction (0 à 23)
        heure = random.randint(0, 23)

        # ---- Génération du profil de transaction selon la stratégie ----------

        if strategie == "montant_explosif":
            # Envoyer un montant très élevé en une seule fois
            # Objectif : dépasser le seuil de montant des détecteurs
            montant   = random.randint(1_000_000, 5_000_000)  # Très gros montant
            frequence = 1           # Une seule transaction pour ne pas éveiller
            ville     = ville_base  # Ville normale pour ne pas cumuler les alertes
            heure     = random.randint(8, 18)  # Heure normale pour paraître légitime

        elif strategie == "fragmentation":
            # Fractionner un gros montant en plusieurs petites transactions
            # Chaque transaction individuelle semble normale
            # Mais la fréquence élevée et le total accumulé sont suspects
            montant   = random.randint(10_000, 50_000)   # Petit montant unitaire
            frequence = random.randint(8, 15)             # Beaucoup de transactions
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "usurpation":
            # Copier exactement le profil d'un client normal pour passer inaperçu
            # C'est la stratégie la plus difficile à détecter car le profil est réel
            if clients_normaux:
                # On copie les caractéristiques d'un vrai client existant
                client_cible = random.choice(clients_normaux)
                montant   = client_cible["montant_moyen"]  # Même montant habituel
                frequence = client_cible["frequence"]      # Même fréquence habituelle
                ville     = client_cible["ville"]          # Même ville d'origine
                heure     = client_cible["heure_habituelle"]  # Même heure habituelle
                # USURPE l'ID du client cible → détectable !
                id_usurpe    = client_cible["client_id"]
            else:
                # Si pas de clients disponibles, profil générique normal
                montant   = random.randint(20_000, 100_000)
                frequence = 2
                ville     = ville_base
                heure     = random.randint(8, 18)

        elif strategie == "rapidite":
            # Envoyer un très grand nombre de transactions en très peu de temps
            # Le montant individuel est normal mais la fréquence est anormalement élevée
            montant   = random.randint(5_000, 30_000)   # Montant normal
            frequence = random.randint(20, 50)           # Fréquence très suspecte
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "localisation":
            # Effectuer des transactions depuis des villes très éloignées
            # en un temps physiquement impossible (ex: Abidjan puis Lagos en 1 minute)
            montant   = random.randint(50_000, 300_000)
            frequence = random.randint(3, 7)
            # Choisir une ville différente de la ville de base du fraudeur
            villes_eloignees = [v for v in villes if v != ville_base]
            ville     = random.choice(villes_eloignees)
            heure     = random.randint(8, 18)

        elif strategie == "compte_dormant":
            # Simuler un compte qui était inactif depuis longtemps
            # et qui soudainement effectue beaucoup de transactions
            # Les détecteurs surveillent ce pattern car il précède souvent une fraude
            montant   = random.randint(100_000, 800_000)  # Montant modéré
            frequence = random.randint(5, 12)              # Fréquence soudainement élevée
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "micro_transactions":
            # Effectuer des milliers de très petits prélèvements
            # Chaque prélèvement est si petit qu'il passe sous les seuils de détection
            # Mais le cumul représente un montant important
            montant   = random.randint(100, 999)    # Très petit montant (sous les radars)
            frequence = random.randint(50, 200)      # Fréquence extrêmement élevée
            ville     = ville_base
            heure     = random.randint(8, 18)

        elif strategie == "horaires_suspects":
            # Effectuer des transactions en pleine nuit (00h - 04h)
            # Peu de gens font des virements à 3h du matin
            # Les détecteurs surveillent les horaires inhabituels
            montant   = random.randint(200_000, 1_000_000)
            frequence = random.randint(2, 6)
            ville     = ville_base
            heure     = random.randint(0, 4)   # Forcer une heure nocturne suspecte

        elif strategie == "round_tripping":
            # Envoyer de l'argent via plusieurs comptes intermédiaires
            # puis le récupérer — technique de blanchiment d'argent
            # Le même montant circule en boucle pour brouiller les pistes
            montant   = random.randint(500_000, 2_000_000)
            frequence = random.randint(4, 8)   # Plusieurs étapes de transfert
            ville     = random.choice(villes)  # Villes variées pour brouiller
            heure     = random.randint(8, 18)

        elif strategie == "mule_account":
            # Utiliser un compte tiers (mule) comme intermédiaire
            # La mule reçoit l'argent frauduleux et le retransmet
            # Rend le fraudeur original difficile à identifier
            montant   = random.randint(150_000, 600_000)
            frequence = random.randint(3, 8)
            ville     = random.choice(villes)  # Ville de la mule (aléatoire)
            heure     = random.randint(8, 18)

        # Incrémente le compteur total de transactions générées
        self.nb_transactions += 1

        # Retourne la transaction sous forme de dictionnaire structuré
        return {
            "id"          : f"F{self.fraudeur_id}_{self.nb_transactions}",
            "fraudeur_id" : self.fraudeur_id,   # Qui a généré cette transaction
            "client_id"   : id_usurpe if strategie == "usurpation" and id_usurpe is not None
                            else -(self.fraudeur_id + 1),  # ID client négatif pour différencier des clients normaux
            "montant"     : montant,             # Montant en FCFA
            "frequence"   : frequence,           # Nombre de transactions associées
            "ville"       : ville,               # Ville d'origine déclarée
            "heure"       : heure,               # Heure de la transaction
            "strategie"   : strategie,           # Stratégie utilisée (connu en interne)
            "est_fraude"  : True                 # Marqueur interne — c'est une fraude
        }

    def apprendre(self, detectee):
        """
        Met à jour le score Q de la stratégie utilisée selon le résultat obtenu.
        C'est le mécanisme central d'apprentissage par renforcement.

        Formule Q-Learning simplifiée :
        Q(strategie) = Q(strategie) + alpha * (reward - Q(strategie))

        Interprétation :
        - reward = +1 : la fraude a réussi → on renforce cette stratégie
        - reward = -1 : la fraude a été détectée → on pénalise cette stratégie

        :param detectee: True si la transaction a été détectée, False sinon
        """

        # Récompense positive si la fraude passe, négative si elle est bloquée
        reward = -1 if detectee else +1

        # Mise à jour du score Q selon la formule d'apprentissage
        # L'ancien score est progressivement corrigé vers la nouvelle réalité
        ancien_score = self.q_scores[self.strategie_actuelle]
        self.q_scores[self.strategie_actuelle] = (
            ancien_score + self.alpha * (reward - ancien_score)
        )

        # Mise à jour des compteurs de statistiques
        if detectee:
            self.nb_echecs += 1     # Transaction bloquée par un détecteur
        else:
            self.nb_reussites += 1  # Transaction passée sans être détectée

    def get_taux_reussite(self):
        """
        Calcule le taux de réussite global du fraudeur.
        Taux = nb_reussites / nb_transactions_total

        :return: Float entre 0.0 et 1.0 (ex: 0.72 signifie 72% de réussite)
        """

        # Protection contre la division par zéro au tout début de la simulation
        if self.nb_transactions == 0:
            return 0.0

        return self.nb_reussites / self.nb_transactions

    def enregistrer_round(self):
        """
        Enregistre le taux de réussite du round actuel dans l'historique.
        Appelé à la fin de chaque round par le superviseur.
        Permet de tracer la courbe d'évolution du fraudeur sur l'interface.
        """

        # Ajoute le taux de réussite actuel à la liste historique
        self.historique_reussite.append(self.get_taux_reussite())