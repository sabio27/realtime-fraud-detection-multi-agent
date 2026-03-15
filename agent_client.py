# =============================================================================
# agent_client.py
# Rôle : Définit le comportement d'un agent client normal dans la simulation.
#        Les clients normaux génèrent des transactions légitimes qui servent
#        de "bruit de fond" réaliste pour compliquer la tâche des détecteurs.
#        Un bon système de détection doit distinguer ces transactions normales
#        des transactions frauduleuses — sans bloquer les clients innocents.
# =============================================================================

import random  # Pour générer des variations naturelles dans les transactions

class AgentClient:
    """
    Un agent client normal qui :
    - Génère des transactions légitimes avec un profil cohérent et stable
    - Représente un vrai comportement bancaire (montants, fréquences, horaires)
    - Sert de référence "normale" pour calibrer les détecteurs
    - Permet de mesurer les faux positifs (clients normaux bloqués à tort)
    """

    def __init__(self, client_id):
        """
        Initialise un agent client avec un profil bancaire réaliste et stable.
        Chaque client a ses propres habitudes (montant, ville, heure)
        qui restent cohérentes tout au long de la simulation.

        :param client_id: Identifiant unique du client (ex: 0, 1, 2...)
        """

        # Identifiant unique pour distinguer chaque client dans la simulation
        self.client_id = client_id


        # Graine fixe par client_id
        # → Même client_id = mêmes caractéristiques à chaque simulation
        # → Permet à la base de données d'être utile
        rng = random.Random(client_id * 42)

        # --- Profil bancaire du client ----------------------------------------
        # Ces valeurs sont fixes par client et définissent son comportement normal
        # Elles sont utilisées par les fraudeurs pour la stratégie d'usurpation

        # Liste des villes disponibles dans la simulation
        villes = ["Abidjan", "Dakar", "Lagos", "Accra", "Bamako",
                  "Cotonou", "Lome", "Ouagadougou", "Niamey", "Conakry"]

        # Ville d'origine du client — fixe tout au long de la simulation
        # Un client normal effectue ses transactions depuis la même ville
        self.ville = rng.choice(villes)
        # Montant moyen habituel du client en FCFA
        # Chaque client a sa propre capacité financière (entre 10 000 et 200 000 FCFA)
        # Ce montant reste stable pour représenter un comportement prévisible
        self.montant_moyen = rng.randint(10_000, 200_000)

        # Fréquence habituelle de transactions par round
        # Un client normal effectue entre 1 et 5 transactions par période
        # Au-delà, c'est considéré comme suspect
        self.frequence = rng.randint(1, 5)

        # Heure habituelle des transactions du client
        # Les clients normaux transactent généralement pendant les heures ouvrables
        # Entre 8h et 20h — cohérent avec la vie quotidienne
        self.heure_habituelle = rng.randint(8, 20)

        # --- Statistiques individuelles du client ----------------------------
        # Nombre total de transactions générées par ce client
        self.nb_transactions = 0

        # Nombre de fois où ce client a été bloqué à tort (faux positif)
        # Un taux élevé signifie que les détecteurs sont trop stricts
        self.nb_bloque = 0

    def generer_transaction(self):
        """
        Génère une transaction normale et légitime.
        La transaction respecte le profil habituel du client avec
        de légères variations naturelles pour simuler la réalité.
        Dans la vraie vie, même un client régulier ne fait pas exactement
        le même montant à la même heure tous les jours.

        :return: Dictionnaire représentant la transaction légitime
        """

        # Variation naturelle du montant autour de la moyenne habituelle
        # randint(-20%, +20%) simule les petites variations du quotidien
        # Ex: si montant_moyen = 50 000, la transaction sera entre 40 000 et 60 000
        variation = random.randint(
            -int(self.montant_moyen * 0.2),   # -20% du montant moyen
             int(self.montant_moyen * 0.2)    # +20% du montant moyen
        )
        montant = max(1000, self.montant_moyen + variation)
        # max(1000, ...) garantit que le montant ne descend jamais sous 1000 FCFA

        # Légère variation de l'heure habituelle (±1 heure)
        # Un client peut transacter un peu plus tôt ou plus tard que d'habitude
        heure = max(0, min(23, self.heure_habituelle + random.randint(-1, 1)))
        # max(0, min(23, ...)) garantit que l'heure reste entre 0 et 23

        # Légère variation de la fréquence (±1 transaction)
        # Un client peut faire une transaction de plus ou de moins que d'habitude
        frequence = max(1, self.frequence + random.randint(-1, 1))
        # max(1, ...) garantit au minimum 1 transaction par round

        # Incrémente le compteur total de transactions de ce client
        self.nb_transactions += 1

        # Retourne la transaction sous forme de dictionnaire structuré
        return {
            "id"         : f"C{self.client_id}_{self.nb_transactions}",
            # Identifiant unique : C = Client, suivi de l'id et du numéro de transaction

            "client_id"  : self.client_id,   # Qui a généré cette transaction
            "montant"    : montant,           # Montant en FCFA avec variation naturelle
            "frequence"  : frequence,         # Fréquence avec variation naturelle
            "ville"      : self.ville,        # Ville fixe du client
            "heure"      : heure,             # Heure avec légère variation
            "strategie"  : "normal",          # Pas de stratégie — comportement normal
            "est_fraude" : False              # Marqueur interne — ce n'est PAS une fraude
        }

    def get_profil(self):
        """
        Retourne le profil complet du client sous forme de dictionnaire.
        Utilisé par les agents fraudeurs pour la stratégie d'usurpation —
        ils copient ce profil pour se faire passer pour un client normal.

        :return: Dictionnaire contenant les caractéristiques habituelles du client
        """

        return {
            "client_id"       : self.client_id,
            "ville"           : self.ville,
            "montant_moyen"   : self.montant_moyen,
            "frequence"       : self.frequence,
            "heure_habituelle": self.heure_habituelle
        }

    def enregistrer_blocage(self):
        """
        Enregistre qu'une transaction de ce client a été bloquée à tort.
        Appelée par le superviseur quand un faux positif est détecté.
        Permet de mesurer l'impact négatif d'un détecteur trop strict
        sur les clients innocents.
        """

        # Incrémente le compteur de blocages injustifiés
        self.nb_bloque += 1

    def get_taux_blocage(self):
        """
        Calcule le taux de blocage injustifié de ce client.
        Taux = nb fois bloqué / nb total de transactions

        Un taux élevé signifie que ce client est souvent pris à tort
        pour un fraudeur — mauvaise expérience utilisateur.

        :return: Float entre 0.0 et 1.0 (ex: 0.1 = bloqué 10% du temps)
        """

        # Protection contre la division par zéro au début de la simulation
        if self.nb_transactions == 0:
            return 0.0

        return self.nb_bloque / self.nb_transactions