# =============================================================================
# agent_detecteur.py
# Rôle : Agent détecteur qui analyse chaque transaction et décide
#        si elle est suspecte ou non.
#
# Méthode de détection : Score de risque combiné (0 à 100)
#   → Chaque règle ajoute des points au score
#   → Si score >= 30 → transaction bloquée
#   → Si score <  30 → transaction autorisée
#
# Règles calibrées sur les valeurs EXACTES de agent_fraudeur.py :
#   Règle 1  : montant_explosif   → montant >= 1 000 000
#   Règle 2  : rapidite           → frequence >= 20
#   Règle 3  : micro_transactions → montant < 1000 ET freq > 6
#   Règle 4  : horaires_suspects  → heure <= 4
#   Règle 5  : round_tripping     → montant >= 500k ET freq >= 4
#   Règle 6  : fragmentation      → freq >= 8 ET montant <= 50k
#   Règle 7  : compte_dormant     → montant 100k-800k ET freq >= 8
#   Règle 8  : mule_account       → montant > 240k ET freq >= 3
#   Règle 9  : heure tardive      → heure >= 22
#   Règle 10 : localisation +     → ville différente de l'habitude
#              usurpation           du client (mémoire par client_id)
#
# Zone SAFE des clients normaux (agent_client.py) :
#   montant   : 8 000 - 240 000 FCFA
#   frequence : 1 - 6 transactions
#   heure     : 7h - 21h
# =============================================================================

import time
import random
from base_donnees import charger_villes_clients


class AgentDetecteur:
    """
    Agent détecteur autonome qui :
    - Analyse chaque transaction via un score de risque combiné
    - Mémorise les villes habituelles de chaque client (historique)
    - Détecte les changements de comportement suspects
    - Mesure ses propres performances (détections, ratées, faux positifs)
    """

    def __init__(self, detecteur_id):
        """
        Initialise un agent détecteur.

        :param detecteur_id: Identifiant unique du détecteur (ex: 0, 1, 2...)
        """

        # Identifiant unique pour distinguer chaque détecteur dans la simulation
        self.detecteur_id = detecteur_id

        #self.compteur_round = {}  # nb transactions par client_id ce round

        # --- Mémoire des villes par client -----------------------------------
        # Dictionnaire : client_id → ville habituelle
        # Exemple : {0: "Abidjan", 1: "Dakar", 5: "Lagos"}
        #
        # Utilité :
        # → À la première transaction d'un client, on mémorise sa ville
        # → Aux transactions suivantes, si la ville change → suspect
        # → Permet de détecter : localisation (ville impossible)
        #                         usurpation   (fraudeur dans mauvaise ville)
        self.villes_clients = {}

        # --- Statistiques de performance du détecteur ------------------------
        # Nombre total de transactions analysées
        self.nb_analyses = 0

        # Fraudes correctement bloquées (vrai positif)
        self.nb_fraudes_detectees = 0

        # Fraudes manquées — passées sans être détectées (faux négatif)
        self.nb_fraudes_ratees = 0

        # Clients normaux bloqués à tort (faux positif)
        # → Impact négatif sur l'expérience client
        self.nb_faux_positifs = 0

        # Historique du taux d'échec round par round
        # → Utilisé pour tracer la courbe d'évolution sur l'interface
        self.historique_echec = []

        # Mémoire des villes par client_id
        # Chargée depuis la base de données au démarrage
        # → Les simulations précédentes enrichissent cette mémoire
        # → Dès la 1ère transaction, on connaît déjà les clients connus
        self.villes_clients = charger_villes_clients()

    def calculer_score_risque(self, transaction):
        """
        Calcule un score de risque entre 0 et 100 pour une transaction.

        Principe :
        → Chaque règle vérifie un signal de fraude spécifique
        → Si le signal est présent → points ajoutés au score
        → Le score final détermine si la transaction est bloquée

        Calibration :
        → Les seuils de chaque règle sont choisis pour être
          HORS de la zone normale des clients (agent_client.py)
        → Cela évite les faux positifs sur les clients innocents

        :param transaction: Dictionnaire de la transaction à analyser
        :return            : Score de risque entre 0 et 100
        """

        # Extraction des champs de la transaction
        montant   = transaction["montant"]
        frequence = transaction["frequence"]
        heure     = transaction["heure"]
        ville     = transaction["ville"]
        # client_id peut être absent pour les fraudeurs → None par défaut
        client_id = transaction.get("client_id")


        # Score initial à zéro — augmente si des signaux suspects sont détectés
        score = 0

        # ── RÈGLE 0 : ID dupliqué détecté en amont ────────────────────────
        # Flag calculé centralement dans simulation.py AVANT distribution
        # → Fiable car vu sur la totalité des transactions du round
        # → Un ID qui apparaît 2 fois = vrai client + fraudeur simultanément
        # → Principe "impossible travel" : même compte actif 2 endroits
        if transaction.get("id_duplique", False):
            score += 50

        

        # ── RÈGLE 1 : Montant explosif ────────────────────────────────────
        # Cible la stratégie "montant_explosif" de agent_fraudeur.py
        # Fraudeur génère : 1 000 000 - 5 000 000 FCFA
        # Client max      : 240 000 FCFA
        # → Tout montant >= 500 000 est hors zone client normale
        if montant >= 1_000_000:
            score += 60   # Certitude quasi-totale → montant_explosif
        elif montant >= 500_000:
            score += 40   # Très suspect → round_tripping ou horaires

        # ── RÈGLE 2 : Fréquence anormalement élevée ───────────────────────
        # Cible la stratégie "rapidite" de agent_fraudeur.py
        # Fraudeur génère : frequence 20-50
        # Client max      : frequence 6
        # → Tout freq >= 7 est hors zone client normale
        if frequence >= 20:
            score += 60   # Certitude → rapidite
        elif frequence >= 10:
            score += 40   # Très suspect
        elif frequence >= 7:
            score += 20   # Légèrement hors zone client

        # ── RÈGLE 3 : Micro-transactions à fréquence élevée ──────────────
        # Cible la stratégie "micro_transactions" de agent_fraudeur.py
        # Fraudeur génère : montant 100-999 FCFA ET frequence 50-200
        # Client min      : montant 1000 FCFA, freq max 6
        # → Combinaison montant < 1000 ET freq > 6 = impossible pour un client
        if montant < 1_000 and frequence > 6:
            score += 60   # Certitude → micro_transactions

        # ── RÈGLE 4 : Horaires nocturnes suspects ─────────────────────────
        # Cible la stratégie "horaires_suspects" de agent_fraudeur.py
        # Fraudeur génère : heure 0-4 (pleine nuit)
        # Client min      : heure 7
        # → Toute transaction avant 7h est hors zone client normale
        if heure <= 4:
            score += 50   # Certitude → horaires_suspects
        elif heure <= 6:
            score += 25   # Suspect → légèrement hors zone client

        # ── RÈGLE 5 : Gros montant avec fréquence élevée ─────────────────
        # Cible la stratégie "round_tripping" de agent_fraudeur.py
        # Fraudeur génère : montant 500k-2M ET frequence 4-8
        # Client max      : montant 240k
        # → montant >= 500k suffit car hors zone client
        if montant >= 500_000 and frequence >= 4:
            score += 50   # Certitude → round_tripping

        # ── RÈGLE 6 : Petits montants répétés à haute fréquence ──────────
        # Cible la stratégie "fragmentation" de agent_fraudeur.py
        # Fraudeur génère : montant 10k-50k ET frequence 8-15
        # Client max freq : 6
        # → freq >= 8 suffit car hors zone client (max 6)
        if frequence >= 8 and montant <= 50_000:
            score += 40   # Certitude → fragmentation

        # ── RÈGLE 7 : Montant modéré avec fréquence élevée ───────────────
        # Cible la stratégie "compte_dormant" de agent_fraudeur.py
        # Fraudeur génère : montant 100k-800k ET frequence 5-12
        # On exige freq >= 8 pour éviter les faux positifs
        # (un client peut avoir montant 100k-240k ET freq 5-6 normalement)
        if 100_000 <= montant <= 800_000 and frequence >= 8:
            score += 35   # Suspect → compte_dormant

        # ── RÈGLE 8 : Montant intermédiaire hors zone client ─────────────
        # Cible la stratégie "mule_account" de agent_fraudeur.py
        # Fraudeur génère : montant 150k-600k ET frequence 3-8
        # Client max      : montant 240k
        # → On exige montant > 240k (hors zone client) pour éviter faux positifs
        if montant > 240_000 and frequence >= 3:
            score += 25   # Suspect → mule_account

        # ── RÈGLE 9 : Heure tardive en soirée ────────────────────────────
        # Signal faible complémentaire
        # Client max heure : 21h (heure_habituelle max 20h + variation 1h)
        # → heure >= 22 est légèrement hors zone client
        if heure >= 22:
            score += 15   # Signal faible — combiné avec autres règles

        # ── RÈGLE 10 : Changement de ville détecté ────────────────────────
        # Cible les stratégies "localisation" et "usurpation"
        #
        # Principe de la mémoire des villes :
        # → 1ère transaction d'un client → on mémorise sa ville habituelle
        # → Transactions suivantes → si ville change → très suspect
        #
        # Localisation : fraudeur utilise une ville différente de ville_base
        # Usurpation   : fraudeur copie un client mais est dans une autre ville
        #
        # Note : client_id = None pour les fraudeurs
        #        → cette règle ne s'applique qu'aux vrais clients
        #        → et aux fraudeurs qui usurpent un client connu
        if client_id is not None:

            if client_id in self.villes_clients:
                # Client connu → on compare avec sa ville mémorisée
                ville_habituelle = self.villes_clients[client_id]

                if ville != ville_habituelle:
                    # Ville différente de l'habitude → très suspect
                    # Peut indiquer : usurpation, localisation impossible
                    score += 50

            else:
                # Première transaction de ce client
                # → On mémorise sa ville pour les prochaines comparaisons
                # → Pas de score ajouté (on ne peut pas encore comparer)
                self.villes_clients[client_id] = ville

        # Plafonnement du score à 100 (même si cumul dépasse)
        return min(score, 100)

    def analyser(self, transaction):
        """
        Analyse une transaction et retourne la décision de blocage.

        Seuil de décision : score >= 30
        → Calibré pour être au-dessus du score maximum
          d'un client normal (qui devrait scorer 0 ou très peu)
        → Minimise les faux positifs tout en maximisant la détection

        Simulation du temps de traitement bancaire réaliste :
        → time.sleep(0.02) = 20ms de traitement
        → Justifie l'intérêt du parallélisme (plusieurs détecteurs simultanés)

        :param transaction: Dictionnaire de la transaction à analyser
        :return            : True si suspecte (bloquée), False sinon
        """

        # Simulation du délai de traitement bancaire réel (~20ms)
        #time.sleep(0.02)

        # Calcul CPU réel au lieu de sleep
        # Simule le traitement cryptographique d'une transaction bancaire
        # (hachage, vérification de signature, lookup base de données)
        import math
        _ = sum(math.sqrt(i) for i in range(1_000_000))

        # Incrément du compteur d'analyses
        self.nb_analyses += 1

        # Calcul du score de risque via les 10 règles calibrées
        score = self.calculer_score_risque(transaction)

        # Décision finale : bloquer si score >= 30
        return score >= 30

    def enregistrer_resultat(self, transaction, decision):
        """
        Enregistre le résultat d'une analyse pour les statistiques.
        Compare la décision du détecteur avec la réalité (est_fraude).

        4 cas possibles :
        → Fraude détectée    : est_fraude=True  ET decision=True  →  correct
        → Fraude ratée       : est_fraude=True  ET decision=False →  échec
        → Faux positif       : est_fraude=False ET decision=True  →  erreur
        → Vrai négatif       : est_fraude=False ET decision=False →  correct

        :param transaction: La transaction analysée
        :param decision   : True si le détecteur l'a jugée suspecte
        """

        est_fraude = transaction["est_fraude"]

        if est_fraude and decision:
            # Fraude correctement détectée → bonne performance
            self.nb_fraudes_detectees += 1

        elif est_fraude and not decision:
            # Fraude manquée → le fraudeur a réussi à passer
            self.nb_fraudes_ratees += 1

        elif not est_fraude and decision:
            # Client normal bloqué à tort → mauvaise expérience client
            self.nb_faux_positifs += 1

    def adapter_seuils(self):
        """
        Méthode de compatibilité avec l'ancien système de seuils adaptatifs.
        Les règles calibrées sont fixes → pas d'adaptation nécessaire.
        Conservée pour éviter les erreurs d'appel depuis agent_superviseur.py.
        """
        pass

    def get_taux_echec(self):
        """
        Calcule le taux de fraudes manquées (échec du détecteur).
        Taux = nb_fraudes_ratees / total_fraudes_vues

        :return: Float entre 0.0 et 1.0 (ex: 0.3 = 30% de fraudes ratées)
        """

        total = self.nb_fraudes_detectees + self.nb_fraudes_ratees

        # Protection contre division par zéro au début de la simulation
        if total == 0:
            return 0.0

        return self.nb_fraudes_ratees / total

    def enregistrer_round(self):
        """
        Sauvegarde le taux d'échec du round actuel dans l'historique.
        Appelée à la fin de chaque round par agent_superviseur.py.
        Permet de tracer la courbe d'évolution sur l'interface Streamlit.
        """
        self.historique_echec.append(self.get_taux_echec())
        #self.compteur_round = {}