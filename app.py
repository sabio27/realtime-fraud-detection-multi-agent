# =============================================================================
# app.py
# Rôle : Interface visuelle de la simulation avec Streamlit.
#        Affiche en temps réel les transactions, l'état des agents,
#        les courbes d'apprentissage et le speedup parallèle.
# =============================================================================

import streamlit as st        # Framework web pour l'interface visuelle
import matplotlib.pyplot as plt  # Pour tracer les graphiques et courbes
import pandas as pd           # Pour afficher les tableaux de données
from simulation import Simulation  # Moteur de simulation multi-agents


# Protection obligatoire sur Windows pour multiprocessing
# Sans cette ligne, chaque processus fils recrée l'app Streamlit
# → récursion infinie → crash
if __name__ == "__main__":
    pass  # Le code de l'interface Streamlit est en dehors de ce bloc pour s'exécuter normalement

# --- Configuration de la page Streamlit --------------------------------------
# layout="wide" utilise toute la largeur de l'écran
st.set_page_config(page_title="Simulation Fraude Multi-Agents", layout="wide")

st.title("Simulation de Detection de Fraude Multi-Agents Parallele")
st.markdown("---")

# =============================================================================
# SIDEBAR — Panneau de controle
# Permet à l'utilisateur de configurer la simulation avant de la lancer
# =============================================================================
# --- Panneau de controle (barre laterale gauche) ------------------------------
st.sidebar.header("Parametres de simulation")

# Calcul du nombre de coeurs physiques disponibles
# cpu_count() // 2 = coeurs physiques (hyperthreading exclu)
import multiprocessing
coeurs_physiques = multiprocessing.cpu_count() // 2

# Affichage informatif des ressources disponibles
st.sidebar.info(
    f"Coeurs physiques disponibles : {coeurs_physiques}\n\n"
    f"Maximum recommande pour les detecteurs : {coeurs_physiques}"
)

# Slider clients normaux
nb_clients = st.sidebar.slider(
    "Nombre de clients normaux",
    min_value=5, max_value=1000, value=20
)

# Slider fraudeurs
nb_fraudeurs = st.sidebar.slider(
    "Nombre de fraudeurs",
    min_value=1, max_value=500, value=5
)

# Slider detecteurs — limité aux coeurs physiques
nb_detecteurs = st.sidebar.slider(
    "Nombre de detecteurs (processus paralleles)",
    min_value=1, max_value=coeurs_physiques, value=min(3, coeurs_physiques)
)

# Slider rounds
nb_rounds = st.sidebar.slider(
    "Nombre de rounds",
    min_value=5, max_value=30, value=15
)

# --- Validations et avertissements -------------------------------------------

# Calcul de la charge par detecteur
charge_par_detecteur = (nb_clients + nb_fraudeurs) / nb_detecteurs

# Calcul du ratio clients/fraudeurs
ratio = nb_clients / nb_fraudeurs if nb_fraudeurs > 0 else 0

# Verification 1 : charge minimale par detecteur
if charge_par_detecteur < 5:
    st.sidebar.warning(
        f"Charge trop faible par detecteur "
        f"({charge_par_detecteur:.1f} transactions). "
        f"Augmente les clients/fraudeurs ou reduis les detecteurs."
    )

# Verification 2 : ratio clients/fraudeurs
if ratio < 2:
    st.sidebar.warning(
        f"Ratio clients/fraudeurs trop faible ({ratio:.1f}). "
        f"Dans la realite bancaire, ce ratio est de 3 a 5 minimum."
    )

# Verification 3 : nombre de rounds suffisant
if nb_rounds < 10:
    st.sidebar.warning(
        "Moins de 10 rounds : l'apprentissage des agents "
        "ne sera pas bien visible sur les courbes."
    )

# Affichage du resume des parametres valides
st.sidebar.markdown("---")
st.sidebar.markdown("**Resume de la simulation :**")
st.sidebar.markdown(
    f"- Transactions par round : **{nb_clients + nb_fraudeurs}**\n"
    f"- Charge par detecteur : **{charge_par_detecteur:.1f}** transactions\n"
    f"- Processus paralleles : **{nb_detecteurs}**\n"
    f"- Memoire estimee : **~{nb_detecteurs * 50}MB**"
)

# Bouton de lancement
lancer = st.sidebar.button("Lancer la simulation")

# =============================================================================
# ZONE PRINCIPALE — Affichage des résultats
# S'affiche uniquement après avoir cliqué sur "Lancer la simulation"
# =============================================================================
if lancer:

    # --- Création de la simulation avec les paramètres choisis ---------------
    sim = Simulation(
        nb_clients    = nb_clients,
        nb_fraudeurs  = nb_fraudeurs,
        nb_detecteurs = nb_detecteurs,
        nb_rounds     = nb_rounds
    )

    # --- Barre de progression et stockage des métriques par round ------------
    st.subheader("Simulation en cours...")

    # Barre de progression visuelle (0% → 100% au fil des rounds)
    barre = st.progress(0)

    # Texte dynamique indiquant le round en cours
    statut = st.empty()

    # Liste pour stocker les métriques de chaque round
    # Remplie par le callback appelé à chaque round
    historique_rounds = []

    historique_transactions  = []  # Stocke les transactions de chaque round

    def callback_round(metriques):
        """
        Fonction appelée automatiquement à chaque round par simulation.py.
        Met à jour la barre de progression et stocke les métriques.

        :param metriques: Dictionnaire des métriques du round terminé
        """

        # Ajoute les métriques du round à l'historique local
        historique_rounds.append(metriques)

        # Calcule la progression en pourcentage (0.0 à 1.0)
        #progression = len(historique_rounds) / nb_rounds

        # Met à jour la barre de progression
        #barre.progress(progression)

        # Met à jour le texte de statut
        # Stocke les transactions du round pour l'animation D3.js

        transactions_round = [
            {
                "client_id" : t.get("client_id"),
                "montant"   : t["montant"],
                "ville"     : t["ville"],
                "heure"     : t["heure"],
                "frequence" : t["frequence"],
                "est_fraude": t["est_fraude"],
                "strategie" : t.get("strategie", "normal"),
                "decision"  : d,
                "detecteur_assign": t.get("detecteur_assign", 0)
            }
            for t, d in sim.resultats
        ]
        historique_transactions.append(transactions_round)

        progression = len(historique_rounds) / nb_rounds
        barre.progress(progression)
        statut.text(
            f"Round {len(historique_rounds)}/{nb_rounds} — "
            f"Taux de detection : "
            f"{metriques['taux_detection']:.1%}"
        )


    # --- Lancement de la simulation complète ---------------------------------
    # La simulation appelle callback_round() à chaque round terminé
    bilan = sim.lancer(callback=callback_round)

    # Message de fin une fois tous les rounds terminés
    statut.text("Simulation terminee !")
    st.success("Simulation terminee avec succes !")
    st.markdown("---")

    # =========================================================================
    # SECTION 1 — Metriques globales (chiffres clés en haut de page)
    # =========================================================================
    st.subheader("Bilan global de la simulation")

    # Affichage en 5 colonnes pour les métriques principales
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Transactions totales",
        bilan["total_transactions"]
    )
    c2.metric(
        "Fraudes detectees",
        bilan["total_fraudes_detectees"]
    )
    c3.metric(
        "Fraudes ratees",
        bilan["total_fraudes_ratees"]
    )
    c4.metric(
        "Faux positifs",
        bilan["total_faux_positifs"]
    )
    c5.metric(
        "Taux detection global",
        f"{bilan['taux_detection_global']:.1%}"
    )

    st.markdown("---")

    # =========================================================================
    # ONGLETS — Organisation visuelle
    # =========================================================================
    import time

    tab1, tab2, tab3, tab4 = st.tabs([
        " Flux en temps réel",
        " Etat des agents",
        " Courbes et graphiques",
        " Strategies des fraudeurs"
    ])

    # ── ONGLET 1 : Flux en temps réel ────────────────────────────────────
    with tab1:
        st.subheader("Animation — Flux de transactions")
        import json

        donnees_animation = json.dumps({
            "rounds"       : historique_transactions,
            "nb_detecteurs": nb_detecteurs,
            "nb_clients"   : nb_clients,
            "nb_fraudeurs" : nb_fraudeurs
        })

        html_animation = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{
                    background: #0e1117;
                    color: white;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    padding: 10px;
                }}
                #stats {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 10px;
                    font-size: 13px;
                    flex-wrap: wrap;
                    align-items: center;
                }}
                .stat-label {{ color: #888; }}
                .stat-val   {{ font-weight: bold; font-size: 15px; }}
                .ok         {{ color: #00cc66; }}
                .bloque     {{ color: #ff4444; }}
                .rate       {{ color: #ffaa00; }}
                #controls {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 10px;
                    flex-wrap: wrap;
                }}
                button {{
                    padding: 7px 18px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 13px;
                    font-weight: bold;
                    transition: opacity 0.2s;
                }}
                button:hover   {{ opacity: 0.85; }}
                #btnPlay       {{ background: #1f77b4; color: white; }}
                #btnPause      {{ background: #555;    color: white; }}
                #btnReplay     {{ background: #e67e22; color: white; }}
                select {{
                    background: #1e2130;
                    color: white;
                    border: 1px solid #555;
                    padding: 6px 10px;
                    border-radius: 6px;
                    font-size: 13px;
                }}
                #info {{ font-size: 12px; color: #aaa; margin-left: auto; }}
                svg   {{ width: 100%; display: block; }}
                #legende {{
                    display: flex;
                    gap: 20px;
                    margin-top: 8px;
                    font-size: 11px;
                    color: #aaa;
                }}
                .leg-dot {{
                    display: inline-block;
                    width: 10px; height: 10px;
                    border-radius: 50%;
                    margin-right: 4px;
                    vertical-align: middle;
                }}
            </style>
        </head>
        <body>

        <div id="stats">
            <span>
                <span class="stat-label">Round : </span>
                <span class="stat-val" id="s-round">0</span>
                <span class="stat-label"> / {nb_rounds}</span>
            </span>
            <span>
                <span class="stat-label">Autorises : </span>
                <span class="stat-val ok" id="s-ok">0</span>
            </span>
            <span>
                <span class="stat-label">Bloques : </span>
                <span class="stat-val bloque" id="s-bloque">0</span>
            </span>
            <span>
                <span class="stat-label">Rates : </span>
                <span class="stat-val rate" id="s-rate">0</span>
            </span>
            <span id="info">Cliquez Play pour demarrer</span>
        </div>

        <div id="controls">
            <button id="btnPlay">&#9654; Play</button>
            <button id="btnPause">&#9646;&#9646; Pause</button>
            <button id="btnReplay">&#8635; Rejouer</button>
            <label style="color:#aaa; font-size:13px;">Vitesse :
                <select id="vitesse">
                    <option value="1200">Lent</option>
                    <option value="600" selected>Normal</option>
                    <option value="250">Rapide</option>
                    <option value="80">Tres rapide</option>
                </select>
            </label>
        </div>

        <svg id="canvas" height="500"></svg>

        <div id="legende">
            <span><span class="leg-dot" style="background:#aaaaaa"></span>Agent (inconnu)</span>
            <span><span class="leg-dot" style="background:#00cc66"></span>Transaction autorisee</span>
            <span><span class="leg-dot" style="background:#ff4444"></span>Transaction bloquee</span>
            <span><span class="leg-dot" style="background:#ffaa00"></span>Fraude ratee</span>
            <span><span class="leg-dot" style="background:#4488ff"></span>Detecteur</span>
            <span><span class="leg-dot" style="background:#ff9900"></span>Superviseur</span>
        </div>

        <script>
        const DATA   = {donnees_animation};
        const ROUNDS = DATA.rounds;
        const NB_DET = DATA.nb_detecteurs;
        const NB_AG  = Math.min(DATA.nb_clients + DATA.nb_fraudeurs, 20);

        let roundIdx    = 0;
        let playing     = false;
        let totalOk     = 0;
        let totalBloque = 0;
        let totalRate   = 0;

        // Abreviations strategies
        const STRAT = {{
            "montant_explosif"  : "EXPLO",
            "fragmentation"     : "FRAG",
            "usurpation"        : "USURP",
            "rapidite"          : "RAPID",
            "localisation"      : "LOCAL",
            "compte_dormant"    : "DORM",
            "micro_transactions": "MICRO",
            "horaires_suspects" : "NUIT",
            "round_tripping"    : "ROUND",
            "mule_account"      : "MULE"
        }};

        // Dimensions
        const W  = window.innerWidth || 960;
        const H  = 500;
        const svg = d3.select("#canvas");
        svg.attr("viewBox", `0 0 ${{W}} ${{H}}`);

        // Positions des 4 colonnes
        const xAgents = 60;
        const xDet    = W * 0.38;
        const xSup    = W * 0.62;
        const xRes    = W - 80;
        const ySup    = H / 2;

        // Fond des colonnes
        const colonnes = [
            {{ x: 0,           w: W*0.18,        couleur: "#0d1f0d", titre: "AGENTS",      tc: "#00aa55" }},
            {{ x: W*0.18+5,    w: W*0.22-5,      couleur: "#0d1428", titre: "DETECTEURS",  tc: "#4488ff" }},
            {{ x: W*0.40+5,    w: W*0.22-5,      couleur: "#1a1000", titre: "SUPERVISEUR", tc: "#ff9900" }},
            {{ x: W*0.62+5,    w: W*0.38-5,      couleur: "#0d0d1f", titre: "RESULTATS",   tc: "#aaaaff" }}
        ];

        colonnes.forEach(c => {{
            svg.append("rect")
                .attr("x", c.x).attr("y", 28)
                .attr("width", c.w).attr("height", H - 38)
                .attr("fill", c.couleur).attr("rx", 8);

            // Ligne decorative sous titre
            svg.append("line")
                .attr("x1", c.x + 10).attr("y1", 52)
                .attr("x2", c.x + c.w - 10).attr("y2", 52)
                .attr("stroke", c.tc).attr("stroke-width", 1.5)
                .attr("opacity", 0.5);

            svg.append("text")
                .attr("x", c.x + c.w / 2).attr("y", 46)
                .attr("text-anchor", "middle")
                .attr("fill", c.tc)
                .attr("font-size", 12)
                .attr("font-weight", "bold")
                .attr("letter-spacing", 2)
                .text(c.titre);
        }});

        // Noeuds agents fixes (max 20 affiches)
        const agentY = (i) => {{
            const esp = Math.min(22, (H - 100) / NB_AG);
            const top = 65;
            return top + i * esp;
        }};

        const gAgents = svg.append("g");
        for (let i = 0; i < NB_AG; i++) {{
            const y = agentY(i);

            // Noeud agent (gris neutre — identite inconnue)
            gAgents.append("circle")
                .attr("cx", xAgents).attr("cy", y)
                .attr("r", 6)
                .attr("fill", "#888888")
                .attr("stroke", "#aaaaaa")
                .attr("stroke-width", 1);

            // Label agent
            gAgents.append("text")
                .attr("x", xAgents + 10).attr("y", y + 4)
                .attr("fill", "#666").attr("font-size", 8)
                .text("A" + (i + 1));
        }}

        // Indication si plus d'agents
        if (DATA.nb_clients + DATA.nb_fraudeurs > 20) {{
            svg.append("text")
                .attr("x", xAgents).attr("y", H - 40)
                .attr("text-anchor", "middle")
                .attr("fill", "#555").attr("font-size", 9)
                .text("+" + (DATA.nb_clients + DATA.nb_fraudeurs - 20) + " agents");
        }}

        // Detecteurs fixes
        const yDetect = (i) => {{
            const esp = Math.min(80, (H - 100) / NB_DET);
            const top = (H - esp * (NB_DET - 1)) / 2;
            return top + i * esp;
        }};

        const gDet = svg.append("g");
        for (let i = 0; i < NB_DET; i++) {{
            const y = yDetect(i);

            // Halo
            gDet.append("circle")
                .attr("id", `halo${{i}}`)
                .attr("cx", xDet).attr("cy", y).attr("r", 28)
                .attr("fill", "none")
                .attr("stroke", "#4488ff")
                .attr("stroke-width", 1.5)
                .attr("opacity", 0.25);

            // Cercle principal
            gDet.append("circle")
                .attr("id", `det${{i}}`)
                .attr("cx", xDet).attr("cy", y).attr("r", 22)
                .attr("fill", "#1a3a8a")
                .attr("stroke", "#4488ff")
                .attr("stroke-width", 2);

            // Label
            gDet.append("text")
                .attr("x", xDet).attr("y", y + 5)
                .attr("text-anchor", "middle")
                .attr("fill", "white")
                .attr("font-size", 12)
                .attr("font-weight", "bold")
                .text("D" + (i + 1));
        }}

        // Superviseur fixe (noeud central)
        const gSup = svg.append("g");

        // Halo superviseur
        gSup.append("circle")
            .attr("id", "halo-sup")
            .attr("cx", xSup).attr("cy", ySup).attr("r", 38)
            .attr("fill", "none")
            .attr("stroke", "#ff9900")
            .attr("stroke-width", 1.5)
            .attr("opacity", 0.2);

        // Cercle superviseur
        gSup.append("circle")
            .attr("id", "sup-cercle")
            .attr("cx", xSup).attr("cy", ySup).attr("r", 28)
            .attr("fill", "#3a2000")
            .attr("stroke", "#ff9900")
            .attr("stroke-width", 2.5);

        gSup.append("text")
            .attr("x", xSup).attr("y", ySup - 4)
            .attr("text-anchor", "middle")
            .attr("fill", "#ff9900")
            .attr("font-size", 10)
            .attr("font-weight", "bold")
            .text("SUP");

        gSup.append("text")
            .attr("x", xSup).attr("y", ySup + 10)
            .attr("text-anchor", "middle")
            .attr("fill", "#ff9900")
            .attr("font-size", 8)
            .text("VISEUR");

        // Compteur decisions sur le superviseur
        gSup.append("text")
            .attr("id", "sup-count")
            .attr("x", xSup).attr("y", ySup + 48)
            .attr("text-anchor", "middle")
            .attr("fill", "#888").attr("font-size", 9)
            .text("0 decisions");

        // Couche transactions (par-dessus tout)
        const gTx = svg.append("g");

        // Liste resultats (colonne droite)
        let resultatsAffich = [];
        const maxResultats  = 12;

        function afficherResultat(texte, couleur) {{
            resultatsAffich.unshift({{ texte, couleur }});
            if (resultatsAffich.length > maxResultats) {{
                resultatsAffich.pop();
            }}

            // Efface et redessine la liste
            svg.selectAll(".res-item").remove();
            resultatsAffich.forEach((r, i) => {{
                svg.append("text")
                    .attr("class", "res-item")
                    .attr("x", xRes - 10)
                    .attr("y", 75 + i * 18)
                    .attr("text-anchor", "end")
                    .attr("fill", r.couleur)
                    .attr("font-size", 10)
                    .attr("opacity", 1 - i * 0.07)
                    .text(r.texte);
            }});
        }}

        // Pulse detecteur
        function pulseDet(i, couleur) {{
            d3.select(`#det${{i}}`)
                .transition().duration(120)
                .attr("r", 28).attr("fill", couleur)
                .transition().duration(280)
                .attr("r", 22).attr("fill", "#1a3a8a");

            d3.select(`#halo${{i}}`)
                .transition().duration(120)
                .attr("r", 40).attr("opacity", 0.6)
                .attr("stroke", couleur)
                .transition().duration(350)
                .attr("r", 28).attr("opacity", 0.25)
                .attr("stroke", "#4488ff");
        }}

        // Pulse superviseur
        let supCount = 0;
        function pulseSup(couleur) {{
            supCount++;
            document.getElementById("sup-count").textContent =
                supCount + " decisions";

            d3.select("#sup-cercle")
                .transition().duration(120)
                .attr("r", 36).attr("fill", couleur).attr("opacity", 0.9)
                .transition().duration(300)
                .attr("r", 28).attr("fill", "#3a2000").attr("opacity", 1);

            d3.select("#halo-sup")
                .transition().duration(120)
                .attr("r", 50).attr("opacity", 0.5)
                .attr("stroke", couleur)
                .transition().duration(400)
                .attr("r", 38).attr("opacity", 0.2)
                .attr("stroke", "#ff9900");
        }}

        // Anime une transaction complete
        function animerTx(tx, detIdx, done) {{
            const estFraude = tx.est_fraude === true || tx.est_fraude === "True";
            const decision  = tx.decision  === true || tx.decision  === "True";
            const vitesse   = parseInt(document.getElementById("vitesse").value);

            // Agent source (noeud aleatoire parmi les noeuds affiches)
            const agIdx  = Math.floor(Math.random() * NB_AG);
            const yAgent = agentY(agIdx);
            const yD     = yDetect(detIdx);

            // Couleur neutre au depart
            const coulNeutre = "#aaaaaa";

            // Couleur finale selon decision
            let coulFin, symbole, resTexte;
            if (decision && estFraude) {{
                coulFin  = "#ff4444";
                symbole  = "BLOQUE";
                resTexte = "BLOQUE — " + Math.round(tx.montant/1000) + "k FCFA";
                totalBloque++;
                document.getElementById("s-bloque").textContent = totalBloque;
            }} else if (!decision && estFraude) {{
                coulFin  = "#ffaa00";
                symbole  = "RATE";
                resTexte = "RATE — " + Math.round(tx.montant/1000) + "k FCFA";
                totalRate++;
                document.getElementById("s-rate").textContent = totalRate;
            }} else if (decision && !estFraude) {{
                coulFin  = "#ff8800";
                symbole  = "FP";
                resTexte = "FAUX POSITIF";
            }} else {{
                coulFin  = "#00cc66";
                symbole  = "OK";
                resTexte = "OK — " + Math.round(tx.montant/1000) + "k FCFA";
                totalOk++;
                document.getElementById("s-ok").textContent = totalOk;
            }}

            // PHASE 1 : Agent → Detecteur (boule neutre)
            const ligne1 = gTx.append("line")
                .attr("x1", xAgents).attr("y1", yAgent)
                .attr("x2", xAgents).attr("y2", yAgent)
                .attr("stroke", coulNeutre)
                .attr("stroke-width", 1)
                .attr("stroke-dasharray", "3,3")
                .attr("opacity", 0.35);

            const boule = gTx.append("circle")
                .attr("cx", xAgents).attr("cy", yAgent)
                .attr("r", 8)
                .attr("fill", coulNeutre)
                .attr("stroke", "white")
                .attr("stroke-width", 0.5)
                .attr("opacity", 0.9);

            const labelMontant = gTx.append("text")
                .attr("x", xAgents).attr("y", yAgent - 12)
                .attr("text-anchor", "middle")
                .attr("fill", "#aaa").attr("font-size", 8)
                .text(Math.round(tx.montant/1000) + "k");

            // Badge strategie sur la boule (fraudeur uniquement)
            const abbr = (estFraude && tx.strategie && STRAT[tx.strategie])
                ? STRAT[tx.strategie] : "";

            const badge = abbr ? gTx.append("text")
                .attr("x", xAgents).attr("y", yAgent + 20)
                .attr("text-anchor", "middle")
                .attr("fill", "#ffaa00")
                .attr("font-size", 8)
                .attr("font-weight", "bold")
                .text(abbr) : null;

            // Animation Phase 1 : vers detecteur
            ligne1.transition().duration(vitesse).ease(d3.easeLinear)
                .attr("x2", xDet).attr("y2", yD);

            boule.transition().duration(vitesse).ease(d3.easeLinear)
                .attr("cx", xDet).attr("cy", yD)
                .on("end", function() {{

                    // Detecteur analyse → pulse + changement couleur
                    boule.attr("fill", coulFin);
                    pulseDet(detIdx, coulFin);

                    // Affiche decision sur detecteur
                    svg.select("#res-d" + detIdx).remove();
                    gDet.append("text")
                        .attr("id", "res-d" + detIdx)
                        .attr("x", xDet + 32).attr("y", yD + 4)
                        .attr("fill", coulFin)
                        .attr("font-size", 11)
                        .attr("font-weight", "bold")
                        .text(symbole);

                    // PHASE 2 : Detecteur → Superviseur
                    setTimeout(() => {{
                        const ligne2 = gTx.append("line")
                            .attr("x1", xDet).attr("y1", yD)
                            .attr("x2", xDet).attr("y2", yD)
                            .attr("stroke", coulFin)
                            .attr("stroke-width", 1)
                            .attr("stroke-dasharray", "3,3")
                            .attr("opacity", 0.4);

                        ligne2.transition().duration(vitesse * 0.6)
                            .ease(d3.easeLinear)
                            .attr("x2", xSup).attr("y2", ySup);

                        boule.transition().duration(vitesse * 0.6)
                            .ease(d3.easeLinear)
                            .attr("cx", xSup).attr("cy", ySup)
                            .on("end", function() {{

                                // Superviseur pulse
                                pulseSup(coulFin);

                                // PHASE 3 : Superviseur → Resultats
                                setTimeout(() => {{
                                    const ligne3 = gTx.append("line")
                                        .attr("x1", xSup).attr("y1", ySup)
                                        .attr("x2", xSup).attr("y2", ySup)
                                        .attr("stroke", coulFin)
                                        .attr("stroke-width", 1)
                                        .attr("stroke-dasharray", "3,3")
                                        .attr("opacity", 0.4);

                                    ligne3.transition().duration(vitesse * 0.4)
                                        .ease(d3.easeLinear)
                                        .attr("x2", xRes).attr("y2", ySup);

                                    boule.transition().duration(vitesse * 0.4)
                                        .ease(d3.easeLinear)
                                        .attr("cx", xRes).attr("cy", ySup)
                                        .attr("opacity", 0)
                                        .on("end", function() {{
                                            // Affiche dans la liste resultats
                                            afficherResultat(resTexte, coulFin);

                                            // Nettoyage
                                            boule.remove();
                                            labelMontant.remove();
                                            ligne1.remove();
                                            ligne2.remove();
                                            ligne3.remove();
                                            if (badge) badge.remove();
                                            done();
                                        }});
                                }}, 150);
                            }});
                    }}, 200);
                }});

            labelMontant.transition().duration(vitesse).ease(d3.easeLinear)
                .attr("x", xDet).attr("y", yD - 12);
            
            if (badge) {{
                badge.transition().duration(vitesse).ease(d3.easeLinear)
                    .attr("x", xDet).attr("y", yD + 20);
            }}
        }}

        // Joue un round
        function jouerRound(ri, done) {{
            if (ri >= ROUNDS.length) {{ done(); return; }}
            const txs = ROUNDS[ri];
            document.getElementById("s-round").textContent = ri + 1;
            document.getElementById("info").textContent =
                "Round " + (ri+1) + "/" + ROUNDS.length +
                " — " + txs.length + " transactions";

            // Efface decisions detecteurs
            for (let i = 0; i < NB_DET; i++) {{
                svg.select("#res-d" + i).remove();
            }}

            // Groupe par detecteur
            const groupes = {{}};
            for (let i = 0; i < NB_DET; i++) groupes[i] = [];
            txs.forEach(tx => {{
                const det = tx.detecteur_assign !== undefined
                    ? tx.detecteur_assign
                    : Math.floor(Math.random() * NB_DET);
                groupes[det].push(tx);
            }});

            const maxTx = Math.max(...Object.values(groupes).map(g => g.length));
            let step = 0;

            function prochainStep() {{
                if (!playing) return;
                if (step >= maxTx) {{ setTimeout(done, 600); return; }}

                let lances = 0, termines = 0;

                for (let i = 0; i < NB_DET; i++) {{
                    if (step < groupes[i].length) {{
                        lances++;
                        animerTx(groupes[i][step], i, () => {{
                            termines++;
                            if (termines === lances) {{
                                step++;
                                prochainStep();
                            }}
                        }});
                    }}
                }}
                if (lances === 0) {{ step++; prochainStep(); }}
            }}
            prochainStep();
        }}

        // Joue tous les rounds
        function jouerTout() {{
            if (roundIdx >= ROUNDS.length) {{
                playing = false;
                document.getElementById("info").textContent = "Simulation terminee !";
                return;
            }}
            jouerRound(roundIdx, () => {{
                roundIdx++;
                if (playing) jouerTout();
            }});
        }}

        // Controles
        document.getElementById("btnPlay").onclick = () => {{
            if (!playing) {{ playing = true; jouerTout(); }}
        }};

        document.getElementById("btnPause").onclick = () => {{
            playing = false;
            document.getElementById("info").textContent = "En pause";
        }};

        document.getElementById("btnReplay").onclick = () => {{
            playing = false; roundIdx = 0; supCount = 0;
            totalOk = 0; totalBloque = 0; totalRate = 0;
            resultatsAffich = [];
            document.getElementById("s-ok").textContent     = 0;
            document.getElementById("s-bloque").textContent = 0;
            document.getElementById("s-rate").textContent   = 0;
            document.getElementById("s-round").textContent  = 0;
            document.getElementById("sup-count").textContent = "0 decisions";
            gTx.selectAll("*").remove();
            svg.selectAll(".res-item").remove();
            for (let i = 0; i < NB_DET; i++) {{
                svg.select("#res-d" + i).remove();
            }}
            document.getElementById("info").textContent = "Cliquez Play pour rejouer";
        }};
        </script>
        </body>
        </html>
        """

        st.components.v1.html(html_animation, height=600, scrolling=False)

    # ── ONGLET 2 : Etat des agents ────────────────────────────────────────
    with tab2:
        st.subheader("Etat des agents")
        col_f, col_d = st.columns(2)

        with col_f:
            st.markdown("**Fraudeurs**")
            donnees_fraudeurs = []
            for f in sim.fraudeurs:
                meilleure_strategie = max(f.q_scores, key=f.q_scores.get)
                donnees_fraudeurs.append({
                    "ID"                 : f"Fraudeur {f.fraudeur_id + 1}",
                    "Strategie active"   : f.strategie_actuelle,
                    "Meilleure strategie": meilleure_strategie,
                    "Transactions"       : f.nb_transactions,
                    "Reussites"          : f.nb_reussites,
                    "Echecs"             : f.nb_echecs,
                    "Taux reussite"      : f"{f.get_taux_reussite():.1%}"
                })
            st.dataframe(pd.DataFrame(donnees_fraudeurs),
                        use_container_width=True)

        with col_d:
            st.markdown("**Detecteurs**")
            donnees_detecteurs = []
            for d in sim.detecteurs:
                donnees_detecteurs.append({
                    "ID"               : f"Detecteur {d.detecteur_id + 1}",
                    "Analyses"         : d.nb_analyses,
                    "Fraudes detectees": d.nb_fraudes_detectees,
                    "Fraudes ratees"   : d.nb_fraudes_ratees,
                    "Faux positifs"    : d.nb_faux_positifs,
                    "Taux echec"       : f"{d.get_taux_echec():.1%}",
                    "Seuil montant"    : "1,000,000",
                    "Seuil frequence"  : "20"
                })
            st.dataframe(pd.DataFrame(donnees_detecteurs),
                        use_container_width=True)

    # ── ONGLET 3 : Courbes et graphiques ──────────────────────────────────
    with tab3:
        st.subheader("Courbes et graphiques")

        rounds         = [m["round"]             for m in historique_rounds]
        taux_detection = [m["taux_detection"]     for m in historique_rounds]
        taux_fp        = [m["taux_faux_positifs"] for m in historique_rounds]
        speedups       = [m["speedup"]            for m in historique_rounds]
        colors         = plt.cm.tab10.colors

        g1, g2 = st.columns(2)

        with g1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(rounds, taux_detection,
                    marker="o", color="green",
                    linewidth=2, label="Taux detection")
            ax1.plot(rounds, taux_fp,
                    marker="s", color="orange",
                    linewidth=2, label="Taux faux positifs")
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Taux")
            ax1.set_title("Evolution de la detection par round")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            st.pyplot(fig1)

        with g2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(rounds, speedups,
                    marker="o", color="blue",
                    linewidth=2, label="Speedup reel")
            ax2.axhline(y=1.0, color="red",
                       linestyle="--", linewidth=1,
                       label="Reference sequentielle")
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Speedup")
            ax2.set_title("Speedup parallele vs sequentiel")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        g3, g4 = st.columns(2)

        with g3:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            for i, fraudeur in enumerate(sim.fraudeurs):
                ax3.plot(
                    range(len(fraudeur.historique_reussite)),
                    fraudeur.historique_reussite,
                    marker="o", color=colors[i % 10],
                    linewidth=2,
                    label=f"Fraudeur {fraudeur.fraudeur_id + 1}"
                )
            ax3.set_xlabel("Round")
            ax3.set_ylabel("Taux de reussite")
            ax3.set_title("Apprentissage des fraudeurs par round")
            #ax3.legend(fontsize=7)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)

            # Legende uniquement si peu de fraudeurs
            # Au-dela de 10, elle deborde sur le graphique
            if nb_fraudeurs <= 10:
                ax3.legend(fontsize=7)
            else:
                ax3.text(
                    0.98, 0.98,
                    f"{nb_fraudeurs} fraudeurs",
                    transform=ax3.transAxes,
                    fontsize=8, color="gray",
                    ha="right", va="top"
                )

            st.pyplot(fig3)

        with g4:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            for i, detecteur in enumerate(sim.detecteurs):
                ax4.plot(
                    range(len(detecteur.historique_echec)),
                    detecteur.historique_echec,
                    marker="s", color=colors[i % 10],
                    linewidth=2,
                    label=f"Detecteur {detecteur.detecteur_id + 1}"
                )
            ax4.set_xlabel("Round")
            ax4.set_ylabel("Taux d echec")
            ax4.set_title("Adaptation des detecteurs par round")
            ax4.legend(fontsize=7)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            st.pyplot(fig4)

    # ── ONGLET 4 : Strategies des fraudeurs ───────────────────────────────
    with tab4:
        st.subheader("Strategies des fraudeurs")
        g5, g6 = st.columns(2)

        with g5:
            fig5, ax5 = plt.subplots(figsize=(6, 5))
            for i, fraudeur in enumerate(sim.fraudeurs):
                strategies = list(fraudeur.q_scores.keys())
                scores     = list(fraudeur.q_scores.values())
                ax5.bar(
                    [s[:8] for s in strategies],
                    scores,
                    alpha=0.6,
                    label=f"Fraudeur {fraudeur.fraudeur_id + 1}",
                    color=colors[i % 10]
                )
            ax5.set_xlabel("Strategie")
            ax5.set_ylabel("Score Q (apprentissage)")
            ax5.set_title("Scores Q finaux par strategie")
            # Legende uniquement si peu de fraudeurs
            if nb_fraudeurs <= 10:
                ax5.legend(fontsize=7)
            else:
                ax5.text(
                    0.98, 0.98,
                    f"{nb_fraudeurs} fraudeurs",
                    transform=ax5.transAxes,
                    fontsize=8, color="gray",
                    ha="right", va="top"
                )
            ax5.tick_params(axis="x", rotation=45)
            ax5.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig5)

        with g6:
            fig6, ax6 = plt.subplots(figsize=(6, 5))
            labels  = ["Sequentiel\n(1 thread)",
                      f"Parallele\n({nb_detecteurs} threads)"]
            valeurs = [bilan["temps_sequentiel"], bilan["temps_parallele"]]
            couleurs = ["red", "green"]
            barres = ax6.bar(labels, valeurs, color=couleurs, alpha=0.8)
            for barre_item, valeur in zip(barres, valeurs):
                ax6.text(
                    barre_item.get_x() + barre_item.get_width() / 2,
                    barre_item.get_height() + 0.01,
                    f"{valeur:.3f}s",
                    ha="center", va="bottom", fontsize=10
                )
            ax6.set_ylabel("Temps d execution (secondes)")
            ax6.set_title(f"Speedup final : {bilan['speedup']:.2f}x")
            ax6.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig6)