"""Affiche le bilan des simulations enregistrées dans la base.

Depuis la racine du projet :
    python -m scripts.check_db
"""

from core.base_donnees import charger_historique_simulations

print("Historique des simulations")
print("=" * 50)

for s in charger_historique_simulations():
    print(f"Sim {s['simulation_id']:2d} | "
          f"Détection : {s['taux_detection']*100:.1f}% | "
          f"Détectées : {s['fraudes_detectees']:3d} | "
          f"Ratées : {s['fraudes_ratees']:3d} | "
          f"FP : {s['faux_positifs']:2d}")
