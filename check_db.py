from base_donnees import charger_historique_simulations

sims = charger_historique_simulations()

print("="*50)
print("HISTORIQUE DES SIMULATIONS")
print("="*50)

for s in sims:
    print(f"Sim {s['simulation_id']:2d} | "
          f"Détection : {s['taux_detection']*100:.1f}% | "
          f"Détectées : {s['fraudes_detectees']:3d} | "
          f"Ratées : {s['fraudes_ratees']:3d} | "
          f"FP : {s['faux_positifs']:2d}")