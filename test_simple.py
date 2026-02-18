import numpy as np
import random
from tsp_metaheuristics import TSPInstance, SimulatedAnnealing, HillClimbingBest

# Fixe les graines pour reproductibilité
np.random.seed(42)
random.seed(42)

print("="*60)
print("TEST RAPIDE - TSP avec 15 villes")
print("="*60)

# Crée une petite instance
instance = TSPInstance(n_cities=15)
print(f"Instance créée avec {instance.n_cities} villes")

# Visualise l'instance
print("\n📍 Affichage de l'instance...")
instance.visualize(title="Instance Test - 15 villes")

# Test Hill Climbing
print("\n🔍 Test Hill Climbing Best...")
hc = HillClimbingBest(instance)
solution_hc = hc.optimize(max_evaluations=2000)
print(f"✓ Longueur trouvée: {solution_hc.length:.2f}")
instance.visualize(solution_hc, title="Solution Hill Climbing")

# Test Recuit Simulé
print("\n🔥 Test Recuit Simulé...")
sa = SimulatedAnnealing(instance, T0=200, alpha=0.98)
solution_sa = sa.optimize(max_evaluations=5000)
print(f"✓ Longueur trouvée: {solution_sa.length:.2f}")
instance.visualize(solution_sa, title="Solution Recuit Simulé")

print("\n" + "="*60)
print("✅ TESTS TERMINÉS")
print("="*60)