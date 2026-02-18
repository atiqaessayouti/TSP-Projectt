"""
Projet M1 - Comparaison de métaheuristiques pour le TSP
Auteurs: Essayouti Atiqa - Timourti Sanae - Misbah Kaoutar
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random

# Force l'utilisation d'un mode sans fenêtre pour enregistrer les fichiers directement
import matplotlib
matplotlib.use('Agg') 

class Tour:
    def __init__(self, cities, distances):
        self.cities = list(cities)
        self.distances = distances
        self.length = self._calculate_length()
    def _calculate_length(self):
        total = 0
        n = len(self.cities)
        for i in range(n):
            total += self.distances[self.cities[i]][self.cities[(i + 1) % n]]
        return total
    def copy(self): return Tour(self.cities, self.distances)
    def swap(self, i, j):
        self.cities[i], self.cities[j] = self.cities[j], self.cities[i]
        self.length = self._calculate_length()

class TSPInstance:
    def __init__(self, n_cities):
        self.n_cities = n_cities
        self.coords = np.random.uniform(0, 100, (n_cities, 2))
        self.dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                self.dist_matrix[i,j] = np.linalg.norm(self.coords[i]-self.coords[j])
    def random_tour(self):
        cities = list(range(self.n_cities))
        random.shuffle(cities)
        return Tour(cities, self.dist_matrix)

class HillClimbingBest:
    def __init__(self, inst): self.inst = inst
    def optimize(self, evals):
        curr = self.inst.random_tour()
        hist = [curr.length]
        e = 1
        while e < evals:
            best_n = curr
            n = len(curr.cities)
            for i in range(n-1):
                for j in range(i+1, n):
                    temp = curr.copy(); temp.swap(i,j); e += 1
                    if temp.length < best_n.length: best_n = temp
            if best_n.length < curr.length: curr = best_n; hist.append(curr.length)
            else: break
        return curr, hist

class SimulatedAnnealing:
    def __init__(self, inst): self.inst = inst
    def optimize(self, evals):
        curr = self.inst.random_tour()
        best = curr.copy()
        hist = [curr.length]
        T, alpha, e = 100, 0.98, 1
        while e < evals and T > 0.01:
            for _ in range(50):
                i, j = random.sample(range(len(curr.cities)), 2)
                temp = curr.copy(); temp.swap(i,j); e += 1
                delta = temp.length - curr.length
                if delta < 0 or random.random() < np.exp(-delta/T):
                    curr = temp
                    if curr.length < best.length: best = curr.copy()
                hist.append(best.length)
            T *= alpha
        return best, hist

def run_exp(n_cities, name):
    inst = TSPInstance(n_cities)
    print(f"\n--- {name} ---")
    
    # HC
    start = time.time()
    sol_hc, hist_hc = HillClimbingBest(inst).optimize(10000)
    t_hc = time.time() - start
    print(f"HC Best: {sol_hc.length:.2f} in {t_hc:.2f}s")
    
    # SA
    start = time.time()
    sol_sa, hist_sa = SimulatedAnnealing(inst).optimize(10000)
    t_sa = time.time() - start
    print(f"Recuit Simulé: {sol_sa.length:.2f} in {t_sa:.2f}s")
    
    # Sauvegarde du graphe
    plt.figure(figsize=(10,6))
    plt.plot(hist_hc, label=f"Hill Climbing ({sol_hc.length:.1f})", color='blue')
    plt.plot(hist_sa, label=f"Recuit Simulé ({sol_sa.length:.1f})", color='red')
    plt.title(f"Convergence TSP - {n_cities} villes")
    plt.xlabel("Évaluations"); plt.ylabel("Distance"); plt.legend(); plt.grid(True)
    
    # Sauvegarde dans le dossier screenshots
    filename = f"convergence_{n_cities}.png"
    plt.savefig(os.path.join("screenshots", filename))
    plt.close()
    print(f"✅ Graphe sauvegardé: screenshots/{filename}")

if __name__ == "__main__":
    # Crée le dossier screenshots s'il n'existe pas
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
        
    run_exp(20, "Instance A")
    run_exp(50, "Instance B")
    print("\n🚀 TOUT EST PRÊT. Vérifie le dossier 'screenshots' !")