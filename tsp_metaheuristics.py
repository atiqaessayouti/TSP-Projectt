"""
Projet M1 - Comparaison de métaheuristiques pour le TSP
========================================================
Auteur: Etudiant M1
Date: 2026
Université Hassan II de Casablanca - ENSET Mohammedia
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Callable
import copy
import random

# ============================================================================
# ÉTAPE 1: CLASSE TOUR - Représentation d'une solution TSP
# ============================================================================

class Tour:
    """
    Représente un tour (solution) pour le TSP.
    Un tour est une permutation des villes.
    """
    
    def __init__(self, cities: List[int], distances: np.ndarray):
        """
        Args:
            cities: Liste ordonnée des indices de villes (permutation)
            distances: Matrice de distances entre les villes
        """
        self.cities = cities.copy()
        self.distances = distances
        self.length = self._calculate_length()
    
    def _calculate_length(self) -> float:
        """Calcule la longueur totale du tour"""
        total = 0
        n = len(self.cities)
        for i in range(n):
            city_a = self.cities[i]
            city_b = self.cities[(i + 1) % n]  # Retour à la ville de départ
            total += self.distances[city_a][city_b]
        return total
    
    def copy(self):
        """Crée une copie du tour"""
        return Tour(self.cities.copy(), self.distances)
    
    def swap(self, i: int, j: int):
        """Échange deux villes dans le tour"""
        self.cities[i], self.cities[j] = self.cities[j], self.cities[i]
        self.length = self._calculate_length()
    
    def get_neighbors_swap(self) -> List['Tour']:
        """
        Génère tous les voisins en utilisant l'opérateur swap.
        Pour n villes, il y a n(n-1)/2 voisins possibles.
        """
        neighbors = []
        n = len(self.cities)
        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = self.copy()
                neighbor.swap(i, j)
                neighbors.append(neighbor)
        return neighbors
    
    def __str__(self):
        return f"Tour(length={self.length:.2f}, cities={self.cities[:5]}...)"


# ============================================================================
# ÉTAPE 2: CLASSE TSP_INSTANCE - Gestion des instances du problème
# ============================================================================

class TSPInstance:
    """
    Représente une instance du problème TSP avec n villes.
    """
    
    def __init__(self, n_cities: int, coordinates: np.ndarray = None):
        """
        Args:
            n_cities: Nombre de villes
            coordinates: Coordonnées (x, y) des villes. Si None, génération aléatoire.
        """
        self.n_cities = n_cities
        
        if coordinates is None:
            # Génération aléatoire dans [0, 100] x [0, 100]
            self.coordinates = np.random.uniform(0, 100, (n_cities, 2))
        else:
            self.coordinates = coordinates
        
        # Calcul de la matrice de distances euclidiennes
        self.distances = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Calcule la matrice de distances euclidiennes entre toutes les villes"""
        n = self.n_cities
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Distance euclidienne
                dx = self.coordinates[i][0] - self.coordinates[j][0]
                dy = self.coordinates[i][1] - self.coordinates[j][1]
                dist = np.sqrt(dx**2 + dy**2)
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def random_tour(self) -> Tour:
        """Génère un tour aléatoire"""
        cities = list(range(self.n_cities))
        random.shuffle(cities)
        return Tour(cities, self.distances)
    
    def visualize(self, tour: Tour = None, title: str = "TSP Instance"):
        """Visualise l'instance et optionnellement un tour"""
        plt.figure(figsize=(10, 8))
        
        # Affiche les villes
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                    c='red', s=100, zorder=2, label='Villes')
        
        # Affiche les numéros des villes
        for i, (x, y) in enumerate(self.coordinates):
            plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='bottom')
        
        # Affiche le tour si fourni
        if tour is not None:
            for i in range(len(tour.cities)):
                city_a = tour.cities[i]
                city_b = tour.cities[(i + 1) % len(tour.cities)]
                
                x_values = [self.coordinates[city_a][0], self.coordinates[city_b][0]]
                y_values = [self.coordinates[city_a][1], self.coordinates[city_b][1]]
                
                plt.plot(x_values, y_values, 'b-', alpha=0.6, zorder=1)
            
            plt.title(f"{title}\nLongueur: {tour.length:.2f}")
        else:
            plt.title(title)
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================================
# ÉTAPE 3: MÉTAHEURISTIQUES
# ============================================================================

class Metaheuristic:
    """Classe abstraite pour les métaheuristiques"""
    
    def __init__(self, instance: TSPInstance):
        self.instance = instance
        self.best_solution = None
        self.best_length = float('inf')
        self.history = []  # Historique des longueurs pour visualisation
        self.evaluations = 0  # Nombre d'évaluations de la fonction objectif
    
    def optimize(self, max_evaluations: int) -> Tour:
        """Méthode à implémenter par les sous-classes"""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# 3.1: HILL CLIMBING - BEST IMPROVEMENT
# ----------------------------------------------------------------------------

class HillClimbingBest(Metaheuristic):
    """
    Hill Climbing avec stratégie Best Improvement.
    À chaque itération, explore TOUS les voisins et choisit le meilleur.
    """
    
    def optimize(self, max_evaluations: int = 10000) -> Tour:
        # Initialisation avec un tour aléatoire
        current = self.instance.random_tour()
        self.best_solution = current.copy()
        self.best_length = current.length
        self.evaluations = 1
        self.history = [current.length]
        
        improved = True
        
        while improved and self.evaluations < max_evaluations:
            improved = False
            
            # Génère tous les voisins
            neighbors = current.get_neighbors_swap()
            self.evaluations += len(neighbors)
            
            if not neighbors: break

            # Trouve le meilleur voisin
            best_neighbor = min(neighbors, key=lambda t: t.length)
            
            # Si le meilleur voisin est meilleur que la solution actuelle
            if best_neighbor.length < current.length:
                current = best_neighbor
                improved = True
                
                # Met à jour la meilleure solution
                if current.length < self.best_length:
                    self.best_solution = current.copy()
                    self.best_length = current.length
                
                self.history.append(current.length)
        
        return self.best_solution


# ----------------------------------------------------------------------------
# 3.2: HILL CLIMBING - FIRST IMPROVEMENT
# ----------------------------------------------------------------------------

class HillClimbingFirst(Metaheuristic):
    """
    Hill Climbing avec stratégie First Improvement.
    S'arrête au PREMIER voisin améliorant trouvé.
    """
    
    def optimize(self, max_evaluations: int = 10000) -> Tour:
        current = self.instance.random_tour()
        self.best_solution = current.copy()
        self.best_length = current.length
        self.evaluations = 1
        self.history = [current.length]
        
        improved = True
        
        while improved and self.evaluations < max_evaluations:
            improved = False
            
            # Génère les voisins de manière itérative
            n = len(current.cities)
            # On shuffle l'ordre de recherche pour éviter les biais
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in range(n - 1):
                if self.evaluations >= max_evaluations:
                    break
                    
                for j in range(i + 1, n):
                    # Crée et évalue un voisin
                    neighbor = current.copy()
                    neighbor.swap(i, j)
                    self.evaluations += 1
                    
                    # Si amélioration trouvée, accepte immédiatement
                    if neighbor.length < current.length:
                        current = neighbor
                        improved = True
                        
                        if current.length < self.best_length:
                            self.best_solution = current.copy()
                            self.best_length = current.length
                        
                        self.history.append(current.length)
                        break  # Sort de la boucle j
                
                if improved:
                    break  # Sort de la boucle i
        
        return self.best_solution


# ----------------------------------------------------------------------------
# 3.3: MULTI-START HILL CLIMBING
# ----------------------------------------------------------------------------

class MultiStartHC(Metaheuristic):
    """
    Multi-Start Hill Climbing.
    Lance plusieurs fois Hill Climbing depuis différentes solutions initiales.
    """
    
    def __init__(self, instance: TSPInstance, hc_type: str = 'best', n_starts: int = 10):
        """
        Args:
            hc_type: 'best' pour Best Improvement, 'first' pour First Improvement
            n_starts: Nombre de redémarrages (par défaut 10)
        """
        super().__init__(instance)
        self.hc_type = hc_type
        self.n_starts = n_starts 
    
    def optimize(self, max_evaluations: int = 50000) -> Tour:
        self.best_length = float('inf')
        self.evaluations = 0
        self.history = []
        
        # On divise le budget par le nombre de starts
        evals_per_start = max_evaluations // self.n_starts
        if evals_per_start == 0: evals_per_start = 1 # Sécurité
        
        for start in range(self.n_starts):
            if self.evaluations >= max_evaluations:
                break
            
            # Crée une instance de HC
            if self.hc_type == 'best':
                hc = HillClimbingBest(self.instance)
            else:
                hc = HillClimbingFirst(self.instance)
            
            # Lance l'optimisation pour ce 'start'
            solution = hc.optimize(evals_per_start)
            
            # Met à jour les statistiques globales
            self.evaluations += hc.evaluations
            self.history.extend(hc.history)
            
            # Garde la meilleure solution trouvée parmi tous les starts
            if solution.length < self.best_length:
                self.best_solution = solution.copy()
                self.best_length = solution.length
        
        return self.best_solution


# ----------------------------------------------------------------------------
# 3.4: RECUIT SIMULÉ (SIMULATED ANNEALING)
# ----------------------------------------------------------------------------

class SimulatedAnnealing(Metaheuristic):
    """
    Recuit Simulé (Simulated Annealing).
    Accepte des solutions dégradantes avec une probabilité décroissante.
    """
    
    def __init__(self, instance: TSPInstance, T0: float = 100, alpha: float = 0.95, 
                 T_min: float = 0.01):
        """
        Args:
            T0: Température initiale
            alpha: Coefficient de refroidissement (0 < alpha < 1)
            T_min: Température minimale
        """
        super().__init__(instance)
        self.T0 = T0
        self.alpha = alpha
        self.T_min = T_min
    
    def optimize(self, max_evaluations: int = 50000) -> Tour:
        # Initialisation
        current = self.instance.random_tour()
        self.best_solution = current.copy()
        self.best_length = current.length
        self.evaluations = 1
        self.history = [current.length]
        
        T = self.T0
        
        while T > self.T_min and self.evaluations < max_evaluations:
            # Pour chaque température, fait plusieurs itérations
            for _ in range(100):  # 100 itérations par palier de température
                if self.evaluations >= max_evaluations:
                    break
                
                # Génère un voisin aléatoire
                n = len(current.cities)
                i, j = random.sample(range(n), 2)
                neighbor = current.copy()
                neighbor.swap(i, j)
                self.evaluations += 1
                
                # Calcule la variation de coût
                delta = neighbor.length - current.length
                
                # Critère d'acceptation
                if delta <= 0:
                    # Amélioration: accepte toujours
                    current = neighbor
                else:
                    # Dégradation: accepte avec probabilité exp(-delta/T)
                    probability = np.exp(-delta / T)
                    if random.random() < probability:
                        current = neighbor
                
                # Met à jour la meilleure solution
                if current.length < self.best_length:
                    self.best_solution = current.copy()
                    self.best_length = current.length
                
                self.history.append(current.length)
            
            # Refroidissement
            T *= self.alpha
        
        return self.best_solution


# ============================================================================
# ÉTAPE 4: PROTOCOLE EXPÉRIMENTAL
# ============================================================================

class Experiment:
    """
    Gère l'exécution des expériences et la collecte des résultats.
    """
    
    def __init__(self, instance: TSPInstance, instance_name: str):
        self.instance = instance
        self.instance_name = instance_name
        self.results = {}
    
    def run_algorithm(self, algo_class, algo_name: str, n_runs: int = 30, 
                      max_evaluations: int = 10000, **kwargs):
        """
        Exécute un algorithme plusieurs fois et collecte les statistiques.
        """
        print(f"\n{'='*60}")
        print(f"Exécution: {algo_name} sur {self.instance_name}")
        print(f"{'='*60}")
        
        best_lengths = []
        times = []
        all_histories = []
        
        for run in range(n_runs):
            # Crée une instance de l'algorithme
            algo = algo_class(self.instance, **kwargs)
            
            # Lance l'optimisation
            start_time = time.time()
            solution = algo.optimize(max_evaluations)
            elapsed_time = time.time() - start_time
            
            # Collecte les résultats
            best_lengths.append(solution.length)
            times.append(elapsed_time)
            all_histories.append(algo.history)
            
            # Affichage périodique
            if n_runs < 10 or (run + 1) % 5 == 0:
                print(f"  Run {run + 1}/{n_runs} - Meilleur: {solution.length:.2f}")
        
        # Calcule les statistiques
        self.results[algo_name] = {
            'best': np.min(best_lengths),
            'mean': np.mean(best_lengths),
            'std': np.std(best_lengths),
            'worst': np.max(best_lengths),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'all_lengths': best_lengths,
            'histories': all_histories
        }
        
        print(f"\n  Résultats:")
        print(f"    Meilleur: {self.results[algo_name]['best']:.2f}")
        print(f"    Moyen: {self.results[algo_name]['mean']:.2f} ± {self.results[algo_name]['std']:.2f}")
        print(f"    Temps: {self.results[algo_name]['time_mean']:.3f}s ± {self.results[algo_name]['time_std']:.3f}s")
    
    def print_summary(self):
        """Affiche un tableau récapitulatif des résultats"""
        print(f"\n{'='*80}")
        print(f"RÉSUMÉ - {self.instance_name}")
        print(f"{'='*80}")
        print(f"{'Algorithme':<25} {'Meilleur':<12} {'Moyen ± Std':<20} {'Temps (s)':<15}")
        print(f"{'-'*80}")
        
        for algo_name, results in self.results.items():
            print(f"{algo_name:<25} {results['best']:<12.2f} "
                  f"{results['mean']:>7.2f} ± {results['std']:<7.2f} "
                  f"{results['time_mean']:>7.3f} ± {results['time_std']:<5.3f}")
    
    def plot_convergence(self):
        """Trace les courbes de convergence"""
        plt.figure(figsize=(12, 6))
        
        for algo_name, results in self.results.items():
            # Moyenne des historiques
            max_len = max(len(h) for h in results['histories'])
            
            # Remplit les historiques courts avec leur dernière valeur
            padded_histories = []
            for h in results['histories']:
                if len(h) < max_len:
                    h_padded = h + [h[-1]] * (max_len - len(h))
                else:
                    h_padded = h
                padded_histories.append(h_padded)
            
            mean_history = np.mean(padded_histories, axis=0)
            
            plt.plot(mean_history, label=algo_name, linewidth=2, alpha=0.8)
        
        plt.xlabel('Itérations', fontsize=12)
        plt.ylabel('Longueur du tour', fontsize=12)
        plt.title(f'Courbes de convergence - {self.instance_name}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================================
# ÉTAPE 5: EXEMPLE D'UTILISATION COMPLÈTE
# ============================================================================

def main():
    """Fonction principale pour exécuter les expériences"""
    
    # Fixe la graine aléatoire pour la reproductibilité
    np.random.seed(42)
    random.seed(42)
    
    # ========================================================================
    # Création des instances
    # ========================================================================
    print("Création des instances TSP...")
    
    instance_A = TSPInstance(n_cities=20)
    instance_B = TSPInstance(n_cities=50)
    
    # ========================================================================
    # Expériences sur Instance A (20 villes)
    # ========================================================================
    print("\n" + "="*80)
    print("INSTANCE A - 20 VILLES")
    print("="*80)
    
    exp_A = Experiment(instance_A, "Instance A (20 villes)")
    
    # 1. Hill Climbing Best
    exp_A.run_algorithm(HillClimbingBest, "HC Best Improvement", 
                        n_runs=10, max_evaluations=5000)
    
    # 2. Hill Climbing First
    exp_A.run_algorithm(HillClimbingFirst, "HC First Improvement", 
                        n_runs=10, max_evaluations=5000)
    
    # 3. Multi-Start HC (CORRIGÉ: n_starts est maintenant accepté)
    exp_A.run_algorithm(MultiStartHC, "Multi-Start HC (Best)", 
                        n_runs=10, max_evaluations=5000, 
                        hc_type='best', n_starts=5)
    
    # 4. Recuit Simulé
    exp_A.run_algorithm(SimulatedAnnealing, "Recuit Simulé", 
                        n_runs=10, max_evaluations=5000,
                        T0=100, alpha=0.95, T_min=0.01)
    
    # Affiche les résultats et graphes
    exp_A.print_summary()
    exp_A.plot_convergence()
    
    # ========================================================================
    # Expériences sur Instance B (50 villes)
    # ========================================================================
    print("\n" + "="*80)
    print("INSTANCE B - 50 VILLES")
    print("="*80)
    
    exp_B = Experiment(instance_B, "Instance B (50 villes)")
    
    # Pour B, on utilise un budget plus grand
    exp_B.run_algorithm(HillClimbingBest, "HC Best Improvement", 
                        n_runs=5, max_evaluations=20000)
    
    exp_B.run_algorithm(SimulatedAnnealing, "Recuit Simulé", 
                        n_runs=5, max_evaluations=20000,
                        T0=150, alpha=0.98, T_min=0.01)
    
    exp_B.print_summary()
    
    print("\n" + "="*80)
    print("EXPÉRIENCES TERMINÉES")
    print("="*80)


if __name__ == "__main__":
    main()