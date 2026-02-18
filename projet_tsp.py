import math
import random
import time
import statistics
import copy

# --- PARTIE 1: MODELISATION (Villes & Instance) ---

class City:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def distance_to(self, other_city):
        # Distance Euclidienne
        return math.sqrt((self.x - other_city.x)**2 + (self.y - other_city.y)**2)

class TSPInstance:
    def __init__(self, num_cities, size=100):
        self.num_cities = num_cities
        self.cities = []
        self.generate_random_instance(size)

    def generate_random_instance(self, size):
        # Génération aléatoire dans un carré [0, size] x [0, size]
        self.cities = [City(i, random.uniform(0, size), random.uniform(0, size)) for i in range(self.num_cities)]

    def calculate_tour_cost(self, tour):
        # Coût total d'une solution (tour)
        total_distance = 0
        for i in range(len(tour)):
            from_city = self.cities[tour[i]]
            to_city = self.cities[tour[(i + 1) % len(tour)]] # Retour au départ
            total_distance += from_city.distance_to(to_city)
        return total_distance

# --- PARTIE 2: METAHEURISTIQUES ---

class TSPSolver:
    def __init__(self, instance):
        self.instance = instance

    def generate_random_solution(self):
        tour = list(range(self.instance.num_cities))
        random.shuffle(tour)
        return tour

    def get_neighbors(self, tour):
        # Voisinage par SWAP (échange de 2 villes)
        neighbors = []
        n = len(tour)
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = tour[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    # 1. Hill Climbing (Best Improvement)
    def hill_climbing_best(self):
        current_solution = self.generate_random_solution()
        current_cost = self.instance.calculate_tour_cost(current_solution)
        
        while True:
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_cost = float('inf')

            # Chercher le MEILLEUR voisin
            for neighbor in neighbors:
                cost = self.instance.calculate_tour_cost(neighbor)
                if cost < best_neighbor_cost:
                    best_neighbor_cost = cost
                    best_neighbor = neighbor

            if best_neighbor_cost < current_cost:
                current_solution = best_neighbor
                current_cost = best_neighbor_cost
            else:
                break # Optimum local atteint
        return current_solution, current_cost

    # 2. Hill Climbing (First Improvement)
    def hill_climbing_first(self):
        current_solution = self.generate_random_solution()
        current_cost = self.instance.calculate_tour_cost(current_solution)
        
        improved = True
        while improved:
            improved = False
            neighbors = self.get_neighbors(current_solution)
            random.shuffle(neighbors) # Pour éviter le biais de l'ordre

            for neighbor in neighbors:
                cost = self.instance.calculate_tour_cost(neighbor)
                if cost < current_cost:
                    current_solution = neighbor
                    current_cost = cost
                    improved = True
                    break # On prend le PREMIER qui améliore
        return current_solution, current_cost

    # 3. Multi-Start Hill Climbing
    def multi_start_hc(self, iterations=10):
        best_global_solution = None
        best_global_cost = float('inf')

        for _ in range(iterations):
            sol, cost = self.hill_climbing_best() # Ou First, au choix
            if cost < best_global_cost:
                best_global_cost = cost
                best_global_solution = sol
        
        return best_global_solution, best_global_cost

    # 4. Simulated Annealing (Recuit Simulé)
    def simulated_annealing(self, T0=1000, alpha=0.99, max_iter=1000):
        current_solution = self.generate_random_solution()
        current_cost = self.instance.calculate_tour_cost(current_solution)
        
        best_solution = list(current_solution)
        best_cost = current_cost
        
        T = T0
        for _ in range(max_iter):
            # Générer UN voisin aléatoire (swap)
            i, j = random.sample(range(len(current_solution)), 2)
            neighbor = list(current_solution)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_cost = self.instance.calculate_tour_cost(neighbor)

            delta = neighbor_cost - current_cost

            # Règle d'acceptation
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_solution = neighbor
                current_cost = neighbor_cost

                # Mise à jour du global best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution)
            
            # Refroidissement
            T *= alpha
            if T < 1e-3: break
            
        return best_solution, best_cost

    # 5. BONUS: Tabu Search (Recherche Tabou)
    def tabu_search(self, max_iter=100, tabu_tenure=5):
        current_solution = self.generate_random_solution()
        best_solution = list(current_solution)
        best_cost = self.instance.calculate_tour_cost(current_solution)
        
        tabu_list = [] # Liste des mouvements interdits (paires i, j)

        for _ in range(max_iter):
            neighbors = []
            # Générer voisinage avec info sur le mouvement (i, j)
            n = len(current_solution)
            for i in range(n):
                for j in range(i + 1, n):
                    neighbor = list(current_solution)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost = self.instance.calculate_tour_cost(neighbor)
                    neighbors.append((neighbor, cost, (i, j)))

            # Filtrer les voisins (non tabous ou critère d'aspiration)
            neighbors.sort(key=lambda x: x[1]) # Trier par meilleur coût
            
            best_move = None
            
            for sol, cost, move in neighbors:
                if move not in tabu_list or cost < best_cost: # Aspiration
                    best_move = (sol, cost, move)
                    break
            
            if best_move:
                current_solution, current_cost, move = best_move
                if current_cost < best_cost:
                    best_solution = list(current_solution)
                    best_cost = current_cost
                
                # Mise à jour liste tabou
                tabu_list.append(move)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
            else:
                break 

        return best_solution, best_cost

# --- PARTIE 3: PROTOCOLE EXPERIMENTAL ---

def run_experiment():
    instance_sizes = [20, 50, 80] # A, B, C
    num_runs = 30 # Comme demandé
    
    print(f"{'Instance':<10} | {'Algo':<15} | {'Best':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Time (s)':<10}")
    print("-" * 80)

    for n in instance_sizes:
        tsp = TSPInstance(n)
        solver = TSPSolver(tsp)
        
        algorithms = {
            "HC (Best)": solver.hill_climbing_best,
            "HC (First)": solver.hill_climbing_first,
            "Multi-Start": solver.multi_start_hc,
            "Sim. Annealing": solver.simulated_annealing,
            "Tabu (Bonus)": solver.tabu_search
        }

        for algo_name, algo_func in algorithms.items():
            costs = []
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                _, cost = algo_func()
                end_time = time.time()
                
                costs.append(cost)
                times.append(end_time - start_time)
            
            best_res = min(costs)
            mean_res = statistics.mean(costs)
            std_res = statistics.stdev(costs) if len(costs) > 1 else 0
            mean_time = statistics.mean(times)

            print(f"{n:<10} | {algo_name:<15} | {best_res:<10.2f} | {mean_res:<10.2f} | {std_res:<10.2f} | {mean_time:<10.4f}")

if __name__ == "__main__":
    run_experiment()