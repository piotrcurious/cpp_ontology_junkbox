import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sympy import symbols, Matrix, groebner
from multiprocessing import Pool, cpu_count, current_process
import os

# 1. Function to extract C++ function prototypes using regular expressions.
def extract_function_prototypes(code):
    pattern = r'\b[\w\*\&]+\s+[\w\*\&]+\s*\([^)]*\)\s*;'
    return re.findall(pattern, code)

# 2. Fitness Function with Lyapunov Stability for Parallel Execution
def fitness_with_lyapunov(candidate):
    # Simple Lyapunov function check
    x = symbols('x:3')
    Q = Matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    V_expr = Matrix(x).T * Q * Matrix(x)

    # Evaluate the Lyapunov function at a fixed point (e.g., all ones)
    V_val = V_expr[0,0].subs({x[0]: 1, x[1]: 1, x[2]: 1})

    if 'return' in candidate and any(kw in candidate for kw in ['int', 'double', 'float']):
        # Use the value of the Lyapunov function to influence the stability score
        stability_score = -float(V_val) / (len(candidate) + 1)
    else:
        stability_score = np.inf

    return stability_score

# 3. Worker for parallel evolutionary strategy (defined at top-level for pickling)
def evolutionary_worker(prototype):
    score = fitness_with_lyapunov(prototype)
    return prototype, score

# 4. Parallel Evolutionary Strategy
def evolutionary_strategy(prototypes, generations=10, population_size=50):
    population = prototypes * (population_size // len(prototypes) + 1)
    population = population[:population_size]

    results = []
    for _ in range(generations):
        with Pool(processes=min(cpu_count(), len(population))) as pool:
            results = pool.map(evolutionary_worker, population)

        # Select top 50% based on fitness (lower is better in this stability-based score)
        sorted_results = sorted(results, key=lambda x: x[1])
        selected_candidates = [result[0] for result in sorted_results[:population_size // 2]]

        # Crossover and Mutation
        new_candidates = []
        for i in range(len(selected_candidates) // 2):
            parent1 = selected_candidates[i]
            parent2 = selected_candidates[-i-1]
            length = min(len(parent1), len(parent2))
            crossover_point = length // 2
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            new_candidates.extend([child1, child2])

        mutated_candidates = []
        for candidate in new_candidates:
            mutated_candidate = list(candidate)
            mutation_chance = 0.1
            for i in range(len(mutated_candidate)):
                if np.random.rand() < mutation_chance:
                    mutated_candidate[i] = chr(np.random.randint(32, 127))
            mutated_candidates.append(''.join(mutated_candidate))

        population = selected_candidates + mutated_candidates

    # Return the best candidate based on fitness
    if not results:
        with Pool(processes=min(cpu_count(), len(population))) as pool:
            results = pool.map(evolutionary_worker, population)
    best_prototype, _ = sorted(results, key=lambda x: x[1])[0]
    return best_prototype

# 5. Worker for parallel feature extraction (defined at top-level for pickling)
def compute_groebner_worker(prototype):
    tokens = re.findall(r'[a-zA-Z_]\w*', prototype)[:5] # Limit variables
    if not tokens: return np.array([0.0], dtype=np.float64)
    variables = symbols(tokens)
    equations = [var**2 + 1 for var in variables]
    try:
        G = groebner(equations, variables)
        return np.array([len(str(g)) for g in G], dtype=np.float64)
    except:
        return np.array([0.0], dtype=np.float64)

# 6. Parallel Algebraic Geometry Feature Extraction
def algebraic_geometry_features(prototypes):
    with Pool(processes=min(cpu_count(), len(prototypes))) as pool:
        features = pool.map(compute_groebner_worker, prototypes)

    return features

# Main execution logic
def main():
    # Text2Vec for Vectorization and Similarity Calculation
    vectorizer = TfidfVectorizer()

    # Example C++ code snippets.
    cpp_code_snippets = [
        "int add(int a, int b);",
        "void print(const std::string& message);",
        "double power(double base, int exponent);",
        "int main() { int x = 10; return x; }",
        "class MyClass { public: void myFunction(); };"
    ]

    # Extracting function prototypes from the code snippets.
    function_prototypes = []
    for code in cpp_code_snippets:
        prototypes = extract_function_prototypes(code)
        function_prototypes.extend(prototypes)

    # Apply parallel evolutionary strategy to generate the best function prototype.
    if not function_prototypes:
        print("No function prototypes found in the provided code snippets.")
    else:
        best_prototype = evolutionary_strategy(function_prototypes)

        # Vectorize the best prototype (include original prototypes to ensure vectorizer has enough data for PCA)
        all_for_fit = function_prototypes + [best_prototype]
        vectorizer.fit(all_for_fit)

        polynomial_features = PolynomialFeatures(degree=2)
        all_prototype_vectors = vectorizer.transform(all_for_fit).toarray()
        all_transformed = polynomial_features.fit_transform(all_prototype_vectors)

        n_samples, n_features = all_transformed.shape
        n_comp = min(2, n_samples - 1, n_features)

        if n_comp > 0:
            pca = PCA(n_components=n_comp)
            reduced_all = pca.fit_transform(all_transformed)
            reduced_vector = reduced_all[-1:]
        else:
            reduced_vector = all_transformed[-1:]

        # Extract algebraic geometry features in parallel.
        algebraic_features = algebraic_geometry_features([best_prototype])

        # Combine reduced vector with algebraic features.
        final_vector = np.hstack((reduced_vector.flatten(), algebraic_features[0]))

        # Display the best function prototype and its feature vector.
        print(f"Best Function Prototype: '{best_prototype}'")
        print(f"Feature Vector: {final_vector}")

if __name__ == "__main__":
    main()
