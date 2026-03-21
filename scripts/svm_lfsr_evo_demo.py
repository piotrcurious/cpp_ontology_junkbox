
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from multiprocessing import Pool, cpu_count

class LFSR:
    def __init__(self, seed, taps):
        self.state = seed
        self.taps = taps
        self.n_bits = len(bin(seed)) - 2  # Number of bits in the seed

    def step(self):
        # Perform one LFSR step and update the state
        xor = 0
        for tap in self.taps:
            xor ^= (self.state >> tap) & 1
        self.state = (self.state >> 1) | (xor << (self.n_bits - 1))
        return self.state

    def generate_sequence(self, length):
        return [self.step() for _ in range(length)]

# Global instance for worker to access (simplification for example)
svm_with_lfsr_instance = None

def lfsr_worker(prototype):
    score = svm_with_lfsr_instance.svm_fitness(prototype)
    return prototype, score

class SVMWithLFSR:
    def __init__(self, svm_model, vectorizer, poly_features, lfsr):
        self.svm_model = svm_model
        self.vectorizer = vectorizer
        self.poly_features = poly_features
        self.lfsr = lfsr

    def svm_fitness(self, candidate):
        candidate_vector = self.vectorizer.transform([candidate]).toarray()
        high_dim_vector = self.poly_features.transform(candidate_vector)

        decision_scores = self.svm_model.decision_function(high_dim_vector)

        # Fitness is based on the magnitude of the decision score
        return np.max(decision_scores)

    def mutate_candidate(self, candidate):
        mutation_sequence = self.lfsr.generate_sequence(len(candidate))
        mutated_candidate = list(candidate)
        for i in range(len(mutated_candidate)):
            if mutation_sequence[i] % 2 == 1:  # Use LFSR to decide mutation
                mutated_candidate[i] = chr((ord(mutated_candidate[i]) + mutation_sequence[i]) % 127)
        return ''.join(mutated_candidate)

    def crossover_candidates(self, parent1, parent2):
        # Use the shorter length to avoid IndexError
        length = min(len(parent1), len(parent2))
        crossover_sequence = self.lfsr.generate_sequence(length)
        child1 = list(parent1)
        child2 = list(parent2)
        for i in range(length):
            if crossover_sequence[i] % 2 == 1:  # Use LFSR to decide crossover points
                child1[i], child2[i] = child2[i], child1[i]
        return ''.join(child1), ''.join(child2)

    def evolutionary_strategy(self, prototypes, generations=10, population_size=50):
        global svm_with_lfsr_instance
        svm_with_lfsr_instance = self

        population = prototypes * (population_size // len(prototypes) + 1)
        population = population[:population_size]

        results = []
        for _ in range(generations):
            with Pool(processes=min(cpu_count(), len(population))) as pool:
                results = pool.map(lfsr_worker, population)

            # Select top 50% based on fitness
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            selected_candidates = [result[0] for result in sorted_results[:population_size // 2]]

            # Crossover and Mutation using LFSR sequences
            new_candidates = []
            for i in range(len(selected_candidates) // 2):
                parent1 = selected_candidates[i]
                parent2 = selected_candidates[-i-1]
                child1, child2 = self.crossover_candidates(parent1, parent2)
                new_candidates.extend([self.mutate_candidate(child1), self.mutate_candidate(child2)])

            population = selected_candidates + new_candidates

        # Return the best candidate based on fitness
        if not results:
             with Pool(processes=min(cpu_count(), len(population))) as pool:
                results = pool.map(lfsr_worker, population)
        best_prototype, _ = sorted(results, key=lambda x: x[1], reverse=True)[0]
        return best_prototype

def extract_function_prototypes(code):
    pattern = r'\b[\w\*\&]+\s+[\w\*\&]+\s*\([^)]*\)\s*;'
    return re.findall(pattern, code)

def main():
    # Example labeled dataset
    cpp_code_snippets_dataset = [
        ("int add(int a, int b);", "math"),
        ("void print(const std::string& message);", "io"),
        ("double power(double base, int exponent);", "math"),
        ("int main() { int x = 10; return x; }", "main"),
        ("class MyClass { public: void myFunction(); };", "class_method"),
        ("void logError(const std::string& error);", "io"),
        ("double sqrt(double x);", "math"),
        ("void init();", "init"),
    ]

    # 1. Extract function prototypes and their labels.
    function_prototypes = [snippet[0] for snippet in cpp_code_snippets_dataset]
    labels = [snippet[1] for snippet in cpp_code_snippets_dataset]

    # 2. Vectorize the function prototypes.
    vectorizer = TfidfVectorizer()
    prototype_vectors = vectorizer.fit_transform(function_prototypes).toarray()

    # 3. Apply polynomial kernel to enhance dimensionality.
    polynomial_features = PolynomialFeatures(degree=3, interaction_only=False)
    high_dim_vectors = polynomial_features.fit_transform(prototype_vectors)

    # 5. Train SVM model
    svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    svm_classifier.fit(high_dim_vectors, labels)

    # 6. Evaluate the model on training data
    y_pred = svm_classifier.predict(high_dim_vectors)
    print("Classification Report (Training Data):")
    print(classification_report(labels, y_pred, zero_division=0))

    # 7. Initialize the LFSR
    lfsr = LFSR(seed=0b10111001, taps=[7, 5, 4, 3])  # Example taps

    # 8. Instantiate the SVM with LFSR Class
    svm_with_lfsr = SVMWithLFSR(svm_model=svm_classifier.named_steps['svc'], vectorizer=vectorizer, poly_features=polynomial_features, lfsr=lfsr)

    # 9. Example Evolutionary Strategy Application
    if not function_prototypes:
        print("No function prototypes found in the provided code snippets.")
    else:
        best_prototype = svm_with_lfsr.evolutionary_strategy(function_prototypes)
        print(f"Best Function Prototype based on SVM with LFSR: '{best_prototype}'")

if __name__ == "__main__":
    main()
