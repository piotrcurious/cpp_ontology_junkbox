
import re
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from multiprocessing import Pool, cpu_count

# Ignore runtime warnings for correlation calculation with zero variance
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 0. Function to extract C++ function prototypes using regular expressions.
def extract_function_prototypes(code):
    pattern = r'\b[\w\*\&]+\s+[\w\*\&]+\s*\([^)]*\)\s*;'
    return re.findall(pattern, code)

# 1. Global objects for parallel workers to access (for demonstration).
vectorizer = TfidfVectorizer()
polynomial_features = PolynomialFeatures(degree=3, interaction_only=False)
svm_classifier_global = None

# 8. SVM Fitness Function using Correlated Dimensions
def svm_fitness(candidate):
    candidate_vector = vectorizer.transform([candidate]).toarray()
    high_dim_vector = polynomial_features.transform(candidate_vector)

    # Use the pipeline to transform and then predict decision function
    decision_scores = svm_classifier_global.decision_function(high_dim_vector)

    # Fitness is based on the magnitude of the decision score
    return np.max(decision_scores)

# 9. Evolutionary Strategy Worker (defined at top-level for pickling)
def evolutionary_worker(prototype):
    score = svm_fitness(prototype)
    return prototype, score

# 10. Evolutionary Strategy leveraging SVM with Correlation
def evolutionary_strategy_svm(prototypes, generations=10, population_size=50):
    population = prototypes * (population_size // len(prototypes) + 1)
    population = population[:population_size]

    for _ in range(generations):
        with Pool(processes=min(cpu_count(), population_size)) as pool:
            results = pool.map(evolutionary_worker, population)

        # Select top 50% based on fitness
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        selected_candidates = [result[0] for result in sorted_results[:population_size // 2]]

        # Crossover and Mutation
        new_candidates = []
        for i in range(len(selected_candidates) // 2):
            parent1 = selected_candidates[i]
            parent2 = selected_candidates[-i-1]
            crossover_point = len(parent1) // 2
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
    best_prototype, _ = sorted(results, key=lambda x: x[1], reverse=True)[0]
    return best_prototype

# 7. Correlate extra dimensions with the SVM decision function
def correlate_dimensions_with_svm(model, X, labels):
    decision_function = model.decision_function(X)
    # Handle multi-class decision function (n_samples, n_classes * (n_classes-1) / 2) or (n_samples, n_classes)
    # For SVC linear with n_classes > 2, it's typically (n_samples, n_classes)
    correlations = np.corrcoef(X.T, decision_function.T)[:X.shape[1], X.shape[1]:]

    # Use actual number of classes for argsort
    n_output_dims = correlations.shape[1]
    most_correlated_dims = np.argsort(-np.abs(np.nan_to_num(correlations)), axis=0)[:3, :]
    return most_correlated_dims, correlations

def main():
    global svm_classifier_global
    # Example labeled dataset
    cpp_code_snippets = [
        ("int add(int a, int b);", "math"),
        ("void print(const std::string& message);", "io"),
        ("double power(double base, int exponent);", "math"),
        ("int main() { int x = 10; return x; }", "main"),
        ("class MyClass { public: void myFunction(); };", "class_method"),
        ("void logError(const std::string& error);", "io"),
        ("double sqrt(double x);", "math"),
        ("void init();", "init"),
    ]

    # Extract function prototypes and their labels.
    function_prototypes = [snippet[0] for snippet in cpp_code_snippets]
    labels = [snippet[1] for snippet in cpp_code_snippets]

    # Vectorize the function prototypes.
    prototype_vectors = vectorizer.fit_transform(function_prototypes).toarray()

    # Apply polynomial kernel to enhance dimensionality.
    high_dim_vectors = polynomial_features.fit_transform(prototype_vectors)

    # Train SVM model
    svm_classifier_global = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    svm_classifier_global.fit(high_dim_vectors, labels)

    # Evaluate the model
    y_pred = svm_classifier_global.predict(high_dim_vectors)
    print("Classification Report (on training data for demonstration):")
    print(classification_report(labels, y_pred, zero_division=0))

    # Correlate training data dimensions with SVM decision function
    correlated_dims, correlations = correlate_dimensions_with_svm(svm_classifier_global, high_dim_vectors, labels)

    print("\nTop correlated dimensions with SVM decision function:")
    unique_labels = svm_classifier_global.classes_
    # The decision function for multi-class linear SVM might have different shape depending on ovo/ovr
    # sklearn SVC uses OVO by default, but decision_function can be ovr.
    # For SVC, n_classes > 2, it's (n_samples, n_classes) if decision_function_shape='ovr'

    n_output_cols = correlations.shape[1]
    for i in range(min(len(unique_labels), n_output_cols)):
        label = unique_labels[i]
        dims = correlated_dims[:, i]
        corrs = correlations[dims, i]
        print(f"Class/Decision Dimension '{label}': Dimensions {dims} with correlations {corrs}")

    # Example C++ code snippets for evolution.
    cpp_code_snippets_raw = [
        "int add(int a, int b);",
        "void print(const std::string& message);",
        "double power(double base, int exponent);",
        "int main() { int x = 10; return x; }",
        "class MyClass { public: void myFunction(); };"
    ]

    # Extracting function prototypes from the code snippets.
    function_prototypes_evolve = []
    for code in cpp_code_snippets_raw:
        prototypes = extract_function_prototypes(code)
        function_prototypes_evolve.extend(prototypes)

    # Apply the parallel evolutionary strategy with SVM fitness function.
    if not function_prototypes_evolve:
        print("No function prototypes found in the provided code snippets.")
    else:
        best_prototype = evolutionary_strategy_svm(function_prototypes_evolve)
        print(f"\nBest Function Prototype based on SVM: '{best_prototype}'")

if __name__ == "__main__":
    main()
