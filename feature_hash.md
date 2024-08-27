from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Define a simple ontology for C++ code structures.
ontology = {
    'class': "Defines a class with member functions and attributes.",
    'function': "Defines a function that can perform operations or return a value.",
    'loop': "Defines a loop structure such as for, while, or do-while.",
    'conditional': "Defines a conditional structure such as if-else or switch-case.",
    'variable_declaration': "Declares a variable with a specific type.",
}

# 2. C++ code snippets representing different structures.
cpp_code_snippets = [
    "class MyClass { public: void myFunction() {} };",    # Class definition
    "int main() { int x = 0; for(int i = 0; i < 10; i++) { x += i; } return x; }",  # Loop in main function
    "if (x > 0) { y = 1; } else { y = -1; }",   # Conditional
    "int x = 10;",    # Variable declaration
    "double square(double num) { return num * num; }"   # Function definition
]

# 3. Use HashingVectorizer to convert text into a fixed-size hashed vector representation.
vectorizer = HashingVectorizer(n_features=20, norm=None, alternate_sign=False)

# Combine all text for vectorization
ontology_descriptions = list(ontology.values())

# Vectorize the ontology and C++ code snippets.
ontology_vectors = vectorizer.fit_transform(ontology_descriptions)
cpp_code_vectors = vectorizer.transform(cpp_code_snippets)

# 4. Function to identify the closest C++ structure based on the hashed vectors using cosine similarity
def identify_cpp_structure(code_vector):
    similarities = cosine_similarity(code_vector, ontology_vectors)
    most_similar_index = np.argmax(similarities)
    return list(ontology.keys())[most_similar_index]

# 5. Identify and categorize each C++ code snippet
for idx, code_snippet in enumerate(cpp_code_snippets):
    code_vector = cpp_code_vectors[idx]
    identified_structure = identify_cpp_structure(code_vector)
    print(f"Code Snippet: '{code_snippet}'\nIdentified as: {identified_structure}\n")
