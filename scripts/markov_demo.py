import re
import math
from collections import defaultdict, Counter

# 1. Define a simple ontology for C++ code structures with corresponding example sequences.
ontology = {
    'class': "class { public: private: protected: };",
    'function': "return type name() { statements }",
    'loop': "for() {} while() {} do {} while();",
    'conditional': "if() {} else {} switch() {} case:",
    'variable_declaration': "type name = value;",
}

# 2. Tokenize the ontology examples to create Markov chains.
def tokenize(text):
    # Simple tokenization based on splitting by whitespace and punctuation.
    return re.findall(r'\b\w+\b|[{}();]', text)

def build_markov_chain(tokens, order=1):
    # Build a Markov chain of the given order from a sequence of tokens.
    markov_chain = defaultdict(Counter)
    for i in range(len(tokens) - order):
        state = tuple(tokens[i:i + order])
        next_token = tokens[i + order]
        markov_chain[state][next_token] += 1

    # Normalize the transition counts to probabilities
    normalized_chain = {}
    for state, transitions in markov_chain.items():
        total = sum(transitions.values())
        normalized_chain[state] = {token: count / total for token, count in transitions.items()}
    return normalized_chain

# 3. Define function to calculate the likelihood of a sequence matching a Markov chain.
def calculate_likelihood(chain, tokens, order=1):
    log_likelihood = 0
    # Small probability for unseen transitions to avoid log(0)
    epsilon = 0.0001
    for i in range(len(tokens) - order):
        state = tuple(tokens[i:i + order])
        next_token = tokens[i + order]
        if state in chain and next_token in chain[state]:
            log_likelihood += math.log(chain[state][next_token])
        else:
            log_likelihood += math.log(epsilon)
    return log_likelihood

def main():
    ontology_chains = {}
    for structure, example in ontology.items():
        tokens = tokenize(example)
        ontology_chains[structure] = build_markov_chain(tokens)

    # 4. C++ code snippets representing different structures.
    cpp_code_snippets = [
        "class MyClass { public: void myFunction() {} };",  # Class definition
        "int main() { int x = 0; for(int i = 0; i < 10; i++) { x += i; } return x; }",  # Loop in main function
        "if (x > 0) { y = 1; } else { y = -1; }",  # Conditional
        "int x = 10;",  # Variable declaration
        "double square(double num) { return num * num; }"  # Function definition
    ]

    # 5. Identify the structure of each C++ code snippet using the Markov chains.
    for code_snippet in cpp_code_snippets:
        tokens = tokenize(code_snippet)
        likelihoods = {}

        for structure, chain in ontology_chains.items():
            likelihood = calculate_likelihood(chain, tokens)
            likelihoods[structure] = likelihood

        identified_structure = max(likelihoods, key=likelihoods.get)
        print(f"Code Snippet: '{code_snippet}'\nIdentified as: {identified_structure}\n")

if __name__ == "__main__":
    main()
