import os
import sys
import argparse
import numpy as np
import re
import json
from tqdm import tqdm
from typing import Pattern

from tree_sitter import Parser, Language
import tree_sitter_java as tsjava
import graphviz

# Import NeuroX transformers extractor
import neurox.data.extraction.transformers_extractor as transformers_extractor

# Initialize the Java language
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)
# parser.set_language(JAVA_LANGUAGE)

def read_source_code(file_path: str) -> str:
    """Reads the source code from the given file path."""
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    return source_code

def extract_leaf_tokens(node, depth=0):
    """Recursively extracts tokens from the leaf nodes of the AST, including depth."""
    tokens = []
    if len(node.children) == 0:
        tokens.append((node.type, node.text.decode('utf-8'), depth))
    else:
        for child in node.children:
            tokens.extend(extract_leaf_tokens(child, depth + 1))
    return tokens

def visualize_ast(node, graph, parent_id=None):
    """Recursively builds the AST visualization."""
    node_id = str(id(node))
    label = f"{node.type} [{node.start_point}-{node.end_point}]"
    graph.node(node_id, label)
    if parent_id:
        graph.edge(parent_id, node_id)
    for child in node.children:
        visualize_ast(child, graph, node_id)

def _create_binary_data(tokens_with_depth, activations, binary_filter, balance_data=False):
    """Creates a binary labeled dataset based on the binary_filter."""
    if isinstance(binary_filter, set):
        filter_fn = lambda x: x in binary_filter
    elif isinstance(binary_filter, Pattern):
        filter_fn = lambda x: binary_filter.match(x)
    elif callable(binary_filter):
        filter_fn = binary_filter
    else:
        raise NotImplementedError("ERROR: The binary_filter must be a set, a regex pattern, or a callable function.")

    positive_class_data = []
    negative_class_data = []

    for (token, depth), activation in zip(tokens_with_depth, activations):
        if filter_fn(token):
            positive_class_data.append(((token, depth), activation))
        else:
            negative_class_data.append(((token, depth), activation))

    if len(negative_class_data) == 0 or len(positive_class_data) == 0:
        raise ValueError("ERROR: Positive or Negative class examples are zero")

    # Combine positive and negative data
    combined_data = positive_class_data + negative_class_data

    # If balancing is required
    if balance_data:
        # Implement balancing if needed
        pass  # Skipping balancing for simplicity

    words = [t for (t, _), _ in combined_data]
    depths = [d for (_, d), _ in combined_data]
    labels = ['positive'] * len(positive_class_data) + ['negative'] * len(negative_class_data)
    activations = [a for _, a in combined_data]

    return list(zip(words, depths)), labels, activations

def annotate_data(tokens_with_depth, activations, binary_filter, output_prefix, balance_data=False):
    """Creates binary data and saves it."""
    tokens_depths, labels, activations = _create_binary_data(tokens_with_depth, activations, binary_filter, balance_data=balance_data)

    # Prepare output directory at the same level as src
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the files
    words_file = os.path.join(output_dir, f"{output_prefix}_tokens.txt")
    labels_file = os.path.join(output_dir, f"{output_prefix}_labels.txt")
    activations_file = os.path.join(output_dir, f"{output_prefix}_activations.txt")

    with open(words_file, "w", encoding='utf-8') as f_words, open(labels_file, "w", encoding='utf-8') as f_labels:
        current_depth = -1
        for (word, depth), label in zip(tokens_depths, labels):
            # Start a new line if depth decreases
            if depth <= current_depth:
                f_words.write('\n')
                f_labels.write('\n')
            current_depth = depth
            # Write token and label with a space separator
            f_words.write(f"{word} ")
            f_labels.write(f"{label} ")

    # Save activations to text file (one per line)
    with open(activations_file, 'w', encoding='utf-8') as f:
        for activation_vector in activations:
            if isinstance(activation_vector, np.ndarray):
                activation_vector = activation_vector.tolist()
            activation_str = ' '.join(map(str, activation_vector))
            f.write(activation_str + '\n')

    print(f"Words saved to '{words_file}'.")
    print(f"Labels saved to '{labels_file}'.")
    print(f"Activations saved to '{activations_file}'.")

def main():
    # Set up argument parser
    parser_arg = argparse.ArgumentParser(description='Parse Java code and generate AST visualization and activations.')
    parser_arg.add_argument('file', help='Path to the Java source file.')
    parser_arg.add_argument('--model', default='bert-base-uncased', help='Transformer model to use for activations.')
    parser_arg.add_argument('--device', default='cpu', help='Device to run the model on ("cpu" or "cuda").')
    parser_arg.add_argument('--binary_filter', default='set:public,static', help='Binary filter for labeling.')
    parser_arg.add_argument('--output_prefix', default='output', help='Prefix for output files.')
    args = parser_arg.parse_args()

    java_file_path = args.file

    if not os.path.isfile(java_file_path):
        print(f"Error: File '{java_file_path}' does not exist.")
        sys.exit(1)

    # Read the source code from the file
    source_code = read_source_code(java_file_path)

    # Parse the source code
    tree = parser.parse(source_code.encode('utf-8'))
    root_node = tree.root_node

    # Extract tokens from leaf nodes, including depth
    tokens_tuples = extract_leaf_tokens(root_node)
    tokens = [token_text for _, token_text, _ in tokens_tuples]

    # Print tokens, types, and depths
    print("Extracted Tokens:")
    for token_type, token_text, depth in tokens_tuples:
        print(f"Type: {token_type}, Text: {token_text}, Depth: {depth}")

    # Prepare output directory at the same level as src
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write tokens to input file
    input_file = os.path.join(output_dir, 'input_sentences.txt')
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(tokens) + '\n')

    # Generate activations using NeuroX transformers_extractor
    model = args.model
    output_file = os.path.join(output_dir, 'activations.json')

    transformers_extractor.extract_representations(
        model,
        input_file,
        output_file,
        aggregation="average",
        output_type="json"
    )

    # Read activations from the output file
    with open(output_file, 'r', encoding='utf-8') as f:
        activation_data = json.load(f)

    # Adjusted code to parse the new format
    activations = []
    extracted_tokens = []

    for feature in activation_data['features']:
        token = feature['token'].replace('Ġ', '')
        layers = feature['layers']
        activation_values = []
        for layer in layers:
            activation_values.extend(layer['values'])
        activations.append(np.array(activation_values))
        extracted_tokens.append(token)

    # Handle binary filter input
    binary_filter = args.binary_filter
    if binary_filter.startswith("re:"):
        binary_filter = re.compile(binary_filter[3:])
    elif binary_filter.startswith("set:"):
        binary_filter = set(binary_filter[4:].split(","))
    else:
        raise NotImplementedError("Filter must start with 're:' for regex or 'set:' for a set of words.")

    # Annotate the data
    tokens_with_depth = [(t, d) for (_, t, d) in tokens_tuples]
    annotate_data(tokens_with_depth, activations, binary_filter, args.output_prefix, balance_data=True)

    # Visualize the AST
    graph = graphviz.Digraph(format='png')
    visualize_ast(root_node, graph)

    # Create an output filename based on input file name
    input_filename = os.path.splitext(os.path.basename(java_file_path))[0]
    ast_output_path = os.path.join(output_dir, f'{input_filename}_ast')
    graph.render(ast_output_path, format='png', cleanup=True)

    print(f"\nAST visualization saved as '{ast_output_path}.png'.")

if __name__ == "__main__":
    main()
