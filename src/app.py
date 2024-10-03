import os
import sys
import argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import Pattern
import re


from tree_sitter import Parser, Language
import tree_sitter_java as tsjava
import graphviz

# Initialize the Java language
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

def read_source_code(file_path: str) -> str:
    """
    Reads the source code from the given file path.

    Args:
        file_path (str): Path to the source code file.

    Returns:
        str: The source code as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    return source_code

def extract_leaf_tokens(node):
    """
    Recursively extracts tokens from the leaf nodes of the AST.

    Args:
        node: The current node in the AST.

    Returns:
        list: A list of tuples containing token types and their text.
    """
    tokens = []
    if len(node.children) == 0:  # If the node is a leaf node
        tokens.append((node.type, node.text.decode('utf-8')))
    else:
        for child in node.children:
            tokens.extend(extract_leaf_tokens(child))
    return tokens

def visualize_ast(node, graph, parent_id=None):
    """
    Recursively builds the AST visualization.

    Args:
        node: The current node in the AST.
        graph: The Graphviz graph object.
        parent_id: The identifier of the parent node.
    """
    node_id = str(id(node))
    label = f"{node.type} [{node.start_point}-{node.end_point}]"
    graph.node(node_id, label)
    if parent_id:
        graph.edge(parent_id, node_id)
    for child in node.children:
        visualize_ast(child, graph, node_id)

def generate_activations(tokens, model_name='bert-base-uncased', device='cpu'):
    """
    Generates activations using a transformer model based on the tokens.

    Args:
        tokens (List[List[str]]): A list of lists containing tokens from the source code.
        model_name (str): The name of the transformer model to use.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        activations (List[List[List[float]]]): A list of activations corresponding to the tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    activations = []

    for sentence_tokens in tqdm(tokens, desc="Generating activations"):
        sentence = ' '.join(sentence_tokens)
        inputs = tokenizer(sentence, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of layers

        # Aggregate activations for each token
        token_activations = []
        for idx in range(len(sentence_tokens)):
            # We take the activations from the last layer for simplicity
            # You can modify this to include multiple layers or different aggregation
            token_activation = hidden_states[-1][0, idx + 1, :].cpu().numpy()  # +1 to skip [CLS] token
            token_activations.append(token_activation.tolist())
        activations.append(token_activations)

    return activations

def _create_binary_data(tokens, activations, binary_filter, balance_data=False):
    """
    Given a list of tokens, their activations, and a binary_filter, create a binary labeled dataset.
    A binary filter can be a set, regex, or a function.

    Args:
        tokens (List[List[str]]): A list of lists containing tokens from the source code.
        activations (List[List[List[float]]]): A list of lists containing activation vectors corresponding to the tokens.
        binary_filter (set or Pattern or callable): Defines the positive class.
        balance_data (bool): Whether to balance the dataset.

    Returns:
        Tuple[List[str], List[str], List[List[float]]]: Words, labels, and activations.
    """
    if isinstance(binary_filter, set):
        filter_fn = lambda x: x in binary_filter
    elif isinstance(binary_filter, Pattern):
        filter_fn = lambda x: binary_filter.match(x)
    elif callable(binary_filter):
        filter_fn = binary_filter
    else:
        raise NotImplementedError("ERROR: The binary_filter must be a set, a regex pattern, or a callable function.")

    positive_class_words = []
    positive_class_activations = []
    negative_class_words = []
    negative_class_activations = []

    print("Creating binary dataset ...")
    for s_idx, sentence_tokens in enumerate(tokens):
        for w_idx, word in enumerate(sentence_tokens):
            if filter_fn(word):
                positive_class_words.append(word)
                positive_class_activations.append(activations[s_idx][w_idx])
            else:
                negative_class_words.append(word)
                negative_class_activations.append(activations[s_idx][w_idx])

    if len(negative_class_words) == 0 or len(positive_class_words) == 0:
        raise ValueError("ERROR: Positive or Negative class examples are zero")
    elif len(negative_class_words) < len(positive_class_words):
        print("WARNING: The negative class examples are less than the positive class examples")
        print("Positive class examples: ", len(positive_class_words), "Negative class examples: ", len(negative_class_words))

    if balance_data:
        # Implement balancing if needed
        pass  # For now, we skip balancing for simplicity

    print("Number of Positive examples: ", len(positive_class_words))

    words = positive_class_words + negative_class_words
    labels = ['positive'] * len(positive_class_words) + ['negative'] * len(negative_class_words)
    activations = positive_class_activations + negative_class_activations

    return words, labels, activations

def annotate_data(tokens, activations, binary_filter, output_prefix, balance_data=False):
    """
    Given tokens, activations, a binary_filter, and output_prefix, creates binary data and saves it.

    Args:
        tokens (List[List[str]]): Tokens from the source code.
        activations (List[List[List[float]]]): Activation vectors corresponding to the tokens.
        binary_filter (set or Pattern or callable): Defines the positive class.
        output_prefix (str): Prefix for the output files.
        balance_data (bool): Whether to balance the dataset.
    """
    words, labels, activations = _create_binary_data(tokens, activations, binary_filter, balance_data=balance_data)

    # Prepare output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, '..', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the files
    words_file = os.path.join(output_dir, f"{output_prefix}_tokens.txt")
    labels_file = os.path.join(output_dir, f"{output_prefix}_labels.txt")
    activations_file = os.path.join(output_dir, f"{output_prefix}_activations.npy")

    with open(words_file, "w", encoding='utf-8') as f:
        f.write("\n".join(words))

    with open(labels_file, "w", encoding='utf-8') as f:
        f.write("\n".join(labels))

    np.save(activations_file, np.array(activations))

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

    # Extract tokens from leaf nodes
    tokens_tuples = extract_leaf_tokens(root_node)
    tokens = [[token_text for _, token_text in tokens_tuples]]

    # Print tokens and labels
    print("Extracted Tokens:")
    for token_type, token_text in tokens_tuples:
        print(f"Type: {token_type}, Text: {token_text}")

    # Generate activations using a transformer model
    activations = generate_activations(tokens, model_name=args.model, device=args.device)

    # Handle binary filter input
    binary_filter = args.binary_filter
    if binary_filter.startswith("re:"):
        binary_filter = re.compile(binary_filter[3:])
    elif binary_filter.startswith("set:"):
        binary_filter = set(binary_filter[4:].split(","))
    else:
        raise NotImplementedError("Filter must start with 're:' for regex or 'set:' for a set of words.")

    # Annotate the data
    annotate_data(tokens, activations, binary_filter, args.output_prefix, balance_data=True)

    # Visualize the AST
    graph = graphviz.Digraph(format='png')
    visualize_ast(root_node, graph)

    # Create an output filename based on input file name
    input_filename = os.path.splitext(os.path.basename(java_file_path))[0]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, '..', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ast_output_path = os.path.join(output_dir, f'{input_filename}_ast')
    graph.render(ast_output_path, format='png', cleanup=True)

    print(f"\nAST visualization saved as '{ast_output_path}.png'.")

if __name__ == "__main__":
    main()
