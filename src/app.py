import os
import sys
import argparse
from tree_sitter import Parser, Language
import tree_sitter_java as tsjava
import graphviz

# Initialize the Java language
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)

def read_source_code(file_path: str) -> bytes:
    """
    Reads the source code from the given file path.

    Args:
        file_path (str): Path to the source code file.

    Returns:
        bytes: The source code encoded in UTF-8.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()
    return source_code.encode('utf-8')

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

def main():
    # Set up argument parser
    parser_arg = argparse.ArgumentParser(description='Parse Java code and generate AST visualization.')
    parser_arg.add_argument('file', help='Path to the Java source file.')
    args = parser_arg.parse_args()

    java_file_path = args.file

    if not os.path.isfile(java_file_path):
        print(f"Error: File '{java_file_path}' does not exist.")
        sys.exit(1)

    # Read the source code from the file
    source_code = read_source_code(java_file_path)

    # Parse the source code
    tree = parser.parse(source_code)
    root_node = tree.root_node

    # Extract tokens from leaf nodes
    tokens = extract_leaf_tokens(root_node)

    # Print tokens and labels
    print("Extracted Tokens and Labels:")
    for token_type, token_text in tokens:
        print(f"Type: {token_type}, Text: {token_text}")

    # Prepare output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, '..', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write tokens to a file
    tokens_file_path = os.path.join(output_dir, 'tokens.txt')
    with open(tokens_file_path, 'w', encoding='utf-8') as token_file:
        for _, token_text in tokens:
            token_file.write(f"{token_text}\n")

    # Write labels to a file
    labels_file_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_file_path, 'w', encoding='utf-8') as label_file:
        for token_type, _ in tokens:
            label_file.write(f"{token_type}\n")

    print(f"\nTokens saved to '{tokens_file_path}'.")
    print(f"Labels saved to '{labels_file_path}'.")

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
