import re
from typing import List, Tuple
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import pandas as pd
import graphviz


cases = {
    'single_letter': '^[a-zA-Z]$',
    'camel_case': '^[a-z]+([A-Z][a-z]+)+$',
    'snake_case': '^[a-z]+(_[a-z]+)*$',
    'screaming_snake_case': '^[A-Z]+(_[A-Z]+)*$',
    'prefix': '^(get|set)[A-Za-z]+$',
    'numeric': '^[a-zA-Z].+[0-9]+$',
}


def check_token(token, regex):
    pattern = cases[regex]
    return bool(re.match(pattern, token))


def find_label_with_regex(token):
    for key in cases:
        if check_token(token, key):
            return key
    return 'N/A'


def extract_tokens_with_bio(node, parent_label=None) -> Tuple[List[str], List[str]]:
    """
    Extracts tokens from the AST with enhanced BIO labeling that considers the context of parent nodes.

    Parameters
    ----------
    node : tree_sitter.Node
        The root node of the AST.
    parent_label : str, optional
        The label to be used for the BIO tag, typically derived from the parent node's type.

    Returns
    -------
    Tuple[List[str], List[str]]
        - tokens: A list of tokens extracted from the AST.
        - bio_labels: A list of BIO labels corresponding to each token.
    """
    tokens = []
    bio_labels = []
    leaf_list = []

    def recurse_node(node, leaf_list, parent_label=None):
        if node.is_named:
            token_text = node.text.decode('utf-8')
            if parent_label:
                bio_labels.append(f"B-{parent_label}")
                tokens.append(token_text)
            else:
                bio_labels.append("O")
                tokens.append(token_text)

        # Recursively process child nodes
        for child in node.children:
            if node.is_named:
                # Use the current node's type as the label prefix for its children
                child_label = parent_label or node.type
                if child.is_named and len(child.children) > 0:
                    recurse_node(child, leaf_list, child_label)
                else:
                    child_text = child.text.decode('utf-8')
                    if len(child_text.split()) > 1:
                        # Split into B- and I- for multi-token labels
                        bio_labels.append(f"B-{child_label}")
                        tokens.append(child_text.split()[0])
                        for part in child_text.split()[1:]:
                            bio_labels.append(f"I-{child_label}")
                            tokens.append(part)
                    else:
                        bio_labels.append(f"B-{child_label}")
                        tokens.append(child_text)
                        # leaf_list.append(str(child)[2:-1])
                        leaf_list.append(child)
            else:
                recurse_node(child, leaf_list)

    recurse_node(node, leaf_list, parent_label)
    return tokens, leaf_list, bio_labels


def visualize_ast(node, graph, parent_id=None):
    """
    Visualizes the AST using Graphviz.

    Parameters
    ----------
    node : tree_sitter.Node
        The root node of the AST.
    graph : graphviz.Digraph
        The Graphviz Digraph object used to visualize the AST.
    parent_id : str, optional
        The ID of the parent node, used to create edges between nodes.
    """
    node_id = str(id(node))
    label = f"{node.text}\nnode type {node.type}"# [{node.start_point}-{node.end_point}]"
    graph.node(node_id, label)
    if parent_id:
        graph.edge(parent_id, node_id)
    for child in node.children:
        visualize_ast(child, graph, node_id)


def main():
    # Load Tree-sitter Java language
    # JAVA_LANGUAGE = Language(tsjava.language())
    # parser = Parser(JAVA_LANGUAGE)
    PYTHON_LANGUAGE = Language(tspython.language())
    parser = Parser(PYTHON_LANGUAGE)

    # source_code = b'''
    # public class HelloWorld {
    #     public static void main(String[] args) {
    #         System.out.println("Hello, World!");
    #     }
    # }
    # '''

    # source_code = b'''
    # for (int i = 0; i < 10; i++) {
    #     System.out.println(i);
    # }
    # '''

    source_code = b'''
    def add_numbers (a, b):
        return a + b
    '''

    # source_code = b'''
    # public addNumbers(a, b) {
    #     return a + b;
    # }
    # '''

    # Parse the source code
    tree = parser.parse(source_code)
    root_node = tree.root_node

    tokens, leaves, bio_labels = extract_tokens_with_bio(root_node)

    # Print tokens and corresponding BIO labels
    # print("Extracted Tokens and BIO Labels:")
    # for token, bio in zip(tokens, bio_labels):
    #     print(f"Token: {token}, BIO Label: {bio}")

    # Create a Graphviz Digraph object and visualize the AST
    graph = graphviz.Digraph(format="png")
    visualize_ast(root_node, graph)
    graph.render("java_ast")

    leaf_text = []
    leaf_labels = []
    for l in leaves:
        text = str(l.text)[2:-1]
        leaf_text.append(text)
        leaf_labels.append(find_label_with_regex(text) if l.type == 'identifier' else 'N/A')
    print(leaf_text)
    print(leaf_labels)


if __name__ == "__main__":
    main()
