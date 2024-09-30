import re
from typing import List, Tuple
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import pandas as pd
import graphviz


def get_leaf_nodes(node):
    leaf_nodes = []

    # Base case: If the node has no children, it is a leaf node
    if len(node.children) == 0:
        leaf_nodes.append(node)

    # Recursively check the children for leaf nodes
    for child in node.children:
        leaf_nodes.extend(get_leaf_nodes(child))

    return leaf_nodes


def find_bio_label_type(l):
    if (l.type == str(l.text)[2:-1] or l.type == "identifier") and l.parent is not None:
        return l.parent.type
    return l.type


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
    # # Comment about function
    # '''

    # source_code = b'''
    # public addNumbers(a, b) {
    #     return a + b;
    # }
    # '''

    # Parse the source code
    tree = parser.parse(source_code)
    root_node = tree.root_node
    leaf_nodes = get_leaf_nodes(root_node)
    bio = []
    prev = None
    O_type = None
    for i, l in enumerate(leaf_nodes):
        name = find_bio_label_type(l)
        leaf_text = str(l.text)[2:-1]
        # print("Leaf:", leaf_text, "\tName:", name, "\tType:", l.type, "\tParent Type:", l.parent.type, "Prev:", prev)
        # _, b, _ = extract_tokens_with_bio(l)
        if l.type == leaf_text and i > 0 and (prev != name or O_type is None):
            if l.parent.child_count == 2:
                bio.append(leaf_text + ': B' + '-' + name)
            else:
                bio.append(leaf_text + ': O' + '-' + name)
            prev = None
            O_type = name
        else:
            bio_type = 'B' if prev != name else 'I'
            bio.append(leaf_text + ': ' + bio_type + '-' + name)
            prev = name
            O_type = None


    print(bio)

    # tokens, bio_labels, leaf_labels = extract_tokens_with_bio(root_node)

    # print(tokens[:-(len(leaf_nodes))], "\n", bio_labels[:-(len(leaf_nodes))])

    # Print tokens and corresponding BIO labels
    # print("\nExtracted Tokens and BIO Labels:")
    # for token, bio in zip(tokens, bio_labels):
    #     # print(f"Token: {token}, BIO Label: {bio}")
    #     print(f"{token}, {bio}")
    # print(leaf_labels)


if __name__ == "__main__":
    main()
