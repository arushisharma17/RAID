import re
from typing import List
from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import pandas as pd

cases = {
    'single_letter': '^[a-zA-Z]$',
    'camel_case': '^[a-z][a-z]*(?:[A-Z][a-z0-9]+)*[a-zA-Z]?$',
    'pascal_case': '^([A-Z][a-z]+)*[A-Z][a-z]*$',
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


def get_nodes_at_level(node, target_level) -> List[Node]:
    """
    Extracts nodes at a given number of levels down.

    Parameters
    ----------
    node : tree_sitter.Node
        The root node of the AST.
    target_level : int
        The desired depth to retrieve.

    Returns
    -------
    List[Node]
        A list of knows at the target level down.
    """

    def retrieve_nodes(node, target_level, current_level=0):
        nodes_at_level = []

        # if level is beyond tree, appends the leaf
        if current_level == target_level or node.child_count == 0:
            nodes_at_level.append(node)
        else:
            for child in node.children:
                nodes_at_level.extend(retrieve_nodes(child, target_level, current_level + 1))

        return nodes_at_level

    return retrieve_nodes(node, target_level)


def find_bio_label_type(node) -> str:
    if (node.type == str(node.text)[2:-1] or node.type == 'identifier') and node.parent is not None:
        return node.parent.type
    return node.type


def extract_bio_labels_from_source_code(source_code, language, depth=-1):
    """
    Parses the source code, then generates and displays the separate tokens and labels

    Parameters
    ----------
    source_code : bytes
        The code snippet to be parsed.
    language : str
        The language in which the code snippet should be parsed.
    depth : int, optional
        The desired depth to retrieve from the tree; the default value retrieves leaf nodes.
    """
    if language == 'java':
        code_language = Language(tsjava.language())
    elif language == 'python':
        code_language = Language(tspython.language())
    else:
        print("Please pick Java or Python as a language.")
        return
    parser = Parser(code_language)

    tree = parser.parse(source_code)
    root_node = tree.root_node

    leaf_nodes = get_nodes_at_level(root_node, depth)
    leaf_labels = []
    bio = []
    prev = None
    o_type = None
    for i, node in enumerate(leaf_nodes):
        name = find_bio_label_type(node)
        leaf_text = str(node.text)[2:-1]
        leaf_labels.append(find_label_with_regex(leaf_text) if node.type == 'identifier' else 'N/A')

        if node.type == leaf_text and i > 0 and (prev != name or o_type is None):
            if ((language == 'python' and node.parent.child_count == 2)
                    or (language == 'java' and node.parent.child_count == 3 and node.parent.child(2).type == ';')):
                bio.append(leaf_text + ": B" + '-' + name)
            else:
                bio.append(leaf_text + ": O" + '-' + name)
            prev = None
            o_type = name
        else:
            bio_type = 'B' if prev != name else 'I'
            bio.append(leaf_text + ": " + bio_type + '-' + name)
            prev = name
            o_type = None

    token_data = []
    label_data = []
    for element in bio:
        split_element = element.split(": ")
        token_data.append(split_element[0])
        label_data.append(split_element[1])

    data = {"TOKEN": token_data,
            "LABEL": label_data,
            "REGEX": leaf_labels}

    df = pd.DataFrame(data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


def main():
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
    # # '''

    # source_code = b'''
    # def add_numbers (a, b):
    #     return a + b
    # '''

    # source_code = b'''
    # # Comment about function
    # '''

    source_code = b'''
    public int addNumbers(a, b) {
        return a + b;
    }
    '''

    extract_bio_labels_from_source_code(source_code, 'java')


if __name__ == "__main__":
    main()
