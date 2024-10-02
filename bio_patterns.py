import re
from typing import List, Tuple
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import pandas as pd
import graphviz


def get_nodes_at_level(node, target_level=-1, current_level=0):
    nodes_at_level = []

    # if level is beyond tree, appends the leaf
    if current_level == target_level or node.child_count == 0:
        nodes_at_level.append(node)
    else:
        for child in node.children:
            nodes_at_level.extend(get_nodes_at_level(child, target_level, current_level + 1))

    return nodes_at_level


def find_bio_label_type(l):
    if (l.type == str(l.text)[2:-1] or l.type == 'identifier') and l.parent is not None:
        return l.parent.type
    return l.type


def bio_label_str(source_code, language):
    # Load Tree-sitter Java language
    if language == 'java':
        code_language = Language(tsjava.language())
    else:
        code_language = Language(tspython.language())
    parser = Parser(code_language)

    # Parse the source code
    tree = parser.parse(source_code)
    root_node = tree.root_node
    leaf_nodes = get_nodes_at_level(root_node)
    bio = []
    prev = None
    o_type = None
    for i, l in enumerate(leaf_nodes):
        name = find_bio_label_type(l)
        leaf_text = str(l.text)[2:-1]
        # print("Leaf:", leaf_text, "\tName:", name, "\tType:", l.type, "\tParent Type:", l.parent.type, "Prev:", prev)
        if l.type == leaf_text and i > 0 and (prev != name or o_type is None):
            if ((language == 'python' and l.parent.child_count == 2)
                    or (language == 'java' and l.parent.child_count == 3 and l.parent.child(2).type == ';')):
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

    print('{0:11}  {1}'.format('TOKEN', 'LABEL'))
    for element in bio:
        split_element = element.split(':')
        print('{0:10}  {1}'.format(split_element[0], split_element[1]))


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

    bio_label_str(source_code, 'java')


if __name__ == "__main__":
    main()
