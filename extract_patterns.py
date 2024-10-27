import re
from typing import List
from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
import pandas as pd
import json

from label_dictionary import LabelDictionary


class PatternExtractor:

    def __init__(self):

        self.cases = {
            'single_letter': '^[a-zA-Z]$',
            'camel_case': '^[a-z][a-z]*(?:[A-Z][a-z0-9]+)*[a-zA-Z]?$',
            'pascal_case': '^([A-Z][a-z]+)*[A-Z][a-z]*$',
            'snake_case': '^[a-z]+(_[a-z]+)*$',
            'screaming_snake_case': '^[A-Z]+(_[A-Z]+)*$',
            'prefix': '^(get|set)[A-Za-z]+$',
            'numeric': '^[a-zA-Z].+[0-9]+$',
        }




    def check_token(self, token, regex):
        pattern = self.cases[regex]
        return bool(re.match(pattern, token))


    def find_label_with_regex(self, token):
        for key in self.cases:
            if self.check_token(token, key):
                return key
        return 'O'


    def search_for_ancestor_type(self, leaf_node, type_to_search):
        cursor = leaf_node
        while cursor.parent:
            if cursor.type == type_to_search:
                return cursor.type
            else:
                cursor = cursor.parent
        if cursor.type == type_to_search:
            return cursor.type
        return leaf_node.type# if leaf_node.type != str(leaf_node.text)[2:-1] else leaf_node.parent.type




    def get_nodes_at_level(self, node, target_level) -> List[Node]:
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
            A list of nodes at the target level down.
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

        # print('target level:', target_level)
        return retrieve_nodes(node, target_level)


    def find_bio_label_type(self, node) -> str:
        return node.grammar_name


    def extract_bio_labels_from_layer(self, source_code, language, depth=-1):
        """
        Parses the source code, then generates and displays the separate tokens and labels for a specific layer.
        Ignores specific labels.

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

        leaf_nodes = self.get_nodes_at_level(root_node, depth)
        leaf_labels = []
        bio = []
        prev = None
        b_clause = False
        for i, node in enumerate(leaf_nodes):
            name = self.find_bio_label_type(node)
            leaf_text = str(node.text)[2:-1]
            leaf_labels.append(self.find_label_with_regex(leaf_text) if node.type == 'identifier' else 'O')

            if node.child_count == 0 or (node.type == leaf_text and i > 0 and (prev != name)
                    and len(leaf_text) == 1 and not leaf_labels[-1] == 'single_letter'):
                bio.append(leaf_text + ": O-" + name)
                b_clause = False
            elif i > 0 and b_clause and not (node == node.parent.child(0)):
                bio.append(leaf_text + ": I-" + name)
            else:
                bio_type = 'B'
                bio.append(leaf_text + ": " + bio_type + '-' + name)
                prev = name
                b_clause = True

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
        print('\n')

        return token_data, label_data, leaf_labels


    def create_tree_json(self, source_code, language, name):
        """
        Generates JSON in tree format of tokens and labels for source code.

        Parameters
        ----------
        source_code : bytes
            The code snippet to be parsed.
        language : str
            The language in which the code snippet should be parsed.
        name : str
            The desired name of the json file.
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
        label_dictionary = LabelDictionary()

        def tree_to_dict(node):
            # Convert the current node's attributes into a dictionary
            node_dict = {
                'token': str(node.text)[2:-1],
                'label': label_dictionary.convert_label(self.find_bio_label_type(node)),
                'sub_tokens': []
            }

            # Recursively convert each child and append to the 'children' list
            for child in node.children:
                node_dict['sub_tokens'].append(tree_to_dict(child))

            return node_dict


        tree_dict = tree_to_dict(root_node)
        with open(name + '.json', 'w') as json_file:
            json.dump(tree_dict, json_file, indent=4)


    def search_for_type(self, node_type, leaf_nodes, label_dictionary):
        """
        For the given node type, returns list of labels for all leaves.

        Parameters
        ----------
        node_type : str
            The node type (non-leaf) to be parsed.
        leaf_nodes : List[Node]
            The list of leaf nodes.
        label_dictionary : LabelDictionary
            Label dictionary to be used for converting labels to desired format.

        Returns
        -------
        List[str]
            A list of labels with BIO label and parameter for each leaf.
        """
        bio = []
        prev = None
        for node in leaf_nodes:
            label_type = self.search_for_ancestor_type(node, node_type)
            if label_type == node_type:
                bio_label = 'I' if prev == label_type else 'B'
            else:
                bio_label = 'O'
            bio.append(bio_label + '-' + label_dictionary.convert_label(label_type))

            if prev == node_type and prev != label_type:
                bio[-2] = 'O' + bio[-2][1:]  # last label is always O

            prev = label_type

        if prev == node_type:
            bio[-1] = 'O' + bio[-1][1:]  # last label is always O

        return bio


    def get_all_bio_labels(self, source_code, language, file_name):
        """
        For all non-leaf labels, generates BIO labels with parameters for tokens and sends to CSV file.

        Parameters
        ----------
        source_code : bytes
            The source code to be parsed.
        language : str
            Language for the source code to be parsed in.
        file_name : str
            File name for generated CSV.
        """
        if language == 'java':
            code_language = Language(tsjava.language())
        elif language == 'python':
            code_language = Language(tspython.language())
        else:
            print("Please pick Java or Python as a language.")
            return
        parser = Parser(code_language)
        label_dictionary = LabelDictionary()
        tree = parser.parse(source_code)
        root_node = tree.root_node
        leaf_nodes = self.get_nodes_at_level(root_node, -1)

        data = {
            'TOKEN': [str(node.text)[2:-1] for node in leaf_nodes],
            'REGEX': [self.find_label_with_regex(str(node.text)[2:-1]) for node in leaf_nodes]
        }

        for non_leaf_type in label_dictionary.non_leaf_types:
            bio = self.search_for_type(non_leaf_type, leaf_nodes, label_dictionary)
            data[non_leaf_type.upper()] = bio

        df = pd.DataFrame(data)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        print('\n')

        df.to_csv(file_name + '.csv', index=False)


# def main():
#     # could try splitting into an array by '\n', that could make it easier for leaves at least?
#     source_code = b'''
#     public class HelloWorld {
#         public static void main(String[] args) {
#             System.out.println("Hello, World!");
#         }
#     }
#     '''
#
#     # source_code = b'''
#     # for (int i = 0; i < 10; i++) {
#     #     System.out.println(i);
#     # }
#     # # '''
#
#     # source_code = b'''
#     # def add_numbers (a, b):
#     #     return a + b
#     # '''
#
#     # source_code = b'''
#     # # Comment about function
#     # '''
#
#     # source_code = b'''
#     # public int addNumbers(a, b) {
#     #     return a + b;
#     # }
#     # '''
#     e = PatternExtractor()
#     # e.extract_bio_labels_from_source_code(source_code, 'java')
#
#     e.get_all_bio_labels(source_code, 'java')
#     # e.create_tree_json(source_code, 'java', 'test')
#
#
# if __name__ == "__main__":
#     main()
