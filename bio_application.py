import re
from typing import List, Tuple
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import pandas as pd


cases = {
    'camel': '^[a-z]+([A-Z][a-z]+)+$',
    'snake': '^[a-z]+(_[a-z]+)*$',
    'screaming_snake': '^[A-Z]+(_[A-Z]+)*$',
    'prefix': '^(get|set)[A-Za-z]+$',
    'numeric': '^[a-zA-Z].+[0-9]+$'
}


def check_token(token, regex):
    pattern = cases[regex]
    return bool(re.match(pattern, token))


def extract_tokens_with_bio(node, label=None) -> Tuple[List[str], List[str]]:
    tokens = []
    bio_labels = []

    def recurse(node, current_label, space=0):
        if node.is_named:
            token_text = node.text.decode('utf-8')
            token_list = re.findall(r'\w+|[^\s\w]', token_text)  # Splits into words and symbols

            for i, token in enumerate(token_list):
                tokens.append(token)
                if i == 0:
                    bio_labels.append(f"{space}-B-{current_label}" if current_label else f"{space}-O")
                else:
                    bio_labels.append(f"{space}-I-{current_label}" if current_label else f"{space}-O")

        for child in node.children:
            s = space + 1
            if node.is_named:
                recurse(child, node.type, s)
            else:
                recurse(child, current_label, s)

    recurse(node, label, 0)
    return tokens, bio_labels


def main():
    # Load Tree-sitter Java language
    JAVA_LANGUAGE = Language(tsjava.language())
    parser = Parser(JAVA_LANGUAGE)

    # Source code to be parsed (example Java code)
    source_code = b'''
    public class HelloWorld {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }
    '''

    # Parse the source code
    tree = parser.parse(source_code)
    root_node = tree.root_node

    # First iteration of BIO labeling
    tokens_1, bio_labels_1 = extract_tokens_with_bio(root_node)

    # Second iteration of BIO labeling (using the same function to simulate a second pass)
    tokens_2, bio_labels_2 = extract_tokens_with_bio(root_node)

    # Create a DataFrame to show the tokens and their BIO labels for both iterations
    data = {
        "Token": tokens_1,
        "First Iteration BIO Label": bio_labels_1,
        "Second Iteration BIO Label": bio_labels_2,
    }

    df = pd.DataFrame(data)
    # print(df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


if __name__ == "__main__":
    main()
