import os
import sys
import argparse
import numpy as np
import re
import json
from typing import Pattern

from tree_sitter import Parser, Language
import tree_sitter_java as tsjava
import graphviz

# Import NeuroX transformers extractor
import neurox.data.extraction.transformers_extractor as transformers_extractor

# Initialize the Java language
JAVA_LANGUAGE = Language(tsjava.language())

class JavaASTProcessor:
    """Class to handle AST parsing and visualization."""
    def __init__(self, java_file_path, output_dir):
        self.java_file_path = java_file_path
        self.output_dir = output_dir
        self.source_code = None
        self.tree = None
        self.root_node = None
        self.tokens_tuples = None
        self.tokens = None
        self.parser = Parser(JAVA_LANGUAGE)

    def process_ast(self):
        """Reads and parses the source code, extracts tokens, and writes tokens to file."""
        self.read_source_code()
        self.parse_source_code()
        self.tokens_tuples = self.extract_leaf_tokens()
        self.tokens = [token_text for _, token_text, _ in self.tokens_tuples]
        self.write_tokens_to_file()

    def read_source_code(self):
        """Reads the source code from the given file path."""
        with open(self.java_file_path, 'r', encoding='utf-8') as file:
            self.source_code = file.read()

    def parse_source_code(self):
        """Parses the source code into an AST."""
        self.tree = self.parser.parse(self.source_code.encode('utf-8'))
        self.root_node = self.tree.root_node

    def extract_leaf_tokens(self, node=None, depth=0):
        """Recursively extracts tokens from the leaf nodes of the AST, including depth."""
        if node is None:
            node = self.root_node
        tokens = []
        if len(node.children) == 0:
            tokens.append((node.type, node.text.decode('utf-8'), depth))
        else:
            for child in node.children:
                tokens.extend(self.extract_leaf_tokens(child, depth + 1))
        return tokens

    def write_tokens_to_file(self):
        """Writes tokens to input_sentences.txt in the output directory."""
        input_file = os.path.join(self.output_dir, 'input_sentences.txt')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(self.tokens) + '\n')
        print(f"Tokens written to '{input_file}'.")
        # Also, print the tokens, types, and depths
        print("Extracted Tokens:")
        for token_type, token_text, depth in self.tokens_tuples:
            print(f"Type: {token_type}, Text: {token_text}, Depth: {depth}")

    def visualize_ast(self, input_filename):
        """Visualizes the AST and saves it as a PNG file."""
        graph = graphviz.Digraph(format='png')
        self._visualize_ast(self.root_node, graph)
        ast_output_path = os.path.join(self.output_dir, f'{input_filename}_ast')
        graph.render(ast_output_path, format='png', cleanup=True)
        print(f"\nAST visualization saved as '{ast_output_path}.png'.")

    def _visualize_ast(self, node, graph, parent_id=None):
        """Helper method for visualizing the AST."""
        node_id = str(id(node))
        label = f"{node.type} [{node.start_point}-{node.end_point}]"
        graph.node(node_id, label)
        if parent_id:
            graph.edge(parent_id, node_id)
        for child in node.children:
            self._visualize_ast(child, graph, node_id)

class ActivationAnnotator:
    """Class to handle activation generation and data annotation."""
    def __init__(self, model_name, device='cpu', binary_filter='set:public,static', output_prefix='output'):
        self.model_name = model_name
        self.device = device
        self.binary_filter = binary_filter
        self.output_prefix = output_prefix
        self.binary_filter_compiled = None

    def process_activations(self, tokens_tuples, output_dir):
        """Generates activations, parses them, handles binary filter, and annotates data."""
        input_file = os.path.join(output_dir, 'input_sentences.txt')
        output_file = os.path.join(output_dir, 'activations.json')
        # Generate activations
        self.generate_activations(input_file, output_file)
        # Parse activations
        extracted_tokens, activations = self.parse_activations(output_file)
        # Handle binary filter
        self.handle_binary_filter()
        # Prepare tokens with depth
        tokens_with_depth = [(t, d) for (_, t, d) in tokens_tuples]
        # Annotate data
        self.annotate_data(tokens_with_depth, activations, output_dir)
        # Aggregate activations using scalar aggregation
        token_to_scalar_activation = self.aggregate_activations_scalar(tokens_with_depth, activations)
        # Output mapping to file
        self.write_token_activation_mapping(token_to_scalar_activation, output_dir)
        # Keep the mean vector function for future reference (not used by default)
        # token_to_mean_activation = self.aggregate_activations(tokens_with_depth, activations)

    def generate_activations(self, input_file, output_file):
        """Generates activations using the specified transformer model."""
        transformers_extractor.extract_representations(
            self.model_name,
            input_file,
            output_file,
            aggregation="average",
            output_type="json",
            device=self.device
        )

    def parse_activations(self, activation_file):
        """Parses the activations from the JSON output."""
        with open(activation_file, 'r', encoding='utf-8') as f:
            activation_data = json.load(f)
        activations = []
        extracted_tokens = []
        for feature in activation_data['features']:
            token = feature['token'].replace('Ġ', '')
            layers = feature['layers']
            # Only use activations from the last layer
            last_layer_values = layers[-1]['values']
            activations.append(np.array(last_layer_values))
            extracted_tokens.append(token)
        return extracted_tokens, activations

    def handle_binary_filter(self):
        """Compiles the binary filter based on user input."""
        if self.binary_filter.startswith("re:"):
            self.binary_filter_compiled = re.compile(self.binary_filter[3:])
        elif self.binary_filter.startswith("set:"):
            self.binary_filter_compiled = set(self.binary_filter[4:].split(","))
        else:
            raise NotImplementedError("Filter must start with 're:' for regex or 'set:' for a set of words.")

    def annotate_data(self, tokens_with_depth, activations, output_dir):
        """Creates binary data and saves it with tokens and labels organized by AST depth."""
        tokens_depths, labels, activations = self._create_binary_data(
            tokens_with_depth, activations, self.binary_filter_compiled, balance_data=True
        )

        # Save the files
        words_file = os.path.join(output_dir, f"{self.output_prefix}_tokens.txt")
        labels_file = os.path.join(output_dir, f"{self.output_prefix}_labels.txt")
        activations_file = os.path.join(output_dir, f"{self.output_prefix}_activations.txt")

        with open(words_file, "w", encoding='utf-8') as f_words, open(labels_file, "w", encoding='utf-8') as f_labels:
            previous_depth = None
            for (word, depth), label in zip(tokens_depths, labels):
                if previous_depth is None:
                    # First token, no need to write a newline
                    pass
                elif depth > previous_depth:
                    # Depth increased, start a new line
                    f_words.write('\n')
                    f_labels.write('\n')
                # Write token and label with a space separator
                f_words.write(f"{word} ")
                f_labels.write(f"{label} ")
                previous_depth = depth

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


    def _create_binary_data(self, tokens_with_depth, activations, binary_filter, balance_data=False):
        """Creates a binary labeled dataset based on the binary_filter."""
        if isinstance(binary_filter, set):
            filter_fn = lambda x: x in binary_filter
        elif isinstance(binary_filter, Pattern):
            filter_fn = lambda x: binary_filter.match(x)
        elif callable(binary_filter):
            filter_fn = binary_filter
        else:
            raise NotImplementedError("ERROR: The binary_filter must be a set, a regex pattern, or a callable function.")

        words = []
        depths = []
        labels = []
        final_activations = []

        for (token, depth), activation in zip(tokens_with_depth, activations):
            words.append(token)
            depths.append(depth)
            if filter_fn(token):
                labels.append('positive')
            else:
                labels.append('negative')
            final_activations.append(activation)

        return list(zip(words, depths)), labels, final_activations

    def aggregate_activations_scalar(self, tokens_with_depth, activations):
        """Aggregates activations for each token by computing a scalar value."""
        from collections import defaultdict
        token_to_activations = defaultdict(list)
        for (token, depth), activation in zip(tokens_with_depth, activations):
            token_to_activations[token].append(activation)
        token_to_scalar_activation = {}
        for token, activation_list in token_to_activations.items():
            # Stack activation arrays
            stacked_activations = np.stack(activation_list)
            # Compute mean activation vector
            mean_activation_vector = np.mean(stacked_activations, axis=0)
            # Compute scalar value (e.g., mean of the mean activation vector)
            scalar_activation = np.mean(mean_activation_vector)
            token_to_scalar_activation[token] = scalar_activation
        return token_to_scalar_activation

    def write_token_activation_mapping(self, token_to_activation, output_dir):
        """Writes the token to scalar activation mapping to a JSON file."""
        # Construct output file path
        mapping_file = os.path.join(output_dir, f"{self.output_prefix}_scalar_token_activation_mapping.json")
        # Write to file
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(token_to_activation, f, indent=2)
        print(f"Token activation mapping saved to '{mapping_file}'.")

    # Existing mean vector aggregation function (kept for future reference)
    def aggregate_activations(self, tokens_with_depth, activations):
        """Aggregates activations for each token by computing the mean activation vector."""
        from collections import defaultdict
        token_to_activations = defaultdict(list)
        for (token, depth), activation in zip(tokens_with_depth, activations):
            token_to_activations[token].append(activation)
        token_to_mean_activation = {}
        for token, activation_list in token_to_activations.items():
            # Stack activation arrays and compute mean along axis 0
            mean_activation = np.mean(np.stack(activation_list), axis=0)
            token_to_mean_activation[token] = mean_activation
        return token_to_mean_activation

# def main():
#     # Set up argument parser
#     parser_arg = argparse.ArgumentParser(description='Parse Java code and generate AST visualization and activations.')
#     parser_arg.add_argument('file', help='Path to the Java source file.')
#     parser_arg.add_argument('--model', default='bert-base-uncased', help='Transformer model to use for activations.')
#     parser_arg.add_argument('--device', default='cpu', help='Device to run the model on ("cpu" or "cuda").')
#     parser_arg.add_argument('--binary_filter', default='set:public,static', help='Binary filter for labeling.')
#     parser_arg.add_argument('--output_prefix', default='output', help='Prefix for output files.')
#     args = parser_arg.parse_args()

#     java_file_path = args.file

#     if not os.path.isfile(java_file_path):
#         print(f"Error: File '{java_file_path}' does not exist.")
#         sys.exit(1)

#     # Prepare output directory at the same level as src
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     output_dir = os.path.join(base_dir, 'output')
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Initialize classes
#     ast_processor = JavaASTProcessor(java_file_path, output_dir)
#     activation_annotator = ActivationAnnotator(
#         model_name=args.model,
#         device=args.device,
#         binary_filter=args.binary_filter,
#         output_prefix=args.output_prefix
#     )

#     # Process AST and write tokens
#     ast_processor.process_ast()

#     # Process activations and annotate data
#     activation_annotator.process_activations(ast_processor.tokens_tuples, output_dir)

#     # Visualize the AST
#     input_filename = os.path.splitext(os.path.basename(java_file_path))[0]
#     ast_processor.visualize_ast(input_filename)

# if __name__ == "__main__":
#     main()
