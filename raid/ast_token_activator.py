import os
import sys
import argparse
import numpy as np
import re
import json
from typing import Pattern

from tree_sitter import Parser, Language
import tree_sitter_java as tsjava
# import graphviz

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

    # def visualize_ast(self, input_filename):
    #     """Visualizes the AST and saves it as a PNG file."""
    #     graph = graphviz.Digraph(format='png')
    #     self._visualize_ast(self.root_node, graph)
    #     ast_output_path = os.path.join(self.output_dir, f'{input_filename}_ast')
    #     graph.render(ast_output_path, format='png', cleanup=True)
    #     print(f"\nAST visualization saved as '{ast_output_path}.png'.")

    # def _visualize_ast(self, node, graph, parent_id=None):
    #     """Helper method for visualizing the AST."""
    #     node_id = str(id(node))
    #     label = f"{node.type} [{node.start_point}-{node.end_point}]"
    #     graph.node(node_id, label)
    #     if parent_id:
    #         graph.edge(parent_id, node_id)
    #     for child in node.children:
    #         self._visualize_ast(child, graph, node_id)

class ActivationAnnotator:
    """Class to handle activation generation and data annotation."""
    def __init__(self, model_name, device='cpu', binary_filter='set:public,static', output_prefix='output', aggregation_method='mean'):
        self.model_name = model_name
        self.device = device
        self.binary_filter = binary_filter
        self.output_prefix = output_prefix
        self.binary_filter_compiled = None
        self.aggregation_method = aggregation_method

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
        # Write aggregated activations
        self.write_aggregated_activations(tokens_with_depth, activations, output_dir)
        # Aggregate phrase activations
        phrase_activations = self.aggregate_phrase_activations(tokens_with_depth, activations, method=self.aggregation_method)
        # Output mapping to file
        self.write_phrase_activations(phrase_activations, output_dir)

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
            # Collect activations from all layers
            token_activations = []
            for layer in layers:
                layer_index = layer['index']
                layer_values = layer['values']
                token_activations.append((layer_index, np.array(layer_values)))
            activations.append(token_activations)
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
        tokens_depths, labels, flat_activations = self._create_binary_data(
            tokens_with_depth, activations, self.binary_filter_compiled, balance_data=False
        )

        # Collect tokens and labels by depth
        from collections import defaultdict
        depth_to_tokens = defaultdict(list)
        depth_to_labels = defaultdict(list)

        for (token, depth), label in zip(tokens_depths, labels):
            depth_to_tokens[depth].append(token)
            depth_to_labels[depth].append(label)

        # Determine the maximum depth
        max_depth = max(depth_to_tokens.keys())

        # Save the files
        words_file = os.path.join(output_dir, f"{self.output_prefix}_tokens.txt")
        labels_file = os.path.join(output_dir, f"{self.output_prefix}_labels.txt")
        activations_file = os.path.join(output_dir, f"{self.output_prefix}_activations.txt")

        with open(words_file, "w", encoding='utf-8') as f_words, open(labels_file, "w", encoding='utf-8') as f_labels:
            # Iterate over all depths from 0 to max_depth
            for depth in range(max_depth + 1):
                tokens_line = ' '.join(depth_to_tokens.get(depth, []))
                labels_line = ' '.join(depth_to_labels.get(depth, []))
                f_words.write(tokens_line + '\n')
                f_labels.write(labels_line + '\n')

        # Save activations to text file (order maintained as per tokens_with_depth)
        # Since activations now contain per-layer activations, we can save only the last layer if needed
        with open(activations_file, 'w', encoding='utf-8') as f:
            for token_activations in flat_activations:
                # Get activation from the last layer
                last_layer_activation = token_activations[-1][1].tolist()
                activation_str = ' '.join(map(str, last_layer_activation))
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

        for (token, depth), token_activations in zip(tokens_with_depth, activations):
            words.append(token)
            depths.append(depth)
            if filter_fn(token):
                labels.append('positive')
            else:
                labels.append('negative')
            final_activations.append(token_activations)

        return list(zip(words, depths)), labels, final_activations

    def aggregate_activation_list(self, activations, method='mean'):
        """
        Aggregates a list of activations using the specified method.
        """
        activations_array = np.stack(activations)
        if method == 'mean':
            return np.mean(activations_array, axis=0)
        elif method == 'max':
            return np.max(activations_array, axis=0)
        elif method == 'sum':
            return np.sum(activations_array, axis=0)
        elif method == 'concat':
            return np.concatenate(activations_array, axis=0)
        else:
            raise ValueError("Unsupported aggregation method")

    def aggregate_phrase_activations(self, tokens_with_depth, activations, method='mean'):
        """
        Aggregates token-level activations into phrase-level activations.
        Phrases are defined as tokens at the same depth level.
        """
        from collections import defaultdict

        # Group tokens and activations by depth (phrases)
        depth_to_tokens = defaultdict(list)
        depth_to_activations = defaultdict(list)
        for (token, depth), token_layers in zip(tokens_with_depth, activations):
            depth_to_tokens[depth].append(token)
            depth_to_activations[depth].append(token_layers)  # token_layers is list of (layer_index, activation)

        # For each depth (phrase), aggregate activations per layer
        phrase_activations = []
        for depth in sorted(depth_to_tokens.keys()):
            tokens = depth_to_tokens[depth]
            tokens_layers_list = depth_to_activations[depth]

            # Aggregate activations per layer index
            layer_to_activations = defaultdict(list)
            for token_layers in tokens_layers_list:
                for layer_index, activation in token_layers:
                    layer_to_activations[layer_index].append(activation)

            # Prepare aggregated layers
            aggregated_layers = []
            for layer_index in sorted(layer_to_activations.keys()):
                activations_list = layer_to_activations[layer_index]
                # Aggregate activations using the specified method
                aggregated_activation = self.aggregate_activation_list(activations_list, method=method)
                aggregated_layers.append({
                    "index": layer_index,
                    "values": aggregated_activation.tolist()
                })

            # Use the tokens as a phrase key (joined tokens)
            phrase_key = ' '.join(tokens)
            phrase_feature = {
                "phrase": phrase_key,
                "layers": aggregated_layers
            }
            phrase_activations.append(phrase_feature)

        # Build the output data structure
        output_data = {
            "linex_index": 0,
            "features": phrase_activations
        }

        return output_data

    def write_phrase_activations(self, phrase_activations, output_dir):
        """Writes the phrase activations to a JSON file in the same format as activations.json."""
        # Construct output file path
        mapping_file = os.path.join(output_dir, f"{self.output_prefix}_phrasal_activations.json")
        # Write to file
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(phrase_activations, f, indent=2)
        print(f"Phrase activations saved to '{mapping_file}'.")

    def write_aggregated_activations(self, tokens_with_depth, activations, output_dir):
        """Writes the mean-aggregated activations across all layers to a JSON file."""
        aggregated_file = os.path.join(output_dir, f"{self.output_prefix}_aggregated_activations.json")
        
        # Create a list to store token-activation mappings
        token_activations = []
        
        # Iterate through tokens and their activations together
        for (token, _), token_layers in zip(tokens_with_depth, activations):
            # Get activations from all layers for this token
            layer_activations = [layer[1] for layer in token_layers]
            # Aggregate across layers using the specified method
            aggregated = self.aggregate_activation_list(layer_activations, method=self.aggregation_method)
            
            # Create feature object for this token
            token_feature = {
                "token": token,  # Use actual token from tokens_with_depth
                "aggregated_values": aggregated.tolist()
            }
            token_activations.append(token_feature)
        
        # Build the output data structure
        output_data = {
            "aggregation_method": self.aggregation_method,
            "features": token_activations
        }
        
        # Write to JSON file
        with open(aggregated_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Aggregated activations saved to '{aggregated_file}'.")
