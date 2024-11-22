"""
Java AST Processing and Neural Activation Analysis Tools

This module provides tools for processing Java Abstract Syntax Trees (AST) and analyzing neural network
activations for code understanding tasks. It includes two main classes: JavaASTProcessor for AST
manipulation and ActivationAnnotator for neural activation analysis.

"""

import os
import sys
import argparse
import numpy as np
import re
import json
from typing import Pattern, List, Tuple, Dict, Any, Union

from tree_sitter import Parser, Language
import tree_sitter_java as tsjava
import neurox.data.extraction.transformers_extractor as transformers_extractor

# Initialize the Java language
JAVA_LANGUAGE = Language(tsjava.language())

class JavaASTProcessor:
    """
    Process Java source code to extract Abstract Syntax Tree (AST) information.
    
    This class handles parsing Java source files, extracting tokens from the AST,
    and optionally visualizing the tree structure.
    
    Attributes:
        java_file_path (str): Path to the Java source file
        output_dir (str): Directory where output files will be saved
        source_code (str): Content of the Java source file
        tree (tree_sitter.Tree): Parsed AST
        root_node (tree_sitter.Node): Root node of the AST
        tokens_tuples (List[Tuple[str, str, int]]): List of (token_type, token_text, depth) tuples
        tokens (List[str]): List of extracted token texts
        parser (tree_sitter.Parser): Tree-sitter parser instance
    """

    def __init__(self, java_file_path: str, output_dir: str):
        """
        Initialize the JavaASTProcessor with file path and output directory.
        
        Args:
            java_file_path: Path to the Java source file
            output_dir: Directory where output files will be saved
        """
        self.java_file_path = java_file_path
        self.output_dir = output_dir
        self.source_code = None
        self.tree = None
        self.root_node = None
        self.tokens_tuples = None
        self.tokens = None
        self.parser = Parser(JAVA_LANGUAGE)

    def process_ast(self) -> None:
        """
        Process the Java source file to extract AST information.
        
        This method performs the complete processing pipeline:
        1. Reads the source code
        2. Parses it into an AST
        3. Extracts tokens
        4. Writes tokens to output file
        """
        self.read_source_code()
        self.parse_source_code()
        self.tokens_tuples = self.extract_leaf_tokens()
        self.tokens = [token_text for _, token_text, _ in self.tokens_tuples]
        self.write_tokens_to_file()

    def read_source_code(self) -> None:
        """
        Read and store the Java source code from the file.
        
        Raises:
            FileNotFoundError: If the Java file doesn't exist
            UnicodeDecodeError: If the file encoding is not UTF-8
        """
        with open(self.java_file_path, 'r', encoding='utf-8') as file:
            self.source_code = file.read()

    def parse_source_code(self) -> None:
        """
        Parse the source code into an AST using tree-sitter.
        
        The parsed tree is stored in self.tree and the root node in self.root_node.
        """
        self.tree = self.parser.parse(self.source_code.encode('utf-8'))
        self.root_node = self.tree.root_node

    def extract_leaf_tokens(self, node=None, depth: int = 0) -> List[Tuple[str, str, int]]:
        """
        Recursively extract tokens from AST leaf nodes.
        
        Args:
            node: Current AST node (defaults to root node if None)
            depth: Current depth in the AST
            
        Returns:
            List of tuples containing (token_type, token_text, depth)
        """
        if node is None:
            node = self.root_node
        tokens = []
        if len(node.children) == 0:
            tokens.append((node.type, node.text.decode('utf-8'), depth))
        else:
            for child in node.children:
                tokens.extend(self.extract_leaf_tokens(child, depth + 1))
        return tokens

    def write_tokens_to_file(self) -> None:
        """
        Write extracted tokens to a file in the output directory.
        
        Creates 'input_sentences.txt' in the output directory and prints token information
        to the console.
        
        Raises:
            IOError: If unable to write to the output directory
        """
        input_file = os.path.join(self.output_dir, 'input_sentences.txt')
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(self.tokens) + '\n')
        print(f"Tokens written to '{input_file}'.")
        print("Extracted Tokens:")
        for token_type, token_text, depth in self.tokens_tuples:
            print(f"Type: {token_type}, Text: {token_text}, Depth: {depth}")


class ActivationAnnotator:
    """
    Generate and analyze neural network activations for code tokens.
    
    This class handles the generation of neural network activations for tokens,
    applies filtering, and provides various analysis methods.
    
    Attributes:
        model_name (str): Name of the transformer model to use
        device (str): Computing device ('cpu' or 'cuda')
        binary_filter (str): Filter specification for binary classification
        output_prefix (str): Prefix for output files
        binary_filter_compiled (Union[Pattern, set]): Compiled filter
        aggregation_method (str): Method for aggregating activations
    """

    def __init__(self, model_name: str, device: str = 'cpu', 
                 binary_filter: str = 'set:public,static', 
                 output_prefix: str = 'output',
                 aggregation_method: str = 'mean'):
        """
        Initialize the ActivationAnnotator.
        
        Args:
            model_name: Name of the transformer model
            device: Computing device ('cpu' or 'cuda')
            binary_filter: Filter specification (starts with 're:' for regex or 'set:' for word set)
            output_prefix: Prefix for output files
            aggregation_method: Method for aggregating activations ('mean', 'max', 'sum', 'concat')
        """
        self.model_name = model_name
        self.device = device
        self.binary_filter = binary_filter
        self.output_prefix = output_prefix
        self.binary_filter_compiled = None
        self.aggregation_method = aggregation_method

    def process_activations(self, tokens_tuples: List[Tuple[str, str, int]], 
                          output_dir: str) -> None:
        """
        Process token activations through the complete pipeline.
        
        Args:
            tokens_tuples: List of (token_type, token_text, depth) tuples
            output_dir: Directory for output files
            
        Raises:
            ValueError: If binary_filter format is invalid
            IOError: If unable to write to output directory
        """
        input_file = os.path.join(output_dir, 'input_sentences.txt')
        output_file = os.path.join(output_dir, 'activations.json')
        self.generate_activations(input_file, output_file)
        extracted_tokens, activations = self.parse_activations(output_file)
        self.handle_binary_filter()
        tokens_with_depth = [(t, d) for (_, t, d) in tokens_tuples]
        self.annotate_data(tokens_with_depth, activations, output_dir)
        self.write_aggregated_activations(tokens_with_depth, activations, output_dir)
        phrase_activations = self.aggregate_phrase_activations(
            tokens_with_depth, activations, method=self.aggregation_method)
        self.write_phrase_activations(phrase_activations, output_dir)

    def generate_activations(self, input_file: str, output_file: str) -> None:
        """
        Generate neural network activations using the specified transformer model.
        
        Args:
            input_file: Path to input file containing tokens
            output_file: Path where activations will be saved
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If model fails to generate activations
        """
        transformers_extractor.extract_representations(
            self.model_name,
            input_file,
            output_file,
            aggregation="average",  
            output_type="json",
            device=self.device
        )

    def parse_activations(self, activation_file: str) -> Tuple[List[str], List[List[Tuple[int, np.ndarray]]]]:
        """
        Parse activations from the JSON output file.
        
        Args:
            activation_file: Path to the activation JSON file
            
        Returns:
            Tuple containing:
                - List of extracted tokens
                - List of activations per token (each activation is a list of layer tuples)
                
        Raises:
            FileNotFoundError: If activation file doesn't exist
            JSONDecodeError: If activation file is not valid JSON
        """
        with open(activation_file, 'r', encoding='utf-8') as f:
            activation_data = json.load(f)
        activations = []
        extracted_tokens = []
        for feature in activation_data['features']:
            token = feature['token'].replace('Ġ', '')
            layers = feature['layers']
            token_activations = []
            for layer in layers:
                layer_index = layer['index']
                layer_values = layer['values']
                token_activations.append((layer_index, np.array(layer_values)))
            activations.append(token_activations)
            extracted_tokens.append(token)
        return extracted_tokens, activations

    def handle_binary_filter(self) -> None:
        """
        Compile the binary filter based on user input.
        
        The filter can be either a regex pattern (prefixed with 're:')
        or a set of words (prefixed with 'set:').
        
        Raises:
            NotImplementedError: If filter prefix is neither 're:' nor 'set:'
            re.error: If regex pattern is invalid
        """
        if self.binary_filter.startswith("re:"):
            self.binary_filter_compiled = re.compile(self.binary_filter[3:])
        elif self.binary_filter.startswith("set:"):
            self.binary_filter_compiled = set(self.binary_filter[4:].split(","))
        else:
            raise NotImplementedError("Filter must start with 're:' for regex or 'set:' for a set of words.")

    def annotate_data(self, tokens_with_depth: List[Tuple[str, int]], 
                     activations: List[List[Tuple[int, np.ndarray]]], 
                     output_dir: str) -> None:
        """
        Create binary data and save tokens and labels organized by AST depth.
        
        Args:
            tokens_with_depth: List of (token, depth) tuples
            activations: List of activations for each token
            output_dir: Directory to save output files
            
        Raises:
            IOError: If unable to write to output directory
        """
        tokens_depths, labels, flat_activations = self._create_binary_data(
            tokens_with_depth, activations, self.binary_filter_compiled, balance_data=False
        )

        from collections import defaultdict
        depth_to_tokens = defaultdict(list)
        depth_to_labels = defaultdict(list)

        for (token, depth), label in zip(tokens_depths, labels):
            depth_to_tokens[depth].append(token)
            depth_to_labels[depth].append(label)

        max_depth = max(depth_to_tokens.keys())

        words_file = os.path.join(output_dir, f"{self.output_prefix}_tokens.txt")
        labels_file = os.path.join(output_dir, f"{self.output_prefix}_labels.txt")
        activations_file = os.path.join(output_dir, f"{self.output_prefix}_activations.txt")

        with open(words_file, "w", encoding='utf-8') as f_words, \
             open(labels_file, "w", encoding='utf-8') as f_labels:
            for depth in range(max_depth + 1):
                tokens_line = ' '.join(depth_to_tokens.get(depth, []))
                labels_line = ' '.join(depth_to_labels.get(depth, []))
                f_words.write(tokens_line + '\n')
                f_labels.write(labels_line + '\n')

        with open(activations_file, 'w', encoding='utf-8') as f:
            for token_activations in flat_activations:
                last_layer_activation = token_activations[-1][1].tolist()
                activation_str = ' '.join(map(str, last_layer_activation))
                f.write(activation_str + '\n')

    def _create_binary_data(self, tokens_with_depth: List[Tuple[str, int]], 
                          activations: List[List[Tuple[int, np.ndarray]]], 
                          binary_filter: Union[Pattern, set], 
                          balance_data: bool = False) -> Tuple[List[Tuple[str, int]], List[str], List[List[Tuple[int, np.ndarray]]]]:
        """
        Create binary labeled dataset based on the binary_filter.
        
        Args:
            tokens_with_depth: List of (token, depth) tuples
            activations: List of activations for each token
            binary_filter: Compiled regex pattern or set of words
            balance_data: Whether to balance positive and negative samples
            
        Returns:
            Tuple containing:
                - List of (word, depth) tuples
                - List of labels ('positive' or 'negative')
                - List of token activations
                
        Raises:
            NotImplementedError: If binary_filter is not a set, regex pattern, or callable
        """
        if isinstance(binary_filter, set):
            filter_fn = lambda x: x in binary_filter
        elif isinstance(binary_filter, Pattern):
            filter_fn = lambda x: binary_filter.match(x)
        elif callable(binary_filter):
            filter_fn = binary_filter
        else:
            raise NotImplementedError("The binary_filter must be a set, a regex pattern, or a callable function.")

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

    def aggregate_activation_list(self, activations: List[np.ndarray], 
                                method: str = 'mean') -> np.ndarray:
        """
        Aggregate a list of activations using the specified method.
        
        Args:
            activations: List of activation arrays to aggregate
            method: Aggregation method ('mean', 'max', 'sum', or 'concat')
            
        Returns:
            numpy.ndarray: Aggregated activation array
            
        Raises:
            ValueError: If unsupported aggregation method is specified
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

    def aggregate_phrase_activations(self, tokens_with_depth: List[Tuple[str, int]],
                                   activations: List[List[Tuple[int, np.ndarray]]], 
                                   method: str = 'mean') -> Dict[str, Any]:
        """
        Aggregate token-level activations into phrase-level activations.
        
        Phrases are defined as tokens at the same depth level in the AST.
        
        Args:
            tokens_with_depth: List of (token, depth) tuples
            activations: List of activation layers for each token
            method: Aggregation method ('mean', 'max', 'sum', or 'concat')
            
        Returns:
            Dict containing:
                - linex_index: Index of the line
                - features: List of phrase features with their aggregated activations
                
        Raises:
            ValueError: If unsupported aggregation method is specified
        """
        from collections import defaultdict

        depth_to_tokens = defaultdict(list)
        depth_to_activations = defaultdict(list)
        
        for (token, depth), token_layers in zip(tokens_with_depth, activations):
            depth_to_tokens[depth].append(token)
            depth_to_activations[depth].append(token_layers)

        phrase_activations = []
        for depth in sorted(depth_to_tokens.keys()):
            tokens = depth_to_tokens[depth]
            tokens_layers_list = depth_to_activations[depth]

            layer_to_activations = defaultdict(list)
            for token_layers in tokens_layers_list:
                for layer_index, activation in token_layers:
                    layer_to_activations[layer_index].append(activation)

            aggregated_layers = []
            for layer_index in sorted(layer_to_activations.keys()):
                activations_list = layer_to_activations[layer_index]
                aggregated_activation = self.aggregate_activation_list(activations_list, method=method)
                aggregated_layers.append({
                    "index": layer_index,
                    "values": aggregated_activation.tolist()
                })

            phrase_key = ' '.join(tokens)
            phrase_feature = {
                "phrase": phrase_key,
                "layers": aggregated_layers
            }
            phrase_activations.append(phrase_feature)

        output_data = {
            "linex_index": 0,
            "features": phrase_activations
        }

        return output_data

    def write_phrase_activations(self, phrase_activations: Dict[str, Any], 
                               output_dir: str) -> None:
        """
        Write phrase activations to a JSON file.
        
        Args:
            phrase_activations: Dictionary containing phrase activation data
            output_dir: Directory to save the output file
            
        Raises:
            IOError: If unable to write to output directory
            TypeError: If phrase_activations is not JSON-serializable
        """
        mapping_file = os.path.join(output_dir, f"{self.output_prefix}_phrasal_activations.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(phrase_activations, f, indent=2)
        print(f"Phrase activations saved to '{mapping_file}'.")

    def write_aggregated_activations(self, tokens_with_depth: List[Tuple[str, int]], 
                                   activations: List[List[Tuple[int, np.ndarray]]], 
                                   output_dir: str) -> None:
        """
        Write mean-aggregated activations across all layers to a JSON file.
        
        Args:
            tokens_with_depth: List of (token, depth) tuples
            activations: List of activation layers for each token
            output_dir: Directory to save the output file
            
        Raises:
            IOError: If unable to write to output directory
            ValueError: If token and activation lengths don't match
        """
        aggregated_file = os.path.join(output_dir, f"{self.output_prefix}_aggregated_activations.json")
        
        token_activations = []
        
        for (token, _), token_layers in zip(tokens_with_depth, activations):
            layer_activations = [layer[1] for layer in token_layers]
            aggregated = self.aggregate_activation_list(layer_activations, method=self.aggregation_method)
            
            token_feature = {
                "token": token,
                "aggregated_values": aggregated.tolist()
            }
            token_activations.append(token_feature)
        
        output_data = {
            "aggregation_method": self.aggregation_method,
            "features": token_activations
        }
        
        with open(aggregated_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Aggregated activations saved to '{aggregated_file}'.")