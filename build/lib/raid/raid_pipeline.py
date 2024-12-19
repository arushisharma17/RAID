"""
RAID Pipeline Main Script

This module serves as the main entry point for the RAID (Recognition and Analysis of Identifiers) pipeline,
which processes Java source code for neural network analysis. It coordinates the use of various components
including AST processing, activation generation, and token labeling.

The pipeline performs the following steps:
1. Processes Java source code to extract AST information
2. Generates neural network activations for code tokens
3. Creates labeled datasets for further analysis
4. Generates input and label files in BIO format

Command Line Arguments:
    file: Path to the Java source file
    --model: Transformer model name (default: 'bert-base-uncased')
    --device: Computing device ('cpu' or 'cuda', default: 'cpu')
    --binary_filter: Filter for token labeling (default: 'set:public,static')
    --output_prefix: Prefix for output files (default: 'output')
    --aggregation_method: Method for aggregating activations (default: 'mean')
    --label: Non-leaf type for token categorization (default: 'leaves')
    --layer: Specific transformer layer to extract (default: all layers)
"""

import os
import sys
import argparse
from typing import NoReturn
from pathlib import Path

# Import necessary classes from other modules
from .ast_token_activator import JavaASTProcessor, ActivationAnnotator
from .extract_patterns import PatternExtractor
from .generate_files import TokenLabelFilesGenerator


def main() -> NoReturn:
    """
    Main entry point for the RAID pipeline.
    
    This function:
    1. Parses command line arguments
    2. Sets up the processing environment
    3. Initializes and runs the AST processor
    4. Generates and processes neural network activations
    5. Creates labeled token files
    
    Raises:
        FileNotFoundError: If the specified Java file doesn't exist
        IOError: If unable to create output directory
        Exception: For various processing errors in the pipeline stages
    """
    # Set up argument parser
    parser_arg = argparse.ArgumentParser(description='RAID pipeline for processing Java code.')
    parser_arg.add_argument('file', 
                           help='Path to the Java source file.')
    parser_arg.add_argument('--model', 
                           default='bert-base-uncased',
                           help='Transformer model for activations.')
    parser_arg.add_argument('--device', 
                           default='cpu',
                           help='Device to run the model on ("cpu" or "cuda").')
    parser_arg.add_argument('--binary_filter',
                           default='set:public,static',
                           help='Binary filter for labeling.')
    parser_arg.add_argument('--output_prefix',
                           default='output',
                           help='Prefix for output files.')
    parser_arg.add_argument('--aggregation_method',
                           default='mean',
                           help='Aggregation method for activations (mean, max, sum, concat).')
    parser_arg.add_argument('--label',
                           default='leaves',
                           help='Desired non-leaf type to categorize tokens.')
    parser_arg.add_argument('--layer',
                           type=int,
                           default=None,
                           help='Specific transformer layer to extract (default: all layers)')
    args = parser_arg.parse_args()

    java_file_path = args.file

    # Validate input file existence
    if not os.path.isfile(java_file_path):
        print(f"Error: File '{java_file_path}' does not exist.")
        sys.exit(1)

    # Create output directory in the current working directory instead of package directory
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)

    # Initialize and process AST
    ast_processor = JavaASTProcessor(java_file_path, output_dir)
    ast_processor.process_ast()

    # Process activations and annotate data
    activation_annotator = ActivationAnnotator(
        model_name=args.model,
        device=args.device,
        binary_filter=args.binary_filter,
        output_prefix=os.path.join(output_dir, args.output_prefix),
        layer=args.layer
    )
    activation_annotator.process_activations(ast_processor.tokens_tuples, output_dir)

    # Generate .in and .label files using TokenLabelFilesGenerator
    generator = TokenLabelFilesGenerator()
    generator.generate_in_label_bio_files(java_file_path, 'java', args.label)


if __name__ == "__main__":
    main()