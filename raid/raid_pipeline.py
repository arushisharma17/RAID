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


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser with help descriptions.
    """
    parser = argparse.ArgumentParser(
        description='RAID (Rapid Automated Interpretability Datasets) tool for analyzing code using transformer models.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required argument
    parser.add_argument(
        'input_file',
        help='Path to the source code file to analyze'
    )

    # Optional arguments
    parser.add_argument(
        '--model',
        default='bert-base-uncased',
        help='Transformer model to use (default: bert-base-uncased)'
    )

    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Computing device to use (default: cpu)'
    )

    parser.add_argument(
        '--binary_filter',
        default='set:public,static',
        help='Filter specification (format: "type:pattern"). Types: set (comma-separated) or re (regex)'
    )

    parser.add_argument(
        '--output_prefix',
        default='output',
        help='Prefix for output files (default: output)'
    )

    parser.add_argument(
        '--aggregation_method',
        default='mean',
        choices=['mean', 'max', 'sum', 'concat'],
        help='Method to aggregate activations (default: mean)'
    )

    parser.add_argument(
        '--label',
        default='leaves',
        choices=['program', 'class_declaration', 'class_body', 'method_declaration', 
                'formal_parameters', 'block', 'method_invocation', 'leaves'],
        help='Type of AST label to analyze (default: leaves)'
    )

    parser.add_argument(
        '--layer',
        type=int,
        default=None,
        help='Specific transformer layer to analyze (0-12, default: all layers)'
    )

    return parser

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
    parser = setup_parser()
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Validate layer number if specified
    if args.layer is not None and not (0 <= args.layer <= 12):
        parser.error("Layer must be between 0 and 12")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    java_file_path = args.input_file

    # Initialize and process AST
    ast_processor = JavaASTProcessor(java_file_path, args.output_prefix)
    ast_processor.process_ast()

    # Process activations and annotate data
    activation_annotator = ActivationAnnotator(
        model_name=args.model,
        device=args.device,
        binary_filter=args.binary_filter,
        output_prefix=args.output_prefix,
        layer=args.layer
    )
    activation_annotator.process_activations(ast_processor.tokens_tuples, args.output_prefix)

    # Generate .in and .label files using TokenLabelFilesGenerator
    generator = TokenLabelFilesGenerator()
    generator.generate_in_label_bio_files(java_file_path, 'java', args.label)


if __name__ == "__main__":
    main()