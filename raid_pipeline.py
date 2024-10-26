import os
import sys
import argparse

# Add the 'src' directory to sys.path to allow module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary classes from other modules
from ast_token_activator import JavaASTProcessor, ActivationAnnotator
from extract_patterns import PatternExtractor
from generate_files import TokenLabelFilesGenerator


def main():
    # Set up argument parser
    parser_arg = argparse.ArgumentParser(description='RAID pipeline for processing Java code.')
    parser_arg.add_argument('file', help='Path to the Java source file.')
    parser_arg.add_argument('--model', default='bert-base-uncased', help='Transformer model for activations.')
    parser_arg.add_argument('--device', default='cpu', help='Device to run the model on ("cpu" or "cuda").')
    parser_arg.add_argument('--binary_filter', default='set:public,static', help='Binary filter for labeling.')
    parser_arg.add_argument('--output_prefix', default='output', help='Prefix for output files.')
    parser_arg.add_argument('--aggregation_method', default='mean', help='Aggregation method for activations (mean, max, sum, concat).')
    parser_arg.add_argument('--label', default='leaves', help='Desired non-leaf type to categorize tokens.')
    args = parser_arg.parse_args()

    java_file_path = args.file

    if not os.path.isfile(java_file_path):
        print(f"Error: File '{java_file_path}' does not exist.")
        sys.exit(1)

    # Prepare output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize and process AST
    ast_processor = JavaASTProcessor(java_file_path, output_dir)
    ast_processor.process_ast()

    # Visualize the AST
    input_filename = os.path.splitext(os.path.basename(java_file_path))[0]
    ast_processor.visualize_ast(input_filename)

    # Process activations and annotate data
    activation_annotator = ActivationAnnotator(
        model_name=args.model,
        device=args.device,
        binary_filter=args.binary_filter,
        output_prefix=args.output_prefix
    )
    activation_annotator.process_activations(ast_processor.tokens_tuples, output_dir)

    # Generate .in and .label files using TokenLabelFilesGenerator
    generator = TokenLabelFilesGenerator()
    # Generate files based on the Java file
    generator.generate_in_and_label_files(java_file_path, 'java', args.label)


if __name__ == "__main__":
    main()