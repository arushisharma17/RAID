# RAID (Rapid Automated Interpretability Datasets) tool
Designed to generate binary and multiclass datasets rapidly using regular expressions, AST labels, and semantic concepts. This tool enabled us to perform fine-grained analyses of concepts learned by the model through its latent representations. 

### Features
1. Labels of different granularities: tokens, phrases, blocks, other semantic chunks.
2. Corresponding activations for tokens, phrases, blocks.
4. B-I-0 Labelling for higher-level semantic concepts (Phrase and Block level chunks)
5. Activation aggregation for higher-level semantic concepts (Phrase and Block level chunks): YOu can generate activations once and experiment with different granularities by aggregatingt the activations.
6. Integration with static analysis tools to create custom labels
   a. Tree-sitter parsers (Abstract Syntax Tree based labels) - Syntactic
   b. CK metrics (Object Oriented metrics/Design patterns) - Structural
   c. CFG, DFG, AST Nesting/Parent-child relationships, etc -Hierarchical
   d. SE datasets, Ontologies, Design Patterns - Semantic
   e. Regular Expressions to create datasets, filter datasets, edit datasets.

## Temp Installation Instructions

Install RAID from TestPyPI, and its NeuroX dependency:

```bash
pip install -i https://test.pypi.org/simple/ raid==1.0.0
pip install git+https://github.com/arushisharma17/NeuroX.git@fe7ab9c2d8eb1b4b3f93de73b8eaae57a6fc67b7
```

### Usage

Run RAID on a Java file:

```bash
raid path/to/your/file.java --model bert-base-uncased --device cpu --binary_filter "set:public,static" --output_prefix output --aggregation_method mean --label class_body --layer 5
```
   $ raid C:/Users/hkhri/OneDrive/Desktop/SD/RAID/input/sample_input.java --model bert-base-uncased --device cpu --binary_filter "set:public,static" --output_prefix output --aggregation_method mean --label class_body --layer 5

## How to Run
In terminal, run the following command:
```
python raid_pipeline.py [filepath to code file] [model] [device] [binary filter] [output prefix] [aggregation method] [label]
```
An example input:
```
python raid_pipeline.py C:/Users/User/OneDrive/Desktop/SD/RAID/input/sample_input.java --model bert-base-uncased --device cpu --binary_filter 'set:public,static' --output_prefix output --aggregation_method mean --label class_body
```
Allowed commands for --label are: program, class_declaration, class_body, method_declaration, formal_parameters, block, method_invocation, leaves

## Link to Colab notebook for tutorial and intial instructions

https://colab.research.google.com/drive/1MfTbOMrZnQ_FkC65CCJyUE4v21u5pJ4G?usp=sharing

Note: Please keep updating readme as you add code. 


[![RAID-Workflow](https://github.com/user-attachments/assets/fdf16639-70f5-4f7e-b20b-762a5cdcaaba)](https://docs.google.com/drawings/d/1LEqqQ_1dJ7MWrBR_2kRZF71bF8Sv6TiDTebAvaGkFX0/edit?usp=sharing)

## Features
1. Integrate static analysis tools. 
2. Generate AST nodes and labels
3. Extract layerwise and 'all' layers activations
4. Split phrasal nodes into B-I-O tokens
5. Aggregate activations
6. Lexical Patterns or features
7. Support for multiple languages
8. Code Ontologies/HPC Ontologies
9. Software engineering datasets
10. Hierarchical Properties/Structural Properties
11. Add support for autoencoders NeuroX
12. Train, validation, test splits for probing tasks support.
