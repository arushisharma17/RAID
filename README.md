# RAID (Rapid Automated Interpretability Datasets) tool
Designed to generate binary and multiclass datasets rapidly using regular expressions, AST labels, and semantic concepts. This tool enabled us to perform fine-grained analyses of concepts learned by the model through its latent representations. 

### Features
1. Labels of different granularities: tokens, phrases, blocks, other semantic chunks.
2. Corresponding activations for tokens, phrases, blocks.
4. B-I-0 Labelling for higher-level semantic concepts (Phrase and Block level chunks)
5. Activation aggregation for higher-level semantic concepts (Phrase and Block level chunks): You can generate activations once and experiment with different granularities by aggregating the activations.
6. Integration with static analysis tools to create custom labels
   a. Tree-sitter parsers (Abstract Syntax Tree based labels) - Syntactic
   b. CK metrics (Object Oriented metrics/Design patterns) - Structural
   c. CFG, DFG, AST Nesting/Parent-child relationships, etc -Hierarchical
   d. SE datasets, Ontologies, Design Patterns - Semantic
   e. Regular Expressions to create datasets, filter datasets, edit datasets.

## Installation Instructions

Install RAID and its dependencies:

```bash
pip install git+https://github.com/arushisharma17/NeuroX.git@fe7ab9c2d8eb1b4b3f93de73b8eaae57a6fc67b7
pip install raid-tool
```

### Usage

Run RAID on a Java file:

```bash
raid path/to/your/file.java --model bert-base-uncased --device cpu --binary_filter "set:public,static" --output_prefix output --aggregation_method mean --label class_body --layer 5
```


### Available Labels
The following labels are supported for the `--label` parameter:
- program
- class_declaration 
- class_body
- method_declaration
- formal_parameters
- block
- method_invocation
- leaves

### Aggregation Methods
Available methods for `--aggregation_method`:
- mean (default)
- max
- sum
- concat

## Link to Colab notebook for tutorial and initial instructions
https://colab.research.google.com/drive/1MfTbOMrZnQ_FkC65CCJyUE4v21u5pJ4G?usp=sharing

Note: Please keep updating readme as you add code.

[![RAID-Workflow](https://colab.research.google.com/drive/165SuE7ZWAAfUBcTuH6dVlQqoMFSL7ja-)](https://docs.google.com/drawings/d/1LEqqQ_1dJ7MWrBR_2kRZF71bF8Sv6TiDTebAvaGkFX0/edit?usp=sharing)

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

