# RAID (Rapid Automated Interpretability Datasets) tool
Designed to generate binary and multiclass datasets rapidly using regular expressions, AST labels, and semantic concepts. This tool enabled us to perform fine-grained analyses of concepts learned by the model through its latent representations. 

### Features
1. Labels of different granularities: tokens, phrases, blocks, other semantic chunks.
2. Corresponding activations for tokens, phrases, blocks.
3. Support for labels using Regular expressions, Abstract Syntax Trees, Semantic datasets.
4. Integration with Tree-sitter parsers (Abstract Syntax Tree based labels)
5. B-I-0 Labelling for higher-level semantic concepts (Phrase and Block level chunks)
6. Activation aggregation for higher-level semantic concepts (Phrase and Block level chunks): YOu can generate activations once and experiment with different granularities by aggregatingt the activations. 
   

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
