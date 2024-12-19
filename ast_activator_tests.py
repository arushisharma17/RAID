import unittest
import os
import tempfile
import json
import numpy as np
from unittest.mock import patch, Mock
from tree_sitter import Node, Tree

# Import the classes to test
from RAID.raid.ast_token_activator import JavaASTProcessor, ActivationAnnotator

class TestJavaASTProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a sample Java file
        self.java_content = """
        public class Test {
            public static void main(String[] args) {
                System.out.println("Hello");
            }
        }
        """
        self.java_file = os.path.join(self.test_dir, "Test.java")
        with open(self.java_file, "w") as f:
            f.write(self.java_content)
        
        self.processor = JavaASTProcessor(self.java_file, self.test_dir)

    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_read_source_code(self):
        """Test if source code is read correctly"""
        self.processor.read_source_code()
        self.assertEqual(self.processor.source_code.strip(), self.java_content.strip())

    def test_parse_source_code(self):
        """Test if source code is parsed into AST"""
        self.processor.read_source_code()
        self.processor.parse_source_code()
        self.assertIsNotNone(self.processor.root_node)
        self.assertIsInstance(self.processor.tree, Tree)

    def test_write_tokens_to_file(self):
        """Test if tokens are written to file correctly"""
        self.processor.process_ast()
        output_file = os.path.join(self.test_dir, 'input_sentences.txt')
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            content = f.read().strip()
            self.assertGreater(len(content.split()), 0)

class TestActivationAnnotator(unittest.TestCase):
    def setUp(self):
        self.annotator = ActivationAnnotator(
            model_name="bert-base-uncased",
            device="cpu",
            binary_filter="re:^(if|while|for)$",
            output_prefix="test",
            aggregation_method="mean"
        )
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.test_dir)

    def test_parse_activations(self):
        # Create sample activation file
        activation_file = os.path.join(self.test_dir, "test_activations.json")
        sample_data = {
            "tokens": ["if", "condition", "{", "statement", "}"],
            "activations": [
                [[0, [0.1, 0.2]], [1, [0.3, 0.4]]],  # Layer activations for first token
                [[0, [0.5, 0.6]], [1, [0.7, 0.8]]],  # Layer activations for second token
                [[0, [0.9, 1.0]], [1, [1.1, 1.2]]],
                [[0, [1.3, 1.4]], [1, [1.5, 1.6]]],
                [[0, [1.7, 1.8]], [1, [1.9, 2.0]]]
            ]
        }
        with open(activation_file, 'w') as f:
            json.dump(sample_data, f)

        tokens, activations = self.annotator.parse_activations(activation_file)
        
        self.assertEqual(len(tokens), 5)
        self.assertEqual(len(activations), 5)
        self.assertEqual(tokens[0], "if")
        self.assertEqual(len(activations[0]), 2)  # Two layers

    def test_aggregate_activation_list(self):
        # Test different aggregation methods
        activations = [
            (0, np.array([0.1, 0.2])),
            (1, np.array([0.3, 0.4]))
        ]

        # Test mean aggregation
        self.annotator.aggregation_method = "mean"
        result = self.annotator.aggregate_activation_list(activations)
        expected_mean = np.array([0.2, 0.3])  # Mean of [0.1, 0.3] and [0.2, 0.4]
        np.testing.assert_array_almost_equal(result, expected_mean)

        # Test max aggregation
        self.annotator.aggregation_method = "max"
        result = self.annotator.aggregate_activation_list(activations)
        expected_max = np.array([0.3, 0.4])  # Max of [0.1, 0.3] and [0.2, 0.4]
        np.testing.assert_array_almost_equal(result, expected_max)

    def test_handle_binary_filter(self):
        # Test regex filter
        tokens_with_depth = [
            ("if", 1),
            ("while", 2),
            ("for", 1),
            ("else", 2),
            ("print", 3)
        ]
        
        binary_labels = self.annotator.handle_binary_filter(tokens_with_depth)
        expected = [1, 1, 1, 0, 0]  # if, while, for match the regex, others don't
        self.assertEqual(binary_labels, expected)

        # Test set filter
        self.annotator.binary_filter = "set:if,while"
        binary_labels = self.annotator.handle_binary_filter(tokens_with_depth)
        expected = [1, 1, 0, 0, 0]  # Only if and while match the set
        self.assertEqual(binary_labels, expected)

    def test_write_aggregated_activations(self):
        tokens_with_depth = [("if", 1), ("condition", 2)]
        activations = [
            [(0, np.array([0.1, 0.2])), (1, np.array([0.3, 0.4]))],
            [(0, np.array([0.5, 0.6])), (1, np.array([0.7, 0.8]))]
        ]

        self.annotator.write_aggregated_activations(
            tokens_with_depth, 
            activations,
            self.test_dir
        )

        output_file = os.path.join(self.test_dir, f"{self.annotator.output_prefix}_aggregated.json")
        self.assertTrue(os.path.exists(output_file))

        with open(output_file) as f:
            data = json.load(f)
            self.assertIn("tokens", data)
            self.assertIn("depths", data)
            self.assertIn("activations", data)
            self.assertEqual(len(data["tokens"]), 2)

    def test_invalid_aggregation_method(self):
        self.annotator.aggregation_method = "invalid"
        activations = [(0, np.array([0.1, 0.2]))]
        
        with self.assertRaises(ValueError):
            self.annotator.aggregate_activation_list(activations)

    def test_invalid_binary_filter(self):
        tokens_with_depth = [("if", 1)]
        self.annotator.binary_filter = "invalid:pattern"
        
        with self.assertRaises(ValueError):
            self.annotator.handle_binary_filter(tokens_with_depth)

if __name__ == '__main__':
    unittest.main()