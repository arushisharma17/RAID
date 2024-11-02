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
        self.test_dir = tempfile.mkdtemp()
        self.annotator = ActivationAnnotator(
            model_name="bert-base-uncased",
            binary_filter="set:public,static"
        )

    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_handle_binary_filter_set(self):
        """Test binary filter handling with set"""
        self.annotator.handle_binary_filter()
        self.assertEqual(self.annotator.binary_filter_compiled, {"public", "static"})

    def test_handle_binary_filter_regex(self):
        """Test binary filter handling with regex"""
        self.annotator.binary_filter = "re:public|static"
        self.annotator.handle_binary_filter()
        self.assertTrue(self.annotator.binary_filter_compiled.match("public"))
        self.assertTrue(self.annotator.binary_filter_compiled.match("static"))
        self.assertFalse(self.annotator.binary_filter_compiled.match("private"))

    def test_aggregate_activation_list(self):
        """Test activation aggregation methods"""
        test_activations = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        
        # Test mean aggregation
        mean_result = self.annotator.aggregate_activation_list(test_activations, method='mean')
        np.testing.assert_array_equal(mean_result, np.array([2.0, 3.0]))
        
        # Test max aggregation
        max_result = self.annotator.aggregate_activation_list(test_activations, method='max')
        np.testing.assert_array_equal(max_result, np.array([3.0, 4.0]))

    @patch('neurox.data.extraction.transformers_extractor.extract_representations')
    def test_generate_activations(self, mock_extract):
        """Test activation generation"""
        input_file = os.path.join(self.test_dir, "input.txt")
        output_file = os.path.join(self.test_dir, "output.json")
        
        self.annotator.generate_activations(input_file, output_file)
        
        mock_extract.assert_called_once_with(
            "bert-base-uncased",
            input_file,
            output_file,
            aggregation="average",
            output_type="json",
            device="cpu"
        )

if __name__ == '__main__':
    unittest.main()