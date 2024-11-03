import unittest
from extract_patterns import PatternExtractor


hello_world_j = b'''
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
'''

for_loop_j = b'''
for (int i = 0; i < 10; i++) {
    System.out.println(i);
}
# '''

add_number_j = b'''
public int addNumbers(a, b) {
    return a + b;
}
'''

add_numbers_p = b'''
def add_numbers (a, b):
    return a + b
'''

comment_p = b'''
# Comment about function
'''

all_source_code = [hello_world_j, for_loop_j, add_number_j, add_numbers_p, comment_p]


class TestExtractPatterns(unittest.TestCase):
    def test_all_running(self):
        e = PatternExtractor()
        # e.extract_bio_labels_from_source_code(source_code, 'java')
        for source_code in all_source_code[:3]:
            e.get_all_bio_labels(source_code, 'java', 'test')
        for source_code in all_source_code[3:]:
            e.get_all_bio_labels(source_code, 'python', 'test')
        self.assertTrue(True)
        # e.create_tree_json(source_code, 'java', 'test')

    def test_cannot_run_different_language(self):
        e = PatternExtractor()
        # e.extract_bio_labels_from_source_code(source_code, 'java')

        for source_code in all_source_code[3:]:
            self.assertFalse(e.get_all_bio_labels(source_code, 'java', 'test'))
        for source_code in all_source_code[:3]:
            self.assertFalse(e.get_all_bio_labels(source_code, 'python', 'test'))
        self.assertTrue(True)
        # e.create_tree_json(source_code, 'java', 'test')


if __name__ == '__main__':
    unittest.main()
