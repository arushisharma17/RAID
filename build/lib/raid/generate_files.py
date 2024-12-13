import csv
import re
import time

from .extract_patterns import PatternExtractor
from .label_dictionary import LabelDictionary
import os

start_time = time.time()

class TokenLabelFilesGenerator:
    def read_file(self, file_name):
        """
        Helper method that parses the contexts of a text file line by line and returns each element as a separate string.

        Parameters
        ----------
        file_name : str
            Name of the file.
        Returns
        -------
        List[str]
            A list of each element in the given text file.
        """
        with open(file_name, encoding="utf-8") as file:
            strings = []
            st = ''
            for line in file:
                line = re.sub(r'[^\x00-\x7F]+', '', line)
                # line = line.decode('utf-8', 'ignore').encode("utf-8")
                st += line
            strings.append(st)
        return strings


    def write_file(self, file_name, string, tokens, labels):
        """
        Helper method that writes to the .in and .label files.

        Parameters
        ----------
        file_name : str
            The name of the text file containing elements to be written to the .in and .label files.
        string : str
            The element to be parsed.
        tokens : List[str]
            The list of tokens to be parsed through.
        labels : List[str]
            The list of labels to be parsed through.
        """
        prev = 0
        token_index = 0
        token_str = ''
        tokens_len = len(tokens)  # Get total length of tokens
        
        with (open(file_name + '.in', 'a') as file_in, 
              open(file_name + '.label', 'a') as file_labels,
              open(file_name + '.bio', 'w') as file_bio):
            
            lines = string.split('\n')
            for i, line in enumerate(lines):
                if len(line.strip()) == 0:  # if line is blank, add newline and skip
                    file_in.write('\n')
                    file_labels.write('\n')
                    file_bio.write('\n')
                    continue
                    
                # Handle multiline comments
                if line.lstrip()[:2] == '/*' and prev < token_index + 1 and token_index < tokens_len:
                    file_in.write(' '.join(tokens[prev:token_index+1]).strip() + '\n')
                    file_labels.write(' '.join(label[2:] for label in labels[prev:token_index+1]).replace(' ', '') + '\n')
                    file_bio.write(' '.join(label[0] for label in labels[prev:token_index+1]).replace(' ', '') + '\n')
                    token_str = ''
                    prev = token_index + 1
                    token_index += 1
                    continue

                line = line.replace(" ", "")  # don't worry about spaces
                cut_to_next_line = True
                
                # Process tokens for this line
                while token_index < tokens_len and abs(prev - token_index) < 500:  # Add bounds check
                    current_token = tokens[token_index]
                    stripped_token = current_token.replace(" ", "")
                    
                    # Handle escaped characters
                    if stripped_token.startswith('\\\\') and len(stripped_token) > 2:
                        stripped_token = stripped_token.replace('\\\\', '\\')
                        
                    token_str += stripped_token
                    test_token_str = token_str.replace('\\', '').strip()
                    test_line = line.replace('\\', '').strip().replace('\t', '')
                    
                    t_count = current_token.count('\n')
                    if t_count > 0:  # if token is multiline
                        cut_to_next_line = line in current_token or line == token_str[:token_str.find('\n')]
                        break
                    elif test_line.endswith(test_token_str):  # if the line and token_str are a match
                        if not test_line.startswith(test_token_str) or test_line == test_token_str:
                            cut_to_next_line = True
                            break
                    elif len(test_token_str) > len(test_line):  # if token_str has become longer than the line
                        cut_to_next_line = True
                        break
                        
                    token_index += 1
                    if token_index >= tokens_len:
                        break
                
                # Write accumulated tokens
                if prev < token_index and prev < tokens_len:
                    file_in.write(' '.join(tokens[prev:token_index]) + ' ')
                    file_labels.write(' '.join(label[2:] for label in labels[prev:token_index]) + ' ')
                    file_bio.write(' '.join(label[0] for label in labels[prev:token_index]) + ' ')
                
                if cut_to_next_line:  # Add newlines if needed
                    file_in.write('\n')
                    file_labels.write('\n')
                    file_bio.write('\n')
                
                token_str = ''  # Reset for next line
                prev = token_index  # Update token start position


    def generate_in_label_bio_files(self, source_file, language, label_type):
        """
        Generates .in, .label, and .bio files for the given text file.
        """
        label_dictionary = LabelDictionary()
        
        # Use os.path.join for proper path handling
        output_dir = os.path.join(os.getcwd(), 'output')
        base_name = os.path.basename(source_file).split('.')[0]
        file_name = os.path.join(output_dir, base_name)
        
        tokens = []
        bio_labels = []

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize files
        with (open(file_name + '.in', 'w') as file_in, 
              open(file_name + '.label', 'w') as file_label,
              open(file_name + '.bio', 'w') as file_bio):
            file_in.write('')
            file_label.write('')
            file_bio.write('')

        strings = self.read_file(source_file)
        csv_file_path = file_name + '.csv'

        if not os.path.isfile(csv_file_path):
            print(f"Generating CSV file at {csv_file_path}")
            extractor = PatternExtractor()
            for st in strings:
                extractor.get_all_bio_labels(bytes(st, encoding='utf8'), language, file_name)
        
        print("CSV Finished")
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        try:
            with open(csv_file_path, mode='r') as file:
                csv_file = csv.reader(file)
                iter_csv = iter(csv_file)
                next(iter_csv)  # Skip header
                for i, lines in enumerate(iter_csv):
                    if len(lines) > 0:  # Make sure we have data
                        tokens.append(lines[0])
                        label_index = label_dictionary.non_leaf_types.get(label_type)
                        if label_index is None:
                            raise ValueError(f"Invalid label type: {label_type}")
                        if label_index < len(lines):
                            bio_labels.append(lines[label_index])
                        else:
                            print(f"Warning: Line {i} does not have enough columns")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise

        print("Appending Finished")
        print(f"Found {len(tokens)} tokens and {len(bio_labels)} labels")
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        for st in strings:
            self.write_file(file_name, st, tokens, bio_labels)
        
        print("Writing Finished")
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")


    def generate_json_file(self, source_file, language):
        """
        Generates .json file for the given file, listing tokens, their labels, and children recursively.

        Parameters
        ----------
        source_file : str
            The text file containing elements to be written to the .in and .label files.
        language : str
            The language to extra labels in.
        """
        extractor = PatternExtractor()
        file_name = 'output/' + os.path.basename(source_file).split('.')[0]
        strings = self.read_file(source_file)
        code = '\n'.join(strings)
        extractor.create_tree_json(bytes(code, encoding='utf8'), language, file_name)


# def main():
#     g = TokenLabelFilesGenerator()
#     print("Generating In/Label/Bio")
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed time: {elapsed_time:.2f} seconds")
#     g.generate_in_label_bio_files('input/source-code-cleaned.txt', 'java', 'program')
#     # g.generate_in_label_bio_files('input/for.txt', 'java', 'program')
#     # g.generate_json_file('input/small-src-chunck1.txt', 'java')
#
#
# if __name__ == "__main__":
#     main()
