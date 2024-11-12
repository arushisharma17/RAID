import csv
import re
import time

from extract_patterns import PatternExtractor
from label_dictionary import LabelDictionary
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
        token_iter = iter(tokens)
        cut_to_next_line = True
        with (open(file_name + '.in', 'a') as file_in, open(file_name + '.label', 'a') as file_labels,
              open(file_name + '.bio', 'w') as file_bio):
            lines = string.split('\n')
            line_enum = iter(lines)
            for i, line in enumerate(line_enum):
                if i == 5:
                    pass
                print(i)
                if len(line.strip()) == 0:
                    file_in.write('\n')
                    file_labels.write('\n')
                    file_bio.write('\n')
                    continue
                elif line.lstrip()[:2] == '/*' and len(tokens[prev:token_index+1]) > 0:
                    file_in.write(' '.join(tokens[prev:token_index+1]).strip() + '\n')
                    file_labels.write(' '.join(label[2:] for label in labels[prev:token_index+1]).replace(' ', '') + '\n')
                    file_bio.write(' '.join(label[0] for label in labels[prev:token_index+1]).replace(' ', '') + '\n')
                    token_str = ''
                    for it in range(''.join(tokens[prev:token_index+1]).count('\n')):
                        next(line_enum)
                    prev = token_index + 1
                    token_index += 1
                    next(token_iter)
                    continue
                line = line.replace(" ", "")
                while abs(prev - token_index < 500):  # temp condition
                    t = next(token_iter)
                    token_index += 1
                    stripped_t = t.replace(" ", "")
                    if stripped_t.startswith('\\\\') and len(stripped_t) > 2:
                        stripped_t = stripped_t.replace('\\\\', '\\')
                    token_str += stripped_t
                    test_token_str = token_str.replace('\\', '').strip()
                    test_line = line.replace('\\', '').strip().replace('\t', '')
                    t_count = t.count('\n')
                    if t_count > 0:
                        for it in range(t_count-1):
                            next(line_enum)
                        cut_to_next_line = line in t or line == token_str[:token_str.find('\n')]
                        break
                    elif test_line.endswith(test_token_str):
                        if not test_line.startswith(test_token_str) or test_line == test_token_str:
                            cut_to_next_line = True
                            break
                    elif len(test_token_str) > len(test_line):
                        cut_to_next_line = True
                        break
                file_in.write(' '.join(tokens[prev:token_index]) + ' ')
                file_labels.write(' '.join(label[2:] for label in labels[prev:token_index]) + ' ')
                file_bio.write(' '.join(label[0] for label in labels[prev:token_index]) + ' ')
                print(f'{token_str}\nVS\n{line}')
                print('line break', cut_to_next_line)
                if cut_to_next_line:
                    file_in.write('\n')
                    file_labels.write('\n')
                    file_bio.write('\n')
                if token_str.find('\n') > -1 and cut_to_next_line:
                    next(line_enum)
                token_str = ''
                prev = token_index


    def generate_in_label_bio_files(self, source_file, language, label_type):
        """
        Generates .in, .label, and .bio files for the given text file.

        Parameters
        ----------
        source_file : str
            The text file containing elements to be written to the .in and .label files.
        language : str
            The language to extra labels in.
        label_type : str
            The desired label (non-leaf) to be parsed.
        """
        label_dictionary = LabelDictionary()
        file_name = 'output/' + os.path.basename(source_file).split('.')[0]
        tokens = []
        bio_labels = []
        with (open(file_name + '.in', 'w') as file_in, open(file_name + '.label', 'w') as file_label,
              open(file_name + '.bio', 'w') as file_bio):
            file_in.write('')
            file_label.write('')
            file_bio.write('')


        strings = self.read_file(source_file)

        if not os.path.isfile(file_name + '.csv'):
            print("No CSV Found")
            extractor = PatternExtractor()
            # strings = self.read_file(source_file)
            for st in strings:
                extractor.get_all_bio_labels(bytes(st, encoding='utf8'), language, file_name)
        print("CSV Finished")
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        with open(file_name + '.csv', mode='r') as file:
            csv_file = csv.reader(file)
            iter_csv = iter(csv_file)
            next(iter_csv)
            for i, lines in enumerate(iter_csv):
                tokens.append(lines[0])
                bio_labels.append(lines[label_dictionary.non_leaf_types[label_type]])
        print("Appending Finished")
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
