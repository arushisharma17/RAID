import csv

from .extract_patterns import PatternExtractor
from .label_dictionary import LabelDictionary
import os


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
        with open(file_name) as file:
            strings = []
            st = ''
            for line in file:
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
        label_dict = LabelDictionary()
        decreased = False
        prev = 0
        with (open(file_name + '.in', 'a') as file_in, open(file_name + '.label', 'a') as file_labels,
              open(file_name + '.bio', 'w') as file_bio):
            lines = string.split('\n')
            token_index = 0
            for line in lines:
                if line == '':
                    file_in.write('\n')
                    file_labels.write('\n')
                    file_bio.write('\n')
                    continue
                for count, t in enumerate(tokens[prev:]):
                    ti = line.find(t, token_index)
                    if ti >= token_index:
                        token_index = ti + 1
                    else:
                        if count >= 2:
                            count = count - 1
                            decreased = True
                        break
                if count == 1 and not decreased:
                    count = count - 1
                for (token, label) in zip(tokens[prev:prev + count + 1], labels[prev:prev + count + 1]):
                    file_in.write(token + ' ')
                    file_labels.write(label_dict.convert_label(label[2:]) + ' ')
                    file_bio.write(label[0] + ' ')
                file_in.write('\n')
                file_labels.write('\n')
                file_bio.write('\n')
                prev = prev + count + 1
                token_index = 0
                decreased = False


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

        if not os.path.isfile('output/' + file_name + '.csv'):
            extractor = PatternExtractor()
            strings = self.read_file(source_file)
            for st in strings:
                extractor.get_all_bio_labels(bytes(st, encoding='utf8'), language, file_name)

        with open(file_name + '.csv', mode='r') as file:
            csv_file = csv.reader(file)
            for i, lines in enumerate(csv_file):
                if i == 0:
                    continue
                tokens.append(lines[0])
                bio_labels.append(lines[label_dictionary.non_leaf_types[label_type]])

        for st in strings:
            self.write_file(file_name, st, tokens, bio_labels)


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


def main():
    g = TokenLabelFilesGenerator()
    g.generate_in_label_bio_files('input/small-src-chunck1.txt', 'java', 'class_body')
    g.generate_json_file('input/small-src-chunck1.txt', 'java')


if __name__ == "__main__":
    main()
