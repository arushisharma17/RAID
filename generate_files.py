from extract_patterns import PatternExtractor
from label_dictionary import LabelDictionary
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
                if line != '\n':
                    st += line
                elif line != '\n':
                    strings.append(st.strip())
                    st = ''
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
            file_in.write('\n')
            file_labels.write('\n')
            file_bio.write('\n')


    def generate_in_and_label_files(self, source_file, language, depth=-1):
        """
        Generates .in and .label files for the given text file.

        Parameters
        ----------
        source_file : str
            The text file containing elements to be written to the .in and .label files.
        language : str
            The language to extra labels in.
        depth : int
            The desired depth of the AST tree to parse.
        """
        extractor = PatternExtractor()
        strings = self.read_file(source_file)
        file_name = 'output/' + os.path.basename(source_file).split('.')[0]
        with (open(file_name + '.in', 'w') as file_in, open(file_name + '.label', 'w') as file_label,
              open(file_name + '.bio', 'w') as file_bio):
            file_in.write('')
            file_label.write('')
            file_bio.write('')
        for st in strings:
            tokens, labels, _ = extractor.extract_bio_labels_from_source_code(bytes(st, encoding='utf8'), language, depth)
            self.write_file(file_name, st, tokens, labels)


    def generate_json_file(self, source_file, language):
        extractor = PatternExtractor()
        file_name = 'output/' + os.path.basename(source_file).split('.')[0]
        strings = self.read_file(source_file)
        code = '\n'.join(strings)
        extractor.create_tree_json(bytes(code, encoding='utf8'), language, file_name)



# if __name__ == "__main__":
#     main()
