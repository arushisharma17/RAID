from extract_patterns import extract_bio_labels_from_source_code


def read_file(file_name):
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
            else:
                strings.append(st.strip())
                st = ''
        strings.append(st)
    return strings


def write_file(file_name, string, tokens, labels):
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
    with open(file_name + '.in', 'a') as file_in, open(file_name + '.label', 'a') as file_labels:
        lines = string.split('\n')
        token_index = 0
        for l in lines:
            for count, t in enumerate(tokens[prev:]):
                ti = l.find(t, token_index)
                if ti >= token_index:
                    token_index = ti
                else:
                    break
            if count == 0:
                count = count + 1
            for i in range(prev, count + prev):
                file_in.write(tokens[i] + ' ')
                file_labels.write(labels[i] + ' ')
            file_in.write('\n')
            file_labels.write('\n')
            prev = prev + count if count > 1 or prev > 0 else 0
            token_index = 0
        file_in.write('\n')
        file_labels.write('\n')


def generate_in_and_label_files(file_name, language):
    """
    Generates .in and .label files for the given text file.

    Parameters
    ----------
    file_name : str
        The text file containing elements to be written to the .in and .label files.
    language : str
        The language to extra labels in.
    """
    if file_name[-4:] != '.txt':
        raise Exception("Input a text file.")

    strings = read_file(file_name)
    file = file_name[:-4]
    with open(file + '.in', 'w') as file_in, open(file + '.label', 'w') as file_label:
        file_in.write('')
        file_label.write('')
    for st in strings:
        tokens, labels, _ = extract_bio_labels_from_source_code(bytes(st, encoding='utf8'), language)
        write_file(file, st, tokens, labels)


def main():
    generate_in_and_label_files('code.txt', 'java')


if __name__ == "__main__":
    main()
