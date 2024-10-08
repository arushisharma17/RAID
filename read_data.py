from extract_patterns import extract_bio_labels_from_source_code


def read_file(file_name):
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


def write_file(file_name, strings, tokens, labels):
    file = file_name[:-4]
    prev = 0
    with open(file + '.in', 'a') as file_in, open(file + '.label', 'a') as file_labels:
        lines = strings.split('\n')
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
            prev = prev + count
            token_index = 0
        file_in.write('\n')
        file_labels.write('\n')


def generate_in_and_label_files(file_name, language):
    strings = read_file(file_name)
    file = file_name[:-4]
    with open(file + '.in', 'w') as file_in, open(file + '.label', 'w') as file_label:
        file_in.write('')
        file_label.write('')
    for st in strings:
        tokens, labels, _ = extract_bio_labels_from_source_code(bytes(st, encoding='utf8'), language)
        write_file(file_name, st, tokens, labels)


def main():
    generate_in_and_label_files('java.txt', 'java')


if __name__ == "__main__":
    main()
