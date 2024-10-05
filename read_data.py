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
    prev = 0
    with open(file_name, 'a') as file:
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
                file.write(labels[i] + ' ')
            file.write('\n')
            prev = prev + count
            token_index = 0
        file.write('\n')



def main():
    strings = read_file('java.in')
    print(strings)
    with open('java.label', 'w') as file:
        file.write('')
    for st in strings:
        tokens, labels, _ = extract_bio_labels_from_source_code(bytes(st, encoding='utf8'), 'java')
        write_file('java.label', st, tokens, labels)


if __name__ == "__main__":
    main()
