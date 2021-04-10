from re import compile

GOODLINE = compile('[a-zA-Z]+ [a-zA-Z]+ [a-zA-Z]+')
SCRUBBER = compile("[^\w .,;:!?']")
WHITESPACE = compile('\s+')

def main(input_filename: str, output_filename: str):
    with open(input_filename, 'r', encoding='us-ascii') as input_file, open(output_filename, 'wb') as output_file:
        for (index, line) in enumerate(input_file):
            goodline = GOODLINE.search(line) is not None
            scrubbed = SCRUBBER.sub('', WHITESPACE.sub(' ', line))
            if 0 == index % 100000:
                print(f'Line {index}: keep: {str(goodline)}, scrubbed: {scrubbed}')
            if goodline:
                output_file.write(scrubbed.encode('us-ascii', 'replace'))

if __name__ == "__main__":
    main("data/all.txt", "data/scrubbed.txt")

