from requests import get
from time import sleep
from typing import BinaryIO, List

def fetch(url: str, output: BinaryIO, s: int) -> bool:
    sleep(s)
    try:
        response = get(url, stream=True, timeout=30)
        if 200 == response.status_code:
            output.write(response.text.encode('us-ascii', 'replace'))
            return True
        else:
            print(f'{url}  {response.status_code}')
    except Exception as e:
        print(e)
    return False

def main(out_file: str, *path_files: str):
    failures: List[str] = []
    with open(out_file, 'wb') as output:
        for (f_num, filename) in enumerate(path_files):
            with open(filename) as paths:
                for (l_num, line) in enumerate(map(str.strip, paths)):
                    if line:
                        print(f'{f_num} {l_num}')
                        if not fetch(line, output, len(failures)):
                            failures.append(line)
        for (e_num, failure) in enumerate(failures):
            print(f'f {e_num}')
            fetch(failure, output, len(failures))


if __name__ == "__main__":
    main("data/all.txt", "data/paths.txt")

