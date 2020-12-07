import sys

from main.parser import *

filename = 'svclass.sav'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('You should run the program as ./test.py <xlsx_file>')

    dataset_name = sys.argv[1]
    df = parse_dataset(dataset_name)
    df = encode_data(df)

    test(df, None, None)
