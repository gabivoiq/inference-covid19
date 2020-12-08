from main.parser import *


def process_data(file):
    df_test = parse_dataset(file)
    df_test = encode_data(df_test)

    return test(df_test, None, None)
