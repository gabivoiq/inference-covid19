import sys

from main.parser import *

filename = 'svclass.sav'


# def test(df_exe):
#     svclass = pickle.load(open(filename, 'rb'))
#
#     data = df_exe.drop([e.test_result], axis=1)
#     target = df_exe[e.test_result]
#     y_pred = svclass.predict(data)
#     print(classification_report(target, y_pred))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('You should run the program as ./test.py <xlsx_file>')

    dataset_name = sys.argv[1]
    df = parse_dataset(dataset_name)
    df = encode_data(df)

    test(df, None, None)
