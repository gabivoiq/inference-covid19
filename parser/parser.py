import pandas as pd
import dataset_entry as entry

dataset_file = "mps.dataset.xlsx"


def parse_dataset():
    df = pd.read_excel(dataset_file)
    df_parsed = df[df[entry.test_result].notnull() &
                   ~df[entry.test_result].isin({"NECONCLUDENT"}) &
                   df[entry.declared_symptoms].notnull()]
    print(df_parsed)


if __name__ == "__main__":
    parse_dataset()
