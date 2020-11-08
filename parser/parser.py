import pandas as pd
import dataset_entry as entry

dataset_file = "mps.dataset.xlsx"


def parse_dataset():
    df = pd.read_excel(dataset_file)

    df = df[df[entry.test_result].isin({"POZITIV", "NEGATIV"})]
    df = df.dropna(subset=[entry.declared_symptoms, entry.test_result])

    df = df.applymap(lambda x: x.lower() if type(x) == str else x)

    df.to_excel("output_dataset.xlsx")
    df.to_csv("output_dataset.csv")


if __name__ == "__main__":
    parse_dataset()
