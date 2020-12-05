import pandas as pd
import dataset_entry as e
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

dataset_file = "mps.dataset.xlsx"


def parse_dataset():
    _df = pd.read_excel(dataset_file)

    _df = _df[_df[e.test_result].isin({"POZITIV", "NEGATIV"})]
    _df = _df.dropna(subset=[e.declared_symptoms, e.test_result])
    _df = _df.drop([e.source_institution, e.diagnosis_and_hospitalization_signs], 1)

    _df = _df.fillna(" ")
    _df = _df.applymap(lambda x: x.lower() if type(x) == str else x)
    _df = _df.applymap(lambda x: str(x))
    _df.loc[_df[e.declared_symptoms].str.contains('asimpt', case=False), e.declared_symptoms] = 'asimptomatic'
    _df.loc[_df[e.reported_symptoms_hospitalization]
        .str.contains('asimpt', case=False), e.reported_symptoms_hospitalization] = 'asimptomatic'

    print("NUMBER OF ENTRIES:", _df.shape)
    print("NEGATIVE RESULTS:", _df[_df[e.test_result] == 'negativ'].shape)
    print("POSITIVE RESULTS:", _df[_df[e.test_result] == 'pozitiv'].shape)
    _df.to_excel("output_dataset.xlsx")
    # _df.to_csv("output_dataset.csv")

    return _df


def encode_data(_df):
    _df = _df.apply(LabelEncoder().fit_transform)
    # print(_df)
    return _df


def next(_df):
    data = _df.drop([e.test_result], axis=1)
    target = _df[e.test_result]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=1 / 3, random_state=0,
                                                        stratify=target)

    # dff = pd.DataFrame(y_train)
    # dfff = pd.DataFrame(y_test)
    # print(dff[dff[e.test_result] == 1].size)
    # print(dfff[dfff[e.test_result] == 1].size)

    svclassifier = SVC(C=20, kernel='rbf', gamma=0.01)
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = parse_dataset()
    df = encode_data(df)
    next(df)
