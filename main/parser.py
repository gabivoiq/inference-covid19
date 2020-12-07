import pandas as pd
import dataset_entry as e
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

dataset_file1 = "mps.dataset.xlsx"
dataset_exe = "exemplu.xlsx"
filename = 'svclass.sav'
label = LabelEncoder()
file_x = "x_test_dataset.csv"
file_y = "y_test_dataset.csv"


def parse_dataset(dataset_file):
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
    # _df.to_excel("output_dataset.xlsx")
    # _df.to_csv("output_dataset.csv")

    return _df


def encode_data(_df):

    _df = _df.apply(label.fit_transform)
    # _df = _df.apply(label.inverse_transform)
    # print(_df)
    return _df


def split_and_train(_df):
    data = _df.drop([e.test_result], axis=1)
    target = _df[e.test_result]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=1 / 3, random_state=0,
                                                        stratify=target)

    # _df = _df.apply(LabelEncoder().inverse_transform)
    # x_exe = LabelEncoder().inverse_transform(x_train)

    # _df.to_excel("train_dataset.xlsx")
    # print(y_test.shape)
    # print(x_test.shape)
    # frames = [x_test, y_test]
    # h = pd.concat(frames)
    # h = _df.apply(label.inverse_transform)
    # h.to_csv("x_test_dataset.csv")

    x_test.to_csv(file_x, index=False)
    y_test.to_csv(file_y, index=False)

    # dfff = pd.DataFrame(y_test)
    # print(dff[dff[e.test_result] == 1].size)
    # print(dfff[dfff[e.test_result] == 1].size)

    svclassifier = SVC(C=10, kernel='rbf', gamma=0.01)
    svclassifier.fit(x_train, y_train)
    pickle.dump(svclassifier, open(filename, 'wb'))


def test(_df, file_x, file_y):
    svclass = pickle.load(open(filename, 'rb'))

    if file_x is None and file_y is None:
        x_test = _df.drop([e.test_result], axis=1)
        y_test = _df[e.test_result]
    else:
        x_test = pd.read_csv(file_x)
        y_test = pd.read_csv(file_y)

    y_pred = svclass.predict(x_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print(metrics.auc(fpr, tpr))


if __name__ == "__main__":
    df = parse_dataset(dataset_file1)
    df = encode_data(df)
    split_and_train(df)
    test(df, file_x, file_y)