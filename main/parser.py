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
    _df = _df.dropna(subset=[e.test_result])
    _df = _df.drop([e.source_institution, e.diagnosis_and_hospitalization_signs], 1)

    _df = _df.fillna(" ")
    _df = _df.applymap(lambda x: x.lower() if type(x) == str else x)
    _df = _df.applymap(lambda x: str(x))
    _df.loc[_df[e.declared_symptoms].str.contains('asimpt', case=False), e.declared_symptoms] = 'asimptomatic'
    _df.loc[_df[e.reported_symptoms_hospitalization].str.contains('asimpt',
                                                                  case=False), e.reported_symptoms_hospitalization] = 'asimptomatic'

    print("NUMBER OF ENTRIES:", _df.shape)
    print("NEGATIVE RESULTS:", _df[_df[e.test_result] == 'negativ'].shape)
    print("POSITIVE RESULTS:", _df[_df[e.test_result] == 'pozitiv'].shape)

    return _df


def encode_data(_df):
    _df = _df.apply(label.fit_transform)
    return _df


def split_and_train(_df):
    data = _df.drop([e.test_result], axis=1)
    target = _df[e.test_result]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=1 / 3, random_state=0,
                                                        stratify=target)

    print("POSITIVE RESULTS TRAIN:", y_train.loc[y_train == 1].size)
    print("POSITIVE RESULTS TEST:", y_test.loc[y_test == 1].size)

    x_test.to_csv(file_x, index=False)
    y_test.to_csv(file_y, index=False)

    svclassifier = SVC(C=10, kernel='rbf', gamma=0.01)
    svclassifier.fit(x_train, y_train)
    pickle.dump(svclassifier, open(filename, 'wb'))


def test(_df):
    svclass = pickle.load(open(filename, 'rb'))

    if file_x is None and file_y is None:
        x_test = _df.drop([e.test_result], axis=1)
        y_test = _df[e.test_result]
    else:
        x_test = pd.read_csv(file_x)
        y_test = pd.read_csv(file_y)

    y_pred = svclass.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", matrix)
    accuracy = (matrix[0][0] + matrix[0][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
    print("Accuracy:", accuracy)
    precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    print("Precision:", precision)
    rappel = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    print("Rappel:", rappel)
    f1_score = (2 * precision * rappel) / (precision + rappel)
    print("F1 score:", f1_score)
    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    print("AUC:", auc)
    data = {
        "accuracy" : accuracy,
        "confusion_matrix": [
            [int(matrix[0][0]), int(matrix[0][1])],
            [int(matrix[1][0]), int(matrix[1][1])]
        ],
        "precision": precision,
        "rappel": rappel,
        "f1_score": f1_score,
        "auc": auc
    }
    print(data)

    return data


if __name__ == "__main__":
    df = parse_dataset(dataset_file1)
    df = encode_data(df)
    split_and_train(df)
    test(df)
