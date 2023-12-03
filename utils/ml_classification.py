from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score
import pickle
import numpy as np


RESULT_PATH = Path("result/")
DATA_PATH = Path("data2/")


classifiers = {
    'logistic_regression':LogisticRegression(random_state=42,max_iter=10),
    'support_vector_machine':SVC(random_state=42,probability=True),
    'random_forest':RandomForestClassifier(n_estimators=2,random_state=42),
    'neural_network':MLPClassifier(random_state=42,max_iter=10),
    }



if __name__ =='__main__':
    features = []
    labels = []
    root_train = './dataset/adni1_train.pkl' # adni2_train.pkl
    root_test = './dataset/adni1_test.pkl'# adni2_test.pkl
    with open(root_train, 'rb') as load_data1:
        data_dict_train = pickle.load(load_data1)
    with open(root_test, 'rb') as load_data2:
        data_dict_test = pickle.load(load_data2)
    keys = ['CN', 'MCI', 'AD']
    for i in range(len(keys)):
        list1 = data_dict_train[keys[i]]
        features.extend(list1)
        labels.extend([i] * len(list1))
        list2 = data_dict_test[keys[i]]
        features.extend(list2)
        labels.extend([i] * len(list2))
    stratifiedKFolds = StratifiedKFold(n_splits=10, shuffle=False)
    test_performance = {}
    for m in ['logistic_regression','support_vector_machine','random_forest','neural_network']:
        print('model: ',m)
        # m = 'random_forest'
        for (trn_idx, val_idx) in stratifiedKFolds.split(features, labels):
            x_train = [features[id1] for id1 in trn_idx]
            y_train = [labels[id1] for id1 in trn_idx]
            x_train = [x[0] for x in x_train]

            x_test = [features[id1] for id1 in val_idx]
            y_test = [labels[id1] for id1 in val_idx]
            x_test = [x[0] for x in x_test]

            model = classifiers[m]
            # training
            model_trn = model.fit(x_train, y_train)
            # test
            predictions = model_trn.predict(x_test)
            class_report = classification_report(y_test, predictions)
            print(class_report)




