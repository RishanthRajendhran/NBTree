import numpy as np
from sklearn import tree 
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
import argparse
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor

def getAccuracy(Y, valWeights, Preds):
    if len(Y) != len(Preds):
        print("Y and Preds are not of same size!")
        return 0
    weights = np.array(valWeights).astype(np.float64)
    correctPreds = (np.array(Preds).reshape(len(Preds),)==np.array(Y).reshape(len(Y),))
    return np.sum(weights*correctPreds)/len(Y)

parser = argparse.ArgumentParser()
parser.add_argument("-trainX", help="File path to preprocessed train data matrix")
parser.add_argument("-trainXWeights", help="File path to preprocessed train data weights matrix")
parser.add_argument("-testX", help="File path to preprocessed test data matrix")
parser.add_argument("-preds", help="File path to predictions")
parser.add_argument("-trainY", help="File path to preprocessed train data target labels matrix")
parser.add_argument("-testID", help="File path to preprocessed test data ID matrix")
parser.add_argument("-TVsplit", action="store_true", default=False)
parser.add_argument("-validX",help="File path to preprocessed validation data matrix")
parser.add_argument("-validY",help="File path to preprocessed validation data target labels matrix")
parser.add_argument("-validXWeights",help="File path to preprocessed validation data weights matrix")

args = parser.parse_args()

if args.trainX:
    train_x_path = args.trainX
else:
    train_x_path = "./Data/train_x_final.csv"

if args.trainXWeights:
    weights_path = args.trainXWeights
else:
    weights_path = "./Data/weights_final.csv"

if args.testX:
    test_x_path = args.testX
else:
    test_x_path = "./Data/test_x_final.csv"

if args.preds:
    preds_path = args.testX
else:
    preds_path = "./Data/preds_final.csv"

if args.trainY:
    train_y_path = args.trainY
else:
    train_y_path = "./Data/train_y_final.csv"

if args.testID:
    test_id_path = args.testID
else:
    test_id_path = "./Data/test_id_final.csv"

TVsplit = args.TVsplit

if TVsplit:
    print("Performing train-validation split...")
    if not args.trainX:
        train_x_path = "./Data/train_x_finalTV.csv"
    if not args.trainY:
        train_y_path = "./Data/train_y_finalTV.csv"
    if not args.trainXWeights:
        weights_path = "./Data/weights_finalTV.csv"

if args.validX:
    valid_x_path = args.validX
else: 
    valid_x_path = "./Data/valid_x_final.csv"

if args.validY:
    valid_y_path = args.validY
else: 
    valid_y_path = "./Data/valid_y_final.csv"

if args.validXWeights:
    valid_weights_path = args.validXWeights 
else:
    valid_weights_path = "./Data/valid_weights_final.csv"


with open(train_x_path, newline="") as csvFile:
    X = np.array(list(csv.reader(csvFile, delimiter=",")))

with open(weights_path, newline="") as csvFile:
    weights = np.ndarray.flatten(np.array(list(csv.reader(csvFile, delimiter=","))))
    weights = np.ndarray.flatten(weights)

with open(test_x_path, newline="") as csvFile:
    test = np.array(list(csv.reader(csvFile, delimiter=",")))

with open(train_y_path, newline="") as csvFile:
    Y = np.array(list(csv.reader(csvFile, delimiter=",")))
    Y = np.ndarray.flatten(Y)

with open(test_id_path, newline="") as csvFile:
    ID = np.array(list(csv.reader(csvFile, delimiter=",")))
    ID = np.ndarray.flatten(ID)

if TVsplit:
    with open(valid_x_path, newline="") as csvFile:
        valX = np.array(list(csv.reader(csvFile, delimiter=",")))

    with open(valid_y_path, newline="") as csvFile:
        valY = np.ndarray.flatten(np.array(list(csv.reader(csvFile, delimiter=","))))
        valY = np.ndarray.flatten(valY)

    with open(valid_weights_path, newline="") as csvFile:
        valWeights = np.ndarray.flatten(np.array(list(csv.reader(csvFile, delimiter=","))))
        valWeights = np.ndarray.flatten(valWeights)
else:
    valX = X.copy()
    valY = Y.copy()
    valWeights = weights.copy()

clf = BaggingClassifier(n_estimators=100, random_state=0)
# clf = tree.DecisionTreeClassifier()
# clf = RandomForestRegressor(n_estimators=100, random_state=0)
clf = clf.fit(X,Y,sample_weight=weights)

# print(X.shape)
# print(clf.feature_importances_)
# print(len(clf.predict(X)))
# print(len(Y))
print(f"Train Data Accuracy: {getAccuracy(clf.predict(X), weights, Y)}")
print(f"Validation Data Accuracy: {getAccuracy(clf.predict(valX), valWeights, valY)}")
preds = clf.predict(test)
preds = preds.reshape(len(preds),1)
ID = ID.reshape(len(ID),1)
IDpreds = np.concatenate((ID,preds),axis=1)
with open(preds_path,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Prediction'])
    w.writerows(IDpreds)
