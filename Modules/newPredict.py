import numpy as np
import sys
import csv

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm

import tensorflow as tf
import keras as ke

model = sys.argv[1]

trainXPath = "./Data/New/trainX.csv"
trainYPath = "./Data/New/trainY.csv"
valXPath = "./Data/New/valX.csv"
valYPath = "./Data/New/valY.csv"
testXPath = "./Data/New/testX.csv"
testIDPath = "./Data/New/testID.csv"

preds_path = "./Data/preds_final.csv"

feature_names = ["age", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country"]

categorical_column_indices = [1, 3, 5, 6, 7, 8, 9, 13]
numerical_column_indices = [0, 2, 4, 10, 11, 12]

with open(trainXPath,"r") as f:
    X = list(csv.reader(f, delimiter=","))
X = np.array(X)

with open(trainYPath,"r") as f:
    Y = list(csv.reader(f, delimiter=","))
Y = np.array(Y)
Y = np.ndarray.flatten(Y)

with open(testXPath,"r") as f:
    testX = list(csv.reader(f, delimiter=","))
testX = np.array(testX).astype(np.float64)

with open(testIDPath,"r") as f:
    testID = list(csv.reader(f, delimiter=","))
testID = np.array(testID)
testID = np.ndarray.flatten(testID)

trainX = X[:-int(len(X)*0.3),:].copy().astype(np.float64)
trainY = Y[:-int(len(X)*0.3)].copy().astype(int)

valX = X[-int(len(X)*0.3):,:].copy().astype(np.float64)
valY = Y[-int(len(X)*0.3):].copy().astype(np.float64)

attrsType = []
for i in range(X.shape[1]):
    if i in categorical_column_indices:
        attrsType.append("categorical")
    else:
        attrsType.append("numerical")

if model == "decisionTree":
    clf = tree.DecisionTreeClassifier()
    # print(tree.export_text(clf))
elif model == "adaBoost":
    clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
elif model == "bagging":
    clf = BaggingClassifier(n_estimators=500, random_state=0)
elif model == "randomForest":
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
elif model == "gradientBoosting":
    clf = GradientBoostingClassifier(n_estimators=750, random_state=0, min_samples_leaf=25)
elif model == "linearSVM":
    clf = svm.LinearSVC()
elif model == "nonLinearSVM":
    clf = svm.SVC(kernel="poly")

if model == "neuralNetwork":
    clf = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="tanh"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(512, activation="tanh"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(512, activation="tanh"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(512, activation="tanh", kernel_regularizer=ke.regularizers.l2(0.1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    clf.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate = 1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics="accuracy"
    )

    clf.fit(trainX, trainY, epochs=5000)

    print(clf.summary())

    testLoss, testAcc = clf.evaluate(valX, valY, verbose=2)

    print("\nTest Accuracy: ",testAcc)
else:
    clf = clf.fit(trainX, trainY)
    print(f"Train Data Accuracy: {np.sum(clf.predict(trainX)==trainY)/len(trainY)}")
    print(f"Validation Data Accuracy: {np.sum(clf.predict(valX)==valY)/len(valY)}")

preds = clf.predict(testX)
preds = np.ndarray.flatten(preds)
preds = np.ndarray.flatten(preds)
preds = np.array([int(round(p)) for p in preds])
preds = preds.reshape(len(preds),1)
testID = testID.reshape(len(testID),1)
IDpreds = np.concatenate((testID,preds),axis=1)
with open(preds_path,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Prediction'])
    w.writerows(IDpreds)