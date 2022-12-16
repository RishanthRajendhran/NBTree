from Modules.NBTree.helper.imports.packageImports import np, csv
from Modules.NBTree.helper.classes.NBTree import NBTree

trainXPath = "../Data/NBTree/trainX.csv"
trainYPath = "../Data/NBTree/trainY.csv"
valXPath = "../Data/NBTree/valX.csv"
valYPath = "../Data/NBTree/valY.csv"
testXPath = "../Data/NBTree/testX.csv"
testIDPath = "../Data/NBTree/testID.csv"

preds_path = "../Data/preds_final.csv"

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
testX = np.array(testX).astype(object)

with open(testIDPath,"r") as f:
    testID = list(csv.reader(f, delimiter=","))
testID = np.array(testID)
testID = np.ndarray.flatten(testID)

trainX = X[:-int(len(X)*0.3),:].copy().astype(object)
trainY = Y[:-int(len(X)*0.3)].copy().astype(int)

valX = X[-int(len(X)*0.3):,:].copy().astype(object)
valY = Y[-int(len(X)*0.3):].copy().astype(int)

attrsType = []
for i in range(X.shape[1]):
    if i in categorical_column_indices:
        attrsType.append("categorical")
    else:
        attrsType.append("numerical")

clf = NBTree()

clf = clf.fit(trainX, trainY, attrsType)

# clf.printTree()

print(f"Train Data Accuracy: {np.sum(clf.predict(trainX, attrsType)==trainY)/len(trainY)}")
print(f"Validation Data Accuracy: {np.sum(clf.predict(valX, attrsType)==valY)/len(valY)}")


preds = clf.predict(testX, attrsType)
preds = np.ndarray.flatten(preds)
preds = np.array([int(round(p)) for p in preds])
preds = preds.reshape(len(preds),1)
testID = testID.reshape(len(testID),1)
IDpreds = np.concatenate((testID,preds),axis=1)
with open(preds_path,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Prediction'])
    w.writerows(IDpreds)