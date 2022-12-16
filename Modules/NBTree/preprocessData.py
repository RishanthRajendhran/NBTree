from Modules.NBTree.helper.imports.packageImports import np, csv, SimpleImputer, KNNImputer, preprocessing

trainPath = "../Data/train_final.csv"
testPath = "../Data/test_final.csv"
testIDPath = "../Data/test_id_final.csv"
preds_path = "../Data/preds_final.csv"

outTrainXPath = "../Data/NBTree/trainX.csv"
outTrainYPath = "../Data/NBTree/trainY.csv"
outTestXPath = "../Data/NBTree/testX.csv"
outTestIDPath = "../Data/NBTree/testID.csv"

categorical_column_indices = [1, 3, 5, 6, 7, 8, 9, 13]
numerical_column_indices = [0, 2, 4, 10, 11, 12]

def writeToCsvFile(x, x_path):
    if x.ndim == 1:
        x = x.reshape(x.shape[0], 1)
    with open(x_path, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(x)
        csv_file.close()

with open(trainPath,"r") as f:
    X = list(csv.reader(f, delimiter=","))
X = [np.array(X[i],dtype=object) for i in range(len(X))]
X = np.array(X)

with open(testPath,"r") as f:
    testX = list(csv.reader(f, delimiter=","))
testX = [np.array(testX[i],dtype=object) for i in range(len(testX))]
testX = np.array(testX)

Y = X[1:,-1].copy()
X = X[1:,:-1].copy()

testID = testX[1:,0].copy()
testX = testX[1:,1:].copy()

#Handle missing categorical values
imp = SimpleImputer(missing_values="?",strategy="most_frequent").fit(X[:,categorical_column_indices])
X[:,categorical_column_indices] = imp.transform(X[:,categorical_column_indices])
testX[:,categorical_column_indices] = imp.transform(testX[:,categorical_column_indices])

#Handle missing numerical values
impKNN = KNNImputer(n_neighbors=15, weights="uniform").fit(X[:,numerical_column_indices])
X[:,numerical_column_indices] = impKNN.transform(X[:,numerical_column_indices])
testX[:,numerical_column_indices] = impKNN.transform(testX[:,numerical_column_indices])

for i in categorical_column_indices:
    le = preprocessing.LabelEncoder()
    le.fit(X[:,i])
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    X[:,i] = le.transform(X[:,i])
    for x in range(len(testX)):
        res = le_dict.get(testX[x, i],"unknown")
        if res == "unknown":
            testX[x, i] = -1
        else:
            testX[x, i] = res

writeToCsvFile(X, outTrainXPath)
writeToCsvFile(np.ndarray.flatten(Y), outTrainYPath)
writeToCsvFile(testX, outTestXPath)
writeToCsvFile(testID, outTestIDPath)

