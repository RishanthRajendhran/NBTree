import numpy as np
import csv
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

trainPath = "./Data/train_final.csv"
testPath = "./Data/test_final.csv"
testIDPath = "./Data/test_id_final.csv"
preds_path = "./Data/preds_final.csv"

outTrainXPath = "./Data/New/trainX.csv"
outTrainYPath = "./Data/New/trainY.csv"
outTestXPath = "./Data/New/testX.csv"
outTestIDPath = "./Data/New/testID.csv"

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
impKNN = KNNImputer(n_neighbors=10, weights="uniform").fit(X[:,numerical_column_indices])
X[:,numerical_column_indices] = impKNN.transform(X[:,numerical_column_indices])
testX[:,numerical_column_indices] = impKNN.transform(testX[:,numerical_column_indices])

# min_max_scaler = MinMaxScaler()
# X[:,numerical_column_indices] = min_max_scaler.fit_transform(X[:,numerical_column_indices])

#One-hot encode categorical attributes
catEncs = {}
numCols = X.shape[1]
for col in categorical_column_indices:
    enc = OneHotEncoder(handle_unknown="ignore").fit(X[:,col].reshape(-1,1))
    numCols += (len(enc.categories_[0])-1)-1
    catEncs[col] = enc
newX = np.zeros((X.shape[0],numCols))

k = 0
for col in range(X.shape[1]):
    if col in categorical_column_indices:
        newCols = catEncs[col].transform(X[:,col].reshape(-1,1)).toarray()
        #Handle multicolinearity 
        newCols = newCols[:, 1:].copy()
        newX[:,k:k+newCols.shape[1]] = newCols
        k += newCols.shape[1]
    else:
        newX[:,k] = X[:,col].copy()
        k += 1
X = newX.copy()

k = 0
newX = np.zeros((testX.shape[0],numCols))
for col in range(testX.shape[1]):
    if col in categorical_column_indices:
        newCols = catEncs[col].transform(testX[:,col].reshape(-1,1)).toarray()
        #Handle multicolinearity 
        newCols = newCols[:, 1:].copy()
        newX[:,k:k+newCols.shape[1]] = newCols
        k += newCols.shape[1]
    else:
        newX[:,k] = testX[:,col].copy()
        k += 1
testX = newX.copy()

#Standardize data
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
testX = scaler.transform(testX)

writeToCsvFile(X, outTrainXPath)
writeToCsvFile(np.ndarray.flatten(Y), outTrainYPath)
writeToCsvFile(testX, outTestXPath)
writeToCsvFile(testID, outTestIDPath)

