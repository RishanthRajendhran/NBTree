import numpy as np
from sklearn import tree 
import csv
import matplotlib.pyplot as plt

def getAccuracy(Y, Preds):
    if len(Y) != len(Preds):
        print("Y and Preds are not of same size!")
        return 0
    return np.sum(np.array(Preds).reshape(len(Preds),)==np.array(Y).reshape(len(Y),))/len(Y)

train_x_path = "./Data/train_x_final.csv"
test_x_path = "./Data/test_x_final.csv"
train_y_path = "./Data/train_y_final.csv"
test_id_path = "./Data/test_id_final.csv"
preds_path = "./Data/preds_final.csv"

with open(train_x_path, newline="") as csvFile:
    X = np.array(list(csv.reader(csvFile, delimiter=",")))

with open(test_x_path, newline="") as csvFile:
    test = np.array(list(csv.reader(csvFile, delimiter=",")))

with open(train_y_path, newline="") as csvFile:
    Y = np.array(list(csv.reader(csvFile, delimiter=",")))

with open(test_id_path, newline="") as csvFile:
    ID = np.array(list(csv.reader(csvFile, delimiter=",")))

# X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
# X_train,y_train = X, Y

# clf = tree.DecisionTreeClassifier(random_state=0)
# path = clf.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities

# fig, ax = plt.subplots()
# ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("total impurity of leaves")
# ax.set_title("Total Impurity vs effective alpha for training set")
# plt.show()
# exit(0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
# print(len(clf.predict(X)))
# print(len(Y))
# print(f"Train Accuracy: {getAccuracy(clf.predict(X), Y)}")
preds = clf.predict(test)
preds = preds.reshape(len(preds),1)
ID = ID.reshape(len(ID),1)
IDpreds = np.concatenate((ID,preds),axis=1)
with open(preds_path,'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','Prediction'])
    w.writerows(IDpreds)