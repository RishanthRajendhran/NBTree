from scipy.stats import entropy

def entropy(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

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
