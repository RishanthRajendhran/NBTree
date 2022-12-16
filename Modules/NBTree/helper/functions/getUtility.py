from Modules.NBTree.helper.imports.packageImports import np, KBinsDiscretizer, GaussianNB, KFold
from sklearn import svm

def getUtility(X, weights, Y, attrsType):
    if len(X) <= 1:
        return 1

    if len(np.unique(Y)) == 1:
        return 1

    # clf = svm.SVC(kernel="poly") 

    # clf.fit(X, Y)

    # preds = clf.predict(X)
    # preds = np.ndarray.flatten(preds)
    # preds = np.ndarray.flatten(preds)
    # preds = np.array([int(round(p)) for p in preds])
    # acc = np.sum(preds == Y)/len(Y)
    # return acc

    numerical_attributes = np.where(np.array(attrsType) == "numerical")[0]

    kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')
    discretizedX = X.copy()
    discretizedX[:, numerical_attributes] = kbins.fit_transform(X[:, numerical_attributes])
    gnb = GaussianNB()

    if len(discretizedX) < 5:
        preds = gnb.fit(discretizedX[:-1], Y[:-1], sample_weight=weights[:-1]).predict(discretizedX[-1:])
        acc = np.sum(weights[np.where(np.array(preds)==np.array(Y[-1:]))[0]])/np.sum(weights[-1:])
        return acc

    kf = KFold(n_splits=5)
    acc = 0
    for trainInds, testInds in kf.split(discretizedX):
        trainX, trainW, trainY = discretizedX[trainInds], weights[trainInds], Y[trainInds]
        testX, testW, testY = discretizedX[testInds], weights[testInds], Y[testInds]
        preds = gnb.fit(trainX, trainY, sample_weight=trainW).predict(testX)
        acc += np.sum(testW[np.where(np.array(preds)==np.array(testY))[0]])/np.sum(testW)
    acc /= 5

    return acc