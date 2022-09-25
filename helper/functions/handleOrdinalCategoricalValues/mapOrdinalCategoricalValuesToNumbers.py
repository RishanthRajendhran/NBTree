from helper.imports.packageImports import np, preprocessing, stats
#mapOrdinalCategoricalValuesToNumbers
#Input - X          : Numpy data matrix of data instances
#        weights    : Numpy array of weights of data instances
#        indices    : An array of the attribute indices of X which need to be label encoded
#Output - Modified data matrix X, Modes of columns in indices
#What it does - This function is used to encode ordinal categorical attribute values
#               as numerical values
#Assumption - No NULL attribute values in X
def mapOrdinalCategoricalValuesToNumbers(X, weights, indices):
    colModesCat = []
    for i in indices:
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(X[:,i]))
        X[:,i] = np.array((le.transform(X[:,i])))
        colModesCat.append(stats.mode(X[:,i])[0][0])
    return X, weights, colModesCat