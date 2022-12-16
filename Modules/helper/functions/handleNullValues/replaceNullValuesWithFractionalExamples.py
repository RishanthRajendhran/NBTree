from helper.imports.packageImports import np
#replace_null_values_with_fractional_examples
#Input - X          : Numpy data matrix of data instances
#        weights    : Numpy array of weights of data instances
#        Y          : Numpy matrix of target labels
#        colVals    : Dictionary of unique attribute values with column indices of X as keys
#        attrsType  : String value indicating the data type of the attributes
#                    Options: str, float
#        indices    : An array of the indices of X for which NULL values need to be handled
#Output - Modified data matrix X, Weights of instances in X, Target labels Y
#What it does - This function is used to handle NULL values in case of numerical attributes
#               NULL values are replaced by the fractional examples of all non-NULL values for the attribute
#               under consideration in the data instances in X  
#Assumption - Numerical data is treated as being of np.float64 data type
#             All attributes are of the same type
def replace_null_values_with_fractional_examples(X, weights, Y, colVals, attrsType, indices):
    if len(indices) == 0:
        return X, weights, Y
    newWeights = weights.copy()
    modifiedWeights = []
    newX = X.copy()
    newY = Y.copy()
    modifiedX = []
    modifiedY = []
    for col in indices:
        inds = np.where(X[:,col].astype("str")=="?")[0]
        uniqueVals = colVals[col]
        if attrsType == "str":
            nullInds = np.where(X[:,indices].astype("str")=="?")
            nonNullInds = np.setdiff1d(np.arange(np.array(newX).shape[0]), nullInds)
        else:
            nonNullInds = np.setdiff1d(np.arange(np.array(newX).shape[0]), np.where(np.isnan(X[:,col].astype(np.float64))))
        for i in range(np.array(newX).shape[0]):
            if i not in inds:
                modifiedX.append(newX[i])
                modifiedWeights.append(newWeights[i])
                modifiedY.append(newY[i])
            else: 
                for uVal in uniqueVals:
                    newRow = newX[i].copy()
                    newRow[col] = str(uVal) 
                    modifiedX.append(newRow)
                    if attrsType == "str":
                        valCount = np.sum(np.array(newX)[nonNullInds,col] == str(uVal))
                    else:
                        valCount = np.sum(newX[nonNullInds,col].astype(np.float64) == np.float64(uVal))
                    modifiedWeights.append(newWeights[i]*(valCount/nonNullInds.shape[0]))
                    modifiedY.append(newY[i])
        newX = modifiedX.copy()
        modifiedX = []
        newWeights = modifiedWeights.copy()
        modifiedWeights = []
        newY = modifiedY.copy()
        modifiedY = []
    return np.array(newX), np.array(newWeights), np.array(newY)