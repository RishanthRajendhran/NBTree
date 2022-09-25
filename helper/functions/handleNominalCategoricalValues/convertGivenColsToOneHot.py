from helper.imports.packageImports import np
from helper.functions.handleNominalCategoricalValues.applyOneHotEncoding import apply_one_hot_encoding
#convert_given_cols_to_one_hot
#Input - X                      : Numpy data matrix of data instances
#        weights                : Numpy array of weights of data instances
#        column_indices         : An array of the attribute indices of X which need     
#                                 to be one-hot encoded
#        handleMulticolinearity : Boolean flag indicating whether 
#                                 multicolinearity needs to be handled
#                                 Default: False
#Output - One-hot encoded data matrix X, Numpy array of weights of data instances
#What it does - This function is used to one-hot encode a set of nominal categorical
#               attributes
#Assumption - No NULL attribute values in X
def convert_given_cols_to_one_hot(X, weights, column_indices, colVals, handleMulticolinearity=False):
    newX = np.copy(X)
    colOffset = 0
    for col in column_indices:
        curCol = X[:,col+colOffset]
        enc = np.array(apply_one_hot_encoding(curCol, colVals[col]))
        #Drop the first column to handle multicolinearity 
        if handleMulticolinearity and len(colVals[col]) > 1:
            enc = enc[:,1:].copy()
        if col == 0:
            X = np.concatenate((enc, X[:,(col+1):]),axis=1)
        elif col == len(X[0])-1:
            X = np.concatenate((X[:,0:(col+colOffset)], enc), axis=1)
        else:
            newX = np.concatenate((X[:,0:(col+colOffset)], enc), axis=1)
            X = np.concatenate((newX, X[:,(col+1+colOffset):]),axis=1)
        colOffset += (len(enc[0])-1)
    return X, weights