from helper.imports.packageImports import np
#replace_null_values_with_mean
#Input - X       : Data matrix
#        weights : Numpy array of weights of data instances
#        indices : An array of the indices of X for which NULL values need to be handled
#        colMeans: Numpy array of means of columns under consideration
#                  Default: Compute colMeans internally
#Output - Modified data matrix X, Weights of data instances in X
#What it does - This function is used to handle NULL values in case of numerical attrinutes
#               NULL values are replaced by the weighted mean of all non-NULL values for the attribute
#               under consideration in the data instances in X  
#Assumption - Numerical data is treated as being of np.float64 data type
def replace_null_values_with_mean(X, weights, indices, colMeans = []):
    if len(indices) == 0:
        return X, weights
    if len(colMeans) == 0:
        nullInds = np.where(np.isnan(X[:,indices].astype(np.float64)))
        nonNullInds = np.setdiff1d(np.arange(X.shape[0]), nullInds)
        colMeans = np.average(X[nonNullInds,indices].astype(np.float64), axis=0, weights=weights[nonNullInds])

    inds = np.where(np.isnan(X[:,indices].astype(np.float64)))
    
    X[:,indices][inds] = colMeans[inds[1]].astype(str)
    return X, weights