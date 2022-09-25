from helper.imports.packageImports import np, weighted_mode
#replace_null_values_with_mode
#Input - X       : Data matrix
#        weights : Numpy array of weights of data instances
#        attrsType  : String value indicating the data type of the attributes
#                    Options: str, float
#        indices    : An array of the indices of X for which NULL values need to be handled
#        colModes: Numpy array of modes of columns under consideration
#                  Default: Compute colModes internally
#Output - Modified data matrix X, Weights of data instances in X
#What it does - This function is used to handle NULL values in case of numerical attrinutes
#               NULL values are replaced by the weighted mode of all non-NULL values for the attribute
#               under consideration in the data instances in X
#Assumption - NILL
def replace_null_values_with_mode(X, weights, attrsType, indices, colModes=[]):
    if len(indices) == 0:
        return X, weights
    nan_str = np.array([np.nan]).astype(str)[0]
    if len(colModes) == 0:
        for col in range(X[:,indices].shape[1]):
            counts = []
            xCol = X[:,indices][:,col]
            if attrsType == "str":
                nullInds = np.where(X[:,indices].astype("str")==nan_str)
            else:
                nullInds = np.where(np.isnan(xCol))
            nonNullInds = np.setdiff1d(np.arange(X.shape[0]), nullInds)
            mode, _ = weighted_mode(xCol[nonNullInds], weights[nonNullInds])
            colModes.append(mode)
        colModes = np.array(colModes)
    if attrsType == "str":
        inds = np.where(X[:,indices].astype("str")==nan_str)
    else:
        inds = np.where(np.isnan(X[:,indices]))
    X[:,indices][inds] = colModes[inds[1]]
    return X, weights