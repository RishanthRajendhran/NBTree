from helper.imports.packageImports import np
#standardize
#Input - X              : Numpy data matrix of data instances
#        weights        : Numpy array of weights of data instances
#        column_indices : An array of the attribute indices of X which need to be standardized
#        colMeans       : column means
#                         Default: [] (Computes internally)
#        colSTDs        : column standard deviations
#                         Default: [] (Computes internally)
#Output - Standardized data matrix X, column means, column standard deviations
#What it does - This function is used to standardize numerical attribute values
#Assumption - No NULL attribute values in X
def standardize(X, weights, column_indices, colMeans, colSTDs):
    if len(colMeans)==0 or len(colSTDs)==0:
        colMeans = np.average(X[:,column_indices].astype(np.float64), axis=0, weights=weights)
        colVars = np.average((X[:,column_indices].astype(np.float64)-colMeans)**2, axis=0, weights=weights)
        colSTDs = np.sqrt(colVars)
    for col in column_indices:
        X[:,col] = np.round(((X[:,col].astype(np.float64) - colMeans[column_indices.index(col)])/colSTDs[column_indices.index(col)]),3).astype(str)
    return X, weights, colMeans, colSTDs