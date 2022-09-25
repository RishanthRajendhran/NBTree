from helper.imports.packageImports import np
#mean_normalize
#Input - X : Data matrix
#        column_indices : An array of the attribute indices of X which need to be normalized
#        colMeans : column means
#                   Default: [] (Computes internally)
#        colMins : column minimums
#                   Default: [] (Computes internally)
#        colMaxs : column maximums
#                   Default: [] (Computes internally)
#Output - Normalized data matrix X, column means, column mins, column maxs
#What it does - This function is used to mean normalize numerical attribute values
#Assumption - No NULL attribute values in X
#             Numerical data is treated as being of np.float64 data type
def mean_normalize(X, column_indices, colMeans=[], colMins=[], colMaxs=[]):
    if len(colMeans)==0 or len(colMaxs)==0 or len(colMins)==0:
        cols = X[:, column_indices].astype(np.float64)
        colMeans = np.mean(cols, axis=0)
        colMaxs = np.max(cols, axis=0)
        colMins = np.min(cols, axis=0)
        X[:,column_indices] = ((cols-colMeans)/(colMaxs-colMins)).astype(str)
        return X, colMeans, colMins, colMaxs
    else:
        cols = X[:, column_indices].astype(np.float64)
        X[:,column_indices] = ((cols-colMeans)/(colMaxs-colMins)).astype(str)
        return X, colMeans, colMins, colMaxs