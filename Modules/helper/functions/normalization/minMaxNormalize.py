from helper.imports.packageImports import np
#min_max_normalize
#Input - X : Data matrix
#        column_indices : An array of the attribute indices of X which need to be normalized
#        colMins : column minimums
#                   Default: [] (Computes internally)
#        colMaxs : column maximums
#                   Default: [] (Computes internally)
#Output - Normalized data matrix X, column mins, column maxs
#What it does - This function is used to min-max normalize numerical attribute values
#Assumption - No NULL attribute values in X
#             Numerical data is treated as being of np.float64 data type
def min_max_normalize(X, column_indices, colMins=[], colMaxs=[]):
    if len(colMaxs)==0 or len(colMins)==0:
        colMins = np.min(X, axis=0)
        colMaxs = np.max(X, axis=0)
    for col in column_indices:
        X[:,col] = (X[:,col]-colMins[col])/(colMaxs[col] - colMins[col])
    return X, colMins, colMaxs