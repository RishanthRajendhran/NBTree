from helper.imports.packageImports import np
#get_correlation_matrix
#Input - X                  : Numpy data matrix of data instances
#        Y                  : Numpy data matrix of target labels
#Output - Correlation matrix computed on columns of X and target label Y
#What it does - This function is used to find the correlation coefficient between
#               columns of X and the target label
def get_correlation_matrix(X, Y):
    newX = np.zeros((X.shape[0], X.shape[1]+1))
    newX[:, 0] = np.reshape(Y,[len(Y),1])[:,0]
    newX[:,1:] = X  
    return np.corrcoef(newX.T)