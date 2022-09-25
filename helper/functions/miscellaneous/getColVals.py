from helper.imports.packageImports import np
#getColVals
#Input - X      : Numpy data matrix of data instances
#        cols   : List of column indices 
#                 Default: all columns of X
#Output - Dictionary of unique attribute values with column indices as keys 
#What it does: This function returns a dictionary with column indices as keys and a list of unique      
#              column values as the values
#Assumption - "?" indicates missing/NULL/unknown value, not considered as a unique attribute value
def getColVals(X, cols=None):
    if cols == None:
        cols = np.arange(X.shape[1])
    CVs = {}
    for col in cols:
        curColVal = np.unique(X[:,col])
        if "?" in curColVal:
            curColVal = np.delete(curColVal, np.where(curColVal=="?")[0][0])
        CVs[col] = curColVal
    return CVs