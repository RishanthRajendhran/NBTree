from helper.imports.packageImports import np
#apply_one_hot_encoding
#Input - X : Numpy data matrix containing only the attribute value to be 
#            one-hot encoded
#Output - Modified data matrix X
#What it does - This function is used to one-hot encode one nominal categorical
#               attribute
#Assumption - No NULL attribute values in X
def apply_one_hot_encoding(X, colVal):
    els = np.array(colVal)
    newX = []
    for r in X:
        newR = np.zeros(els.shape[0], dtype=np.int64)
        matchLoc = np.where(els == r)
        if len(matchLoc[0]):
            newR[matchLoc[0][0]] = 1 
        newX.append(newR)
    return newX