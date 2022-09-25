from helper.imports.packageImports import np
#convert_to_numerical_labels
#Input - X : Numpy data matrix containing only the attribute value to be 
#            label encoded
#Output - Modified data matrix X
#What it does - This function is used to encode one ordinal categorical attribute value
#               as numerical value
#Assumption - No NULL attribute values in X
def convert_to_numerical_labels(X):
    els = np.unique(X)
    newX = []
    for e in X:
        newX.append(np.where(els == e)[0][0])
    return newX