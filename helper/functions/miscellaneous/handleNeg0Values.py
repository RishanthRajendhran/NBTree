from helper.imports.packageImports import np 
# #Handling abnormal string values

#handleNeg0Values
#Input - preprocessed_X : Data matrix to be processed
#Output - Modified data matrix preprocessed_X
#What it does - This function is used to replace "-0.0" attribute values within "0.0"
#Assumption - NILL
def handleNeg0Values(preprocessed_X):
    preprocessed_X = preprocessed_X.astype('str')
    preprocessed_X[preprocessed_X == "-0.0"] = "0.0"
    return preprocessed_X