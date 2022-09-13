import numpy as np
import csv
import sys
import pickle
from scipy import stats #For mode
from sklearn import preprocessing #For LabelEncoder

#Hardcoded
continuous_feature_indices = [0, 2, 4, 10, 11, 12]
discrete_feature_indices = []
ordinal_feature_indices = [] 
nominal_feature_indices = [1, 3, 5, 6, 7, 8, 9, 13]

categorical_column_indices = [1, 3, 5, 6, 7, 8, 9, 13]
numerical_column_indices = [0, 2, 4, 10, 11, 12]
#-------------------------------------------
#File operations 

#writeToCsvFile
#Input - x : Data matrix
#        x_path : Path to location of the csv file to be written into
#Output - NILL
#What it does - This function is used to write a data matrix row-by-row
#               into a csv file
#Assumption - NILL
def writeToCsvFile(x, x_path):
    if x.ndim == 1:
        x = x.reshape(x.shape[0], 1)
    with open(x_path, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(x)
        csv_file.close()
#-------------------------------------------
# #Handling abnormal string values

#handle_neg_0_values
#Input - preprocessed_X : Data matrix to be processed
#Output - Modified data matrix preprocessed_X
#What it does - This function is used to replace "-0.0" attribute values within "0.0"
#Assumption - NILL
# def handle_neg_0_values(preprocessed_X):
#     preprocessed_X = preprocessed_X.astype('str')
#     preprocessed_X[preprocessed_X == "-0.0"] = "0.0"
#     return preprocessed_X
#-------------------------------------------
#Handling null values

#replace_null_values_with_mean
#Input - X : Data matrix
#        indices : An array of the indices of X for which NULL values need to be handled
#                  Default: all indices of X
#Output - Modified data matrix X
#What it does - This function is used to handle NULL values in case of numerical attrinutes
#               NULL values are replaced by the mean of all non-NULL values for the attribute
#               under consideration in the data instances in X  
#Assumption - Numerical data is treated as being of np.float64 data type
def replace_null_values_with_mean(X, indices=[]):
    if len(indices) == 0:
        indices = np.arange(X.shape[1])
    colMeans = np.nanmean(X[:,indices].astype(np.float64), axis=0)
    inds = np.where(np.isnan(X[:,indices].astype(np.float64)))
    
    X[:,indices][inds] = colMeans[inds[1]].astype(str)
    return X

#replace_null_values_with_median
#Input - X : Data matrix
#        indices : An array of the indices of X for which NULL values need to be handled
#                  Default: all indices of X
#Output - Modified data matrix X
#What it does - This function is used to handle NULL values in case of numerical attrinutes
#               NULL values are replaced by the median of all non-NULL values for the attribute
#               under consideration in the data instances in X
#Assumption - NILL
def replace_null_values_with_median(X, indices=[]):
    if len(indices) == 0:
        indices = np.arange(X.shape[1])
    colMedians = np.nanmedian(X[:,indices], axis=0)
    inds = np.where(np.isnan(X[:,indices]))
    
    X[:,indices][inds] = colMedians[inds[1]]
    return X

#replace_null_values_with_mode
#Input - X : Data matrix
#        indices : An array of the indices of X for which NULL values need to be handled
#                  Default: all indices of X
#Output - Modified data matrix X
#What it does - This function is used to handle NULL values in case of numerical attrinutes
#               NULL values are replaced by the mode of all non-NULL values for the attribute
#               under consideration in the data instances in X
#Assumption - NILL
def replace_null_values_with_mode(X, indices=[]):
    if len(indices) == 0:
        indices = np.arange(X.shape[1])
    colModes = []
    for col in range(X[:,indices].shape[1]):
        counts = []
        xCol = X[:,indices][:,col]
        xCol = xCol[~np.isnan(xCol)]
        v, c = np.unique(xCol, return_counts=True)
        maxs = v[c == np.max(c)]
        colModes.append(min(maxs))
    colModes = np.array(colModes)
    inds = np.where(np.isnan(X[:,indices]))
    
    X[:,indices][inds] = colModes[inds[1]]
    return X
    
#replace_null_values_with_stringmode
#Input - X : Data matrix
#        indices : An array of the indices of X for which NULL values need to be handled
#                  Default: all indices of X
#Output - Modified data matrix X
#What it does - This function is used to handle NULL values in case of categorical attrinutes
#               NULL values are replaced by the mode of all non-NULL values for the attribute
#               under consideration in the data instances in X
#Assumption - Data is treated as being of str data type
def replace_null_values_with_stringmode(X, indices=[]):
    if len(indices) == 0:
        indices = np.arange(X.shape[1])
    colModes = []
    nan_str = np.array([np.nan]).astype(str)[0]
    for col in range(X[:,indices].shape[1]):
        counts = []
        xCol = X[:,indices][:,col]
        if nan_str in xCol:
            xCol = np.delete(xCol, np.where(xCol == nan_str))
        v, c = np.unique(xCol, return_counts=True)
        maxs = v[c == np.max(c)]
        colModes.append(min(maxs))
    colModes = np.array(colModes)
    inds = np.where(X[:,indices].astype("str")==nan_str)
    
    X[:,indices][inds] = colModes[inds[1]]
    return X  
#-------------------------------------------
#Normalization

#standardize
#Input - X : Data matrix
#        column_indices : An array of the attribute indices of X which need to be standardized
#        colMeans : column means
#                   Default: [] (Computes internally)
#        colSTDs : column standard deviations
#                   Default: [] (Computes internally)
#Output - Standardized data matrix X, column means, column standard deviations
#What it does - This function is used to standardize numerical attribute values
#Assumption - No NULL attribute values in X
def standardize(X, column_indices, colMeans, colSTDs):
    if len(colMeans)==0 or len(colSTDs)==0:
        colMeans = np.nanmean(X, axis=0)
        colSTDs = np.nanstd(X, axis=0)
    for col in column_indices:
        X[:,col] = (X[:,col] - colMeans[col])/colSTDs[col]
    return np.round(X,3), colMeans, colSTDs

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
#-------------------------------------------
#Handle oridinal categorical values

#mapOrdinalCategoricalValuesToNumbers
#Input - X : Numpy data matrix
#        indices : An array of the attribute indices of X which need to be label encoded
#                  Default: all indices of X
#Output - Modified data matrix X
#What it does - This function is used to encode ordinal categorical attribute values
#               as numerical values
#Assumption - No NULL attribute values in X
def mapOrdinalCategoricalValuesToNumbers(X, indices=[]):
    if len(indices)==0:
        indices = np.arange(X.shape[1])
    for i in indices:
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(X[:,i]))
        X[:,i] = np.array((le.transform(X[:,i])))
    return X

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
#-------------------------------------------
#Handle nominal categorical values

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

#convert_given_cols_to_one_hot
#Input - X : Numpy data matrix
#        column_indices : An array of the attribute indices of X which need     
#                         to be one-hot encoded
#        handleMulticolinearity : Boolean flag indicating whether 
#                                 multicolinearity needs to be handled
#                                 Default: False
#Output - One-hot encoded data matrix X
#What it does - This function is used to one-hot encode a set of nominal categorical
#               attributes
#Assumption - No NULL attribute values in X
def convert_given_cols_to_one_hot(X, column_indices, colVals, handleMulticolinearity=False):
    newX = np.copy(X)
    colOffset = 0
    for col in column_indices:
        curCol = X[:,col+colOffset]
        enc = np.array(apply_one_hot_encoding(curCol, colVals[col]))
        #Drop the first column to handle multicolinearity 
        if handleMulticolinearity and len(colVals[col]) > 1:
            enc = enc[:,1:].copy()
        if col == 0:
            X = np.concatenate((enc, X[:,(col+1):]),axis=1)
        elif col == len(X[0])-1:
            X = np.concatenate((X[:,0:(col+colOffset)], enc), axis=1)
        else:
            newX = np.concatenate((X[:,0:(col+colOffset)], enc), axis=1)
            X = np.concatenate((newX, X[:,(col+1+colOffset):]),axis=1)
        colOffset += (len(enc[0])-1)
    return X
#-------------------------------------------

#What it does: This function returns a dictionary with column indices as keys and a list of unique      
#              column values as the values
def getColVals(X):
    CVs = {}
    for col in range(X.shape[1]):
        curColVal = np.unique(X[:,col])
        if "?" in curColVal:
            curColVal = np.delete(curColVal, np.where(curColVal=="?")[0][0])
        CVs[col] = curColVal
    return CVs

def preprocess(X, colVals, continuous_feature_indices, discrete_feature_indices, ordinal_feature_indices, nominal_feature_indices, colMeansC=[], colSTDsC=[], colMeansD=[], colSTDsD=[]):
    #Numerical attributes

    cont = X[:,continuous_feature_indices].astype(np.float64)
    cont = replace_null_values_with_mean(cont)
    # cont = replace_null_values_with_median(cont)
    # cont = replace_null_values_with_mode(cont)
    cont, colMeansC, colSTDsC = standardize(cont, np.arange(cont.shape[1]), colMeansC, colSTDsC)
    # cont = np.round(min_max_normalize(cont, np.arange(cont.shape[1]), colMins, colMaxs), 3)
    # cont = np.round(mean_normalize(cont, np.arange(cont.shape[1]), colMeans, colMins, colMaxs), 3)
    cont = cont.astype("str")
    
    disc = X[:,discrete_feature_indices].astype(np.float64)
    disc = replace_null_values_with_median(disc)
    disc, colMeansD, colSTDsD= standardize(disc, np.arange(disc.shape[1]), colMeansD, colSTDsD)
    disc = disc.astype("str")
    
    #Categorical attributes

    ordinal = X[:,ordinal_feature_indices]
    ordinal = replace_null_values_with_stringmode(ordinal)
    ordinal = mapOrdinalCategoricalValuesToNumbers(ordinal)
    ordinal = ordinal.astype("str")
    
    nominal = X[:,nominal_feature_indices]
    nominal = replace_null_values_with_stringmode(nominal)
    nominal = nominal.astype("str")

    X[:, continuous_feature_indices] = cont
    X[:, discrete_feature_indices] = disc
    X[:, ordinal_feature_indices] = ordinal
    X[:, nominal_feature_indices] = nominal

    X = convert_given_cols_to_one_hot(X, nominal_feature_indices, colVals, True)
    return X.astype(np.float64), colMeansC, colSTDsC, colMeansD, colSTDsD
#-------------------------------------------
#Feature Selection

def get_correlation_matrix(X, Y):
    newX = np.zeros((X.shape[0], X.shape[1]+1))
    newX[:, 0] = np.reshape(Y,[len(Y),1])[:,0]
    newX[:,1:] = X  
    return np.corrcoef(newX.T)

def select_features(corr_mat, T1, T2):
    toSel, toRem = [], []
    for i in range(1,len(corr_mat[:,0])):
        if abs(corr_mat[i][0]) > T1:
            toSel.append(i-1)
    for i in range(len(toSel)):
        for j in range(i+1, len(toSel)):
            f1 = toSel[i]
            f2 = toSel[j]
            if f1 not in toRem and f2 not in toRem:
                if abs(corr_mat[f1][f2]) > T2:
                    toRem.append(f2)
    for r in toRem:
        toSel.remove(r)               
    return toSel
#-------------------------------------------

# def trainValSplit(X,Y):
#     train_X = np.copy(X)
#     train_Y = np.copy(Y)
#     valIndex = -int(validation_split*(train_X.shape[0]))
#     val_X = train_X[valIndex:]
#     val_Y = train_Y[valIndex:]
#     train_X = train_X[:valIndex]
#     train_Y = train_Y[:valIndex]
#     return (train_X, train_Y, val_X, val_Y)

def preprocessTrainData(X, colVals, Y):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    corrMat = get_correlation_matrix(train_X, train_Y)
    T1, T2 = np.mean(corrMat[:,0])/2, 0.4
    selFeatures = select_features(corrMat,T1, T2)
    for c in categorical_column_indices:
        if c not in selFeatures:
            categorical_column_indices.remove(c)
    for c in numerical_column_indices:
        if c not in selFeatures:
            numerical_column_indices.remove(c)
    train_X, train_X_means, train_X_mins, train_X_maxs = mean_normalize(train_X, numerical_column_indices)
    train_X = train_X[:, selFeatures]
    train_X = convert_given_cols_to_one_hot(train_X, categorical_column_indices, colVals)
    return (train_X, train_Y, train_X_means, train_X_mins, train_X_maxs, selFeatures)

def preprocessTestData(X, colVals, train_X_means, train_X_mins, train_X_maxs, selFeatures):
    test_X = np.copy(X)
    test_X = mean_normalize(test_X, numerical_column_indices, train_X_means, train_X_mins, train_X_maxs)
    test_X = test_X[:, selFeatures]
    test_X = convert_given_cols_to_one_hot(test_X, categorical_column_indices, colVals)
    return (test_X)

if len(sys.argv)>1:
    train_path = sys.argv[1]
else:
    train_path = "./Data/train_final.csv"

if len(sys.argv)>2:
    test_path = sys.argv[2]
else:
    test_path = "./Data/test_final.csv"

if len(sys.argv)>3:
    train_x_path = sys.argv[3]
else:
    train_x_path = "./Data/train_x_final.csv"

if len(sys.argv)>5:
    test_x_path = sys.argv[5]
else:
    test_x_path = "./Data/test_x_final.csv"

if len(sys.argv)>4:
    train_y_path = sys.argv[4]
else:
    train_y_path = "./Data/train_y_final.csv"

if len(sys.argv)>6:
    test_id_path = sys.argv[6]
else:
    test_id_path = "./Data/test_id_final.csv"

with open(train_path, newline="") as csvFile:
    train = np.array(list(csv.reader(csvFile, delimiter=",")))[1:,:]

with open(test_path, newline="") as csvFile:
    test = np.array(list(csv.reader(csvFile, delimiter=",")))[1:,:]

train_x = train[:, :-1]
train_y = train[:, -1]

test_id = test[:,0]
test_x = test[:,1:]

# writeToCsvFile(train_x, train_x_path)
# writeToCsvFile(test_x, test_x_path)
# writeToCsvFile(train_y, train_y_path)
# writeToCsvFile(test_y, test_y_path)

colVals = getColVals(train_x)

train_x, colMeansC, colSTDsC, colMeansD, colSTDsD = preprocess(train_x, colVals, continuous_feature_indices, discrete_feature_indices, ordinal_feature_indices, nominal_feature_indices)
test_x, _, _, _, _ = preprocess(test_x, colVals, continuous_feature_indices, discrete_feature_indices, ordinal_feature_indices, nominal_feature_indices,colMeansC,colSTDsC,colMeansD,colSTDsD)

# train_X, train_Y, train_X_means, train_X_mins, train_X_maxs, selFeatures = preprocessTrainData(train_x, colVals, train_y)
# test_X = preprocessTestData(test_x, colVals, train_X_means, train_X_mins, train_X_maxs, selFeatures)

print((train_x).shape)
print((test_x).shape)

writeToCsvFile(train_x, train_x_path)
writeToCsvFile(test_x, test_x_path)
writeToCsvFile(train_y, train_y_path)
writeToCsvFile(test_id, test_id_path)





