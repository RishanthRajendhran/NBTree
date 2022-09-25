from helper.imports.packageImports import np
from helper.functions.handleNullValues.replaceNullValuesWithMean import replace_null_values_with_mean
from helper.functions.normalization.standardize import standardize
from helper.functions.handleNullValues.replaceNullValuesWithMode import replace_null_values_with_mode
from helper.functions.handleOrdinalCategoricalValues.mapOrdinalCategoricalValuesToNumbers import mapOrdinalCategoricalValuesToNumbers
from helper.functions.handleNominalCategoricalValues.convertGivenColsToOneHot import convert_given_cols_to_one_hot
from helper.functions.featureSelection.getCorrelationMatrix import get_correlation_matrix
from helper.functions.featureSelection.selectFeatures import select_features

import helper.config.dataConfig as dataConfig


#preprocessTestData
#Input - X                           : Numpy data matrix of test data instances
#        selFeatures                 : Numpy array of selected feature indices
#        colVals                     : Dictionary of unique attribute values with column indices of X as keys
#        colMeansCont                : List of means of continuous columns of X
#        colSTDsCont                 : List of standard deviations of continuous columns of X
#        colMeansDisc                : List of means of discrete columns of X
#        colSTDsDisc                 : List of standard deviations of discrete columns of X
#        colModesOrdinal             : List of modes of ordinal categorical columns of X
#        colModesNominal             : List of modes of nominal categorical columns of X
#        handleMulticolinearity      : Boolean value indicating whether the first (redundant) column
#                                      in one-hot encoding of an attributed should be dropped
#Output - Preprocessed X, Weights of data instances, list of means of continuous columns, list of standard deviations of continuous columns,
#         list of means of discrete columns, list of standard deviations of discrete columns
#What it does: This function is used to do preprocessing on data matrix X
#Assumption - "?" indicates missing/NULL/unknown value, not considered as a unique attribute value
def preprocessTestData(X, selFeatures, colVals, colMeansCont, colSTDsCont, colMeansDisc, colSTDsDisc, colModesOrdinal, colModesNominal, handleMulticolinearity):
    weights = np.ones((X.shape[0],))
    #Numerical attributes
    
    X, weights = replace_null_values_with_mean(X, weights, dataConfig.continuous_feature_indices, colMeansCont)
    X, weights, colMeansCont, colSTDsCont = standardize(X, weights, dataConfig.continuous_feature_indices, colMeansCont, colSTDsCont)
    
    X, weights = replace_null_values_with_mean(X, weights, dataConfig.discrete_feature_indices, colMeansDisc)
    X, weights, colMeansDisc, colSTDsDisc = standardize(X, weights, dataConfig.discrete_feature_indices, colMeansDisc, colSTDsDisc)
    
    #Categorical attributes
    X, weights = replace_null_values_with_mode(X, weights, "str", dataConfig.ordinal_feature_indices, colModesOrdinal)
    X, weights, _ = mapOrdinalCategoricalValuesToNumbers(X, weights, dataConfig.ordinal_feature_indices)
    
    X, weights = replace_null_values_with_mode(X, weights, "str", dataConfig.nominal_feature_indices, colModesNominal)
    X, weights = convert_given_cols_to_one_hot(X, weights, dataConfig.nominal_feature_indices, colVals, handleMulticolinearity)

    X = X[:, selFeatures].copy()

    return X.astype(np.float64)
