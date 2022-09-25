from helper.imports.packageImports import np, stats
from helper.imports.functionImports import replace_null_values_with_fractional_examples, standardize, mapOrdinalCategoricalValuesToNumbers, convert_given_cols_to_one_hot, get_correlation_matrix, select_features

import helper.config.dataConfig as dataConfig

#preprocessTrainData
#Input - X                           : Numpy data matrix of data instances
#        weights                     : Numpy array of weights of data instances
#        Y                           : Numpy data matrix of target labels
#        colVals                     : Dictionary of unique attribute values with column indices of X as keys
#        handleMulticolinearity      : Boolean value indicating whether the first (redundant) column
#                                      in one-hot encoding of an attributed should be dropped
#        performFeatureSelection     : Boolean value indicating whether feature selection should be
#                                      be performed on the processed input data matrix
#                                      Default: False
#Output - Preprocessed X, Weights of data instances, Y, List of selected features, list of means of continuous columns,
#         list of standard deviations of continuous columns,list of means of discrete columns, list of standard 
#         deviations of discrete columns, list of modes of ordinal categorical columns, list of modes of nominal 
#         categorical columns
#What it does: This function is used to do preprocessing on data matrix X
#Assumption - "?" indicates missing/NULL/unknown value, not considered as a unique attribute value
def preprocessTrainData(X, weights, Y, colVals, handleMulticolinearity, performFeatureSelection=False):
    #Numerical attributes

    X, weights, Y = replace_null_values_with_fractional_examples(X, weights, Y, colVals, "float", dataConfig.continuous_feature_indices)
    X, weights, colMeansCont, colSTDsCont = standardize(X, weights, dataConfig.continuous_feature_indices,[],[])

    X, weights, Y = replace_null_values_with_fractional_examples(X, weights, Y, colVals, "float", dataConfig.discrete_feature_indices)
    X, weights, colMeansDisc, colSTDsDisc = standardize(X, weights, dataConfig.discrete_feature_indices,[],[])

    #Categorical attributes

    X, weights, Y = replace_null_values_with_fractional_examples(X, weights, Y, colVals, "str", dataConfig.ordinal_feature_indices)
    X, weights, colModesOrdinal = mapOrdinalCategoricalValuesToNumbers(X, weights, dataConfig.ordinal_feature_indices)

    X, weights, Y = replace_null_values_with_fractional_examples(X, weights, Y, colVals, "str", dataConfig.nominal_feature_indices)

    X, weights = convert_given_cols_to_one_hot(X, weights, dataConfig.nominal_feature_indices, colVals, handleMulticolinearity)
    colModesNominal = stats.mode(X[:,dataConfig.nominal_feature_indices])[0][0]

    if performFeatureSelection:
        train_X = np.copy(X)
        train_Y = np.copy(Y)
        corrMat = get_correlation_matrix(train_X, train_Y)
        T1, T2 = np.mean(corrMat[:,0])/4, 0.7
        # T1, T2 = 0.65, 0.9
        selFeatures = select_features(corrMat,T1, T2)
        for c in dataConfig.continuous_feature_indices:
            if c not in selFeatures:
                dataConfig.continuous_feature_indices.remove(c)
        for c in dataConfig.discrete_feature_indices:
            if c not in selFeatures:
                dataConfig.discrete_feature_indices.remove(c)
        for c in dataConfig.nominal_feature_indices:
            if c not in selFeatures:
                dataConfig.nominal_feature_indices.remove(c)
        for c in dataConfig.ordinal_feature_indices:
            if c not in selFeatures:
                dataConfig.ordinal_feature_indices.remove(c)
        for c in dataConfig.categorical_column_indices:
            if c not in selFeatures:
                dataConfig.categorical_column_indices.remove(c)
        for c in dataConfig.numerical_column_indices:
            if c not in selFeatures:
                dataConfig.numerical_column_indices.remove(c)
        X = train_X[:, selFeatures].copy()
    else: 
        selFeatures = np.arange(X.shape[1])

    return X.astype(np.float64), weights, Y, selFeatures, colMeansCont, colSTDsCont, colMeansDisc, colSTDsDisc, colModesOrdinal, colModesNominal