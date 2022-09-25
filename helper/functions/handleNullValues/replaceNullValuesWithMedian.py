#Not yet implemented
#Need to comppute weighted medians


# from helper.imports.packageImports import np
# #replace_null_values_with_median
# #Input - X : Data matrix
# #        weights : Numpy array of weights of data instances
# #        indices : An array of the indices of X for which NULL values need to be handled
# #        colMedians: Numpy array of medians of columns under consideration
# #                    Default: Compute colMedians internally
# #Output - Modified data matrix X
# #What it does - This function is used to handle NULL values in case of numerical attrinutes
# #               NULL values are replaced by the median of all non-NULL values for the attribute
# #               under consideration in the data instances in X
# #Assumption - NILL
# def replace_null_values_with_median(X, weights, indices, colMedians = []):
#     if len(indices) == 0:
#         return X
#     if len(colMedians) == 0:
#         colMedians = np.nanmedian(X[:,indices], axis=0)
        
#     inds = np.where(np.isnan(X[:,indices]))
    
#     X[:,indices][inds] = colMedians[inds[1]]
#     return X