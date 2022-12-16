from helper.imports.packageImports import np
#select_features
#Input - corr_mat   : Numpy data matrix of correlation coefficients 
#        T1         : Minimum Threshold for correlation between attribute and target label
#        T2         : Maximum Threshold for correlation between attributes
#Output - List of selected columns
#What it does - This function is used to perform feature selection using correlation
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
                    if abs(corr_mat[f1][0]) > abs(corr_mat[f2][0]):
                        toRem.append(f2)
                    else: 
                        toRem.append(f1)
    for r in toRem:
        toSel.remove(r)  
    # print(toSel)             
    return toSel