from helper.imports.packageImports import np
#trainValSplit
#Input - X                  : Numpy data matrix of data instances
#        weights            : Numpy data matrix of weights of data instances
#        Y                  : Numpy data matrix of target labels
#        validationSplit    : Percentage of train data to be split for validation 
#                             Default: 0.3 (30% validation split)
#        shuffle            : Boolean value to indicate whether training data should be 
#                             shuffled before splitting
#                             Default: False
#Output - Train and validation data sets with the respective weights and target labels
#What it does - This function is used to split the training data into train
#               and validation data
def trainValSplit(X, weights, Y, validationSplit=0.3, shuffle=False):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    train_W = weights.copy()
    if shuffle:
        train_WXY = np.zeros((train_X.shape[0], 1+train_X.shape[1]+1))
        train_WXY[:,0] =train_W
        train_WXY[:,1:-1] = train_X
        train_WXY[:,-1] = train_Y
        np.random.shuffle(train_WXY)
        train_W = np.copy(train_WXY[:,0])
        train_X = np.copy(trainWXY[:,1:-1])
        train_Y = np.copy(trainWXY[:,-1])
    valIndex = -int(validationSplit*(train_X.shape[0]))
    val_X = train_X[valIndex:]
    val_Y = train_Y[valIndex:]
    val_W = train_W[valIndex:]
    train_X = train_X[:valIndex]
    train_Y = train_Y[:valIndex]
    train_W = train_W[:valIndex]
    return (train_X, train_W, train_Y, val_X, val_W, val_Y)