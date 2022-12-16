from Modules.NBTree.helper.functions.buildTree import buildTree
from Modules.NBTree.helper.imports.packageImports import np

class NBTree:
    def __init__(self):
        self.rootNode = None

    def fit(self, X, Y, attrsType, sample_weights=None):
        if sample_weights == None:
            sample_weights = np.ones((len(X),))
        allClasses = np.unique(Y)
        attrsRem = np.arange(X.shape[1])
        allAttrVals = {}
        for a in range(len(attrsRem)):
            if attrsType[attrsRem[a]] == "categorical":
                allAttrVals[attrsRem[a]] = np.unique(X[:, attrsRem[a]])
        return self._fit(X, sample_weights, Y, allClasses, attrsRem, attrsType, allAttrVals)

    def _fit(self, X, weights, Y, allClasses, attrsRem, attrsType, allAttrVals):
        self.rootNode = buildTree(X, weights, Y, allClasses, attrsRem, attrsType, allAttrVals)
        return self

    def printTree(self):
        if self.rootNode == None:
            print("NBTree is empty!")
            return
        print("Printing NBTree...")
        self.rootNode.printNode()
        return

    def predict(self, X, attrsType):
        allPreds = []
        for x in X:
            allPreds.append(self.rootNode.predict(x, attrsType))
        return np.array(allPreds)
    