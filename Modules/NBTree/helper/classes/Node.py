from Modules.NBTree.helper.imports.packageImports import np, KBinsDiscretizer, GaussianNB
from sklearn import svm

class Node:
    def __init__(self, isLeaf, isGNB, maxLabel, attr=None, attrType=None, attrThresh=None):
        self.isLeaf = isLeaf
        self.isGNB = isGNB
        self.maxLabel = maxLabel
        self.gnb = None
        self.kbins = None
        self.attr = attr 
        self.attrType = attrType
        self.attrThresh = attrThresh
        self.children = []

    def addChild(self, childNode, splitVal):
        self.children.append((splitVal, childNode))

    def makeLeaf(self, X, weights, Y, attrsType):
        # clf = svm.SVC(kernel="poly") 

        # clf.fit(X, Y)
        # self.gnb = gnb

        numerical_attributes = np.where(np.array(attrsType) == "numerical")[0]

        kbins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')
        discretizedX = X.copy()
        discretizedX[:, numerical_attributes] = kbins.fit_transform(X[:, numerical_attributes])

        gnb = GaussianNB()
        gnb.fit(discretizedX, Y, sample_weight=weights)

        self.gnb = gnb
        self.kbins = kbins
        return

    def printNode(self, formatting=""):
        if self.isLeaf:
            if self.isGNB:
                print(f"{formatting}GNB Node")
            else:
                print(f"{formatting}Leaf Node")
            return 
        print(f"{formatting}Attribute {self.attr}")
        if self.attrType == "numerical":
            print(f"{formatting} <= {self.attrThresh}")
            self.children[0][1].printNode(formatting+"\t")
            print(f"{formatting} > {self.attrThresh}")
            self.children[1][1].printNode(formatting+"\t")
        else:
            for ch in self.children:
                print(f"{formatting} = {ch[0]}")
                ch[1].printNode(formatting+"\t")

    def predict(self, X, attrsType):
        if self.isLeaf:
            if self.isGNB:
                # return self.gnb.predict(X)
                numerical_attributes = np.where(np.array(attrsType) == "numerical")[0]
                discretizedX = np.array([X.copy()])
                discretizedX[:, numerical_attributes] = self.kbins.transform(np.array([X[numerical_attributes]]))
                return self.gnb.predict(discretizedX)[0]
            return self.maxLabel
        
        if self.attrType == "numerical":
            if X[self.attr] <= self.attrThresh:
                goTo = "le"
            else:
                goTo = "gt"

            for ch in self.children:
                if ch[0] == goTo:
                    return ch[1].predict(X, attrsType)
            return self.maxLabel
            # raise Exception(f"Unexpected side to take: {goTo}!")
        else:
            for ch in self.children:
                if ch[0] == X[self.attr]:
                    return ch[1].predict(X, attrsType)
            return self.maxLabel
            # raise Exception(f"Unexpected attr val: {X[self.attr]}!")

                    
            
