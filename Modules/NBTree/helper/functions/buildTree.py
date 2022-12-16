from Modules.NBTree.helper.classes.Node import Node
from Modules.NBTree.helper.functions.getBestAttrSplit import getBestAttrSplit
from Modules.NBTree.helper.imports.packageImports import np

def buildTree(X, weights, Y, allClasses, attrsRem, attrsType, allAttrVals, prevMaxLabel=None):
    result = getBestAttrSplit(X, weights, Y, allClasses, attrsRem, attrsType, allAttrVals)

    if result["splitSignificant"]:
        attrSplit = result["bestAttr"]
        nonLeafNode = Node(False, False, np.max(Y), attrsRem[attrSplit], attrsType[attrSplit], result["attrThresh"])
        for i in range(len(result["bestSplitsX"])):
            x, w, y = result["bestSplitsX"][i], result["bestSplitsW"][i], result["bestSplitsY"][i] 
            splitVal = result["bestSplitsVal"][i]

            nonLeafNode.addChild(buildTree(x, w, y, allClasses, attrsRem, attrsType, allAttrVals, np.max(Y)), splitVal)
        return nonLeafNode
    else:
        #create a naive bayes classifier
        if len(X) == 0 or len(np.unique(Y))==1:
            leafNode = Node(True, False, prevMaxLabel)
        else:
            leafNode = Node(True, True, np.max(Y))
            leafNode.makeLeaf(X, weights, Y, attrsType)
        return leafNode