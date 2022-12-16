from Modules.NBTree.helper.imports.packageImports import np
from Modules.NBTree.helper.functions.getEntropy import getEntropy
from Modules.NBTree.helper.functions.getUtility import getUtility

def getBestAttrSplit(X, weights, Y, allClasses, attrsRem, attrsType, allAttrVals):
    
    if len(X) < 30:
        return {"splitSignificant":False}

    baseClassCounts = []
    for clas in allClasses:
        baseClassInds = np.where(Y == clas)[0]
        baseClassCounts.append(np.sum(weights[baseClassInds]))
    baseEntropy = getEntropy(baseClassCounts)

    baseUtility = getUtility(X, weights, Y.astype(int), attrsType)

    bestUtility = None
    bestAttr = None
    bestThresh = None
    bestSplitsX = None
    bestSplitsW = None
    bestSplitsY = None
    bestSplitsVal = None

    result = {}

    for a in range(len(attrsRem)):
        if attrsType[a] == "numerical":
            #Get best threshold based on entropy minimization
            allVals = np.unique(X[:,attrsRem[a]])

            bestGainAttr = None
            bestAttrVal = None
            for v in range(len(allVals)):
                gain = baseEntropy

                lessInds = np.where(X[:,attrsRem[a]] <= allVals[v])[0]
                greatInds = np.where(X[:,attrsRem[a]] > allVals[v])[0]
        
                lessClassCounts = []
                for clas in allClasses:
                    lessClassInds = np.where(Y[lessInds] == clas)[0]
                    lessClassCounts.append(np.sum(weights[lessClassInds]))
                gain -= ((len(lessInds))/(len(X)))*getEntropy(lessClassCounts)

                greatClassCounts = []
                for clas in allClasses:
                    greatClassInds = np.where(Y[greatInds] == clas)[0]
                    greatClassCounts.append(np.sum(weights[greatClassInds]))
                gain -= ((len(greatInds))/(len(X)))*getEntropy(greatClassCounts)
                if bestGainAttr == None or gain > bestGainAttr:
                    bestGainAttr = gain
                    bestAttrVal = v
            attrThresh = allVals[bestAttrVal]

            lessInds = np.where(X[:,attrsRem[a]] <= attrThresh)[0]
            greatInds = np.where(X[:,attrsRem[a]] > attrThresh)[0]

            utilitySplit = 0

            utilitySplit += (len(lessInds)/len(X))*getUtility(X[lessInds], weights[lessInds], Y[lessInds], attrsType)
            utilitySplit += (len(greatInds)/len(X))*getUtility(X[greatInds], weights[greatInds], Y[greatInds], attrsType)

            if bestUtility == None or utilitySplit > bestUtility:
                bestUtility = utilitySplit
                bestAttr = a
                bestThresh = attrThresh
                bestSplitsX = np.array([X[lessInds], X[greatInds]], dtype=object)
                bestSplitsW = np.array([weights[lessInds], weights[greatInds]], dtype=object)
                bestSplitsY = np.array([Y[lessInds], Y[greatInds]], dtype=object)
                bestSplitsVal = np.array(["le", "gt"])

        elif attrsType[a] == "categorical":
            utilitySplit = 0
            splitX = []
            splitY = []
            splitW = []
            for v in range(len(allAttrVals[attrsRem[a]])):
                reqInds = np.where(X[:,attrsRem[a]] == allAttrVals[attrsRem[a]][v])[0]
                splitX.append(X[reqInds])
                splitW.append(weights[reqInds])
                splitY.append(Y[reqInds])
                utilitySplit += (len(reqInds)/len(X))*getUtility(X[reqInds], weights[reqInds], Y[reqInds], attrsType)
            
            if bestUtility == None or utilitySplit > bestUtility:
                bestUtility = utilitySplit
                bestAttr = a
                bestThresh = None
                bestSplitsX = np.array(splitX, dtype=object)
                bestSplitsW = np.array(splitW, dtype=object)
                bestSplitsY = np.array(splitY, dtype=object)
                bestSplitsVal = np.array(allAttrVals[attrsRem[a]])
        else:
            raise Exception(f"Unknown attribute type {attrsType[a]} encountered!")

    splitSignificant = False
    if bestUtility > baseUtility and ((bestUtility-baseUtility)/baseUtility) > 0.01:
        splitSignificant = True

    result["splitSignificant"] = splitSignificant
    result["bestAttr"] = bestAttr
    result["attrThresh"] = bestThresh
    result["bestSplitsX"] = bestSplitsX
    result["bestSplitsW"] = bestSplitsW
    result["bestSplitsY"] = bestSplitsY
    result["bestSplitsVal"] = bestSplitsVal
 
    return result

    