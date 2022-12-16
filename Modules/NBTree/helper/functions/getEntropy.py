from Modules.NBTree.helper.imports.packageImports import np, entropy

def getEntropy(counts):
    total = np.sum(counts)
    if total == 0:
        return 0
    return entropy(counts, base=2)  
    # for c in counts:
    #     try:
    #         p = c/total
    #     except:
    #         p = 0
    #     try:
    #         entropy += -(p*np.log2(p))
    #     except:
    #         entropy = 0
    # return entropy
