#Module imports
from helper.imports.packageImports import *
from helper.imports.functionImports import *
#Configuration Imports
import helper.config.dataConfig as dataConfig

#Code
parser = argparse.ArgumentParser()
parser.add_argument("-train", help="Path to train data file")
parser.add_argument("-test", help="Path to test data file")
parser.add_argument("-trainX", help="File path to store preprocessed train data matrix")
parser.add_argument("-weightsX", help="File path to store preprocessed train data weights matrix")
parser.add_argument("-testX", help="File path to store preprocessed test data matrix")
parser.add_argument("-trainY", help="File path to store preprocessed train data target labels matrix")
parser.add_argument("-testID", help="File path to store preprocessed test data ID matrix")

args = parser.parse_args()

if args.train:
    train_path = args.train
else:
    train_path = "./Data/train_final.csv"

if args.test:
    test_path = args.test
else:
    test_path = "./Data/test_final.csv"

if args.trainX:
    train_x_path = args.trainX
else:
    train_x_path = "./Data/train_x_final.csv"

if args.weightsX:
    weights_path = args.weightsX
else:
    weights_path = "./Data/weights_final.csv"

if args.testX:
    test_x_path = args.testX
else:
    test_x_path = "./Data/test_x_final.csv"

if args.trainY:
    train_y_path = args.trainY
else:
    train_y_path = "./Data/train_y_final.csv"

if args.testID:
    test_id_path = args.testID
else:
    test_id_path = "./Data/test_id_final.csv"

with open(train_path, newline="") as csvFile:
    train = np.array(list(csv.reader(csvFile, delimiter=",")))[1:,:]

with open(test_path, newline="") as csvFile:
    test = np.array(list(csv.reader(csvFile, delimiter=",")))[1:,:]

train_x = train[:, :-1]
train_y = train[:, -1]
weights = np.ones((train_x.shape[0],))

test_id = test[:,0]
test_x = test[:,1:]

# writeToCsvFile(train_x, train_x_path)
# writeToCsvFile(test_x, test_x_path)
# writeToCsvFile(train_y, train_y_path)
# writeToCsvFile(test_y, test_y_path)

colVals = getColVals(train_x)

train_x, weights, train_y, selFeatures, colMeansCont, colSTDsCont, colMeansDisc, colSTDsDisc, colModesOrdinal, colModesNominal = preprocessTrainData(train_x, weights, train_y, colVals, dataConfig.handleMulticolinearity, dataConfig.performFeatureSelection)
test_x = preprocessTestData(test_x, selFeatures, colVals, colMeansCont, colSTDsCont, colMeansDisc, colSTDsDisc, colModesOrdinal, colModesNominal, dataConfig.handleMulticolinearity)

writeToCsvFile(train_x, train_x_path)
writeToCsvFile(weights, weights_path)
writeToCsvFile(test_x, test_x_path)
writeToCsvFile(train_y, train_y_path)
writeToCsvFile(test_id, test_id_path)

print((train_x).shape)
print((test_x).shape)
print(np.sum(weights))





