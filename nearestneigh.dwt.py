import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import scipy.io as sio



def dtwCost(a,b):
    cost , path = fastdtw(a,b)
    return cost 
    

def loadData(filename ,trainSetPercentage):
    raw_data = sio.loadmat(filename)
    data_ecg = raw_data['ECGData'][0][0][0]
    labels_ecg = np.array(list(map(lambda x: x[0][0], raw_data['ECGData'][0][0][1])))
    
    dataLen = len(labels_ecg)
    idxs = [i for i in range(dataLen)]
    np.random.seed(1)
    np.random.shuffle(idxs)
    
    training_size = int(trainSetPercentage*len(labels_ecg))
    trainIdx = np.array(idxs[:training_size],dtype='int32')
    testIdx = np.array(idxs[training_size:],dtype='int32')
    print(trainIdx.shape)

    train_data_ecg = data_ecg[trainIdx]
    test_data_ecg = data_ecg[testIdx]
    train_labels_ecg = labels_ecg[trainIdx]
    test_labels_ecg = labels_ecg[testIdx]
    return train_data_ecg ,train_labels_ecg,test_data_ecg,test_labels_ecg

nNeighbors = 1
#Load Data
filename = '.\ECGData\ECGData.mat'
trainData,trainLabel,testData,testLabel = loadData(filename,.6) 

uniqueLabels = np.unique(trainLabel)
labelMap = {}
for idx,label in enumerate(uniqueLabels):
    labelMap[label] = idx
trLabel = []
for label in trainLabel:
    trLabel.append(labelMap[label])
tstLabel = []
for label in testLabel:
    tstLabel.append(labelMap[label])

print("Creating Classifier")
classifier = neighbors.KNeighborsClassifier(nNeighbors,metric=dtwCost,n_jobs=-1)
print("Fitting the training data")
classifier.fit(trainData, trLabel)
print("Scoring the test data")
testClasses = classifier.score(testData,tstLabel)
print("Num of Training Samples: " ,len(trLabel))
print("Num of Test Samples: " ,len(tstLabel))
print("Percentage of Test Data Correctly Classified:" , testClasses)
