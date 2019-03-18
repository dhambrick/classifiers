import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import scipy.io as sio


# def pathCost(a,b):
#     costMatrix , alignmentA,alignmentB = dtw1d(a,b)
#     idxs = list(zip(np.asarray(alignmentA),np.asarray(alignmentB)))
#     alignmentA = None
#     alignmentB = None
#     cost = 0
#     for idx in idxs:
#         cost = cost + costMatrix[idx[0],idx[1]]
#     costMatrix = None
#     return cost 

def dtwCost(a,b):
    ta = np.arange(len(a))
    tb = np.arange(len(b))
    x = list(zip(ta,a))
    y = list(zip(tb,b))
    cost , path = fastdtw(x,y,dist=euclidean)
    del path
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
filename = 'F:\classifiers\ECGData\ECGData.mat'
trainData,trainLabel,testData,testLabel = loadData(filename,.6) 
uniqueLabels = np.unique(trainLabel)
print(dtwCost(trainData[0],trainData[1]))

labelMap = {}
for idx,label in enumerate(uniqueLabels):
    labelMap[label] = idx
trLabel = []
for label in trainLabel:
    trLabel.append(labelMap[label])
tstLabel = []
for label in testLabel:
    tstLabel.append(labelMap[label])


classifier = neighbors.KNeighborsClassifier(nNeighbors , metric = dtwCost)
classifier.fit(trainData, trLabel)