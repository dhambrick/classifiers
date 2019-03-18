from pydtw import dtw1d
import numpy as np
import matplotlib.pyplot as plt
def distance_cost_plot(distances):
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.colorbar()
    plt.show()
def plotAlignment(a,b,ta,tb,alignmentA ,alignmentB):
    plt.figure()
    plt.plot(ta,a)
    plt.plot(tb,b)
    for idx in range(0,len(np.asarray(alignmentA))):
            x  = (a[alignmentA[idx]] ,b[alignmentB[idx]])
            ts = (ta[alignmentA[idx]] ,tb[alignmentB[idx]])
            plt.plot(ts , x)
    plt.show()    
def pathCost(a,b):
    costMatrix , alignmentA,alignmentB = dtw1d(a,b)
    idxs = list(zip(np.asarray(alignmentA),np.asarray(alignmentB)))
    cost = 0
    for idx in idxs:
        cost = cost + costMatrix[idx[0],idx[1]]
    return cost ,costMatrix ,alignmentA ,alignmentB

# a = np.random.rand(10)
# b = np.random.rand(15)
# costMatrix , alignmentA,alignmentB = dtw1d(a,b)

t = np.linspace(0,6.28,100)
ta = tb = t
a = np.sin(ta)
b = np.cos(tb)

# a = np.array([1.0, 1.0, 2.0, 3.0, 2.0, 0.0])
# b = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0])
# ta = [0,1,2,3,4,5]
# tb = [0,1,2,3,4,5,6]

cost ,costMatrix,aPath,bPath = pathCost(a,b)

print("cost: ", cost)
distance_cost_plot(costMatrix)
plotAlignment(a,b,ta,tb,aPath,bPath)



