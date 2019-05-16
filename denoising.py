import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data


data = pywt.data.ecg()
N = len(data)
sigma = 30
legendLabels= [( 'original','smoothed' , 'noisy'),('cA' , 'cD')]
wavelet = 'db1'
noise = np.random.normal(0, sigma, N)
threshold = sigma * np.sqrt(np.log2(N)) 
print(threshold)
noisyData = data + noise
dwtNoisyData = pywt.wavedec(noisyData, wavelet , level=3)

threshWaveCoeff = (dwtNoisyData[0],*tuple(map(lambda x: pywt.threshold(x,threshold,'soft'),dwtNoisyData[1:])))

smoothedData = pywt.waverec(threshWaveCoeff , wavelet)
#fig, axs = plt.subplots(1)
plt.figure()
plt.plot(data , color='black' , linestyle='dashed')
plt.plot(smoothedData, 'r')
plt.plot(noisyData , 'b', alpha = .6)
plt.legend(legendLabels[0])

fig, axs = plt.subplots(len(dwtNoisyData))
for idx,c in enumerate(dwtNoisyData):
    axs[idx].plot(dwtNoisyData[idx])
# axs.plot(dwtNoisyData[1])
# axs.legend(legendLabels[1])

# fig, axs = plt.subplots(2,2)

# axs[0][0].spy(np.diag(dwtNoisyData[0]))
# axs[0][1].spy(np.diag(dwtNoisyData[1]))

# axs[1][0].spy(np.diag(threshWaveCoeff[0]))
# axs[1][1].spy(np.diag(threshWaveCoeff[1]))




plt.show()