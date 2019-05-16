import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

t0 = 0
tN = 1
N =  1023
t = np.linspace(t0,tN,num = N)
dt = t[1] - t[0]
yt = np.piecewise(t,[ (t>=0) & (t<.25) ,
                      (t>=.25) & (t<.5) ,
                      (t>=.5) & (t<.75) ,
                      (t>=.75) & (t<1) ,
                      t >=1],
                      [lambda t : np.sin(2*np.pi * 10 * t),
                       lambda t : np.sin(2*np.pi * 25 * t),
                       lambda t : np.sin(2*np.pi * 75 * t),
                       lambda t : np.sin(2*np.pi * 100 * t),
                       lambda t : 0])

waveletname = 'cmor'
scales = np.arange(1,128)
coefficients, frequencies = pywt.cwt(yt, scales, waveletname, dt)
fftYt = np.fft.fft(yt)
fftPower = (abs(fftYt)) ** 2
power = (abs(coefficients)) ** 2
period = 1. / frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
contourlevels = np.log2(levels)
fig , axs =  plt.subplots( 3, sharex=False, sharey=False)
axs[0].plot(t,yt)
axs[1].plot(fftPower[0:200])
axs[2].contourf(t, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=plt.cm.seismic)
plt.show()

# fig, ax = plt.subplots(figsize=(15, 10))
# im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    

# #Compute multilevel DWT of yt

# dwtYT = pywt.wavedec(yt,'db1')
# print(len(dwtYT[0]))

# fig , axs = plt.subplots(len(dwtYT))
# for i , data in enumerate(dwtYT):
#     axs[i].plot(data)
# plt.show()


# # plt.figure()
# # plt.plot(t,yt)
# # plt.show()

# # yt = np.multiply(np.exp(-t) , np.cos(2*np.pi*t))
# # ecgData = pywt.data.ecg()
# #wp = pywt.WaveletPacket(data = yt , wavelet='db1',mode='symmetric')
# #wpECG = pywt.WaveletPacket(data = ecgData , wavelet='db1',mode='symmetric')
# # nLevels = wp.maxlevel
# # levels = []
# # subplots = []



# for i in range(1,7 ):
#     levels.append([node.path for node in wp.get_level(i, 'natural')])
#     fig , axs =  plt.subplots( len(levels[i-1]), sharex=False, sharey=False)
#     for j , path in enumerate(levels[i-1]):
#         axs[j].plot(wp[path].data)


# plt.show()
