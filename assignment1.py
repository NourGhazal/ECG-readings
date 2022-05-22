from math import floor
from time import time
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal;

def fivePointDifferentiate(ecgData):
    ecgDataDiffer = []
    for i in range(2,ecgData.shape[0]-2):
        y = (-ecgData[i-1]-2*ecgData[i-2]+2*ecgData[i+1]+ecgData[i+2])
        ecgDataDiffer.append(y)
    ecgDataDiffer= np.array(ecgDataDiffer)
    return ecgDataDiffer

def movingAverageFilter(ecgDataDiffersquared,N):
    ecgDataSmoothed = []
    for i in range(N,ecgDataDiffersquared.shape[0]):
        y = np.sum(ecgDataDiffersquared[i-N:i+1])/N
        ecgDataSmoothed.append(y)
    ecgDataSmoothed= np.array(ecgDataSmoothed)
    return ecgDataSmoothed

def detectRthreshold(ecgDataSmoothed):
    return ecgDataSmoothed.max() - (ecgDataSmoothed.std())
    
def detectRWave(ecgDataSmoothed,threshold,f):
    timeStamps = []
    tankenI=[]
    # if(tankenI)
   
    for i in range(1,ecgDataSmoothed.shape[0]-1):
        flag=True
        for j in range(0,len(tankenI)):
            if(tankenI[j]>= i-20 and tankenI[j]<=i+20):
                flag=False
                break
        if(ecgDataSmoothed[i] > threshold  and ecgDataSmoothed[i]>ecgDataSmoothed[i+1] and ecgDataSmoothed[i]>ecgDataSmoothed[i-1]  and flag ):
            timeStamps.append(i/f)
            tankenI.append(i)
    timeStamps= np.array(timeStamps)
    # print(timeStamps)
    return timeStamps
def detectRRInterval(timeStamps,ecgDataSmoothed,f):
    rrInterval = []
    for i in range(0,timeStamps.shape[0]-1):
        # ecgDataSmoothed[round(timeStamps[i]*f)]
        # if((timeStamps[i+1]-timeStamps[i])*1000 > 20):
        rrInterval.append((timeStamps[i+1]-timeStamps[i])*1000)
    rrInterval= np.array(rrInterval)
    # print(rrInterval)
    return rrInterval
        

def startECGAnalysis(f,N):
    ecgData = np.loadtxt("DataN.txt")
    ecgData = notchAndBandFilters(256,50,30,0.1,45,ecgData)
    ecgDataDiffer = fivePointDifferentiate(ecgData)
    ecgDataDiffersquared = np.square(ecgDataDiffer)
    ecgDataSmoothed = movingAverageFilter(ecgDataDiffersquared,N)
    threshold = detectRthreshold(ecgDataSmoothed)
    timeStamps = detectRWave(ecgDataSmoothed,threshold,f)
    rrInterval = detectRRInterval(timeStamps,ecgDataSmoothed,f)
    return {"ecgData":ecgData.tolist(),"ecgDataDiffer":ecgDataDiffer.tolist(),"ecgDataDiffersquared":ecgDataDiffersquared,"ecgDataSmoothed":ecgDataSmoothed.tolist(),"timeStamps":timeStamps.tolist(),"rrInterval":rrInterval.tolist()}

def startECGAnalysisNoFilter(f,N):
    ecgData = np.loadtxt("DataN.txt")
    ecgData = notchAndBandFilters(256,50,30,0.1,45,ecgData)
    # ecgDataDiffer = fivePointDifferentiate(ecgData)
    # ecgDataDiffersquared = np.square(ecgDataDiffer)
    # ecgDataSmoothed = movingAvergeFilter(ecgDataDiffersquared,N)
    threshold = detectRthreshold(ecgData)
    timeStamps = detectRWave(ecgData,threshold,f)
    rrInterval = detectRRInterval(timeStamps,ecgData,f)
    return {"ecgData":ecgData.tolist(),"timeStamps":timeStamps.tolist(),"rrInterval":rrInterval.tolist()}
    # print(timeStamps.shape) 
def ecgAnalysis():
    dict = startECGAnalysis(256,10)
    t = np.arange(len(dict["ecgData"])) /256
    t2 = np.arange(len(dict["ecgDataSmoothed"])) /256
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, dict["ecgData"])
    axs[0].set_xlim(0, 6)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('amplitude')
    axs[1].plot(t2, dict["ecgDataSmoothed"])
    axs[1].set_xlim(0, 6)
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('amplitude')
    fig.tight_layout()
    plt.savefig("Before_After_Filter.jpg")

    keys = [10,15,25]

    for key in keys:
        fig1, axs1 = plt.subplots()
        dict = startECGAnalysis(256,key)
        t = np.arange(len(dict["ecgDataSmoothed"])) /256
        axs1.plot(t,dict["ecgDataSmoothed"])
        rWave =[]
        for i in dict["timeStamps"]:
            rWave.append(dict["ecgDataSmoothed"][round(i*256)])
        t2 = np.array(dict["timeStamps"]) 
        axs1.set_xlim(0, 6)
        axs1.scatter(t2,rWave, marker="*",color="red")
        # plt.show()
        plt.savefig("DetectedR_{}.jpg".format(str(key)))

    fig1, axs1 = plt.subplots()
    dict = startECGAnalysis(256,25)
    t = np.arange(len(dict["rrInterval"]))
    print(len(dict["rrInterval"]))
    axs1.plot(t,dict["rrInterval"]) 
    plt.savefig("RR.jpg".format(str(key)))
    # fig1, axs1 = plt.subplots()
    # axs1.plot(t,dict["rrInterval"]) 
    # plt.savefig("RR.jpg".format(str(key)))




    fig1, axs1 = plt.subplots()
    dict = startECGAnalysisNoFilter(256,25)
    t = np.arange(len(dict["ecgData"])) /256
    axs1.plot(t,dict["ecgData"])
    rWave =[]
    for i in dict["timeStamps"]:
        rWave.append(dict["ecgData"][round(i*256)])
    t2 = np.array(dict["timeStamps"]) 
    axs1.set_xlim(0, 6)
    axs1.scatter(t2,rWave, marker="*",color="red")
    # plt.show()
    plt.savefig("Unfiltered_{}.jpg".format(str(25)))

def detectMissingBeats(rrInterval,timeStamps):
    missingTimeStamps=[]
    threshold = rrInterval.mean()+ rrInterval.std()
    for i in range(rrInterval.shape[0]):
        if(rrInterval[i]>threshold):
           y=(((timeStamps[i+1]*1000)-rrInterval[i])/1000)*256
           missingTimeStamps.append(y) 
    missingTimeStamps= np.array(missingTimeStamps)
    return missingTimeStamps
def sinusArrest():
    ecgData = np.loadtxt("Data2.txt")
    ecgData = notchAndBandFilters(256,50,30,0.1,45,ecgData)
    N=15
    f=256
    ecgDataDiffer = fivePointDifferentiate(ecgData)
    ecgDataDiffersquared = np.square(ecgDataDiffer)
    ecgDataSmoothed = movingAverageFilter(ecgDataDiffersquared,N)
    threshold = detectRthreshold(ecgDataSmoothed)
    timeStamps = detectRWave(ecgDataSmoothed,threshold,f)
    rrInterval = detectRRInterval(timeStamps,ecgDataSmoothed,f)
    missingTimeStamps=detectMissingBeats(rrInterval,timeStamps)
    print(missingTimeStamps)
    np.savetxt('MissingBeats.txt',missingTimeStamps)

def notchAndBandFilters(sampleFrequency,notchFilter,qualityFactor,lowPass,highPAss,noisySignal):
    b, a = signal.iirnotch(notchFilter, qualityFactor, sampleFrequency)
    noisySignal = signal.filtfilt(b, a, noisySignal)
    nyquist_freq = 0.5 * sampleFrequency
    low = lowPass / nyquist_freq
    high = highPAss / nyquist_freq
    b, a = signal.butter(1, [low, high], btype='band')
    noisySignal = signal.filtfilt(b, a, noisySignal)
    return noisySignal



ecgAnalysis()
sinusArrest()




