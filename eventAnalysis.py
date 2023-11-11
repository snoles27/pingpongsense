
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import receiveData as rec

class singeChannelFFT: 
    def __init__(self, freqs:list[float], mags:list[float], label:str):
        self.freqs = freqs.copy()
        self.mags = mags.copy()
        self.label = label

def fftSelect(eventData:rec.event, channelNumber:int, plot = True, nullAvg = True):

    times, values = eventData.getChannelData(channelNumber)
    freqs, mags = dataFft(times, values, plot=False, nullAvg=nullAvg)
    if plot:
        fig = plt.gcf()
        fig.canvas.manager.set_window_title(str(eventData) + ", Channel: " + str(channelNumber))
        plt.plot(freqs, mags)
        plt.title(str(eventData) + ", Channel: " + str(channelNumber))
        plt.xlabel("Frequency (Hz)")
        plt.xlim([0,3000])
        plt.show()

    return freqs, mags

def fftOverlay(eventData:rec.event):
    
    freqs0, values0 = fftSelect(eventData, 0, plot = False)
    freqs1, values1 = fftSelect(eventData, 1, plot=False)
    freqs2, values2 = fftSelect(eventData, 2, plot=False)

    fig = plt.gcf()
    fig.canvas.manager.set_window_title(str(eventData))
    plt.plot(freqs0, values0, label = "ch0")
    plt.plot(freqs1, values1, label = "ch1")
    plt.plot(freqs2, values2, label = "ch2")
    plt.title( str(eventData))
    plt.xlabel("Frequency (Hz)")
    plt.xlim([0,3000])
    plt.legend()
    plt.show()

def dataFft(times:list[int], values:list[int], nullAvg = True, plot = False):
    
    if len(times) != len(values):
        print("NONEQUAL VECTOR SIZES")

    N = len(times)
    timeSeconds = [usTime * 1e-6 for usTime in times]
    timeDeltas = []
    for i in range(1, len(timeSeconds)):
        timeDeltas.append(timeSeconds[i]-timeSeconds[i-1])
    avgSampleSpacing = np.average(timeDeltas)
    #print("Sample Spacing: " + str(avgSampleSpacing))

    if nullAvg:
        average = np.average(values)
        amps = [value - average for value in values]
    else:
        amps = values.copy()
   
    ffts = sp.fft.fft(amps, N, norm = "forward")
    freqs = sp.fft.fftfreq(N, avgSampleSpacing)
    mags = abs(ffts)
    phase = np.angle(ffts)

    if plot:
        plt.plot(freqs, mags)
        plt.xlim([0,3000])
        plt.xlabel("Frequency (Hz)")
        plt.show()

    return freqs, mags

def applyMetricToEvents(events:list[rec.event], metric):
    #events: list of event objects
    #metric: function that takes an event and returns some value characterizing the event
    pass

def compareFFTMetric(events:list[rec.event], metric):
    #events: list of event objects
    #metric: function that takes two lists metric(freqs, mags) and returns some value characterizing the FFT

    #get ffts of ping pong events and not ping pong events
    fftList:list[singeChannelFFT] = []
    for singleEvent in events:
        freqs, mags = fftSelect(singleEvent, 0, plot=False, nullAvg=True)
        fftList.append(singeChannelFFT(freqs, mags, singleEvent.label))
        freqs, mags = fftSelect(singleEvent, 1, plot=False, nullAvg=True)
        fftList.append(singeChannelFFT(freqs, mags, singleEvent.label))
        freqs, mags = fftSelect(singleEvent, 2, plot=False, nullAvg=True)
        fftList.append(singeChannelFFT(freqs, mags, singleEvent.label))
    
    metricResults_p = []
    metricResults_n = []
    for fft in fftList:
        if fft.label == "p":
            metricResults_p.append(metric(fft.freqs, fft.mags))
        else:
            metricResults_n.append(metric(fft.freqs, fft.mags))
        
    return metricResults_p, metricResults_n

def magRMSRange(freqs, mags, minFreq, maxFreq):
    
    freqsarray = np.array(freqs)
    magsarray = np.array(mags)
    totalRMS = np.sqrt(np.mean(magsarray**2))

    minIndex = np.argmin(np.absolute(freqsarray-minFreq))
    maxIndex = np.argmin(np.absolute(freqsarray-maxFreq))

    rangeRMS = np.sqrt(np.mean(magsarray[minIndex:maxIndex]**2))

    return rangeRMS/totalRMS

def meanChannelMagRMSRange(event:rec.event, minFreq, maxFreq):

    fftList:list[singeChannelFFT] = []
    freqs, mags = fftSelect(event, 0, plot=False, nullAvg=True)
    fftList.append(singeChannelFFT(freqs, mags, event.label))
    freqs, mags = fftSelect(event, 1, plot=False, nullAvg=True)
    fftList.append(singeChannelFFT(freqs, mags, event.label))
    freqs, mags = fftSelect(event, 2, plot=False, nullAvg=True)
    fftList.append(singeChannelFFT(freqs, mags, event.label))

    rmsList = []
    for fft in fftList:
        rmsList.append(magRMSRange(fft.freqs, fft.mags, minFreq, maxFreq))

    return np.mean(rmsList)


if __name__ == "__main__":
    folderName = "Data/RawEventData/"
    minFreq = 1000
    maxFreq = 2000

    def rms(freqs, mags) : return magRMSRange(freqs, mags, minFreq, maxFreq)
    
    allEvents = rec.readAllEvents(folderName)
    for event in allEvents:
        print(event.label + ": " + str(meanChannelMagRMSRange(event, minFreq, maxFreq)))

    # ppValue, npValue = compareFFTMetric(allEvents, rms)
    # print(np.mean(ppValue))
    # print(np.mean(npValue))
