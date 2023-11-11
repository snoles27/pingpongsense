
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import receiveData as rec

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
    print("Sample Spacing: " + str(avgSampleSpacing))

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