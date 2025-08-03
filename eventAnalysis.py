
import scipy
import numpy as np
import matplotlib.pyplot as plt
import ReceiveData as rec

class singeChannelFFT: 
    def __init__(self, freqs:list[float], mags:list[float], label:str):
        self.freqs = freqs.copy()
        self.mags = mags.copy()
        self.label = label

#obsolete
def compareFFTMetric(events:list[rec.Event], metric):
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

def fftSelect(eventData:rec.Event, channelNumber:int, plot = True, nullAvg = True):

    times, values = eventData.get_channel_data(channelNumber)
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

def fftOverlay(eventData:rec.Event):
    
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
    
    assert len(times) == len(values), "Vector sizes are not equal"

    N = len(times)
    timeSeconds = [usTime * 1e-6 for usTime in times]
    timeDeltas = []
    for i in range(1, len(timeSeconds)):
        timeDeltas.append(timeSeconds[i]-timeSeconds[i-1])
    avgSampleSpacing = np.average(timeDeltas)
    stdSampleSpacing = np.std(timeDeltas)
    print("Sample Spacing: " + str(avgSampleSpacing) + " +/- " + str(stdSampleSpacing))
    print("Percent Variation: " + str(stdSampleSpacing/avgSampleSpacing))

    if nullAvg:
        average = np.average(values)
        amps = [value - average for value in values]
    else:
        amps = values.copy()
   
    ffts = scipy.fft.fft(amps, N, norm = "forward")
    freqs = scipy.fft.fftfreq(N, avgSampleSpacing)
    mags = abs(ffts)
    phase = np.angle(ffts)

    if plot:
        plt.plot(freqs, mags)
        plt.xlim([0,3000])
        plt.xlabel("Frequency (Hz)")
        plt.show()

    return freqs, mags

def applyMetricToEvents(events:list[rec.Event], metric):
    #events: list of event objects
    #metric: function that takes an event and returns some value characterizing the event

    metricResults_p = []
    metricResults_n = []

    for event in events:
        if event.label == "p":
            metricResults_p.append(metric(event))
        elif event.label == "n":
            metricResults_n.append(metric(event))
        
    return metricResults_p, metricResults_n

def magRMSRange(freqs, mags, minFreq, maxFreq):
    #freqs: frequencies in fourier tranform
    #mags: magnitudes in fourier transform
    #minfreq: minimum frequency
    #maxfreq: maximum frequncy 

    freqsarray = np.array(freqs)
    magsarray = np.array(mags)
    totalRMS = np.sqrt(np.mean(magsarray**2)) #total RMS value of all the magintudes

    minIndex = np.argmin(np.absolute(freqsarray-minFreq))
    maxIndex = np.argmin(np.absolute(freqsarray-maxFreq))

    rangeRMS = np.sqrt(np.mean(magsarray[minIndex:maxIndex]**2)) #RMS of all values in the range 

    return rangeRMS/totalRMS

def meanChannelMagRMSRange(event:rec.Event, minFreq, maxFreq):

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

def evaluateMetric(events:list[rec.Event], metric, cost):

    presults, nresults = applyMetricToEvents(events, metric)
    return cost(presults, nresults)

def classify_RMS(event:rec.Event, minFreq, maxFreq, thresh) -> bool:
    return meanChannelMagRMSRange(event, minFreq, maxFreq) > thresh

if __name__ == "__main__":


    folderName = "Data/RawEventData/"
    event1 = rec.event_file_read(folderName + "LocatingData/6in_(18.0,6.4)_0.txt")
    event2 = rec.event_file_read(folderName + "LocatingData/6in_(18.0,6.4)_1.txt")

    channel = 2
    fig, ax = plt.subplots()
    event1.plot(ax)
    plt.show()
    x,t = event1.get_channel_data(channel)
    x = np.array(x)
    t = np.array(t)

    x = x[15:-250]
    t = t[15:-250]
    t_s = t * 1e-6

    Tx,uTx = event1.sample_rate(channel)
    fs = 1/Tx * 1e6
    N = len(x)

    window_length = 6
    g_std = 2
    # window = scipy.signal.windows.gaussian(
    #     M=window_length,
    #     std =g_std,
    #     sym=True
    # )
    window = scipy.signal.windows.boxcar(window_length)
    SFT = scipy.signal.ShortTimeFFT(
        win=window,
        hop=3,
        fs=fs,
        fft_mode='centered',
        mfft=None,
        scale_to='magnitude'
    )

    Sx = SFT.stft(x)
    t_lo,t_hi,f_lo,f_hi = SFT.extent(
        n=N,
        axes_seq='tf'
    )

    fig1, ax1 = plt.subplots(figsize=(6., 4.)) 

    ax1.set_title(rf"STFT ({SFT.m_num*SFT.T:g}$\,us$ Gaussian window, " +
                rf"$\sigma_t={g_std*SFT.T}\,$us)")
    ax1.set(xlabel=f"Time $t$ in microseconds ({SFT.p_num(N)} slices, " +
                rf"$\Delta t = {SFT.delta_t:g}\,$us)",
            ylabel=f"Freq. $f$ in MHz ({SFT.f_pts} bins, " +
                rf"$\Delta f = {SFT.delta_f:g}\,$MHz)",
            xlim=(t_lo, t_hi))

    im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                    extent=SFT.extent(N), cmap='viridis')
    fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

    # Shade areas where window slices stick out to the side:
    for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                    (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
        ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
    for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
        ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    plt.show()





   


#### Older things pre 8/1/2025
#     folderName = "Data/RawEventData/"
#     def rmsCost(presults, nresults) : return (1 - (np.min(presults) - np.max(nresults))/(np.max(nresults)))
#     allEvents = rec.readAllEvents(folderName)

#     # minFreq = 1415
#     # maxFreq = 1663

#     ## 11/12/23 results
#     # minFreq = 1163.157894736842
#     # maxFreq = 1915.7894736842106
#     # mid = 0.31597582047520656

#     # def rms(freqs, mags) : return magRMSRange(freqs, mags, minFreq, maxFreq)
#     # def channelRMS(event) : return meanChannelMagRMSRange(event, minFreq, maxFreq)
   
#     # presults, nresults = applyMetricToEvents(allEvents, channelRMS)

#     # plt.vlines(presults, -1, 1, colors="blue")
#     # plt.vlines(nresults, -1, 1, colors="red")
#     # plt.vlines(0.312, -1, 1, colors="black")
#     # plt.show()


# #### Evaluating a bunch of possible frequencies 
#     N = 20
#     minFreqs = np.linspace(700, 1500, N)
#     maxFreqs = np.linspace(1200, 2000, N)

#     results = np.zeros((N,N))
#     for i in range(0,N):
#         for j in range(0,N):
#             minfreq = minFreqs[i]
#             maxfreq = maxFreqs[j]
#             if minfreq >= maxfreq-20: #not allowed condition cost --> inf
#                 results[i][j] = np.inf
#             else:
#                 def eval(event) : return meanChannelMagRMSRange(event, minfreq, maxfreq)
#                 results[i][j] = evaluateMetric(allEvents, eval, rmsCost)
            
#     #Dislaying results and plotting the splits on the best one
#     print(results)
#     bestIndex = np.argmin(results)
#     i = int(np.floor(bestIndex/N))
#     j = bestIndex % N
#     print(bestIndex)
#     print(i)
#     print(j)
#     print(results[i][j])
#     minFreq = minFreqs[i]
#     maxFreq = maxFreqs[j]

#     print("Min Freq: " + str(minFreq))
#     print("Max Freq: " + str(maxFreq))
#     def channelRMS(event) : return meanChannelMagRMSRange(event, minFreq, maxFreq)
#     presults, nresults = applyMetricToEvents(allEvents, channelRMS)

#     print("mid: " + str((min(presults) + max(nresults))/2))
#     plt.vlines(presults, -1, 1, colors="blue")
#     plt.vlines(nresults, -1, 1, colors="red")
#     plt.show()
#######

    # 

    # for event in allEvents:
    #     print(event.label + ": " + str(meanChannelMagRMSRange(event, minFreq, maxFreq)))

    # ppValue, npValue = compareFFTMetric(allEvents, rms)
    # print(np.mean(ppValue))
    # print(np.mean(npValue))
