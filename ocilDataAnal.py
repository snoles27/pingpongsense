import csv 
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


TIMESROW_T0 = 2
TIMESROW_TSTEP = 3
ROW_INDEX = 0
ROW_VALUE = 1

class eventData:
    def __init__(self, times: list[float], voltages: list[float]):
        self.times = times.copy()
        self.voltages = voltages.copy()
        self.numPoints = len(times)
        self.sampleSpaceing = times[1] - times[0]

def readDataFromCSV(pathName:str, plot = False, probeRatio = 1.0, smoothing = 0) ->  eventData:
    #pathName: path to read the CSV

    with open(pathName, newline='') as csvfile:
         reader = csv.reader(csvfile, delimiter=',', quotechar='|')
         reader.__next__()
         timesRow = reader.__next__()
         t0 = float(timesRow[TIMESROW_T0])
         tstep = float(timesRow[TIMESROW_TSTEP])
         times = []
         voltages = []
         for row in reader:
              times.append(t0 + tstep * float(row[ROW_INDEX]))
              voltages.append(float(row[ROW_VALUE])/probeRatio)

    if smoothing > 0: 
        voltages = movingAverage(voltages, smoothing)

    if plot:
        plt.plot(times, voltages)
        plt.show()

    return eventData(times, voltages)

def dataFft(data:eventData, plot = False):
    
    N = data.numPoints
    print(data.numPoints)
    print(data.sampleSpaceing)

    ffts = sp.fft.fft(data.voltages, N, norm = "forward")
    freqs = sp.fft.fftfreq(N, data.sampleSpaceing)
    mags = abs(ffts)
    phase = np.angle(ffts)

    if plot:
        plt.plot(freqs, mags)
        plt.xlim([0,3000])
        plt.xlabel("Frequency (Hz)")
        plt.show()

    return freqs, mags

def overlayFfts(pathBase:str, fileNumbers:list[str], smoothing = 0):

    for i in fileNumbers:
        fileName = pathBase + i + ".csv"
        data = readDataFromCSV(fileName, smoothing=smoothing)
        freqs, mags = dataFft(data)
        plt.plot(freqs, mags, label = str(i))
    
    plt.title("FFTs of Data")
    plt.xlim([0,3000])
    plt.xlabel("Freqsuency (Hz)")
    plt.legend()
    plt.show()

def overlayData(pathBase:str, fileNumbers:list[str], smoothing = 0):

    for i in fileNumbers:
        fileName = pathBase + i + ".csv"
        data = readDataFromCSV(fileName, smoothing=smoothing)
        plt.plot(data.times, data.voltages, label = str(i))
    
    plt.title("Voltage vs Time")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
        
def movingAverage(data:list[float], nhalf) -> list[float]:

    dataSmoothed = data.copy()
    for i in range(nhalf, len(data) - nhalf):
        dataSmoothed[i] = sum(data[i - nhalf: i + nhalf - 1])/(nhalf * 2)
    
    # plt.plot(data)
    # plt.plot(dataSmoothed)
    # plt.show()

    return dataSmoothed
    
if __name__ == "__main__":
    print("in ocilDataAnal.py: ")
    pathBase = "/Users/Sam/Code/pingpongsense/Data/Oscilliscope 110423/"
    listToPlot = ["pp7","pp12", "np21", "np25", "np23"]
    overlayData(pathBase, listToPlot, smoothing=5)
    overlayFfts(pathBase, listToPlot, smoothing=5)
    


    # pathName = "/Users/Sam/Code/pingpongsense/Data/Oscilliscope 110423/pp10.csv"
    # ppData = readDataFromCSV(pathName, plot = False, probeRatio=10)
    # movingAverage(ppData.voltages, 5)

