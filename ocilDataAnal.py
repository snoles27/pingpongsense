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


def closestIndex(nums, value):

    return np.argmin(np.abs(np.array(nums)-value))

def readDataFromCSV(pathName:str, plot = False, probeRatio = 1.0, smoothing = 0, onlyPositiveT = False) ->  eventData:
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

    if onlyPositiveT:
        startIndex = closestIndex(times, 0.)
        voltages = voltages[startIndex:]
        times = times[startIndex:]

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

#hardcoded scaling based on how the data was collected assuming <19 is 10X and >19 is 1X probe multiple
def overlayFfts(pathBase:str, fileNumbers:list[str], smoothing = 0, onlyPositiveT = False):

    for i in fileNumbers:
        index = int(i[2:])
        if index < 19:
            print("Scaling: " + i)
            scale = 10
        else:
            scale = 1
        fileName = pathBase + i + ".csv"
        data = readDataFromCSV(fileName, smoothing=smoothing, onlyPositiveT=onlyPositiveT, probeRatio=scale)
        freqs, mags = dataFft(data)
        plt.plot(freqs, mags, label = str(i))
    
    plt.title("FFTs of Data")
    plt.xlim([0,3000])
    plt.xlabel("Freqsuency (Hz)")
    plt.legend()
    plt.show()

#hardcoded scaling based on how the data was collected assuming <19 is 10X and >19 is 1X probe multiple
def overlayData(pathBase:str, fileNumbers:list[str], smoothing = 0, onlyPositiveT = False):

    for i in fileNumbers:
        index = int(i[2:])
        if index < 19:
            print("Scaling " + i)
            scale = 10
        else:
            scale = 1
        fileName = pathBase + i + ".csv"
        data = readDataFromCSV(fileName, smoothing=smoothing, onlyPositiveT=onlyPositiveT, probeRatio=scale)
        plt.plot(data.times, data.voltages, label = str(i))
    
    plt.title("Voltage vs Time")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
        
def movingAverage(data:list[float], nhalf) -> list[float]:

    dataSmoothed = data.copy()
    for i in range(nhalf, len(data) - nhalf):
        dataSmoothed[i] = sum(data[i - nhalf: i + nhalf - 1])/(nhalf * 2)

    return dataSmoothed
    
if __name__ == "__main__":
    print("in ocilDataAnal.py: ")
    pathBase = "/Users/Sam/Code/pingpongsense/Data/Oscilliscope 110423/"
    listToPlot = ["pp10", "pp14", "np25", "np21", "np23"]
    overlayData(pathBase, listToPlot, smoothing=0, onlyPositiveT=True)
    overlayFfts(pathBase, listToPlot, smoothing=0, onlyPositiveT=False)
    


    # pathName = "/Users/Sam/Code/pingpongsense/Data/Oscilliscope 110423/pp10.csv"
    # ppData = readDataFromCSV(pathName, plot = False, probeRatio=10)
    # movingAverage(ppData.voltages, 5)

