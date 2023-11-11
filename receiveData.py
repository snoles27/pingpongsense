import serial
from inputimeout import inputimeout 
import matplotlib.pyplot as plt
import os

SERIALPORT3 = "/dev/tty.usbmodem14301"
BAUDRATE = 9600
READATTEMPTTIMEOUT = 2.0

LINESTART = 2
UUIDSTART = 14
LINEEND = -5 
MINLEN = 4
EVENTENDMARKER = "b'COMPLETE\\r\\n'"

DATA_A0_INDEX = 2
TIME_A0_INDEX = 1
DATA_A1_INDEX = 5
TIME_A1_INDEX = 4
DATA_A2_INDEX = 8
TIME_A2_INDEX = 7

UUIDABREIV = 8

NOLABELSTR = "NO LABEL"

#file info
HEADERLINESNUM = 2
CHANNEL_INDEX = 0
TIME_INDEX = 1
VALUE_INDEX = 2

class eventDataPoint:
    def __init__(self, value:int, time:int):
        self.value = value
        self.time = time

    def __str__(self):
        return "Time: " + str(self.time) + ", Value: " + str(self.value)

class event:
    def __init__(self, uuid:str, a0Data:list[eventDataPoint] = [], a1Data:list[eventDataPoint] = [], a2Data:list[eventDataPoint] = [], label:str = NOLABELSTR):
        self.uuid = uuid
        self.label = label
        self.a0Data = a0Data.copy()
        self.a1Data = a1Data.copy()
        self.a2Data = a2Data.copy()
    
    def __str__(self):
        return "UUID-" + self.uuid[:UUIDABREIV] + "___Label-" + self.label
    
    def getChannelData(self, channelNumber:int):
        times = []
        values = []
        if channelNumber == 0:
            values = [point.value for point in self.a0Data]
            times = [point.time for point in self.a0Data]
        elif channelNumber == 1:
            values = [point.value for point in self.a1Data]
            times = [point.time for point in self.a1Data]
        else:
            values = [point.value for point in self.a2Data]
            times = [point.time for point in self.a2Data]

        return times, values



def readEventData(openPort, requestLabel = False) -> event:
    #openPort: open serial port object
    #returns: event object if event happens within READATTEMPTIMEOUT of function call. None if not. 

    with openPort as ser:
        line = str(ser.readline())
        if len(line) >= MINLEN:
            uuid = line[UUIDSTART:LINEEND]
            eventData = event(uuid = uuid)
            secondLine = str(ser.readline()) #do nothing with the second line (yet)
            line = str(ser.readline()) #read in the frist line to process
            while line != EVENTENDMARKER:
                #process the line that was read in
                lineData = processLine(line)

                #get analog timestammped data points
                a0Data = eventDataPoint(lineData[DATA_A0_INDEX], lineData[TIME_A0_INDEX])
                a1Data = eventDataPoint(lineData[DATA_A1_INDEX], lineData[TIME_A1_INDEX])
                a2Data = eventDataPoint(lineData[DATA_A2_INDEX], lineData[TIME_A2_INDEX])
                
                #append data points to event array 
                eventData.a0Data.append(a0Data)
                eventData.a1Data.append(a1Data)
                eventData.a2Data.append(a2Data)

                #read the next line
                line = str(ser.readline())

            if requestLabel:
                _requestEventLabel(eventData)

            print(eventData)
            return eventData
        else:
            print("NO EVENT")
            return None

def processLine(line) -> list[int]:

    splitList = line[LINESTART:LINEEND].split(", ")
    return [int(ele) for ele in splitList]
    
def _requestEventLabel(eventData:event, timeout = 10):
    try:
        eventData.label = inputimeout("Label Data UUID: " + eventData.uuid[:8] + " (p = ping pong ball, n = not ping pong ball)", timeout=timeout)
    except Exception: 
        print("TIMEOUT: LABEL UNCHANGED")

def plotEvent(eventData:event):

    a0Times = [point.time for point in eventData.a0Data]
    a1Times = [point.time for point in eventData.a1Data]
    a2Times = [point.time for point in eventData.a2Data]

    a0Values = [point.value for point in eventData.a0Data]
    a1Values = [point.value for point in eventData.a1Data]
    a2Values = [point.value for point in eventData.a2Data]

    fig = plt.gcf()
    fig.canvas.manager.set_window_title(str(eventData))
    plt.plot(a0Times, a0Values, label = "Sensor 0")
    plt.plot(a1Times, a1Values, label = "Sensor 1")
    plt.plot(a2Times, a2Values, label = "Sensor 2")
    plt.title(str(eventData))
    plt.xlabel("Time (us)")
    plt.ylabel("Response (a.u.)")
    plt.legend()
    plt.show()

def eventFileWrite(folderLoc:str, eventData:event):

    fileName = str(eventData) + ".txt"
    fullPath = folderLoc + fileName

    file = open(fullPath, mode = 'x')
    file.write("UUID: " + eventData.uuid + "\nLabel: " + eventData.label + "\n")

    times, values = eventData.getChannelData(0)
    for i in range(0, len(times)):
        file.write("0, " + str(times[i]) + ", " + str(values[i]) + "\n")

    times, values = eventData.getChannelData(1)
    for i in range(0, len(times)):
        file.write("1, " + str(times[i]) + ", " + str(values[i]) + "\n")

    times, values = eventData.getChannelData(2)
    for i in range(0, len(times)):
        file.write("2, " + str(times[i]) + ", " + str(values[i]) + "\n")

def eventFileRead(fullPath:str) -> event:

    
    headerLines = []
    file = open(fullPath, 'r')
    for i in range(0, HEADERLINESNUM):
        headerLines.append(str(file.readline()))

    uuid = str(headerLines[0])[:-1].replace("UUID: ", "")
    label = str(headerLines[1])[:-1].replace("Label: ", "")

    axLists = [[], [], []]
    while True:
        line = file.readline()
        if not line:
            break
        
        linestr = str(line)[:-1] #convert to string andd remove new line indexs
        lineItems = linestr.split(", ")

        channel = int(lineItems[CHANNEL_INDEX])
        eventPoint = eventDataPoint(int(lineItems[VALUE_INDEX]), int(lineItems[TIME_INDEX]))
        axLists[channel].append(eventPoint)

    return event(uuid=uuid, a0Data=axLists[0], a1Data=axLists[1], a2Data=axLists[2], label=label)

def readAllEvents(folderLoc:str) -> list[event]:

    events = []
    files = os.listdir(folderLoc)
    files = [f for f in files if os.path.isfile(folderLoc+'/'+f)] #Filtering only the files.
    for file in files:
        fullPath = folderLoc + file
        events.append(eventFileRead(fullPath))
    
    return events
 


    

if __name__ == "__main__":
    folderName = "Data/RawEventData/"
    ser = serial.Serial(SERIALPORT3, BAUDRATE, timeout = READATTEMPTTIMEOUT)

    countMiss = 0
    while True:
        eventData = readEventData(ser, requestLabel=True)
        if eventData is not None:
            eventFileWrite(folderName, eventData)
        else:
            countMiss = countMiss + 1
            if countMiss > 10:
                leave = input("Want to stop? (Y/N)")
                if leave == "Y":
                    break
                else:
                    countMiss = 0
    # fullPath = folderName + str(eventData) + ".txt"
    # dataRead = eventFileRead(fullPath)
    # plotEvent(dataRead)
