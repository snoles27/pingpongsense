import serial
from inputimeout import inputimeout 
import matplotlib.pyplot as plt
import os
import time

SERIALPORT3 = "/dev/tty.usbmodem14301"
SERIALPORT1 = "/dev/tty.usbmodem14201"
SERIALPORT = "/dev/tty.usbmodem1101"
BAUDRATE = 57600 #must match that set in eventRecordandOutput
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
UUID_LINE = 0
LABEL_LINE = 1
CHANNEL_INDEX = 0
TIME_INDEX = 1
VALUE_INDEX = 2


class eventDataPoint:
    """
    Class to store data for a single data point of an event. 
    value: (int) Magnitude of signle (a.u.)s
    time: (int) time reported from microcontroller (us)
    """
    def __init__(self, value:int, time:int):
        self.value = value
        self.time = time

    def __str__(self):
        return "Time: " + str(self.time) + ", Value: " + str(self.value)

class event:
    """
    A "ping pong event". All the information dumped from the microcontroller when the signal crosses a threshold
    uuid: unique tag to name the even for later reference
    label: Label for type of event. 
    a0Data: list of eventDataPoints from a0 channel
    a1Data: list of eventDataPoints from a1 channel
    a2Data: list of eventDataPoints from a2 channel
    """
    def __init__(self, uuid:str, a0Data:list[eventDataPoint] = [], a1Data:list[eventDataPoint] = [], a2Data:list[eventDataPoint] = [], label:str = NOLABELSTR):
        self.uuid = uuid
        self.label = label
        self.a0Data = a0Data.copy()
        self.a1Data = a1Data.copy()
        self.a2Data = a2Data.copy()

    def getShortUUID(self):
        return self.uuid[:UUIDABREIV]
    
    def __str__(self):
        return "UUID-" + self.getShortUUID() + "___Label-" + self.label

    
    def getChannelData(self, channelNumber:int) -> tuple[list[int], list[int]]:
        """
        channelNumber: (int) index of channel data is being requested for
        returns list of times and list of values from channelNumber 
        """

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
    
    def getChannelEvents(self, channelNumber:int) -> list[eventDataPoint]:

        """
        channelNumber: (int) index of channel data is being requested for
        returns list eventDataPoints
        """
        
        if channelNumber == 0:
            return self.a0Data
        elif channelNumber == 1:
            return self.a1Data
        elif channelNumber == 2: 
            return self.a2Data
        else:
            raise("INVALID CHANNEL NUMBER")
        

def readEventData(openPort, requestLabel:bool = False) -> event:
    """
    #openPort: open serial port object
    #returns: event object if event happens within READATTEMPTIMEOUT of function call. None if not. 
    """

    with openPort as ser:
   
        line = str(ser.readline())
        if len(line) >= MINLEN:
            uuid = line[UUIDSTART:LINEEND]
            eventData = event(uuid = uuid)
            secondLine = str(ser.readline()) #do nothing with the second line (yet)
            line = str(ser.readline()) #read in the first line to process
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

            # print(eventData)
            return eventData
        else:
            return None

def processLine(line:str) -> list[int]:
    """
    line: (str) comma-space delimited list of integers
    returns: list of integrers
    """

    splitList = line[LINESTART:LINEEND].split(", ")
    return [int(ele) for ele in splitList]
    
def _requestEventLabel(eventData:event, timeout = 10) -> None:
    try:
        eventData.label = inputimeout("Label Data UUID: " + eventData.uuid[:8] + " (p = ping pong ball, n = not ping pong ball)", timeout=timeout)
    except Exception: 
        print("TIMEOUT: LABEL UNCHANGED")

def eventFileWrite(folderLoc:str, eventData:event) -> None:

    fileName = str(eventData) + ".txt"
    eventFileWriteGenericName(folderLoc, eventData, fileName)

    return

def eventFileWriteGenericName(folderLoc:str, eventData:event, fileName:str) -> None:
    
    """
    Writes data from eventData to a new file in folderLoc with eventData.uuid in the title
    """

    fullPath = folderLoc + fileName

    file = open(fullPath, mode = 'x')
    file.write(fileName + "\n")
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

def eventFileRead(fullPath:str, numHeaderLines = HEADERLINESNUM, uuidLine = UUID_LINE, labelLine = LABEL_LINE) -> event:

    """
    Reads data from fullPath and returns and event object 
    """
    
    headerLines = []
    file = open(fullPath, 'r')
    for i in range(0, numHeaderLines):
        headerLines.append(str(file.readline()))

    uuid = str(headerLines[uuidLine])[:-1].replace("UUID: ", "")
    label = str(headerLines[labelLine])[:-1].replace("Label: ", "")

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
    
    """
    Returns list of all events located in the folderLoc folder
    """

    events = []
    files = os.listdir(folderLoc)
    files = [f for f in files if os.path.isfile(folderLoc+'/'+f)] #Filtering only the files.
    for file in files:
        fullPath = folderLoc + file
        events.append(eventFileRead(fullPath))

    # for item in events:
    #     print(item)

    return events
    
def readSingleEvent(openPort, save:bool = True, folderName:str = "Data/RawEventData/LocatingData/", numAttempt = 10) -> event:

    """
    Reads single event dump from the microcontroller if happens within certain number of read attempts
    Allows options for plotting and saving event with user defined name 
    HARDCODED FOLDER ENTER :(
    """

    countMiss = 0
    while True:
        eventData = readEventData(openPort, requestLabel=False)
        if eventData is not None:

            if save:
                print("Saving to folder " + folderName)
                name = input("File Name: ")
                eventFileWriteGenericName(folderName, eventData, name + ".txt")

            break

        else:
            countMiss = countMiss + 1
            if countMiss > numAttempt:
                leave = input("Want to stop? (Y/N)")
                if leave == "Y":
                    break
                else:
                    countMiss = 0

if __name__ == "__main__":

    folderName = "Data/RawEventData/"
    ser = serial.Serial(SERIALPORT, BAUDRATE, timeout = READATTEMPTTIMEOUT)

    readSingleEvent(ser)