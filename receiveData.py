import serial


SERIALPORT3 = "/dev/tty.usbmodem14301"
BAUDRATE = 9600
READATTEMPTTIMEOUT = 0.5

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

class event:
    def __init__(self, uuid:str, data:list[list[int]] = [[]], times:list[list[int]] = [[]]):
        self.uuid = uuid
        self.data = data.copy()
        self.times = times.copy()

def readEventData(openPort):

    with openPort as ser:
        line = str(ser.readline())
        if len(line) >= MINLEN:
            dataMat = []
            timesMat = []
            uuid = line[UUIDSTART:LINEEND]
            secondLine = str(ser.readline())
            line = str(ser.readline())
            while line != EVENTENDMARKER:
                #process the line that was read in
                lineData = processLine(line)
                print(lineData)
                dataInLine = [lineData[DATA_A0_INDEX], lineData[DATA_A1_INDEX], lineData[DATA_A2_INDEX]]
                timeInLine = [lineData[TIME_A0_INDEX], lineData[TIME_A1_INDEX], lineData[TIME_A2_INDEX]]
                dataMat.append(dataInLine)
                timeInLine.append(timeInLine)
                line = str(ser.readline())
            
            return event(uuid, dataMat, timesMat)
        else:
            print("NO EVENT")
            return None

def processLine(line) -> list[int]:

    splitList = line[LINESTART:LINEEND].split(", ")
    return [int(ele) for ele in splitList]
    

if __name__ == "__main__":
    ser = serial.Serial(SERIALPORT3, BAUDRATE, timeout = READATTEMPTTIMEOUT)
    eventData = readEventData(ser)
    ser.close()

    if eventData is not None:
        for row in eventData.data:
            print(row)