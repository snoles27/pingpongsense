import serial


SERIALPORT3 = "/dev/tty.usbmodem14301"
BAUDRATE = 9600
READATTEMPTTIMEOUT = 0.5

LINESTART = 2
UUIDSTART = 14
LINEEND = -5
MINLEN = 4
EVENTENDMARKER = "b'COMPLETE\\r\\n'"

class event:
    def __init__(self, uuid:str, data:list[int] = [], times:list[int] = []):
        self.uuid = uuid
        self.data = data.copy()
        self.times = times.copy()

def readEventData(openPort):

    with openPort as ser:
        line = str(ser.readline())
        if len(line) >= MINLEN:
            uuid = line[UUIDSTART:LINEEND]
            secondLine = str(ser.readline())
            line = str(ser.readline())
            while line != EVENTENDMARKER:
                print(processLine(line))
                line = str(ser.readline())
        else:
            print("NO EVENT")
            return None

def processLine(line) -> list[int]:

    splitList = line[LINESTART:LINEEND].split(", ")
    return [int(ele) for ele in splitList]
    

if __name__ == "__main__":
    ser = serial.Serial(SERIALPORT3, BAUDRATE, timeout = READATTEMPTTIMEOUT)
    readEventData(ser)
    ser.close()