import eventAnalysis as anl
import receiveData as rec
import serial

if __name__ == "__main__":
    folderName = "Data/RawEventData/"
    ser = serial.Serial(rec.SERIALPORT3, rec.BAUDRATE, timeout = rec.READATTEMPTTIMEOUT)

    thresh = 0.40
    countMiss = 0
    while True:
        eventData = rec.readEventData(ser, requestLabel=False)
        if eventData is not None:
            print(anl.meanChannelMagRMSRange(eventData, 1000, 2000) > thresh)
        else:
            countMiss = countMiss + 1
            if countMiss > 10:
                leave = input("Want to stop? (Y/N)")
                if leave == "Y":
                    break
                else:
                    countMiss = 0