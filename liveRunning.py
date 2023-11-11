import eventAnalysis as anl
import receiveData as rec
import serial

#Classification Constants
MIN_RMS_FREQ = 1415
MAX_RMS_FREQ = 1663
THRESH = 0.312

if __name__ == "__main__":
    folderName = "Data/RawEventData/"
    ser = serial.Serial(rec.SERIALPORT3, rec.BAUDRATE, timeout = rec.READATTEMPTTIMEOUT)

    countMiss = 0
    while True:
        eventData = rec.readEventData(ser, requestLabel=False)
        if eventData is not None:
            print(anl.classify_RMS(eventData, MIN_RMS_FREQ, MAX_RMS_FREQ, THRESH))
        else:
            countMiss = countMiss + 1
            if countMiss > 10:
                leave = input("Want to stop? (Y/N)")
                if leave == "Y":
                    break
                else:
                    countMiss = 0