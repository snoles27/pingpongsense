import receiveData

if __name__ == "__main__":

    filePath = "Data/RawEventData/LocatingData/6in_(9.0,6.4)_0.txt"
    eventData = receiveData.eventFileRead(filePath, numHeaderLines=3)
    receiveData.plotEvent(eventData=eventData)
    