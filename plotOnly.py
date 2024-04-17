import receiveData

if __name__ == "__main__":

    filePath = "Data/RawEventData/UUID-c28adf60___Label-p.txt"
    eventData = receiveData.eventFileRead(filePath)
    receiveData.plotEvent(eventData=eventData)
    