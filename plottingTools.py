import matplotlib.axes
import matplotlib.figure
import receiveData as rd
import matplotlib
import matplotlib.pyplot as plt

def plotEvent(eventData:rd.event) -> None:
    """
    Plot data from an event object 
    """

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

def multEventPlotSingleChannel(events:list[rd.event], chanToPlot:int, showPlt:bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Overlay data from multiple event objects
    events: list of event objects to pull data from
    chanToPlot: Index of channel to plot on all of the events in events
    """
    
    #set up plots
    titleStr = "Multiplot Channel " + str(chanToPlot)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(titleStr)

    #iterate through and plot all events in events
    for index, event in enumerate(events): 
        if chanToPlot == 0:
            times = [point.time for point in event.a0Data]
            vals = [point.value for point in event.a0Data]
        elif chanToPlot == 1: 
            times = [point.time for point in event.a1Data]
            vals = [point.value for point in event.a1Data]
        elif chanToPlot == 2: 
            times = [point.time for point in event.a2Data]
            vals = [point.value for point in event.a2Data]
        else: 
            raise("Invalid chanToPlot value. Must be integer 0, 1 or 2")
    
        ax.plot(times, vals, label = "(" + str(index) + ") " + event.getShortUUID())

    #clean up plot before showing
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Response (a.u.)")
    ax.set_title(titleStr)
    ax.legend()

    if showPlt: 
        plt.show()
    
    return fig, ax

if __name__ == "__main__":

    xFullList = ["9.0",
                "12.0",
                "18.0", 
                "24.0",
                "30.0", 
                "36.0",
                "42.0", 
                "48.0"]
    
    indexesToPlot = [0, 2, 4, 6]
    xPosList = [xFullList[indx] for indx in indexesToPlot]
    
    fullStrList = [("Data/RawEventData/LocatingData/6in_(" + xpos + ",6.4)_0.txt") for xpos in xPosList]

    eventList = [rd.eventFileRead(path, numHeaderLines=3, uuidLine=1, labelLine=2) for path in fullStrList]
    multEventPlotSingleChannel(eventList, chanToPlot=1)
    