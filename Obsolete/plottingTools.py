import matplotlib.axes
import matplotlib.figure
import receiveData as rd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def plotEvent(eventData:rd.event, ax:matplotlib.axes.Axes) -> None:
    """
    Plot data from an event object 
    """

    a0Times = [point.time for point in eventData.a0Data]
    a1Times = [point.time for point in eventData.a1Data]
    a2Times = [point.time for point in eventData.a2Data]

    a0Values = [point.value for point in eventData.a0Data]
    a1Values = [point.value for point in eventData.a1Data]
    a2Values = [point.value for point in eventData.a2Data]

    ax.plot(a0Times, a0Values, label = "Sensor 0")
    ax.plot(a1Times, a1Values, label = "Sensor 1")
    ax.plot(a2Times, a2Values, label = "Sensor 2")
    ax.set_title(str(eventData))
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Response (a.u.)")
    ax.legend()

def overlaySingleChannelPlot(events:list[rd.event], chanToPlot:int, showPlt:bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
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
        times,vals = event.getChannelData(chanToPlot)
    
        ax.plot(times, vals, label = "(" + str(index) + ") " + event.getShortUUID())

    #clean up plot before showing
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Response (a.u.)")
    ax.set_title(titleStr)
    ax.legend()

    if showPlt: 
        plt.show()
    
    return fig, ax

def plot_signal_and_derivative(event:rd.event, chanToPlot:int, ax:matplotlib.axes.Axes, showPlt:bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot a channel from an event and overlay its derivative.
    
    Args:
        event: event object containing the data
        chanToPlot: index of channel to plot (0, 1, or 2)
        ax: matplotlib axes to plot on
        showPlt: whether to show the plot immediately
    
    Returns:
        tuple of (figure, axes) objects
    """
    # Get the channel data
    times, values = event.getChannelData(chanToPlot)
    
    # Convert to numpy arrays for easier manipulation
    times = np.array(times)
    values = np.array(values)
    
    # Calculate the derivative using finite differences
    # Use central differences for interior points, forward/backward for endpoints
    derivative = np.zeros_like(values, dtype=float)
    derivative = derivative[:-1]
    times_derivative = np.zeros_like(derivative)

    #interio points
    for i in range(0, len(values)-1):
        derivative[i] = (values[i+1] - values[i]) / (times[i+1] - times[i])
        times_derivative[i] = (times[i+1] + times[i]) / 2
       
        # Create twin axes for the derivative
    ax2 = ax.twinx()
    
    # Plot the original signal on the primary y-axis
    line1 = ax.plot(times, values, label=f'Channel {chanToPlot}', linewidth=2, color='blue')
    
    # Plot the derivative on the secondary y-axis
    line2 = ax2.plot(times_derivative, derivative, label=f'Channel {chanToPlot} Derivative', 
                    linestyle='--', linewidth=1.5, color='red')
    
    # Set labels and title
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Amplitude (a.u.)', color='blue')
    ax2.set_ylabel('Derivative (a.u./μs)', color='red')
    ax.set_title(f'Signal and Derivative - {event.getShortUUID()}')
    
    # Set grid on primary axes
    ax.grid(True, alpha=0.3)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Color the y-axis labels to match the data
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    if showPlt:
        plt.show()
    
    return ax.figure, ax

def manyEventSubPlot(events:list[rd.event], channelsToPlot:list[int], showPlot:bool = True, timeRangeMicroSeconds:list[int] = [-3500, 3500]) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

    """
    Veritcally stack plots of multiple event objects 
    """

    numPlots = len(events)

    #creat set of verically stacked sublots 
    fig, axes = plt.subplots(numPlots) 

    #iterate through each axes object 
    for index, ax in enumerate(axes):
        event = events[index] #set active event 
        for chanNum in channelsToPlot:

            #get times and values for chanNum
            if chanNum == 0:
                times = [point.time for point in event.a0Data]
                vals = [point.value for point in event.a0Data]
            elif chanNum == 1: 
                times = [point.time for point in event.a1Data]
                vals = [point.value for point in event.a1Data]
            elif chanNum == 2: 
                times = [point.time for point in event.a2Data]
                vals = [point.value for point in event.a2Data]
            else: 
                raise("Invalid chanNum value. Must be integer 0, 1 or 2")
            
            ax.plot(times, vals, label = "Channel " + str(chanNum))
        
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Response (a.u.)")
        ax.set_xlim(timeRangeMicroSeconds)
        ax.set_title(event.getShortUUID())
        ax.legend(bbox_to_anchor=(1.0, 0.5))

    if showPlot: 
        plt.show()
    
    return fig, axes




if __name__ == "__main__":

    xFullList = ["9.0",
                "12.0",
                "18.0", ]
                # "24.0",
                # "30.0", 
                # "36.0",
                # "42.0", 
                #"48.0"]
    
    indexesToPlot = range(0, len(xFullList))
    # indexesToPlot = [3,4,5]
    xPosList = [xFullList[indx] for indx in indexesToPlot]
    
    fullStrList = [("Data/RawEventData/LocatingData/6in_(" + xpos + ",6.4)_1.txt") for xpos in xPosList]

    eventList = [rd.eventFileRead(path, numHeaderLines=3, uuidLine=1, labelLine=2) for path in fullStrList]
    manyEventSubPlot(eventList, channelsToPlot=[0,1])

    fig, ax = plt.subplots()
    plot_signal_and_derivative(eventList[0], chanToPlot=0, ax=ax, showPlt=True)
    