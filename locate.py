import receiveData as rd
import numpy as np
import plottingTools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from scipy.optimize import root

### SENSOR POSITIONS ### 
SENS0_X = 51.5
SENS0_Y = 8.5

SENS1_X = 5.75
SENS1_Y = 6.38

SENS2_X = 25.25
SENS2_Y = 45.75

#wave speed through table in inches/mircosecond
WAVE_SPEED = 0.019 #rough calc based on minimal data 
SPEED_UNCERT = .02 #estimate of speed uncertainty 

MIN_PRERANGE = -5000
MAX_PRERANGE = -2000

def calculateSigTime(event:rd.event, channel:int, baseRange:list[int] = [MIN_PRERANGE, MAX_PRERANGE], triggerMultiple:int = 5) -> tuple[int, int]:
    
    """
    Function to determine time when signal reaches the sensor
    returns: tuple(time solution, uncertainty in time)
    Uses set time range to determine base signal value and noise level.
    """

    default_uncertainty = 125 #setting the default uncertainty at the sample rate. Might be possible to get below this if you are smart

    # get raw time and value data 
    events = event.getChannelEvents(channel)
    times = [event.time for event in events]
    vals = [event.value for event in events]

    indexStart = getIndexFirstGreater(times, baseRange[0])
    indexEnd = getIndexFirstGreater(times, baseRange[1])

    #get average and standard deviation of base value
    baseVal = np.average(vals[indexStart:indexEnd])
    noiseLevel = np.std(vals[indexStart:indexEnd])

    #get a set of values that is centered around 0
    zeroCenteredVals = vals - baseVal

    #find first instance of zeroCenteredVals that is greater than some multiple of the noise
    indexEvent = getIndexFirstGreater(np.abs(zeroCenteredVals), noiseLevel * triggerMultiple)
    
    return (events[indexEvent].time, default_uncertainty)

def getIndexFirstGreater(numbers:list[int], thresh:int) -> int: 
    """
    returns the index of the first instance in numbers to be greater than thresh
    """

    try: 
        index = next(x for x,val in enumerate(numbers) if val >= thresh)
    except StopIteration:
        return -1 

    return index

def zeroAtEvent(x:np.ndarray, sensorLoc:np.ndarray, dt:np.ndarray, speed:float) -> np.ndarray:
    """
    Multivarite function that should equal zero when the equation is solved
    x: [x_e, y_e]. 2x1 np array encoding event position
    sensorLoc: 3x2 np array with each sensor locaiton 
    dt: 2x1 np array [dt10, dt20]
    speed: speed of sound in table with units matching x, sensorLoc and dt
    """

    rho_0 = distance(x, sensorLoc[0, :])
    rho_1 = distance(x, sensorLoc[1, :])
    rho_2 = distance(x, sensorLoc[2, :])

    return np.array([
        rho_1 - rho_0 - speed * dt[0],
        rho_2 - rho_0 - speed * dt[1]
    ])

def jac_zeroAtEvent(x:np.ndarray, sensorLoc:np.ndarray, dt:np.ndarray, speed:float) -> np.ndarray:
    """
    returns jacobian of zeroAtEvent evaluated at x
    """
    rho_0 = distance(x, sensorLoc[0, :])
    rho_1 = distance(x, sensorLoc[1, :])
    rho_2 = distance(x, sensorLoc[2, :])

    return np.array([
        [(x[0] - sensorLoc[1,0])/rho_1 - (x[0] - sensorLoc[0,0])/rho_0, (x[1] - sensorLoc[1,1])/rho_1 - (x[1] - sensorLoc[0,1])/rho_0],
        [(x[0] - sensorLoc[2,0])/rho_2 - (x[0] - sensorLoc[0,0])/rho_0, (x[1] - sensorLoc[2,1])/rho_2 - (x[1] - sensorLoc[0,1])/rho_0]
    ])

def solveLocation(dt:np.ndarray, sensorLoc:np.ndarray, speed:float, guess = np.array([10,10])) -> np.ndarray:

    """
    envoke newtons method to solve for event location. See above functions for what the arguments mean <fill later>
    """

    args = (sensorLoc, dt, speed)

    return root(
        fun = zeroAtEvent,
        jac = jac_zeroAtEvent,
        args = args,
        x0 = guess,
        method = 'lm'
    )

def distance(xf:np.ndarray, x0:np.ndarray) -> np.ndarray:
    "Returns distance between two points, xf and x0"

    return np.linalg.norm(xf - x0)

def plotEventWithSignalTimes(eventData:rd.event, t0:float, ut0:float, t1:float, ut1:float, t2:float, ut2:float) -> None:
    """
    Plot event data with calculated signal times overlaid as semi-transparent rectangles.
    The rectangle colors match the default matplotlib color cycle used in plottingTools.plotEvent.
    """
    # Default matplotlib color cycle (first 3 colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    fig, ax = plt.subplots()
    plottingTools.plotEvent(eventData, ax)
    
    # Get the y-axis limits to determine rectangle height
    y_min, y_max = ax.get_ylim()
    rect_height = y_max - y_min
    
    # Create semi-transparent rectangles centered on tx with width 2 * utx
    # Use the same colors as the plot lines
    rect0 = patches.Rectangle((t0 - ut0, y_min), 2 * ut0, rect_height, 
                             facecolor=colors[0], alpha=0.3, label="t0 ± ut0")
    rect1 = patches.Rectangle((t1 - ut1, y_min), 2 * ut1, rect_height, 
                             facecolor=colors[1], alpha=0.3, label="t1 ± ut1")
    rect2 = patches.Rectangle((t2 - ut2, y_min), 2 * ut2, rect_height, 
                             facecolor=colors[2], alpha=0.3, label="t2 ± ut2")
    
    # Add rectangles to the plot
    ax.add_patch(rect0)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # Focus the plot view on the beginning of the event where the rectangles are
    # Find the earliest and latest times among the signal times and their uncertainties
    earliest_time = min(t0 - ut0, t1 - ut1, t2 - ut2)
    latest_time = max(t0 + ut0, t1 + ut1, t2 + ut2)
    
    # Add some padding around the rectangles for better visibility
    time_padding = (latest_time - earliest_time) * 0.2  # 20% padding
    x_min = earliest_time - time_padding
    x_max = latest_time + time_padding
    
    # Set the x-axis limits to focus on the signal detection region
    ax.set_xlim(x_min, x_max)
    
    ax.legend()
    plt.show()

def plotLocationSolution(sensorLoc:np.ndarray, x_solution:np.ndarray, x_cov:np.ndarray, 
                        ax:matplotlib.axes.Axes = None, data_label:str = None, 
                        plot_sensors:bool = True, color:str = None) -> matplotlib.axes.Axes:
    """
    Plot sensor positions, solution location, and covariance ellipse.
    
    Args:
        sensorLoc: 3x2 array with sensor positions [x, y]
        x_solution: 2x1 array with solution position [x, y]
        x_cov: 2x2 covariance matrix for the solution
        ax: matplotlib axes to plot on (if None, creates new figure)
        data_label: label for this solution in the legend
        plot_sensors: whether to plot sensor positions (only plot once)
        color: color for this solution (if None, uses default cycle)
    
    Returns:
        matplotlib axes object
    """
    # Create new figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_sensors = True  # Always plot sensors for new figure
    
    # Plot sensor positions only once
    if plot_sensors:
        ax.scatter(sensorLoc[:, 0], sensorLoc[:, 1], c=['red', 'green', 'blue'], 
                   s=100, marker='s', label='Sensors', edgecolors='black', linewidth=2)
        
        # Add sensor labels
        for i, (x, y) in enumerate(sensorLoc):
            ax.annotate(f'S{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=12, fontweight='bold')
    
    # Use provided color or get from color cycle
    if color is None:
        color = plt.cm.tab10(len(ax.get_legend_handles_labels()[0]) % 10)
    
    # Plot solution location
    solution_label = f'Solution {data_label}' if data_label else 'Solution'
    ax.scatter(x_solution[0], x_solution[1], c=color, s=150, marker='*', 
               label=solution_label, edgecolors='black', linewidth=2)
    
    # Plot covariance ellipse
    # Calculate eigenvalues and eigenvectors of covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(x_cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Calculate ellipse parameters
    # Use 95% confidence interval (2-sigma)
    confidence_level = 2.0
    major_axis = confidence_level * np.sqrt(eigenvals[0])
    minor_axis = confidence_level * np.sqrt(eigenvals[1])
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    
    # Create ellipse
    from matplotlib.patches import Ellipse
    ellipse_label = f'95% Confidence {data_label}' if data_label else '95% Confidence'
    ellipse = Ellipse(xy=x_solution, width=2*major_axis, height=2*minor_axis, 
                      angle=np.degrees(angle), facecolor=color, alpha=0.3, 
                      edgecolor=color, linewidth=2, label=ellipse_label)
    
    ax.add_patch(ellipse)
    
    # Set equal aspect ratio and labels (only for new figures)
    if plot_sensors:
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (inches)')
        ax.set_ylabel('Y Position (inches)')
        ax.set_title('Ball Impact Location with Uncertainty')
        ax.grid(True, alpha=0.3)
    
    ax.legend()
    
    # Update axis limits to include new data
    all_x = np.concatenate([sensorLoc[:, 0], [x_solution[0]]])
    all_y = np.concatenate([sensorLoc[:, 1], [x_solution[1]]])
    
    # Get current limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Update limits to include new data
    new_x_min = min(x_min, np.min(all_x))
    new_x_max = max(x_max, np.max(all_x))
    new_y_min = min(y_min, np.min(all_y))
    new_y_max = max(y_max, np.max(all_y))
    
    # Add padding
    x_padding = (new_x_max - new_x_min) * 0.1
    y_padding = (new_y_max - new_y_min) * 0.1
    
    ax.set_xlim(new_x_min - x_padding, new_x_max + x_padding)
    ax.set_ylim(new_y_min - y_padding, new_y_max + y_padding)
    
    return ax

if __name__ == "__main__":

    triggerMultiple = 7
    sensorLocList = np.array([[SENS0_X, SENS0_Y],
                            [SENS1_X, SENS1_Y],
                            [SENS2_X, SENS2_Y]])


    testFileName = "Data/RawEventData/LocatingData/6in_(36.0,6.4)_0.txt"

    testEventData = rd.eventFileRead(fullPath=testFileName, numHeaderLines=3, uuidLine=1, labelLine=2)
    
    t0, ut0 = calculateSigTime(testEventData, channel=0, triggerMultiple=triggerMultiple)
    t1, ut1 = calculateSigTime(testEventData, channel=1, triggerMultiple=triggerMultiple)
    t2, ut2 = calculateSigTime(testEventData, channel=2, triggerMultiple=triggerMultiple)

    t10 = t1-t0 
    t20 = t2-t0 

    # Plot the event data with calculated signal times overlaid
    plotEventWithSignalTimes(testEventData, t0, ut0, t1, ut1, t2, ut2)

    soln = solveLocation(np.array([t10,t20]), sensorLoc=sensorLocList, speed=WAVE_SPEED)
    print(soln.x)
    print(soln.cov_x)
    
    # Plot the location solution
    ax = plotLocationSolution(sensorLocList, soln.x, soln.cov_x, data_label="test1")
    
    # Example of overlaying multiple solutions
    # You can call the function multiple times with the same axes
    # ax = plotLocationSolution(sensorLocList, another_solution, another_cov, 
    #                          ax=ax, data_label="Test 2", plot_sensors=False)
    
    plt.tight_layout()
    plt.show()


    # # sensorLocList = np.array([[SENS0_X, SENS0_Y],
    # #                           [SENS1_X, SENS1_Y],
    # #                           [SENS2_X, SENS2_Y]])

    # sensorLocList = np.array([[0, 0],
    #                           [3, 0],
    #                           [-1, 0]])
    
    # event = np.array([1,0])
    # dt = np.array([1,1])
    # c = 1
    # print(solveLocation(dt, sensorLocList, c, guess=np.array([0,0])))


    
    

    # threshMult = 8
    # data = rd.eventFileRead("Data/RawEventData/LocatingData/6in_(9.0,6.4)_1.txt", numHeaderLines=3, uuidLine=1, labelLine=2)
    # t1, ut = calculateSigTime(data, 1, triggerMultiple=threshMult)
    # t0, ut = calculateSigTime(data, 0, triggerMultiple=threshMult)

    # print(t1 - t0)

    