import receiveData as rd
import numpy as np
import newton as newt



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

    indexStart = getIndexFirstGreater(times, MIN_PRERANGE)
    indexEnd = getIndexFirstGreater(times, MAX_PRERANGE)

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

    return newt.newtonSolver(zeroAtEvent, jac_zeroAtEvent, args=args, x0=guess, verbose=True)


def distance(xf:np.ndarray, x0:np.ndarray) -> np.ndarray:
    "Returns distance between two points, xf and x0"

    return np.linalg.norm(xf - x0)

if __name__ == "__main__":


    # sensorLocList = np.array([[SENS0_X, SENS0_Y],
    #                           [SENS1_X, SENS1_Y],
    #                           [SENS2_X, SENS2_Y]])

    sensorLocList = np.array([[0, 0],
                              [3, 0],
                              [-1, 0]])
    
    event = np.array([1,0])
    dt = np.array([1,1])
    c = 1
    print(solveLocation(dt, sensorLocList, c, guess=np.array([0,0])))


    
    

    # threshMult = 8
    # data = rd.eventFileRead("Data/RawEventData/LocatingData/6in_(9.0,6.4)_1.txt", numHeaderLines=3, uuidLine=1, labelLine=2)
    # t1, ut = calculateSigTime(data, 1, triggerMultiple=threshMult)
    # t0, ut = calculateSigTime(data, 0, triggerMultiple=threshMult)

    # print(t1 - t0)

    
