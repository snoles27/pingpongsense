import receiveData as rd
import numpy as np



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

if __name__ == "__main__":

    threshMult = 8
    data = rd.eventFileRead("Data/RawEventData/LocatingData/6in_(9.0,6.4)_1.txt", numHeaderLines=3, uuidLine=1, labelLine=2)
    t1, ut = calculateSigTime(data, 1, triggerMultiple=threshMult)
    t0, ut = calculateSigTime(data, 0, triggerMultiple=threshMult)

    print(t1 - t0)

    
