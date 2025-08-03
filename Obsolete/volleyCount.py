import receiveData as rc
import eventAnalysis as anl
import time
import serial

MIN_RMS_FREQ = 1163
MAX_RMS_FREQ = 1915
THRESH = 0.316

READ_TIME = 0.5
TIMEOUT = 30

TIMEOUT_CODE = 4
POINT_TIMEOUT_CODE = 3
DOUBLE_HIT_CODE = 2
ACTIVE_VOLLEY_CODE = 1
NO_VOLLEY_CODE = 0 

DOUBLE_HIT_TIME = 0.5 #seconds for if two hits happens within 
MAX_BACK_AND_FORTH_TIME = 4.0

def watchPingPongEvent(openPort) -> bool: 
    
    event = rc.readEventData(openPort, requestLabel=False)
    if event is not None: 
        isEvent = anl.classify_RMS(event, MIN_RMS_FREQ, MAX_RMS_FREQ, THRESH)
    else:
        isEvent = False
    
    return isEvent

def activeVolley(openPort) -> (int, int):

    lastStrike = time.time()
    count = 1
    exitCode = ACTIVE_VOLLEY_CODE
    active = True
    while active:
        isEvent = watchPingPongEvent(openPort)
        currentTime = time.time()
        if isEvent:
            if currentTime < lastStrike + DOUBLE_HIT_TIME:
                active = False
                exitCode = DOUBLE_HIT_CODE
            else:
                count = count + 1
                print(count)
                lastStrike = currentTime
        else: 
            if currentTime > lastStrike + MAX_BACK_AND_FORTH_TIME:
                active = False
                exitCode = POINT_TIMEOUT_CODE
            
    return count, exitCode


def waitingForVolley(openPort) -> int: 
    exitCode = NO_VOLLEY_CODE
    enterTime = time.time()
    while exitCode == NO_VOLLEY_CODE: 
        isEvent = watchPingPongEvent(openPort)
        if isEvent:
            exitCode = ACTIVE_VOLLEY_CODE
        else:
            currentTime = time.time()
            if currentTime > enterTime + TIMEOUT:
                exitCode = TIMEOUT_CODE

    return exitCode
        


if __name__ == "__main__":

    ser = serial.Serial(rc.SERIALPORT3, rc.BAUDRATE, timeout = READ_TIME)
   
    exitCode = waitingForVolley(ser)

    while exitCode != TIMEOUT_CODE:
        if exitCode == ACTIVE_VOLLEY_CODE:
            print("Begin Volley")
            count, exitCode = activeVolley(ser)
            print("Final Volley Count: " + str(count))
            print("Exit Code: " + str(exitCode))
        else: 
            exitCode = waitingForVolley(ser)
            print("Exit Code: " + str(exitCode))
