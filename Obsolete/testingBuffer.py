import serial 


ser = serial.Serial("/dev/tty.usbmodem14201", 115200)

char = "k"
while char != "q":
    char = input("What to do:")
    if char == "r":
        while ser.in_waiting !=0:
            val = str(ser.readline())
            print(val)
            # val = val.replace("b'", "")
            # val = val.replace("\\r\\n'", "")
            # numVal = int(val)
            # if numVal % 100 == 0:
            #     print(str(ser.in_waiting) + " bytes in buffer")
            #     print(numVal)
    else:
        print(str(ser.in_waiting) + " bytes are waiting in the buffer")
ser.close()