import eventAnalysis as anl
import ReceiveData as rec
import serial

# Classification Constants (derived with eventAnalysis.py)
MIN_RMS_FREQ = 1163
MAX_RMS_FREQ = 1915
THRESH = 0.316

if __name__ == "__main__":
    folder_name = "Data/RawEventData/"
    ser = serial.Serial(rec.SERIALPORT3, rec.BAUDRATE, timeout=rec.READ_ATTEMPT_TIMEOUT)

    count_miss = 0
    while True:
        event_data = rec.read_event_data(ser, request_label=False)
        if event_data is not None:
            print(anl.classify_RMS(event_data, MIN_RMS_FREQ, MAX_RMS_FREQ, THRESH))
        else:
            count_miss = count_miss + 1
            if count_miss > 10:
                leave = input("Want to stop? (Y/N)")
                if leave == "Y":
                    break
                else:
                    count_miss = 0