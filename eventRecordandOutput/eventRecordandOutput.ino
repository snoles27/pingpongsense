#include <avdweb_AnalogReadFast.h>
#include <limits.h>
#include <UUID.h>


//CONSTANTS
#define NUMSTORE 900
#define NUMCHANNEL 3
#define NUMTIME 1

#define MILLISPERMICRO 0.001

//Thresholds for recording data
#define RECORDTHRESHHIGH 600
#define RECORDTHRESHLOW 400

#define DATAREVERSEKEEP 30 //number of data points before the event was detected to keep in the returned stuff 
#define TIMESTEP 40 //timestep in micros

#define POSTEVENTPAUSE 100 //millis to pause after sending event data over serial

#define DATAENDSTRING "COMPLETE" 

#define BAUDRATE 57600

//GLOBALS
short dataStore[NUMSTORE];
long timeStore[NUMSTORE];
byte pinStore[NUMSTORE];
short incriment; 
short reverseKeepIndex;
short stopIndex;
short eventStartIndex = 0; //value to indicate where the event crosses the threshold
short activeIndex = 0; 
bool notActive = true; 
long currentTime;
long timeThresh;

const byte adcPin0 = A0;  // Analog input pin that the potentiometer is attached to
const byte adcPin1 = A1; 
const byte adcPin2 = A2; 

const byte adcPin[] = {A0, A1, A2};
short adcPinIndex = 0; 
short adcPinIndexStart = 0; 

UUID uuid;

void setup() {

  reverseKeepIndex = DATAREVERSEKEEP * NUMCHANNEL;
  timeThresh = LONG_MIN;
  delay(200);
  Serial.begin(BAUDRATE);
  analogReadResolution(10); //10 bit analog read resolution
  analogReference(AR_DEFAULT); //set the reference voltage to default (3v3 on the SAMD21)

}

void loop() {

  currentTime = micros();
  if(currentTime > timeThresh){
    timeStore[activeIndex] = currentTime;
    dataStore[activeIndex] = analogReadFast(adcPin[adcPinIndex]);
    //pinStore[activeIndex] = adcPin[adcPinIndex];

    //check if over a threshold, if so start the event save 
    if(notActive && (dataStore[activeIndex] > RECORDTHRESHHIGH || dataStore[activeIndex] < RECORDTHRESHLOW)){
      notActive = false; //are in an active event; 
      eventStartIndex = activeIndex;
      int firstSaveIndex = (eventStartIndex - reverseKeepIndex) - ((eventStartIndex - reverseKeepIndex) % NUMCHANNEL);
      stopIndex = (NUMSTORE + firstSaveIndex - 1) % NUMSTORE;
    }

    if(!notActive && activeIndex == stopIndex){

//    //Print Header Info
      uuid.seed(currentTime);
      uuid.generate();
      Serial.print("EVENT UUID: ");
      Serial.println(uuid);
      //dump the data to serial
      int startPrintIndex; //start printing at the value immediatly after the stop index
      startPrintIndex = stopIndex + 1;
      Serial.print("Start Print Index: ");
      Serial.print(startPrintIndex); //should always be % NUMCHANNEL = 0
      Serial.print(", Event Start Index: ");
      Serial.println(eventStartIndex); 
      for(int i = startPrintIndex; i < NUMSTORE; i = i + NUMCHANNEL){

          printRow(dataStore, timeStore, pinStore, i, timeStore[eventStartIndex]);
      }

      for(int i = 0; i < startPrintIndex; i = i + NUMCHANNEL){

          printRow(dataStore, timeStore, pinStore, i, timeStore[eventStartIndex]);
      }

      Serial.println(DATAENDSTRING);

      //reset everthing 
      activeIndex = 0;
      adcPinIndex = 0;
      notActive = true;
      timeThresh = LONG_MIN;
      delay(POSTEVENTPAUSE);
    }
    else{
       //update indicies and values
       adcPinIndex = (adcPinIndex + 1) % 3;
       activeIndex = (activeIndex + 1) % NUMSTORE;
       timeThresh = currentTime + TIMESTEP;
    }
  }
}

void printRow(short data[NUMSTORE], long times[NUMSTORE], byte pins[NUMSTORE], int i, long baseTime){

          Serial.print(i);
          Serial.print(", ");
          Serial.print((times[i] - baseTime));
          Serial.print(", ");
          Serial.print(data[i]);
          Serial.print(", ");
          Serial.print(i+1);
          Serial.print(", ");
          Serial.print((times[i+1] - baseTime));
          Serial.print(", ");
          Serial.print(data[i+1]);
          Serial.print(", ");
          Serial.print(i+2);
          Serial.print(", ");
          Serial.print((times[i+2] - baseTime));
          Serial.print(", ");
          Serial.println(data[i+2]);
}
