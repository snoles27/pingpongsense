#include <avdweb_AnalogReadFast.h>
#include <limits.h>
#include <UUID.h>


#define NUMSTORE 1500
#define NUMCHANNEL 3
#define NUMTIME 1

#define MILLISPERMICRO 0.001

//Thresholds for recording data
#define RECORDTHRESHHIGH 600
#define RECORDTHRESHLOW 400

#define DATAREVERSEKEEP 15 //number of data points before the event was detected to keep in the returned stuff 
#define timeIncriment 50 //number of microseconds to wait between records

#define POSTEVENTPAUSE 100 //millis to pause after sending event data over serial

#define DATAENDSTRING "COMPLETE" 

//GLOBALS

short dataStore[NUMSTORE];
long timeStore[NUMSTORE];
short incriment; 
short reverseKeepIndex;
short stopIndex;
short eventStartIndex = 0; //value to indicate where the event crosses the threshold
short activeIndex = 0; 

long currentTime = 0;
long timeThresh = 0;
long holdTime = 0;
bool notActive = true; 

const byte adcPin0 = A0;  // Analog input pin that the potentiometer is attached to
const byte adcPin1 = A1; 
const byte adcPin2 = A2; 

const byte adcPin[] = {A0, A1, A2};
short adcPinIndex = 0; 
short adcPinIndexStart = 0; 

UUID uuid;

void setup() {

  reverseKeepIndex = DATAREVERSEKEEP * NUMCHANNEL;
  
  delay(200);
  Serial.begin(9600);
  analogReadResolution(10); //10 bit analog read resolution
  analogReference(AR_DEFAULT); //set the reference voltage to default (3v3 on the SAMD21)

}

void loop() {

  currentTime = micros(); //get current time 
  if(currentTime > timeThresh){
    //record value
    dataStore[activeIndex] = analogReadFast(adcPin[adcPinIndex]);
    timeStore[activeIndex] = currentTime; 

    //check if over a threshold, if so start the event save 
    if(notActive && (dataStore[activeIndex] > RECORDTHRESHHIGH || dataStore[activeIndex] < RECORDTHRESHLOW)){
      notActive = false; //are in an active event; 
      eventStartIndex = activeIndex;
      stopIndex = (NUMSTORE + eventStartIndex - reverseKeepIndex - 1) % NUMSTORE; 
//      Serial.println("In the threshold place");
//      Serial.println(eventStartIndex);
//      Serial.println(stopIndex);
    }

    if(!notActive && activeIndex == stopIndex){

//    //Print Header Info
      uuid.generate();
      Serial.print("EVENT UUID: ");
      Serial.println(uuid);
      //dump the data to serial
      int startPrintIndex; //start printing at the first adc0 index 
      startPrintIndex = eventStartIndex - reverseKeepIndex - ((eventStartIndex - reverseKeepIndex) % NUMCHANNEL);
      for(int i = startPrintIndex; i < NUMSTORE; i = i + NUMCHANNEL){
//        Serial.print(i % 3);
//        Serial.print(", ");
//        Serial.print(timeStore[i]);
//        Serial.print(", ");
//        Serial.println(dataStore[i]);

          Serial.print((timeStore[i] - timeStore[eventStartIndex]));
          Serial.print(", ");
          Serial.print(dataStore[i]);
          Serial.print(", ");
          Serial.print((timeStore[i+1] - timeStore[eventStartIndex]));
          Serial.print(", ");
          Serial.print(dataStore[i+1]);
          Serial.print(", ");
          Serial.print((timeStore[i+2] - timeStore[eventStartIndex]));
          Serial.print(", ");
          Serial.println(dataStore[i+2]);
      }

      for(int i = 0; i < startPrintIndex; i = i + NUMCHANNEL){
//        Serial.print(i % 3);
//        Serial.print(", ");
//        Serial.print(timeStore[i]);
//        Serial.print(", ");
//        Serial.println(dataStore[i]);
//
          Serial.print((timeStore[i] - timeStore[eventStartIndex]));
          Serial.print(", ");
          Serial.print(dataStore[i]);
          Serial.print(", ");
          Serial.print((timeStore[i+1] - timeStore[eventStartIndex]));
          Serial.print(", ");
          Serial.print(dataStore[i+1]);
          Serial.print(", ");
          Serial.print((timeStore[i+2] - timeStore[eventStartIndex]));
          Serial.print(", ");
          Serial.println(dataStore[i+2]);
      }

      Serial.println(DATAENDSTRING);
      
      //print key indexes, debugging
//      Serial.print("eventStartIndex: ");
//      Serial.println(eventStartIndex);

      //rest everthing 
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
       timeThresh = currentTime + timeIncriment; 
    }
   
  }
  

}
