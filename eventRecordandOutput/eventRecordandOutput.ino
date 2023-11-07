#include <avdweb_AnalogReadFast.h>


#define NUMSTORE 2000
#define NUMCHANNEL 3
#define NUMTIME 1


//Thresholds for recording data
#define RECORDTHRESHHIGH 600
#define RECORDTHRESHLOW 400

#define DATAREVERSEKEEP 20 //number of data points before the event was detected to keep in the returned stuff 

#define POSTEVENTPAUSE 100 //millis to pause after sending event data over serial


//GLOBALS

short dataStore[NUMSTORE];
short incriment; 
short reverseKeepIndex;
short eventStartIndex = 0; //value to indicate where the event crosses the threshold


const byte adcPin0 = A0;  // Analog input pin that the potentiometer is attached to
const byte adcPin1 = A1; 
const byte adcPin2 = A2; 

void setup() {
  incriment = NUMCHANNEL * NUMSTORE; //number of elements in the array to skip over
  reverseKeepIndex = incriment * DATAREVERSEKEEP;
  delay(1000);
  Serial.begin(9600);
  analogReadResolution(10); //10 bit analog read resolution
  analogReference(AR_DEFAULT); //set the reference voltage to default (3v3 on the SAMD21)

}

void loop() {

  //Record time and analog reads 
  dataStore[eventStartIndex] = (short) millis(); //record time of event recording
  dataStore[eventStartIndex + 1] = analogReadFast(adcPin0);
  dataStore[eventStartIndex + 2] = analogReadFast(adcPin1);
  dataStore[eventStartIndex + 3] = analogReadFast(adcPin2);

  Serial.print(dataStore[i] - dataStore[eventStartIndex]); //t = 0 set at the time when the event crosses the threshold
      Serial.print(", ");
      Serial.print(dataStore[i+1]);
      Serial.print(", ");
      Serial.print(dataStore[i+2]);
      Serial.print(", ");
      Serial.print(dataStore[i+3]);
      Serial.print("\n");
  

//  //check if any of the values cross a threshold
//  if(dataStore[eventStartIndex + 1] > RECORDTHRESHHIGH || dataStore[eventStartIndex + 1] < RECORDTHRESHLOW || dataStore[eventStartIndex + 2] > RECORDTHRESHHIGH || dataStore[eventStartIndex + 2] < RECORDTHRESHLOW || dataStore[eventStartIndex + 3] > RECORDTHRESHHIGH || dataStore[eventStartIndex + 3] < RECORDTHRESHLOW){
//
//    //fast record till end of array
//    for(short i = eventStartIndex + incriment; i < NUMSTORE; i = i + incriment){
//      dataStore[i] = (short) millis();
//      dataStore[i+1] = analogReadFast(adcPin0);
//      dataStore[i+2] = analogReadFast(adcPin1);
//      dataStore[i+3] = analogReadFast(adcPin2);
//    }
//
//    //fast record until we reach the data we want to keep 
//    for(short i = 0; i < eventStartIndex - reverseKeepIndex; i = i + incriment){
//      dataStore[i] = (short) millis();
//      dataStore[i+1] = analogReadFast(adcPin0);
//      dataStore[i+2] = analogReadFast(adcPin1);
//      dataStore[i+3] = analogReadFast(adcPin2);
//    }
//
//    //write the data to serial port 
//    for(short i = eventStartIndex - reverseKeepIndex; i < NUMSTORE; i = i + incriment){
//      Serial.print(dataStore[i] - dataStore[eventStartIndex]); //t = 0 set at the time when the event crosses the threshold
//      Serial.print(", ");
//      Serial.print(dataStore[i+1]);
//      Serial.print(", ");
//      Serial.print(dataStore[i+2]);
//      Serial.print(", ");
//      Serial.print(dataStore[i+3]);
//      Serial.print("\n");
//    }
//    for(short i = 0; i < eventStartIndex - reverseKeepIndex; i = i + incriment){
//      Serial.print(dataStore[i] - dataStore[eventStartIndex]); //t = 0 set at the time when the event crosses the threshold
//      Serial.print(", ");
//      Serial.print(dataStore[i+1]);
//      Serial.print(", ");
//      Serial.print(dataStore[i+2]);
//      Serial.print(", ");
//      Serial.print(dataStore[i+3]);
//      Serial.print("\n");
//    }
//    eventStartIndex = 0; //reset the event index
//    delay(50);
//  }
//  else{
//    eventStartIndex = eventStartIndex + incriment;
//  }

}
