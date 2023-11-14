#include <avdweb_AnalogReadFast.h>

/*
  Analog input, analog output, serial output

  Reads an analog input pin, maps the result to a range from 0 to 255 and uses
  the result to set the pulse width modulation (PWM) of an output pin.
  Also prints the results to the Serial Monitor.

  The circuit:
  - potentiometer connected to analog pin 0.
    Center pin of the potentiometer goes to the analog pin.
    side pins of the potentiometer go to +5V and ground
  - LED connected from digital pin 9 to ground through 220 ohm resistor

  created 29 Dec. 2008
  modified 9 Apr 2012
  by Tom Igoe

  This example code is in the public domain.

  https://www.arduino.cc/en/Tutorial/BuiltInExamples/AnalogInOutSerial
*/


#define NUMSTORE 1500
#define NUMCHANNEL 3
#define RECORDTHRESHHIGH 600
#define RECORDTHRESHLOW 400
#define DAMPINGPAUSEMS 100

// These constants won't change. They're used to give names to the pins used:
const byte adcPin0 = A0;  // Analog input pin that the potentiometer is attached to
const byte adcPin1 = A1; 
const byte adcPin2 = A2; 
const int analogOutPin = 9; // Analog output pin that the LED is attached to


int sensorValue0 = 0; // value read from the pot
int sensorValue1 = 0;
int sensorValue2 = 0;
int outputValue = 0;        // value output to the PWM (analog out)
short dataStore[NUMSTORE];
int trash;

void setup() {
  // initialize serial communications at 9600 bps:
  Serial.begin(9600);
  analogReadResolution(10);
  analogReference(AR_DEFAULT);
  
}

void loop() {
  // read the analog in value:
  sensorValue0 = analogReadFast(adcPin0);
  sensorValue1 = analogReadFast(adcPin1);
  sensorValue2 = analogReadFast(adcPin2);
  if(sensorValue0 > RECORDTHRESHHIGH || sensorValue0 < RECORDTHRESHLOW || sensorValue1 > RECORDTHRESHHIGH || sensorValue1 < RECORDTHRESHLOW || sensorValue2 > RECORDTHRESHHIGH || sensorValue2 < RECORDTHRESHLOW){
    for(int i = 3; i < NUMSTORE; i = i + NUMCHANNEL){
      dataStore[i] = analogReadFast(adcPin0);
      dataStore[i+1] = analogReadFast(adcPin1);
      dataStore[i+2] = analogReadFast(adcPin2);
      //delayMicroseconds(50);
    }
    dataStore[0] = sensorValue0;
    dataStore[1] = sensorValue1;
    dataStore[2] = sensorValue2; 
    for(int i = 0; i < NUMSTORE; i = i + NUMCHANNEL){
      Serial.print(dataStore[i]);
      Serial.print(", ");
      Serial.print(dataStore[i+1]);
      Serial.print(", ");
      Serial.print(dataStore[i+2]);
      Serial.print("\n");
    }
    Serial.print(0);
    delay(DAMPINGPAUSEMS);
  } 
  
 
}
