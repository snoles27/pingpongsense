

## Figuring out inital time for the signal

Whatever we do, we probably need some way to learn the final parameters like we did for classification


#### Ideas
* First time threshold is crossed - Susceptible to noise 
* Short time fourier transform
** Tried in EventAnalysis but didn't really seem to be working. I didn't fully understand what I was doing though so not a great shot
* Define some kind of best fit function (exponential and a sinusoid)
* Do threshold with a check that it keeps growing 
** zero mean the function and then take its absolute value and integrate. Make sure the integral is growing at some rate. 


#### Questions
* Is there anyway to make information from the other signals help? 
