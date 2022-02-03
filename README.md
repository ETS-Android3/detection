This repository contains the android application, which is able to detect poses using the MLkit pose detection API, and classify them using a keras sequential model converted to tensorflow lite.
The code for the keras/tensorflow lite model can be found in Classifier.py.
To run the application, do:
1. clone repository
2. open app folder using Android Studio
3. run application on connected device, or on an AVD (Android virtual device)

Note: to keep the cpu usage as small as possible, no graphic overlay is used. The classification of the pose can be found in the console. 
