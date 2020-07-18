# Bounded Kalman filter method for motion-robust, non-contact heart rate estimation

## Introduction
Rhythmic pulsating action of the heart causes blood volume changes all over the body. This pulsating action results in the generation of cardiac pulse, which can be tracked/observed in the skin, wrist, and fingertips. Photo- plethysmography (PPG) is an optic based plethysmography method, based on the principle that blood absorbs more light than surrounding tissue and hence, variations in blood volume affect transmission or reflectance correspondingly. Prior rPPG methods of pulse-rate measurement from face videos attain high accuracies under well controlled uniformly illuminated and motion-free situations, however, their performance degrades when illumination variations and subjects’ motions are involved. 

## Contribution
In this paper (A Bounded Kalman Filter Method for Motion-Robust, Non-Contact Heart Rate Estimation), a HR measurement method is presented that utilizes facial key-point data to overcome the challenges presented in real world settings as described earlier. In summary, our contributions are:
1.	The ability to identify motion blur and to dynamically (algorithmically) denoise blurred frames to enable frame to frame face capture
2.	The ability to enable motion estimation of feature points with higher accuracy in terms of range and speed
3.	The ability to accurately capture heart rate at distances up to 4ft

## Compiling the script and Understanding the parameters
To run the heart rate estimator, please make sure that you are using only python 2.7 and not python 3 as the estimator relies on an open-source package called geompreds which is not available in python 3 and above. Make sure you have installed all required packages. Then using your terminal run "python Proposed Algorithm_Main Program.py -v (input video)", you can omit the video argument if using the camera.

If you plan on estimating heart rate for a video, please make sure to tweak the following paramaters to ensure that you are getting an accurate estimation.
1. Frames per second (F.P.S) of the video, which can be tweaked in line 578 of the "Proposed Algorithm_Main Program.py" script. In case you are using your camera to estimate pulse rate, please set F.P.S. according to how many frames the program is able to sample in a second as opposed to using the camera's frame sampling rate.
2. Moving window: In the paper, we set window size (line 579 in "Proposed Algorithm_Main Program.py") as 30 seconds for an accurate estimation with the window stride being 1 second (line 605 in "Proposed Algorithm_Main Program.py").
3. Butter bandpass filter paramaters: Order, low-cut frequency and high-cut frequency (line 589-592 in "Proposed Algorithm_Main Program.py"). We find that the order parameter affects estimation accuracy and hence should be tuned according to the dataset.
4. Real-time graph (optional): Please refer to the comments on the script "Proposed Algorithm_Main Program.py" for more details on which lines to uncomment in order to enable the real-time graph.
5. Output pulse rate: By default, the output pulse rates are written to a text file titled "test.txt".

## Citation
If you use any of the resources provided on this page in any of your publications we ask you to cite the following work.

Prakash, S. K. A., & Tucker, C. S. (2018). Bounded Kalman filter method for motion-robust, non-contact heart rate estimation. Biomedical Optics Express, 9(2), 873-897. DOI: 10.1364/BOE.9.000873.

