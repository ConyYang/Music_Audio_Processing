## Python audio signal processing
Achieve music genre classification and feature extraction. 

### Intro
The power spectrum of the music signal is similar to that of the human brain physiological signal, 
which complies with the 1/f signal formula.
 The alpha value of music signal approaches 1 means 
 music is more pleasant.
 A sound is represented in the form of an audio signal having parameters such as frequency, bandwidth, decibel, etc., 
 and a typical audio signal may be represented as a function of amplitude and time. 
 
 
### Install
```shell script
conda install -c conda-forge librosa
# conda env create -f environment.yml
```

### Implementation
In process.py, we do visualization of the audio.
- Waveplot
![waveplot](assets/waveplot.png)
- Spectrum
![Spectrum](assets/spectrum.jpg)
- zero-crossing rate
![zcr](assets/zcrPortion.png)
- Spectral Centroids
![SC](assets/spectralCentroids.png)
- Spectral Roll-Off
![SR](assets/spectralRollOff.png)

In create.py, we create 5 second audio.