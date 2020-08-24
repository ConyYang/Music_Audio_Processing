import numpy as np
import librosa
T= 5.0
sr = 22050
t = np.linspace(start=0.0, stop=T, num=int(T*sr), endpoint=False)
x = 0.5 * np.sin(2*np.pi*220*t)
librosa.output.write_wav('assets/tone_220.wav', y=x, sr=sr)
