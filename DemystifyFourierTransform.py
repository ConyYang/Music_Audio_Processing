import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# load audio file
audio_path = "audio/piano_c.wav"
signal, sr = librosa.load(audio_path)

# Derive spectrum using FT
ft = sp.fft.fft(signal)
magnitude = np.absolute(ft)
frequency = np.linspace(0, sr, len(magnitude))

# plot the waveform and spectrum
def plot_time_frequency_domain():
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(signal,sr,alpha=0.5)
    plt.title("Time Domain")
    plt.subplot(2, 1, 2)
    plt.plot(frequency[:5000], magnitude[:5000])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain")
    plt.savefig('result/time_frequency_domain.png')


def plot_similarity(signal):
    f_fundamental = 523
    phase = 0.57
    t = librosa.samples_to_time(samples=range(len(signal)), sr=sr)
    sin_wave = 0.1 * np.sin(2 * np.pi * (f_fundamental * t - phase))
    plt.figure(figsize=(18, 8))
    plt.plot(t[10000:10400], sin_wave[10000:10400], color="r")
    plt.plot(t[10000:10400], signal[10000:10400], color="y")
    plt.fill_between(t[10000:10400], sin_wave[10000:10400] * signal[10000:10400], color="b")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig('result/Similarity.png')

plot_similarity(signal)
