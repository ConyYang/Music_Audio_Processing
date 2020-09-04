import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "Audio/blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050)  # sr * T -> 22050 * 30

# fft -> spectrum
fft = np.fft.fft(signal)
print(fft.shape)
magnitude = np.abs(fft)
print(magnitude.shape)
frequency = np.linspace(0, sr, len(magnitude))
print(frequency.shape)

left_frequency = frequency[: int(len(frequency) / 2)]
left_magnitude = magnitude[: int(len(magnitude) / 2)]

# stft -> spectrogram
NO_SAMPLE_FFT = 2048  # the window (number of samples) performing a fft
HOP_LENGTH = 512  # how much we are shifting

stft = librosa.core.stft(signal, hop_length=HOP_LENGTH, n_fft=NO_SAMPLE_FFT)
spectrogram = np.abs(stft)
# convert to decibel
log_spectrogram = librosa.amplitude_to_db(spectrogram)


# MFCCs
MFCCs = librosa.feature.mfcc(signal, n_fft=NO_SAMPLE_FFT, hop_length=HOP_LENGTH, n_mfcc=13)


def plot_waveplot():
    librosa.display.waveplot(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig("blueAmplitude.png")


def plot_frequency_magnitude(f, m, filename):
    plt.plot(f, m)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.savefig(filename)


def plot_spectrogram(gram, filename, ylabel="Frequency"):
    librosa.display.specshow(gram, sr=sr, hop_length=HOP_LENGTH)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.savefig(filename)


plot_spectrogram(MFCCs, "MFCC")





