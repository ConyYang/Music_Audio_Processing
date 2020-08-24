import librosa
import librosa.display
# import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np


# load audio files
debussy_file = "audio/debussy.wav"
redhot_file = "audio/redhot.wav"
duke_file = "audio/duke.wav"

debussy, sr = librosa.load(debussy_file)
redhot, _ = librosa.load(redhot_file)
duke, _ = librosa.load(duke_file)

print(debussy.size)

# duration of 1 sample
sample_duration = 1/sr
print("Duration of 1 sample is {:.6f} seconds".format(sample_duration))

# Duration of the audio signal in seconds
duration = sample_duration * len(debussy)
print("Duration of signal is {:.6f} seconds".format(duration))


# Calculate the amplitude envelope
def amplitude_envelope(signal, frame_size, hop_length):
    amp_envelope = []

    # Calculate AE for each frame
    for i in range(0, len(signal), hop_length):
        cur_frame_amp_envelope = max(signal[i:i+frame_size])
        amp_envelope.append(cur_frame_amp_envelope)
    return np.array(amp_envelope)


FRAME_SIZE = 1024
HOP_LENGTH = 512

AE_debussy = amplitude_envelope(debussy, FRAME_SIZE, HOP_LENGTH)
AE_redhot = amplitude_envelope(redhot, FRAME_SIZE, HOP_LENGTH)
AE_duke = amplitude_envelope(duke, FRAME_SIZE, HOP_LENGTH)
print(len(AE_debussy))


# visualize the waveforms
frames = range(0, AE_debussy.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)


def draw_waveplot():
    waveplots = [debussy, redhot, duke]
    AE = [AE_debussy, AE_redhot, AE_duke]
    waveplots_name = ["debussy", "redhot", "duke"]
    plt.figure(figsize=(15, 17))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        librosa.display.waveplot(waveplots[i], alpha=.5)
        plt.plot(t, AE[i], color="r")
        plt.title(waveplots_name[i])
    plt.ylim((-1, 1))
    plt.savefig('result/waveplot.png')
    print("Waveplot draw successfully!")


draw_waveplot()


# visualize amplitude envelope for all the audio files
