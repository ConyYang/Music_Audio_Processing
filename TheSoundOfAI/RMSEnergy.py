import librosa
import librosa.display
# import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

# load audio files
debussy_file = "audio/debussy.wav"
redhot_file = "audio/redhot.wav"
duke_file = "audio/duke.wav"

debussy, _ = librosa.load(debussy_file)
redhot, _ = librosa.load(redhot_file)
duke, _ = librosa.load(duke_file)

# Extract RMSE with librosa
FRAME_LENGTH = 1024
HOP_LENGTH = 512

rms_debussy = librosa.feature.rms(debussy, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
rms_redhot = librosa.feature.rms(redhot, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
rms_duke = librosa.feature.rms(duke, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

# How to calculate rms manually
def rms_calculate(signal, frame_length, hop_length):
    rms_list = []
    for i in range(0, len(signal), hop_length):
        rms_current_frame = np.sqrt(np.sum(signal[i:i+frame_length]**2)/frame_length)
        rms_list.append(rms_current_frame)
    return np.array(rms_list)


rms_debussy_user = rms_calculate(debussy, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
rms_redhot_user = rms_calculate(redhot, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
rms_duke_user = rms_calculate(duke, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)


frames = range(0, rms_debussy.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)


def draw_waveplot():
    waveplots = [debussy, redhot, duke]
    rms = [rms_debussy, rms_redhot, rms_duke]
    rms_User = [rms_debussy_user, rms_redhot_user, rms_duke_user]
    waveplots_name = ["debussy", "redhot", "duke"]
    plt.figure(figsize=(15, 17))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        librosa.display.waveplot(waveplots[i], alpha=.5)
        plt.plot(t, rms[i], color="r")
        plt.plot(t, rms_User[i], color="y")
        plt.title(waveplots_name[i])
    plt.ylim((-1, 1))
    plt.savefig('result/rmsEnergyCompare.png')
    print("Waveplot draw successfully!")


draw_waveplot()