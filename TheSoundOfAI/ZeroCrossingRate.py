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

zrc_debussy = librosa.feature.zero_crossing_rate(debussy, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
zrc_redhot = librosa.feature.zero_crossing_rate(redhot, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
zrc_duke = librosa.feature.zero_crossing_rate(duke, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]


frames = range(0, zrc_debussy.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)


def draw_waveplot(frame_length, actual=True):
    zrc = [zrc_debussy, zrc_redhot, zrc_duke]
    if actual:
        scale_y = 400
    else:
        frame_length = 1
        scale_y = 0.4
    # rms_User = [rms_debussy_user, rms_redhot_user, rms_duke_user]
    color = ["r", "b", "y"]
    plt.figure(figsize=(15, 17))
    for i in range(3):
        plt.plot(t, zrc[i]*frame_length, color=color[i])
    plt.ylim((0, scale_y))
    plt.savefig('result/ZeroCrossingRate.png')
    print("Waveplot draw successfully!")


draw_waveplot(frame_length=FRAME_LENGTH)
