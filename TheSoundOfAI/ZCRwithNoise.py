import librosa
import librosa.display
import matplotlib.pyplot as plt

voice_file = "audio/voice.wav"
noise_file = "audio/noise.wav"

voice, _ = librosa.load(voice_file, duration=15)
noise, _ = librosa.load(noise_file, duration=15)

FRAME_SIZE = 1024
HOP_SIZE = 512

voice_zrc = librosa.feature.zero_crossing_rate(voice, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
noise_zrc = librosa.feature.zero_crossing_rate(noise, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]

frames = range(len(voice_zrc))
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)

plt.figure(figsize=(15, 17))
plt.plot(t, voice_zrc, color="r")
plt.plot(t, noise_zrc, color="y")
plt.ylim((0, 0.6))
plt.savefig('result/ZRC_Noise_Voice.png')