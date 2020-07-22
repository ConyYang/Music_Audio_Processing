import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import sklearn

audio_path = 'beats.wav'
# ipd.Audio(audio_path)
x, sr = librosa.load(audio_path)


class Music(object):
    def __init__(self, path, x, sr):
        self.path = path
        self.x = x
        self.sr = sr

    def print_info(self):
        print(type(self.x), type(self.sr))
        print(self.x.shape, self.sr)

    def draw_waveplot(self):
        plt.figure(figsize=(20, 5))
        librosa.display.waveplot(y=self.x, sr=self.sr)
        plt.savefig('waveplot.png')
        plt.show()

    def draw_spectrum(self):
        X = librosa.stft(y=self.x)
        Xdb= librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(20, 5))
        librosa.display.specshow(data=Xdb, sr=self.sr, x_axis='time', y_axis='log')
        plt.colorbar()
        plt.savefig('spectrum.jpg')
        plt.show()

    def cal_ZCR(self):
        n0 = 39000
        n1 = 39100
        plt.figure(figsize=(20, 5))
        plt.plot(self.x[n0: n1])
        plt.grid()
        plt.savefig('zcrPortion.png')
        plt.show()
        zero_crossings = librosa.zero_crossings(y=self.x[n0:n1],pad=False)
        return sum(zero_crossings)


    def cal_spectral_centroid(self):
        cent = librosa.feature.spectral_centroid(y=self.x, sr=self.sr)
        # print(cent[0].shape)
        # compute time variable for visualization
        frames = range(len(cent[0]))
        t = librosa.frames_to_time(frames)
        normalized_cent = sklearn.preprocessing.minmax_scale(cent[0], axis=0)
        plt.figure(figsize=(20, 5))
        librosa.display.waveplot(y=self.x, sr=self.sr, alpha=0.4)
        plt.plot(t, normalized_cent, color='red')
        # plt.savefig('spectralCentroids.png')
        plt.show()

    def cal_spectral_rolloff(self):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.x+0.01, sr=self.sr)[0]
        print(spectral_rolloff.shape)
        normalized_rolloff = sklearn.preprocessing.minmax_scale(spectral_rolloff, axis=0)
        frames = range(len(spectral_rolloff))
        t = librosa.frames_to_time(frames)
        plt.figure(figsize=(20, 5))
        librosa.display.waveplot(y=self.x, sr=self.sr, alpha=0.4)
        plt.plot(t, normalized_rolloff, color = "red")
        plt.savefig('spectralRollOff.png')
        plt.show()

myaudio = Music(path=audio_path, x=x, sr=sr)
myaudio.cal_spectral_rolloff()
