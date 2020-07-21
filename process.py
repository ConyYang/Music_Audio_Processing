import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

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

myaudio = Music(path=audio_path, x=x, sr=sr)
myaudio.draw_spectrum()