import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn

class MFCC(object):
    def __init__(self, path):
        self.x, self.fs = librosa.load(path)

    def waveplot(self):
        plt.figure(figsize=(20, 5))
        librosa.display.waveplot(y=self.x, sr=self.fs)
        plt.savefig('mfcc_waveplot.png')
        plt.show()

    def cal_mfcc(self):
        mfccs = librosa.feature.mfcc(y=self.x, sr=self.fs)
        print(mfccs.shape)
        plt.figure(figsize=(20, 5))
        librosa.display.specshow(data=mfccs, sr=self.fs, x_axis='time')
        plt.savefig('mfcc_value.png')
        plt.show()

    def scale_mfcc(self):
        mfccs = librosa.feature.mfcc(y=self.x, sr=self.fs)
        mfccs_scale = sklearn.preprocessing.scale(mfccs, axis=1)
        print(mfccs_scale.mean(axis=1))
        print(mfccs_scale.var(axis=1))
        plt.figure(figsize=(20, 5))
        librosa.display.specshow(mfccs_scale, sr=self.fs, x_axis='time')
        plt.savefig('mfcc_scale_value.png')
        plt.show()

class Chroma(MFCC):
    def __init__(self, path):
        super().__init__(path)
    def chroma_stft(self):
        hop_length = 512
        chromagram = librosa.feature.chroma_stft(y=self.x, sr=self.fs, hop_length=hop_length)
        plt.figure(figsize=(20, 5))
        librosa.display.specshow(data=chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length)
        plt.savefig('mfcc_chromagram.png')
        plt.show()



# myMFCC = MFCC(path='loop.wav')
# myMFCC.scale_mfcc()

chroma = Chroma(path=librosa.util.example_audio_file())
chroma.chroma_stft()
