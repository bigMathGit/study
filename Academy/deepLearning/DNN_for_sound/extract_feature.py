import glob
import os
import librosa
import numpy as np
import sys


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    files = []
    print('==DIR', parent_dir)
    for label, sub_dir in enumerate(sub_dirs):
        print('====DIR', label, sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            basename = os.path.basename(fn)
            try:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                label = basename.split('-')[1]
                labels = np.append(labels, label)
                files.append(basename)
                print(fn, label, ext_features.shape)
            except:
                print(fn, 'SKIP')
            
            
    return files, np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    num_data = len(labels)
    num_of_classes = 10 # len(np.unique(labels))
    one_hot = np.zeros((num_data,num_of_classes))
    one_hot[np.arange(num_data), labels] = 1
    return one_hot


parent_dir = 'audio'

sub_dirs = ['data']
files, features, labels = parse_audio_files(parent_dir,sub_dirs)

fp = open('train_features.csv', 'w')
for file, label, feature in zip(files, labels, features):
    fp.write('%s,%d,%s\n'  % (file, label, ','.join(['%f' % f for f in feature])))
fp.close()

print('Audio Data Loading Done')
print("feature. shape",features.shape)
print("label. shape",labels.shape)


# labels = one_hot_encode(labels)
# for f, l in zip(files, labels):
#     print(f, l)
print(one_hot_encode([2,3,1]))