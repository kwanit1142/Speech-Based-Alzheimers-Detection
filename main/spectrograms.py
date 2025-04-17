import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# This file was used to make the spectrogram dataset from the audio dataset
def convert_audio_to_stft(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.wav', '.mp3')):

            input_filepath = os.path.join(input_dir, filename)
            y, sr = librosa.load(input_filepath, sr=None)
            D = librosa.stft(y)
            
            base_name = os.path.splitext(filename)[0]
            output_filepath = os.path.join(output_dir, base_name + '.png')

            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            plt.figure(figsize=(12, 8))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')

            plt.axis('off') 
            plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    convert_audio_to_stft("../data/NCMMSC/AD", "../data/spectrograms/NCMMSC/AD")
    convert_audio_to_stft("../data/NCMMSC/HC", "../data/spectrograms/NCMMSC/HC")
    convert_audio_to_stft("../data/NCMMSC/MCI", "../data/spectrograms/NCMMSC/MCI")
    convert_audio_to_stft("../data/ADReSS/train", "../data/spectrograms/ADReSS/train")
