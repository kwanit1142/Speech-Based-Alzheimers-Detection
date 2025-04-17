import argparse
from model_utils import ModelType, get_model
from data_utils import Datasets, LanguageEnum
import os
import librosa
import math
import torch
import numpy as np
from scipy import stats


MODELS_DIR = "../models"
SAMPLE_RATE = 16000


def split_audio_file(input_file, chunk_duration=10, overlap_factor=0.25):
    """
    Split the audio file into multiple sections of specified duration with a given overlap factor.
    """
    waveform, original_sr = librosa.load(input_file, sr=None, mono=True)

    if original_sr != SAMPLE_RATE:
        waveform = librosa.resample(y=waveform, orig_sr=original_sr, target_sr=SAMPLE_RATE)

    total_duration = len(waveform) / SAMPLE_RATE
    step_size = int(chunk_duration * SAMPLE_RATE * (1 - overlap_factor))
    num_splits = math.ceil((total_duration - chunk_duration) / step_size) + 1
    audio_chunks = []
    
    start_time = 0
    end_time = start_time + chunk_duration

    while end_time < total_duration:
        end_time = start_time + chunk_duration
        if end_time > total_duration:
            end_time = total_duration
            start_time = total_duration - chunk_duration

        start_idx = int(start_time * SAMPLE_RATE)
        end_idx = int(end_time * SAMPLE_RATE)

        # print(start_time, end_time, start_idx, end_idx)

        audio_chunk = waveform[start_idx:end_idx]
        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)
        audio_chunks.append(audio_tensor)
        start_time = start_time + (1-overlap_factor) * chunk_duration

    return audio_chunks


def get_predictions(model, audio_chunks, lang_awareness):
    """
    Perform prediction on each audio chunk using the provided model.
    Perform majority voting to determine the final prediction.
    """
    predictions = []
    lang_predictions = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for chunk in audio_chunks:
        chunk = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(chunk)
            if lang_awareness:
                _, predicted = torch.max(output[0], 1)
                _, lang_predicted = torch.max(output[1], 1)
                predictions.append(predicted.cpu().numpy())
                lang_predictions.append(lang_predicted.cpu().numpy())
            else:
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.cpu().numpy())

    if lang_awareness:
        lang_predicted_label = stats.mode(np.concatenate(lang_predictions), axis=0)
        print(f"Language Detected - {LanguageEnum(lang_predicted_label.mode).name}")

    predicted_label = stats.mode(predictions).mode
    print(f"Predicted Label - {'Alzheimer Dimentia Detected' if not predicted_label else 'Healthy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on an audio file")
    parser.add_argument('--model', type=str, choices=[model.value for model in ModelType], default=ModelType.RAW_AUDIO_LSTM.value, help="Model architecture to use")
    parser.add_argument('--dataset', type=str, choices=[dataset.value for dataset in Datasets], default=Datasets.ALL.value, help="Dataset model is trained on")
    parser.add_argument('--lang_awareness', type=int, choices=[0, 1], default=0, help="If language aware model has to be used")
    parser.add_argument('--input_file', type=str, default="../inference/example.wav", help="Input file for inference")
    args = parser.parse_args()

    MODEL_TYPE = args.model
    LANG_AWARENESS = bool(args.lang_awareness)
    if args.dataset != Datasets.ALL.value:
        LANG_AWARENESS = False
    DATASET = args.dataset + ("_lang_aware" if LANG_AWARENESS else "")

    checkpoint_path = os.path.join(MODELS_DIR, MODEL_TYPE, DATASET, "best_model.pth")

    model = get_model(MODEL_TYPE, checkpoint_path, LANG_AWARENESS)

    audio_chunks = split_audio_file(args.input_file, chunk_duration=10, overlap_factor=0.25)

    get_predictions(model, audio_chunks, LANG_AWARENESS)
