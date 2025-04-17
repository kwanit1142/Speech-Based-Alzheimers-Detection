import os
import argparse
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model_utils import ModelType, get_model
from data_utils import Datasets, LanguageEnum


MODELS_DIR = "../models"
OUTPUT_DIR = "../output"


def make_spectrogram(audio_path):
    """
    Create a spectrogram from the audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return D, S_db, sr, len(y)


def convert_to_tensor(S_db, sr):
    """
    Convert the spectrogram to a tensor and save it as an image.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(S_db, aspect='auto', origin='lower', cmap='magma', vmin=S_db.min(), vmax=S_db.max())
    plt.axis('off')
    plt.savefig(f'{OUTPUT_DIR}/spectrogram.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(f'{OUTPUT_DIR}/spectrogram.png').convert("RGB")
    img_tensor = transform(img)
    return img, img_tensor.unsqueeze(0)


def get_prediction(model, img_tensor, lang_awareness):
    """
    Get the prediction from the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        if lang_awareness:
            _, predicted = torch.max(output[0], 1)
            _, lang_predicted = torch.max(output[1], 1)
        else:
            _, predicted = torch.max(output, 1)

    if lang_awareness:
        print(f"Language Detected - {LanguageEnum(lang_predicted.item()).name}")
    print(f"Predicted Label - {'Alzheimer Dimentia Detected' if not predicted.item() else 'Healthy'}")


def grad_cam_vis(model, img, img_tensor, original_spec_shape, sr):
    """
    Generate Grad-CAM visualization and save the output.
    """
    model.lang_aware = False
    target_layer = model.cnn_backbone[4]
    rgb_img = np.array(img.resize((224, 224))) / 255.0

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(original_spec_shape),
    ])

    grayscale_cam = transform(grayscale_cam).squeeze(0).numpy()
    grayscale_cam_flipped = np.flipud(grayscale_cam)

    plt.figure(figsize=(12, 8))
    plt.imshow(grayscale_cam_flipped, aspect='auto', cmap='gray', origin='lower')
    plt.axis('off')
    plt.savefig(f'{OUTPUT_DIR}/gradcamoutgray.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return grayscale_cam


def save_masked_audio(S_db, D, mask, sr):
    """
    Make masked spectrogram and save the output.
    Also save reconstructed audio from the masked spectrogram.
    """
    threshold = 0.3
    binary_mask = (mask >= threshold).astype(np.uint8)
    binary_mask_flipped = np.flipud(binary_mask)

    plt.figure(figsize=(12, 8))
    plt.imshow(binary_mask_flipped, aspect='auto', cmap='gray', origin='lower')
    plt.axis('off')
    plt.savefig(f'{OUTPUT_DIR}/binarymask.png', format='png', bbox_inches='tight',  pad_inches=0)
    plt.close()

    D_masked = D * binary_mask
    S_db_masked = S_db * binary_mask_flipped
    S_db_visual = np.where(binary_mask_flipped == 0, np.nan, S_db)

    plt.figure(figsize=(12, 8))

    plt.imshow(S_db_visual, aspect='auto', cmap='magma', origin='lower', vmin=S_db.min(), vmax=S_db.max())
    plt.axis('off')
    plt.savefig(f'{OUTPUT_DIR}/masked_spectrogram.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    y_reconstructed = librosa.istft(D_masked)
    sf.write(f'{OUTPUT_DIR}/recon.wav', y_reconstructed, sr)


# All outputs are saved in a directory named outout
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on an audio file")
    parser.add_argument('--model', type=str, choices=[model.value for model in ModelType], default=ModelType.SPEC_CNN.value, help="Model architecture to use")
    parser.add_argument('--dataset', type=str, choices=[dataset.value for dataset in Datasets], default=Datasets.ALL.value, help="Dataset model is trained on")
    parser.add_argument('--lang_awareness', type=int, choices=[0, 1], default=1, help="If language aware model has to be used")
    parser.add_argument('--input_file', type=str, default="../infer_data/adrso011.wav", help="Input file for inference")
    args = parser.parse_args()

    MODEL_TYPE = args.model
    LANG_AWARENESS = bool(args.lang_awareness)
    if args.dataset != Datasets.ALL.value:
        LANG_AWARENESS = False
    DATASET = args.dataset + ("_lang_aware" if LANG_AWARENESS else "")

    checkpoint_path = os.path.join(MODELS_DIR, MODEL_TYPE, DATASET, "best_model.pth")
    model = get_model(MODEL_TYPE, checkpoint_path, LANG_AWARENESS)

    D, S_db, sr, wave_len = make_spectrogram(args.input_file)
    original_spec_shape = S_db.shape

    img, img_tensor = convert_to_tensor(S_db, sr)
    get_prediction(model, img_tensor, LANG_AWARENESS)
    mask = grad_cam_vis(model, img, img_tensor, original_spec_shape, sr)
    save_masked_audio(S_db, D, mask, sr)
