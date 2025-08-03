import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import math
import librosa
import enum
from audiomentations import Compose, PitchShift, GainTransition, AddGaussianNoise, TimeStretch, Shift, Gain, PolarityInversion, HighPassFilter, LowPassFilter


DATA_DIR = "../data"


class LanguageEnum(enum.Enum):
    """Enum for available languages."""
    ENGLISH = 0
    CHINESE = 1


class Datasets(enum.Enum):
    """Enum for available datasets."""
    ALL = "all"
    ADRESS = "adress"
    NCMMSC = "ncmmsc"
    ADRESS_SPEC = "adress_spec"
    NCMMSC_SPEC = "ncmmsc_spec"


class DimentiaDataset(Dataset):
    def __init__(self, split_duration=5.0, overlap_factor=0.5, lang_aware=False):
        """
        Base class for audio datasets.
        """
        self.split_duration = split_duration
        self.overlap_factor = overlap_factor
        self.lang_aware = lang_aware

        self.audio_paths = []  
        self.labels = []
        self.languages = []
        
        self.sample_rate = 16000
    
    def _add_audio_splits(self, audio_name, file_path, label, language=LanguageEnum.ENGLISH):
        """
        Create splits for a single audio file and add them to the dataset.
        """
        try:
            waveform, original_sr = librosa.load(file_path, sr=None, mono=True)
            
            if original_sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=self.sample_rate)
            
            total_duration = len(waveform) / self.sample_rate
            
            step_size = self.split_duration * (1 - self.overlap_factor)
            
            num_splits = math.ceil((total_duration - self.split_duration) / step_size) + 1
            
            for i in range(num_splits):
                start_time = i * step_size
                end_time = start_time + self.split_duration
                
                if end_time > total_duration:
                    if total_duration > self.split_duration:
                        start_time = total_duration - self.split_duration
                        end_time = total_duration
                    else:
                        start_time = 0
                        end_time = total_duration
                
                self.audio_paths.append((audio_name, file_path, start_time, end_time))
                self.labels.append(label)
                self.languages.append(language.value)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_name, file_path, start_time, end_time = self.audio_paths[idx]
        label = self.labels[idx]
        
        waveform, original_sr = librosa.load(file_path, sr=None, mono=True)
        
        if original_sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=self.sample_rate)

        start_sample = int(start_time * self.sample_rate)
        end_sample = min(int(end_time * self.sample_rate), len(waveform))
        
        audio_segment = waveform[start_sample:end_sample]
        
        target_length = int(self.split_duration * self.sample_rate)
        
        if len(audio_segment) < target_length:
            padding = np.zeros(target_length - len(audio_segment))
            audio_segment = np.concatenate([audio_segment, padding])
        elif len(audio_segment) > target_length:
            audio_segment = audio_segment[:target_length]
        
        audio_tensor = torch.FloatTensor(audio_segment).unsqueeze(0) 
        
        if self.lang_aware:
            return audio_tensor, (torch.tensor(label, dtype=torch.long), torch.tensor(self.languages[idx], dtype=torch.long))
        return audio_tensor, torch.tensor(label, dtype=torch.long)


class ADReSSDataset(DimentiaDataset):
    """
    Dataset class for the ADReSS dataset.
    0 - ProbableAD
    1 - Control (Healthy)
    """
    def __init__(self, split_duration=5.0, overlap_factor=0.5, lang_aware=False):
        super().__init__(split_duration, overlap_factor, lang_aware)
        self.base_dir = f"{DATA_DIR}/ADReSS/train"
        self.label_csv = f"{DATA_DIR}/ADReSS/training-groundtruth.csv"

        self.language = LanguageEnum.ENGLISH

        self.class_map = {
            "ProbableAD": 0,
            "Control": 1
        }
        
        self.load_dataset()
    
    def load_dataset(self):
        label_df = pd.read_csv(self.label_csv)
        original_audio_paths = label_df['adressfname'].values
        original_labels = label_df['dx'].values
        
        for audio_path, label in zip(original_audio_paths, original_labels):
            full_path = os.path.join(self.base_dir, f"{audio_path}.mp3")
            self._add_audio_splits(audio_path, full_path, self.class_map[label], self.language)
    

class NCMMSCDataset(DimentiaDataset):
    """
    Dataset class for the NCMMSC dataset.
    0 - ProbableAD - MCI and AD
    1 - Control (Healthy)
    """
    def __init__(self, split_duration=5.0, overlap_factor=0.5, lang_aware=False):
        super().__init__(split_duration, overlap_factor, lang_aware)
        
        self.class_mapping = {
            "MCI": 0, "HC": 1, "AD": 0,
        }

        self.language = LanguageEnum.CHINESE

        self.base_dir = f"{DATA_DIR}/NCMMSC"

        self.load_dataset()


    def load_dataset(self):
        for class_name, class_label in self.class_mapping.items():
            class_dir = os.path.join(self.base_dir, class_name)
            
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found, skipping.")
                continue
            
            for filename in os.listdir(class_dir):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(class_dir, filename)
                    audio_name = os.path.splitext(filename)[0]
                    self._add_audio_splits(audio_name, audio_path, class_label, self.language)


class ADReSSSpectrogramDataset(Dataset):
    """
    Dataset class for the ADReSS dataset.
    0 - ProbableAD
    1 - Control (Healthy)
    """
    def __init__(self, lang_aware=True):
        self.base_dir = f'{DATA_DIR}/spectrograms/ADReSS/train'
        self.label_csv = f'{DATA_DIR}/spectrograms/ADReSS/training-groundtruth.csv'
        self.class_map = {
            "ProbableAD": 0,
            "Control": 1
        }
        
        self.image_paths = []
        self.labels = []
        self.language = LanguageEnum.ENGLISH.value
        self.lang_aware = lang_aware

        self._prepare_file_list()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  
        ])
        
    def _prepare_file_list(self):
        label_df = pd.read_csv(self.label_csv)
        
        for _, row in label_df.iterrows():
            audio_id = row['adressfname']
            label = row['dx']
            
            full_path = os.path.join(self.base_dir, f"{audio_id}.png")
            
            if not os.path.exists(full_path):
                print(f"Warning: Spectrogram {full_path} not found, skipping.")
                continue
            
            self.image_paths.append(full_path)
            self.labels.append(self.class_map[label])    
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        if self.lang_aware:
            return img_tensor, (torch.tensor(label, dtype=torch.long), torch.tensor(self.language, dtype=torch.long))

        return img_tensor, torch.tensor(label, dtype=torch.long)


class NCMMSCSpectrogramDataset(Dataset):
    """
    Dataset class for the NCMMSC dataset.
    0 - ProbableAD - MCI and AD
    1 - Control (Healthy)
    """
    def __init__(self, lang_aware=True):
        self.root_dir = f'{DATA_DIR}/spectrograms/NCMMSC/'

        self.class_map = {
            "AD": 0,
            "HC": 1,
            "MCI": 0
        }

        self.image_paths = []
        self.labels = []
        self.language = LanguageEnum.CHINESE.value
        self.lang_aware = lang_aware

        self._prepare_file_list()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  
        ])

    def _prepare_file_list(self):
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        if self.lang_aware:
            return img_tensor, (torch.tensor(label, dtype=torch.long), torch.tensor(self.language, dtype=torch.long))

        return img_tensor, torch.tensor(label, dtype=torch.long)

def collate_with_augmentation(batch):
    train_aug = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
            # TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
            # PitchShift(min_semitones=-0.25, max_semitones=0.25, p=0.5),
            # GainTransition(p=0.5),
            # Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5),
            # Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.5),
            # PolarityInversion(p=0.5),
            # HighPassFilter(min_cutoff_freq=200.0, max_cutoff_freq=400.0, p=0.5),
            # LowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=6000.0, p=0.5),
        ])
    augmented_batch = []
    for x, y in batch:
        x_aug = train_aug(samples=x.squeeze().numpy(), sample_rate=16000)
        x_aug = torch.FloatTensor(x_aug).unsqueeze(0)
        augmented_batch.append((x_aug, y))
    return torch.utils.data.default_collate(augmented_batch)    
 
def get_dataloaders(dataset_name, batch_size=32, num_workers=4, lang_aware=False):
    """
    To make dataloaders for audio datasets.
    """
    dataset_name = Datasets(dataset_name)
    if dataset_name == Datasets.ALL:
        datasets = [ADReSSDataset(lang_aware=lang_aware), NCMMSCDataset(lang_aware=lang_aware)]
        dataset = ConcatDataset(datasets)
    elif dataset_name == Datasets.ADRESS:
        dataset = ADReSSDataset(lang_aware=lang_aware)
    elif dataset_name == Datasets.NCMMSC:
        dataset = NCMMSCDataset(lang_aware=lang_aware)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator)

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_with_augmentation),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    return dataloader


def get_spectrogram_dataloaders(dataset_name, batch_size=32, num_workers=4, lang_aware=False):
    """
    To make dataloaders for spectrogram datasets.
    """
    dataset_name = Datasets(dataset_name)
    if dataset_name == Datasets.ALL:
        datasets = [ADReSSSpectrogramDataset(lang_aware=lang_aware), NCMMSCSpectrogramDataset(lang_aware=lang_aware)]
        dataset = ConcatDataset(datasets)
    elif dataset_name == Datasets.ADRESS_SPEC:
        dataset = ADReSSSpectrogramDataset(lang_aware=lang_aware)
    elif dataset_name == Datasets.NCMMSC_SPEC:
        dataset = NCMMSCSpectrogramDataset(lang_aware=lang_aware)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator)

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    return dataloader


if __name__ == "__main__":
    # adress_spectrogram_dataset = ADReSSSpectrogramDataset(lang_aware=False)
    # ncmmsc_spectrogram_dataset = NCMMSCSpectrogramDataset(lang_aware=True)

    # adress_loader = DataLoader(ncmmsc_spectrogram_dataset, batch_size=32, shuffle=True)

    # for batch in adress_loader:
    #     images, labels = batch
    #     print(images.shape)
    #     print(labels[0].shape)
    #     break
    pass
