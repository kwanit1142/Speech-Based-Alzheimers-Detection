import torch
import torch.nn as nn
import enum
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class ModelType(enum.Enum):
    """Enum for available model architectures"""
    RAW_AUDIO_CNN = "raw_audio_cnn"
    WAVENET = "wavenet"
    RAW_AUDIO_LSTM = "raw_audio_lstm"
    WAV2VEC2_NN = "wav2vec2_nn"
    SPEC_CNN = "spec_cnn"


class BaseAudioModel(nn.Module):
    def __init__(self, num_disease_classes=2, num_language_classes=2, lang_aware=True):
        super(BaseAudioModel, self).__init__()
        self.lang_aware = lang_aware
        self.base_feature_extractor = nn.Identity()
        self.language_head = nn.Identity()
        self.disease_feature_extractor = nn.Identity()
        self.disease_head = nn.Identity()

    def forward(self, x):
        base_features = self.base_feature_extractor(x)
        
        lang_logits = self.language_head(base_features)

        disease_features = self.disease_feature_extractor(base_features)
        disease_logits = self.disease_head(disease_features)

        if self.lang_aware:
            return disease_logits, lang_logits

        return disease_logits


class RawAudioCNN(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(RawAudioCNN, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=40, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=20, stride=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.dropout3 = nn.Dropout(0.4)

        self.relu = nn.ReLU()

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.pool1, self.dropout1,
            self.conv2, self.bn2, self.relu
        )

        self.language_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.pool2, self.dropout2,
            self.conv3, self.bn3, self.relu,
            self.pool3, self.dropout3,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.disease_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_disease_classes)
        )


class WaveNet(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, lang_aware=True):
        super(WaveNet, self).__init__(num_disease_classes, num_language_classes, lang_aware)

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1 if i == 0 else 64, 64, kernel_size=3, dilation=2**i, padding=2**i),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.1 + 0.1*i)
            ) for i in range(8)
        ])

        self.base_feature_extractor = nn.Sequential(*self.conv_layers[:2])

        self.language_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_language_classes)
        )

        self.remaining_layers = nn.Sequential(*self.conv_layers[2:])
        self.disease_feature_extractor = nn.Sequential(
            self.remaining_layers,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.disease_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_disease_classes)
        )

    def forward(self, x):
        base = x
        for layer in self.base_feature_extractor:
            base = layer(base)

        lang_logits = self.language_head(base)

        residual = base
        for layer in self.remaining_layers:
            base = layer(base)
            base = base + residual  
            residual = base

        disease_features = self.disease_feature_extractor(base)
        disease_logits = self.disease_head(disease_features)

        if self.lang_aware:
            return disease_logits, lang_logits
        return disease_logits


class RawAudioLSTM(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, lang_aware=True):
        super(RawAudioLSTM, self).__init__(num_disease_classes, num_language_classes, lang_aware)

        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=40, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, 
                            num_layers=2, batch_first=True, 
                            bidirectional=True, dropout=0.3)

        self.attention = nn.Linear(256, 1)

        self.relu = nn.ReLU()

        self.base_feature_extractor = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu
        )

        self.language_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.pool1,
            self.conv2,
            self.bn2,
            self.relu,
            self.pool2
        )

        self.disease_head = nn.Sequential(
            nn.Linear(256, num_disease_classes)
        )

    def forward(self, x):
        base_features = self.base_feature_extractor(x)
        lang_logits = self.language_head(base_features)

        x = self.disease_feature_extractor(base_features)

        x = x.permute(0, 2, 1) 
        x, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(x * attn_weights, dim=1)

        disease_logits = self.disease_head(context)

        if self.lang_aware:
            return disease_logits, lang_logits
        return disease_logits


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", num_disease_classes=2, 
                 num_language_classes=2, dropout_prob=0.1, sample_rate=16000, 
                 lang_aware=False, cache_dir=None):
        super().__init__()
        
        self.lang_aware = lang_aware
        self.sample_rate = sample_rate
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir)
        self.hidden_size = self.wav2vec2.config.hidden_size

        self.disease_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size, num_disease_classes)
        )
        
        self.lang_head = nn.Linear(self.hidden_size, num_language_classes)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            return_attention_mask=True,
            cache_dir=cache_dir
        )

    def forward(self, x):
        processed_x = self.process_audio(x)
        input_values = processed_x["input_values"]
        attention_mask = processed_x["attention_mask"]
        
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)

        disease_logits = self.disease_head(pooled)
        lang_logits = self.lang_head(pooled)

        if self.lang_aware:
            return disease_logits, lang_logits
        return disease_logits

    def process_audio(self, audio_tensor):
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(1)

        audio_numpy = audio_tensor.detach().cpu().numpy()
        inputs = self.feature_extractor(
            audio_numpy,
            sampling_rate=self.sample_rate,
            padding="longest",
            return_tensors="pt"
        )
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def classify_audio(self, audio_tensor):
        inputs = self.process_audio(audio_tensor)
        with torch.no_grad():
            logits = self.forward(
                input_values=inputs["input_values"],
                attention_mask=inputs["attention_mask"]
            )
        return logits


class SpectroCNN(nn.Module):
    def __init__(self, num_disease_classes=2, num_language_classes=2, lang_aware=False):
        super(SpectroCNN, self).__init__()
        
        self.lang_aware = lang_aware

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  
        )

        self.flatten = nn.Flatten()  

        self.disease_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_disease_classes)
        )

        self.lang_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_language_classes)
        )

    def forward(self, x):
        x = self.cnn_backbone(x)
        x = self.flatten(x)

        dis_logits = self.disease_head(x)
        
        if self.lang_aware:
            return dis_logits, self.lang_head(x) 

        return dis_logits


def get_model(model_type: ModelType, checkpoint_path: str = None, lang_awareness: bool = True):
    """
    Make and load (if checkpoint_path is provided) the model.
    """
    model_type = ModelType(model_type)
    if model_type == ModelType.RAW_AUDIO_CNN:
        model = RawAudioCNN(lang_aware=lang_awareness)
    elif model_type == ModelType.WAVENET:
        model = WaveNet(lang_aware=lang_awareness)
    elif model_type == ModelType.RAW_AUDIO_LSTM:
        model = RawAudioLSTM(lang_aware=lang_awareness)
    elif model_type == ModelType.WAV2VEC2_NN:
        model = Wav2Vec2Classifier(lang_aware=lang_awareness)
    elif model_type == ModelType.SPEC_CNN:
        model = SpectroCNN(lang_aware=lang_awareness)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        exit(1)
    
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))             
    
    return model

