import torch
import torch.nn as nn
import enum
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class ModelType(enum.Enum):
    """Enum for available model architectures"""
    KURDISH_CNN = "kurdish_cnn"
    GAMMA_ERB_CNN = "gamma_erb_cnn"
    GAMMA_GM_CNN = "gamma_gm_cnn"
    SOUNDNET8 = "soundnet8"
    SOUNDNET5 = "soundnet5"
    M3 = 'm3'
    M5 = 'm5'
    M11 = 'm11'
    M18 = 'm18'
    RAW_AUDIO_CNN = "raw_audio_cnn"
    WAVENET = "wavenet"
    RAW_AUDIO_LSTM = "raw_audio_lstm"
    WAV2VEC2_NN = "wav2vec2_nn"
    SPEC_CNN = "spec_cnn"

class GammaConv1dGreenwoodERB(nn.Module):
    def __init__(self, num_filters=64, kernel_size=401, sr=16000,
                 low_freq=0, high_freq=8000, order=4, freeze=False):
        super(GammaConv1dGreenwoodERB, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sr = sr
        self.order = order

        # Generate ERB-spaced center frequencies
        self.center_freqs = self._erb_space(low_freq, high_freq, num_filters)

        # Create filterbank weights [num_filters, 1, kernel_size]
        filters = []
        for cf in self.center_freqs:
            gt = self._gammatone_ir(cf, kernel_size, sr, order)
            filters.append(gt)
        filters = np.stack(filters)  # shape: [num_filters, kernel_size]
        filters = torch.tensor(filters, dtype=torch.float32).unsqueeze(1)  # [num_filters, 1, kernel_size]

        # Conv1d with fixed weights
        self.conv = nn.Conv1d(in_channels=1, out_channels=num_filters,
                              kernel_size=kernel_size, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(filters)
        if freeze:
            self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

    def _erb_space(self, low_freq, high_freq, num):
        """ERB scale spacing (Glasberg & Moore)"""
        EarQ = 9.26449
        minBW = 24.7
        cf = -(EarQ * minBW) + np.exp(
            np.linspace(1, num, num) *
            (-np.log(high_freq + EarQ * minBW) + np.log(low_freq + EarQ * minBW)) / num
        ) * (high_freq + EarQ * minBW)
        return cf

    def _gammatone_ir(self, cf, length, sr, order):
        """Generate gammatone filter impulse response."""
        t = np.arange(0, length) / sr
        erb = 24.7 + 0.108 * cf
        b = 1.019 * 2 * np.pi * erb
        a = t**(order - 1) * np.exp(-b * t) * np.cos(2 * np.pi * cf * t)
        # Normalize
        a /= np.max(np.abs(a))
        return a

class GammaConv1dGlasbergMoore(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate,
                 order=4, min_freq=50, max_freq=None, freeze=False, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        
        if max_freq is None:
            max_freq = sample_rate / 2
        
        # Generate gammatone filterbank
        weight = self._make_gammatone_filterbank(
            out_channels, kernel_size, sample_rate, order, min_freq, max_freq
        )
        
        with torch.no_grad():
            self.weight.copy_(weight.repeat(in_channels, 1, 1))  # repeat for in_channels
            if self.bias is not None:
                self.bias.zero_()
        
        # Optionally freeze weights
        if freeze:
            self.weight.requires_grad = False
    
    def _make_gammatone_filterbank(self, n_filters, kernel_size, fs, order, min_freq, max_freq):
        freqs = np.geomspace(min_freq, max_freq, n_filters)  # log-spaced
        filters = []
        for fc in freqs:
            filters.append(self._gammatone_ir(fc, fs, kernel_size, order))
        filters = np.stack(filters)
        filters = filters / np.linalg.norm(filters, axis=1, keepdims=True)  # normalize
        return torch.tensor(filters, dtype=torch.float32).unsqueeze(1)  # (out, 1, kernel)
    
    def _gammatone_ir(self, fc, fs, length, order):
        t = np.arange(0, length) / fs
        erb = 24.7 + 0.108 * fc
        b = 1.019 * 2 * np.pi * erb
        gain = ((2 * np.pi * erb) ** order) / np.math.factorial(order - 1)
        g = gain * (t ** (order - 1)) * np.exp(-b * t) * np.cos(2 * np.pi * fc * t)
        return g.astype(np.float32)

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

class GammaERBCNN(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(GammaCNN, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = GammaConv1dGreenwoodERB()
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, stride=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=8)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8, 128)
        self.fc1l = nn.Linear(128 * 8, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout1l = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.25)

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(64),
            nn.ReLU(), self.pool1, self.conv2, # nn.BatchNorm1d(64),
            nn.ReLU(), self.pool2, self.conv3, # nn.BatchNorm1d(64),
            nn.ReLU(), self.conv4, # nn.BatchNorm1d(128),
            nn.ReLU(), self.adaptive_pool
        )

        self.language_head = nn.Sequential(
            self.flattenl, self.fc1l,
            nn.ReLU(), self.dropout1l, nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.flatten, self.fc1,
            nn.ReLU()
        )

        self.disease_head = nn.Sequential(
            self.fc2,
            nn.ReLU(), self.dropout2, nn.Linear(64, num_disease_classes)
        )

class GammaGMCNN(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(GammaCNN, self).__init__(num_disease_classes, num_language_classes, lang_aware)

        self.conv1 = GammaConv1dGlasbergMoore(
            in_channels=1, 
            out_channels=16, 
            kernel_size=64, 
            sample_rate=16000,
            freeze=False
        )
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=8)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8, 128)
        self.fc1l = nn.Linear(128 * 8, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout1l = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.25)

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(16),
            nn.ReLU(), self.pool1, self.conv2, # nn.BatchNorm1d(32),
            nn.ReLU(), self.pool2, self.conv3, # nn.BatchNorm1d(64),
            nn.ReLU(), self.conv4, # nn.BatchNorm1d(128),
            nn.ReLU(), self.adaptive_pool
        )

        self.language_head = nn.Sequential(
            self.flattenl, self.fc1l,
            nn.ReLU(), self.dropout1l, nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.flatten, self.fc1,
            nn.ReLU()
        )

        self.disease_head = nn.Sequential(
            self.fc2,
            nn.ReLU(), self.dropout2, nn.Linear(64, num_disease_classes)
        )

class SoundNet8(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(SoundNet8, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=32)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=16)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=1, padding=0)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=2)
        self.pool5 = nn.MaxPool1d(kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=2)
        self.conv7 = nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=2)
        self.conv8 = nn.Conv1d(1024, 1401, kernel_size=8, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()
        self.fc1 = nn.Linear(1401, 160)
        self.fc1l = nn.Linear(1401, 160)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout1l = nn.Dropout(0.25)
        self.fc2 = nn.Linear(160, 80)
        self.dropout2 = nn.Dropout(0.25)

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(16),
            nn.ReLU(), self.pool1, self.conv2, # nn.BatchNorm1d(32),
            nn.ReLU(), self.pool2, self.conv3, # nn.BatchNorm1d(64),
            nn.ReLU(), self.conv4, # nn.BatchNorm1d(128),
            nn.ReLU(), self.conv5, # nn.BatchNorm1d(256),
            nn.ReLU(), self.pool5, self.conv6, # nn.BatchNorm1d(512),
            nn.ReLU(), self.conv7, # nn.BatchNorm1d(1024),
            nn.ReLU(), self.conv8, # nn.BatchNorm1d(1401),
            nn.ReLU()
        )

        self.language_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), self.flattenl, self.fc1l,
            nn.ReLU(), self.dropout1l, nn.Linear(160, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), self.flatten, self.fc1,
            nn.ReLU(), self.dropout1
        )

        self.disease_head = nn.Sequential(
            self.fc2,
            nn.ReLU(), self.dropout2, nn.Linear(80, num_disease_classes)
        )

class SoundNet5(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(SoundNet5, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=64, stride=2, padding=32)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=16)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8)
        self.pool3 = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4)
        self.conv5 = nn.Conv1d(256, 1401, kernel_size=16, stride=12, padding=4)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()
        self.fc1 = nn.Linear(1401, 160)
        self.fc1l = nn.Linear(1401, 160)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout1l = nn.Dropout(0.25)
        self.fc2 = nn.Linear(160, 80)
        self.dropout2 = nn.Dropout(0.25)

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(32),
            nn.ReLU(), self.pool1, nn.ReLU(), self.conv2, # nn.BatchNorm1d(64),
            nn.ReLU(), self.pool2, self.conv3, # nn.BatchNorm1d(128),
            nn.ReLU(), self.conv4, # nn.BatchNorm1d(256),
            nn.ReLU(), self.conv5, # nn.BatchNorm1d(1401),
            nn.ReLU()
        )

        self.language_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), self.flattenl, self.fc1l,
            nn.ReLU(), self.dropout1l, nn.Linear(160, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), self.flatten, self.fc1,
            nn.ReLU(), self.dropout1
        )

        self.disease_head = nn.Sequential(
            self.fc2, 
            nn.ReLU(), self.dropout2, nn.Linear(80, num_disease_classes)
        )

class KurdishCNN(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(KurdishCNN, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=4)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=4)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=4)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()
        self.fc1 = nn.Linear(100, 80)
        self.fc1l = nn.Linear(100, 80)
        self.fc2 = nn.Linear(80, 40)

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(100),
            nn.ReLU(), self.conv2, # nn.BatchNorm1d(100),
            nn.ReLU(), self.conv3, # nn.BatchNorm1d(100),
            nn.ReLU()
        )

        self.language_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), self.flattenl, self.fc1l,
            nn.ReLU(), nn.Linear(80, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), self.flatten, self.fc1,
            nn.ReLU()
        )

        self.disease_head = nn.Sequential(
            self.fc2,
            nn.ReLU(), nn.Linear(40, num_disease_classes)
        )

class M3(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(M3, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=80, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(256),
            nn.ReLU(), self.pool1, self.conv2, # nn.BatchNorm1d(256),
            nn.ReLU(), self.pool2, self.pool3
        )

        self.language_head = nn.Sequential(
            self.flattenl, nn.Linear(256, 128),
            nn.ReLU(), nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.flatten, nn.Linear(256, 128),
            nn.ReLU()
        )

        self.disease_head = nn.Sequential(
            nn.Linear(128, num_disease_classes)
        )

class M5(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(M5, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=80, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        self.pool5 = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(128),
            nn.ReLU(), self.pool1, self.conv2, # nn.BatchNorm1d(128),
            nn.ReLU(), self.pool2, self.conv3, # nn.BatchNorm1d(256),
            nn.ReLU(), self.pool3, self.conv4, # nn.BatchNorm1d(512),
            nn.ReLU(), self.pool4, self.pool5
        )

        self.language_head = nn.Sequential(
            self.flattenl, nn.Linear(512, 128),
            nn.ReLU(), nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.flatten, nn.Linear(512, 128),
            nn.ReLU()
        )

        self.disease_head = nn.Sequential(
            nn.Linear(128, num_disease_classes)
        )

class M11(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(M11, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv3_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.conv4_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv4_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.conv4_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        self.conv5_1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv5_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool5 = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(64),
            nn.ReLU(), self.pool1, self.conv2_1, # nn.BatchNorm1d(64),
            nn.ReLU(), self.conv2_2, # nn.BatchNorm1d(64),
            nn.ReLU(), self.pool2, self.conv3_1, # nn.BatchNorm1d(128),
            nn.ReLU(), self.conv3_2, # nn.BatchNorm1d(128),
            nn.ReLU(), self.pool3, self.conv4_1, # nn.BatchNorm1d(256),
            nn.ReLU(), self.conv4_2, # nn.BatchNorm1d(256),
            nn.ReLU(), self.conv4_3, # nn.BatchNorm1d(256),
            nn.ReLU(), self.pool4, self.conv5_1, # nn.BatchNorm1d(512),
            nn.ReLU(), self.conv5_2, # nn.BatchNorm1d(512),
            nn.ReLU(), self.pool5
        )

        self.language_head = nn.Sequential(
            self.flattenl, nn.Linear(512, 128),
            nn.ReLU(), nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.flatten, nn.Linear(512, 128),
            nn.ReLU()
        )

        self.disease_head = nn.Sequential(
            nn.Linear(128, num_disease_classes)
        )

class M18(BaseAudioModel):
    def __init__(self, num_disease_classes=2, num_language_classes=2, sample_rate=16000, duration=10, lang_aware=True):
        super(M18, self).__init__(num_disease_classes, num_language_classes, lang_aware)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv2_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.conv3_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv3_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv3_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv3_4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.conv4_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv4_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.conv4_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.conv4_4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        self.conv5_1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv5_2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv5_3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv5_4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool5 = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.flattenl = nn.Flatten()

        # Define heads
        self.base_feature_extractor = nn.Sequential(
            self.conv1, # nn.BatchNorm1d(64),
            nn.ReLU(), self.pool1, self.conv2_1, # nn.BatchNorm1d(64),
            nn.ReLU(), self.conv2_2, # nn.BatchNorm1d(64),
            nn.ReLU(), self.conv2_3, # nn.BatchNorm1d(64),
            nn.ReLU(), self.conv2_4, # nn.BatchNorm1d(64),
            nn.ReLU(), self.pool2, self.conv3_1, # nn.BatchNorm1d(128),
            nn.ReLU(), self.conv3_2, # nn.BatchNorm1d(128),
            nn.ReLU(), self.conv3_3, # nn.BatchNorm1d(128),
            nn.ReLU(), self.conv3_4, # nn.BatchNorm1d(128),
            nn.ReLU(), self.pool3, self.conv4_1, # nn.BatchNorm1d(256),
            nn.ReLU(), self.conv4_2, # nn.BatchNorm1d(256),
            nn.ReLU(), self.conv4_3, # nn.BatchNorm1d(256),
            nn.ReLU(), self.conv4_4, # nn.BatchNorm1d(256),
            nn.ReLU(), self.pool4, self.conv5_1, # nn.BatchNorm1d(512),
            nn.ReLU(), self.conv5_2, # nn.BatchNorm1d(512),
            nn.ReLU(), self.conv5_3, # nn.BatchNorm1d(512),
            nn.ReLU(), self.conv5_4, # nn.BatchNorm1d(512),
            nn.ReLU(), self.pool5
        )

        self.language_head = nn.Sequential(
            self.flattenl, nn.Linear(512, 128),
            nn.ReLU(), nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.flatten, nn.Linear(512, 128),
            nn.ReLU()
        )

        self.disease_head = nn.Sequential(
            nn.Linear(128, num_disease_classes)
        )

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
            self.conv1, # self.bn1,
            self.relu, self.pool1, self.dropout1,
            self.conv2, # self.bn2,
            self.relu
        )

        self.language_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_language_classes)
        )

        self.disease_feature_extractor = nn.Sequential(
            self.pool2, self.dropout2,
            self.conv3, # self.bn3,
            self.relu, self.pool3, self.dropout3,
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
            # self.bn1,
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
            # self.bn2,
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
    if model_type == ModelType.KURDISH_CNN:
        model = KurdishCNN(lang_aware=lang_awareness)
    elif model_type == ModelType.GAMMA_ERB_CNN:
        model = GammaERBCNN(lang_aware=lang_awareness)
    elif model_type == ModelType.GAMMA_GM_CNN:
        model = GammaGMCNN(lang_aware=lang_awareness)
    elif model_type == ModelType.SOUNDNET8:
        model = SoundNet8(lang_aware=lang_awareness)
    elif model_type == ModelType.SOUNDNET5:
        model = SoundNet5(lang_aware=lang_awareness)
    elif model_type == ModelType.M3:
        model = M3(lang_aware=lang_awareness)
    elif model_type == ModelType.M5:
        model = M5(lang_aware=lang_awareness)
    elif model_type == ModelType.M11:
        model = M11(lang_aware=lang_awareness)
    elif model_type == ModelType.M18:
        model = M18(lang_aware=lang_awareness)
    elif model_type == ModelType.RAW_AUDIO_CNN:
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

