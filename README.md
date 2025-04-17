# CSL7770 Project - B21CS073 and B21EE067
The repository contains the code and report for the project of the course CSL7770 - Speech Understanding.

**Directory Structure:**
```
.
├── data
│   ├── ADReSS
│   ├── NCMMSC
│   └── spectrograms
│       ├── ADReSS
│       └── NCMMSC
├── models
│   ├── <model_name>
│   │   ├── <dataset_name>
│   │   |   ├── best_model.pth
│   │   |   ├── training.log
│   │   |   ├── training_history.png
│   └── └── └── confusion_matrix.png
├── infer_data
|   └── <audio_file_name.mp3/wav>
├── infer_results
│   └── <audio_file_name.mp3/wav>
├── main
│   ├── main.py
│   └── main_utils.py
|── output
├── README.md
└── requirements.txt
```

The directory structure explanation is as follows:
- `data`: Contains the sample data directory structure for ADReSS and NCMMSC datasets (one sample from each dataset).
    The `spectrograms` folder contains the spectrograms generated from all the audio files in the respective datasets.

- `models`: Contains the trained models for each dataset. The folder structure is as follows:
    - `<model_name>`: Name of the model architecture used for training.
        - `<dataset_name>`: Name of the dataset used for training.
            - `best_model.pth`: The best model saved during training.
            - `training.log`: The log file containing the training details.
            - `training_history.png`: The plot showing the training history on train and validation set.
            - `confusion_matrix.png`: The confusion matrix plot on test set.

- `infer_data`: Contains three audio files on which gradcam outputs were generated. Each audio file is in `.mp3` or `.wav` format.

- `infer_results`: Contains the gradcam outputs and reconstruction outputs for the audio files in the `infer_data` folder. The gradcam outputs are saved as `.png` files.
    - `spectrogram.png`: The spectrogram of the audio file.
    - `gradcamoutgray.png`: The gradcam output of the audio file in grayscale.
    - `binarymask.png`: The binary mask of the gradcam output.
    - `masked_spectrogram.png`: The masked spectrogram of the audio file.
    - `recon.wav`: The reconstructed audio file after applying the binary mask on the spectrogram.

- `main`: Contains the main code for training and testing the model. The folder contains the following files:
    - `spectrograms.py`: Contains the code for generating the spectrograms from the audio files in a given directory.
    - `data_utils.py`: Contains the code for loading the audio dataset or spectrogram dataset.
    - `model_utils.py`: Contains the code for defining the model architecture or loading the pre-trained model.
    - `train.py`: Contains the code for training the model.
    - `inference_raw.py`: Contains the code for taking inference using an audio based model.
    - `inference_vis.py`: Contains the code for taking inference using a CNN (trained on spectrograms) and generating the gradcam outputs.

- `output`: Currently empty. All the gradcam outputs generated using the `inference_vis.py` file will be saved in this folder.

- `README.md`: This file.
- `requirements.txt`: Contains the list of required packages to run the code. 

## **How to use this repository**


1. **Clone the repository**

    First, clone the repository to your local machine and navigate to the repository using the following commands:

    ```bash
    git clone https://github.com/SohamD34/Speech-Based-Alzheimers-Detection.git
    cd Speech-Based-Alzheimers-Detection
    ```

2. **Prepare the environment**

    ```bash
    conda create -n csl7770 -y
    conda activate csl7770
    pip install -r requirements.txt -y
    ```

    The above command will create a new conda environment named `csl77701` and install all the required packages in it. It will take a few minutes to complete.

3. **Navigate to the `main` folder**

    ```bash
    cd main
    ```

    Make sure the conda environment is activated before running the code.

4. **Train the model**

    Run the `train.py` file to train the model.

    ```bash
    python train.py
    --dataset <dataset_name>
    --model <model_name>
    --batch_size <batch_size>
    --epochs <num_epochs>
    --learning_rate <lr>
    --weight_decay <weight_decay>
    --patience <patience>
    --continue_training <0|1>
    --lang_awareness <0|1>
    ```

    - `<model_name>`: Name of the model architecture to be used for training. 
    Options are `raw_audio_cnn`, `wavenet`, `raw_audio_lstm`, `wav2vec2_nn` and `spec_cnn`.

    - `<dataset_name>`: Name of the dataset to be used for training. Options are `adress`, `ncmmsc`, `adress_spec`, `ncmmsc_spec` and `all`.
    You can use `<dataset_spec>` to train ONLY THE `spec_cnn` model on the spectrograms of the dataset.
    If `all` is selected, the model will be trained on all the datasets. The chosen model if `spec_cnn` will be trained on both the datasets (`adess_spec` and `ncmmsc_spec`) combined randomly.
    The chosen model if any other than `spec_cnn` will be trained on the raw audio files of both the datasets (`adress` and `ncmmsc`) combined randomly.

    - `<batch_size>`: Batch size to be used for training. Default is `32`.

    - `<num_epochs>`: Number of epochs to be used for training. Default is `10`.

    - `<lr>`: Learning rate to be used for training. Default is `3e-3`. If the training is being continued, this will be overridden by the optimizer state of the last epoch.
    
    - `<weight_decay>`: Weight decay to be used for training. Default is `1e-5`.

    - `<patience>`: Number of epochs to wait for the validation loss to improve before stopping the training. Default is `5`.

    - `<continue_training>`: Whether to continue training from the last saved model or not. Default is `0`. If set to `1`, the training will continue from the last saved model.

    - `<lang_awareness>`: Whether to do language aware training or not. Default is `0`. If set to `1`, the model will be trained with language aware training. Applicable on all the models. The dataset used for training will be `all` in this case, otherwise it will be overriden to 0.

    **Note**: The state dict of the best model, optimizer and scheduler and the last epoch model, optimizer and scheduler will be saved in the `models/<model_name>/<dataset_name>` folder. The training log will be saved in the same folder with the name `training.log`. The training history plot will be saved in the same folder with the name `training_history.png`. The confusion matrix plot will be saved in the same folder with the name `confusion_matrix.png`. 


5. **Inference**

- **Audio based model inference**

  Run the `inference_raw.py` file to take inference using the audio based model.

  ```bash
  python inference_raw.py
  --model <model_name>
  --dataset <dataset_name>
  --lang_awareness <0|1>
  --input_file <input_file>
  ```

  - `<model_name>`: Name of the model architecture to be used for inference. Options are `raw_audio_cnn`, `wavenet`, `raw_audio_lstm` and `wav2vec2_nn`.
  - `<dataset_name>`: Name of the dataset the model was trained on. Options are `adress`, `ncmmsc` and `all`.
  - `<lang_awareness>`: Whether to use language aware model or not. Default is `0`. If set to `1`, the model used will be the language aware model. Applicable on all the models but only for `all` dataset.
  - `<input_file>`: Path to the input audio file. The audio file should be in `.mp3` or `.wav` format. 

- **Spectrogram based model inference and GradCAM explanation**

  Run the `inference_vis.py` file to take inference using the spectrogram based model and generate the gradcam outputs.

  ```bash
  python inference_vis.py
  --model <model_name>
  --dataset <dataset_name>
  --input_file <input_file>
  --lang_awareness <0|1>
  ```

  - `<model_name>`: Name of the model architecture to be used for inference. Options are `spec_cnn`.
  - `<dataset_name>`: Name of the dataset the model was trained on. Options are `adress_spec`, `ncmmsc_spec` and `all`.
  - `<input_file>`: Path to the input audio file. The audio file should be in `.mp3` or `.wav` format.
  - `<lang_awareness>`: Whether to use language aware model or not. Default is `0`. If set to `1`, the model used will be the language aware model. Applicable on all the models but only for `all` dataset.

  The gradcam outputs will be saved in the `output` folder.



## Model Performance

<table>
  <tr>
    <th></th>
    <th colspan="2" align="center"><strong>ADReSS</strong></th>
    <th style="border-left: 1px solid #ddd;" colspan="2" align="center"><strong>NCMMSC</strong></th>
    <th style="border-left: 1px solid #ddd;" colspan="2" align="center"><strong>Combined</strong></th>
    <th style="border-left: 1px solid #ddd;" colspan="2" align="center"><strong>Combined with Language Awareness</strong></th>
  </tr>
  <tr>
    <th></th>
    <th align="center">Train</th>
    <th align="center">Val</th>
    <th style="border-left: 1px solid #ddd;" align="center">Train</th>
    <th align="center">Val</th>
    <th style="border-left: 1px solid #ddd;" align="center">Train</th>
    <th align="center">Val</th>
    <th style="border-left: 1px solid #ddd;" align="center">Train</th>
    <th align="center">Val</th>
  </tr>
  <tr>
    <td>CNN on Raw Audio</td>
    <td align="center">63.15</td>
    <td align="center">66.48</td>
    <td style="border-left: 1px solid #ddd;" align="center">92.25</td>
    <td align="center">82.38</td>
    <td style="border-left: 1px solid #ddd;" align="center">75.57</td>
    <td align="center">70.91</td>
    <td style="border-left: 1px solid #ddd;" align="center">70.42</td>
    <td align="center">60.19</td>
  </tr>
  <tr>
    <td>WaveNet</td>
    <td align="center">60.87</td>
    <td align="center">58.94</td>
    <td style="border-left: 1px solid #ddd;" align="center">90.70</td>
    <td align="center">42.66</td>
    <td style="border-left: 1px solid #ddd;" align="center">70.84</td>
    <td align="center">48.30</td>
    <td style="border-left: 1px solid #ddd;" align="center">70.55</td>
    <td align="center">48.30</td>
  </tr>
  <tr>
    <td>LSTM-CNN on Raw Audio</td>
    <td align="center">66.79</td>
    <td align="center">59.08</td>
    <td style="border-left: 1px solid #ddd;" align="center">92.94</td>
    <td align="center">95.68</td>
    <td style="border-left: 1px solid #ddd;" align="center">73.16</td>
    <td align="center">75.77</td>
    <td style="border-left: 1px solid #ddd;" align="center">80.30</td>
    <td align="center">61.34</td>
  </tr>
  <tr>
    <td>Wav2Vec2</td>
    <td align="center">56.21</td>
    <td align="center">58.94</td>
    <td style="border-left: 1px solid #ddd;" align="center">61.39</td>
    <td align="center">58.20</td>
    <td style="border-left: 1px solid #ddd;" align="center">58.62</td>
    <td align="center">60.80</td>
    <td style="border-left: 1px solid #ddd;" align="center">58.82</td>
    <td align="center">60.80</td>
  </tr>
  <tr>
    <td>SpectroCNN</td>
    <td align="center">68.25</td>
    <td align="center">65.22</td>
    <td style="border-left: 1px solid #ddd;" align="center">91.07</td>
    <td align="center">82.14</td>
    <td style="border-left: 1px solid #ddd;" align="center">72.64</td>
    <td align="center">70.59</td>
    <td style="border-left: 1px solid #ddd;" align="center">76.51</td>
    <td align="center">72.55</td>
  </tr>
</table>
