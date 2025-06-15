# Automatic Speech Recognition (ASR) System

An end-to-end deep learning implementation for automatic speech recognition using Connectionist Temporal Classification (CTC) loss and CNN-RNN hybrid architecture.

## Overview

This project implements a state-of-the-art automatic speech recognition system that converts spoken audio into text transcriptions. The system employs a hybrid CNN-RNN architecture with CTC alignment, following established methodologies in speech processing and deep learning.

### Key Features

- **Robust Audio Preprocessing**: Complete pipeline for audio normalization, augmentation, and feature extraction
- **Mel Spectrogram & MFCC Processing**: Advanced spectral feature extraction optimized for human speech
- **Hybrid CNN-RNN Architecture**: Combines convolutional feature extraction with recurrent sequence modeling
- **CTC Alignment**: Automatic sequence alignment without requiring manual character-level annotations
- **Comprehensive Evaluation**: Word Error Rate (WER) and Character Error Rate (CER) metrics

## Architecture

The system implements a multi-stage deep learning pipeline:

1. **Convolutional Feature Extraction**: Residual CNN layers process Mel spectrogram inputs to generate feature maps
2. **Recurrent Sequence Modeling**: Bidirectional LSTM layers convert continuous feature representations into discrete character sequences
3. **CTC Decoding**: Connectionist Temporal Classification for automatic sequence alignment and character prediction

## Project Structure

```
speech-text/
├── README.md                 # Project documentation
├── input/                    # Dataset directory
│   ├── LibriSpeech/         # LibriSpeech corpus
│   │   ├── dev-clean/       # Development set (clean)
│   │   └── test-clean/      # Test set (clean)
│   └── *.tar.gz            # Compressed dataset files
├── model/                   # Trained model artifacts
│   └── best_model_1.00.pt  # Best performing model checkpoint
├── notebook/                # Jupyter notebooks for analysis
│   ├── model_test.ipynb    # Model testing and evaluation
│   └── test.ipynb          # Experimental notebooks
└── src/                    # Source code
    ├── data_info.py
    ├── datapr.py           # Data preprocessing pipeline
    ├── model.py            # Neural network architecture
    └── train.py            # Training loop implementation
```

## Data Processing Pipeline

### Audio Preprocessing

- **Standardization**: Uniform sampling rate, channel configuration, and duration normalization
- **Quality Enhancement**: Noise reduction algorithms for improved signal clarity
- **Data Augmentation**: Time shifting, pitch modulation, and speed variation

### Feature Extraction

- **Mel Spectrograms**: Frequency domain representation optimized for human auditory perception
- **MFCC (Optional)**: Compressed spectral coefficients focusing on speech-relevant frequencies
- **SpecAugment**: Frequency and time masking for improved model generalization

### Label Processing

- Character-level vocabulary construction from transcription data
- Automatic sequence alignment using CTC methodology

## Model Architecture

### CNN Feature Extractor

- Multiple residual convolutional layers
- Processes 2D spectrogram inputs
- Generates hierarchical feature representations

### RNN Sequence Processor

- Bidirectional LSTM layers
- Converts continuous features to discrete character sequences
- Handles variable-length input sequences

### CTC Integration

- **Training**: CTC loss maximizes probability of correct transcriptions
- **Inference**: CTC decoding produces most likely character sequences
- Handles alignment challenges without manual annotation

## Dataset

This implementation uses the **LibriSpeech** corpus, a large-scale dataset of read English speech:

- **Development Set**: Clean speech samples for model validation
- **Test Set**: Clean speech samples for final evaluation
- **Format**: 16kHz WAV files with corresponding text transcriptions

## Evaluation Metrics

### Word Error Rate (WER)

Primary evaluation metric calculating the percentage of word-level errors:

```
WER = (Substitutions + Insertions + Deletions) / Total Words × 100%
```

## Technical Implementation

### CTC Algorithm

- **Training Phase**: Maximizes probability of generating correct transcriptions
- **Inference Phase**: Decodes most probable character sequences
- **Blank Character**: Handles character boundaries and repetitions automatically

### Sequence Alignment

Addresses fundamental ASR challenges:

- Variable character durations in speech
- Silence and pause handling
- Character repetition disambiguation
- Automatic boundary detection
