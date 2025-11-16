# TTS From Scratch

A PyTorch implementation of Tacotron 2 for text-to-speech synthesis from scratch.

## Overview

This project implements Tacotron 2, a neural text-to-speech model that generates mel-spectrograms from text, which can then be converted to audio waveforms. The implementation is built from scratch using PyTorch and trained on the LJSpeech dataset.

## Architecture

Tacotron 2 consists of two main components:

1. **Encoder**: Processes input text sequences into a learned representation
2. **Decoder**: Generates mel-spectrograms from the encoded text representation using attention mechanisms

## Requirements

- Python >= 3.13
- PyTorch
- librosa >= 0.11.0
- pandas >= 2.3.3
- scikit-learn

## Installation

Install dependencies using uv:

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

## Dataset

This project uses the LJSpeech dataset. The dataset should be extracted in the project root directory.

## Data Preparation

Prepare and split the dataset:

```bash
uv run python src/preperation_split.py --path_to_ljspeech LJSpeech-1.1 --path_to_save data/
```

This script:
- Loads metadata from the LJSpeech dataset
- Validates audio file paths
- Computes audio durations
- Splits data into train and test sets
- Saves metadata CSV files to the specified output directory

## Project Structure

```
.
├── src/
│   └── preperation_split.py    # Data preparation and splitting script
├── data/                        # Processed metadata files
├── LJSpeech-1.1/                # Dataset directory
├── pyproject.toml               # Project dependencies
└── README.md
```

## Status

Currently implementing Tacotron 2 architecture and training pipeline.

## License

[Add your license here]

