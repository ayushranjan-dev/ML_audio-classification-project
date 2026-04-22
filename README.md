# Audio Classification System

## Overview
This project is a machine learning-based audio classification system that classifies sound events from `.wav` files using a Convolutional Neural Network (CNN) and MFCC feature extraction.

## Features
- Audio preprocessing using MFCC
- CNN-based model for classification
- Training and evaluation pipeline
- Prediction on custom audio files

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Librosa

## Project Structure
- train.py → model training
- predict.py → prediction on audio files
- model.py → CNN architecture
- data_loader.py → feature extraction
- evaluate.py → performance evaluation

## Setup
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Train model:
```bash
python train.py
```


## Predict:
```bash
python predict.py demo_files/57596-3-1-0.wav
```

## Note

This is a learning project focused on understanding audio classification pipelines and ML workflows.
