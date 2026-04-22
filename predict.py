
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from data_loader import extract_features_from_file
from config import MODEL_DIR, MODEL_NAME, SAMPLES, SAMPLE_RATE

def main(audio_files):
  
    model = load_model(f"{MODEL_DIR}/{MODEL_NAME}", compile=True)
    print(f"Loaded model: {MODEL_NAME}\n")

 
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}\n")
            continue

      
        x = extract_features_from_file(audio_file, duration=SAMPLES / SAMPLE_RATE)
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        
        prob = float(model.predict(x)[0][0])
        print(f"File: {audio_file}")
        print(f"  Raw model output (prob of not_car): {prob:.6f}")

        
        if prob >= 0.5:
            prediction = "not_car"
            confidence = prob
        else:
            prediction = "car"
            confidence = 1 - prob

        
        if confidence < 0.7:
            prediction = "unknown"

        print(f"  Interpreted Prediction: {prediction} | Confidence: {confidence:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file1> [<audio_file2> ...]")
        sys.exit(1)
    main(sys.argv[1:])