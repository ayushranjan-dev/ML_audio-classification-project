import gradio as gr
import numpy as np
from keras.models import load_model
from data_loader import extract_features_from_file
from config import MODEL_DIR, MODEL_NAME, SAMPLES, SAMPLE_RATE


model = load_model(f"{MODEL_DIR}/{MODEL_NAME}", compile=True)

def predict_audio(audio_path):
    x = extract_features_from_file(audio_path, duration=SAMPLES / SAMPLE_RATE)
    x = np.expand_dims(x, axis=0)

    prob = float(model.predict(x)[0][0])
    if prob >= 0.5:
        prediction = "Not Car"
        confidence = prob
    else:
        prediction = "Car"
        confidence = 1 - prob

    if confidence < 0.7:
        prediction = "Unknown"

    return f"Prediction: {prediction}\nConfidence: {confidence:.2f}"

ui = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(type="filepath", label="Upload or Record Sound"),
    outputs="text",
    title="Car Sound Detector 🔊🚗",
    description="Upload or record an audio clip to check if it contains a car sound."
)

if __name__ == "__main__":
    ui.launch(share=True)
    