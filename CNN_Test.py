import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# ----------------- Constants -----------------
MODEL_PATH = "cnn_fer2013.h5"  # CNN model trained from your code
IMG_SIZE = (96, 96)
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Neutral']

# ----------------- Functions -----------------
def preprocess_frame(frame):
    """Convert to grayscale, resize, normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, IMG_SIZE[0], IMG_SIZE[1], 1))
    return reshaped

def predict_emotion(model, frame):
    """Run CNN model prediction."""
    preprocessed = preprocess_frame(frame)
    pred = model.predict(preprocessed)
    emotion = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred)
    return emotion, confidence

def load_image():
    """Open file dialog and predict emotion."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    frame = cv2.imread(file_path)
    if frame is None:
        messagebox.showerror("Error", "Failed to load image.")
        return

    emotion, confidence = predict_emotion(model, frame)
    result_label.config(text=f"Predicted: {emotion} ({confidence:.2%})")

    # Convert OpenCV image to Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_pil = img_pil.resize((400, 400))
    img_tk = ImageTk.PhotoImage(img_pil)

    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep reference

# ----------------- Load Model -----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print(f"Loaded CNN model from {MODEL_PATH}")

# ----------------- Tkinter UI -----------------
root = tk.Tk()
root.title("CNN Emotion Recognition")
root.geometry("500x600")
root.resizable(False, False)

btn_load = tk.Button(root, text="Load Image", command=load_image, font=("Arial", 14))
btn_load.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
