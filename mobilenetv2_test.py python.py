import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ----------------- Constants -----------------
MODEL_PATH = "mobilenetv2_fer2013.h5"  # Model in same folder
IMG_SIZE = (96, 96)
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Neutral']


# ----------------- Functions -----------------
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, IMG_SIZE[0], IMG_SIZE[1], 1))
    return reshaped


def predict_emotion(model, frame):
    preprocessed = preprocess_frame(frame)
    pred = model.predict(preprocessed)
    emotion = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred)
    return emotion, confidence


def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    frame = cv2.imread(file_path)
    if frame is None:
        tk.messagebox.showerror("Error", "Failed to load image.")
        return

    # Predict emotion
    emotion, confidence = predict_emotion(model, frame)
    result_label.config(text=f"Predicted: {emotion} ({confidence:.2%})")

    # Convert OpenCV image to Tkinter image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_pil = img_pil.resize((400, 400))  # Resize for display
    img_tk = ImageTk.PhotoImage(img_pil)

    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep reference


# ----------------- Load Model -----------------
model = load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# ----------------- Tkinter UI -----------------
root = tk.Tk()
root.title("Emotion Recognition")
root.geometry("500x600")
root.resizable(False, False)

btn_load = tk.Button(root, text="Load Image", command=load_image, font=("Arial", 14))
btn_load.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
