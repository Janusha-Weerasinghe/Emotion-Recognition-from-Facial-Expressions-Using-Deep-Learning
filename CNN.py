import os
import tensorflow as tf
import sys
import subprocess
import matplotlib
matplotlib.use('Agg')  # Fix for Tkinter issues

# Install scikit-learn if not installed
try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALIDATION_DIR = os.path.join(DATASET_DIR, 'test')

# Constants
IMG_SIZE = (96, 96)
BATCH_SIZE = 64
NUM_CLASSES = 4
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Neutral']

# CNN Model
def build_cnn(input_shape=(96, 96, 1), num_classes=NUM_CLASSES):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training(history, save_path):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved at {save_path}")

def main():
    print("Starting CNN training...")

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False
    )

    model = build_cnn()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,  # Adjust for 1-hour training limit
        callbacks=[early_stop, reduce_lr]
    )

    # Save CNN model
    model_save_path = os.path.join(BASE_DIR, 'cnn_fer2013.h5')
    model.save(model_save_path)
    print(f"CNN model saved at {model_save_path}")

    # Plot training curves
    plot_path = os.path.join(BASE_DIR, 'cnn_training_curve.png')
    plot_training(history, plot_path)

    # Evaluate
    validation_generator.reset()
    preds = model.predict(validation_generator, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = validation_generator.classes[:len(pred_labels)]

    accuracy = np.mean(pred_labels == true_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(NUM_CLASSES))

    print(f"Validation Accuracy: {accuracy:.4f}")
    for i, label in enumerate(CLASS_NAMES):
        print(f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")

    # Save classification report
    pdf_path = os.path.join(BASE_DIR, 'cnn_classification_report.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.axis('off')
        txt = f"Validation Accuracy: {accuracy:.4f}\n\nClass-wise Metrics:\n\n"
        for i, label in enumerate(CLASS_NAMES):
            txt += f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}\n"
        ax.text(0.01, 0.99, txt, verticalalignment='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    print(f"CNN classification report saved as {pdf_path}")

if __name__ == '__main__':
    main()
