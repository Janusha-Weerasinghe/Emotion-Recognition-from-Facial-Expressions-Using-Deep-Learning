import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALIDATION_DIR = os.path.join(DATASET_DIR, 'test')

IMG_SIZE = (96, 96)
BATCH_SIZE = 64
NUM_CLASSES = 4
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Neutral']

def build_model(input_shape=(96, 96, 1), num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)
    x = Concatenate(axis=-1)([inputs, inputs, inputs])  # grayscale to 3 channels
    base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Starting training process...")
    # Image data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load images from folder structure
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

    model = build_model()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    epochs = 20

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr]
    )

    # Save the trained model to desktop folder inside PyCharm project folder
    model_save_path = os.path.join(BASE_DIR, 'mobilenetv2_fer2013.h5')
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

    # Evaluate performance on validation data
    val_steps = validation_generator.samples // BATCH_SIZE
    validation_generator.reset()
    preds = model.predict(validation_generator, steps=val_steps, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = validation_generator.classes[:val_steps * BATCH_SIZE]

    accuracy = np.mean(pred_labels == true_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=range(NUM_CLASSES))

    print(f"Validation Accuracy: {accuracy:.4f}")
    for i, label in enumerate(CLASS_NAMES):
        print(f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")

    # Save classification report as a PDF file
    pdf_path = os.path.join(BASE_DIR, 'classification_report.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.axis('off')
        txt = f"Validation Accuracy: {accuracy:.4f}\n\n"
        txt += "Class-wise Metrics:\n\n"
        for i, label in enumerate(CLASS_NAMES):
            txt += f"{label}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}\n"
        ax.text(0.01, 0.99, txt, verticalalignment='top', fontsize=12)
        pdf.savefig(fig)
        plt.close()
    print(f"Classification report saved as {pdf_path}")

if __name__ == '__main__':
    main()
