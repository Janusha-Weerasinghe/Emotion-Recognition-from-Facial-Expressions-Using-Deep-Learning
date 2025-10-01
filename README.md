# Emotion-Recognition-from-Facial-Expressions-Using-Deep-Learning
Hosted link - https://huggingface.co/spaces/Janusha/Emotion_Recognition_from_Facial_Expressions


A Python-based **Facial Emotion Recognition** system using **Convolutional Neural Networks (CNN)** and **MobileNetV2**, with a **real-world Tkinter GUI** for interactive testing. This project allows users to load images and see predicted emotions with confidence scores.

Features

- Predicts facial emotions: `Angry`, `Happy`, `Sad`, `Neutral`.
- Supports both:
  - Custom **CNN model** trained on FER2013 dataset.
  - Pre-trained **MobileNetV2 model** for emotion recognition.
- **Tkinter GUI**:
  - Load images interactively.
  - Display emotion prediction and confidence.
  - Show the image with overlayed results.
- Training scripts include:
  - Data augmentation.
  - Model checkpointing.
  - Early stopping and learning rate reduction.
  - Training & validation metric plots.
  - PDF classification reports.

## Requirements

Python 3.8+ and the following packages:
tensorflow
numpy
opencv-python
scikit-learn
matplotlib
Pillow
tkinter

## Folder Structure

<img width="360" height="406" alt="image" src="https://github.com/user-attachments/assets/1494b78c-e629-47f6-83cf-7144429a16c7" />

##Result


MobileNetV2
<img width="646" height="799" alt="image" src="https://github.com/user-attachments/assets/8bc93de5-df98-4616-b5b2-be7325929766" />

<img width="643" height="808" alt="image" src="https://github.com/user-attachments/assets/cf5e02c1-50ab-4eb8-ba6b-0e68fd44155c" />

<img width="675" height="815" alt="image" src="https://github.com/user-attachments/assets/fab63e8d-ebc4-4912-a798-67568cfac418" />

<img width="670" height="810" alt="image" src="https://github.com/user-attachments/assets/d50e9304-7616-440b-897a-f83bbf841647" />

Custom CNN
<img width="660" height="811" alt="image" src="https://github.com/user-attachments/assets/44f3dd3d-4212-44e0-b13d-54515cd6d3eb" />

<img width="671" height="812" alt="image" src="https://github.com/user-attachments/assets/6cad8c40-01a8-4f58-9fea-3a5fbe5e9577" />

<img width="661" height="826" alt="image" src="https://github.com/user-attachments/assets/38d7f86a-6d97-44c0-b088-962c253ce059" />

<img width="681" height="812" alt="image" src="https://github.com/user-attachments/assets/8a8edff8-96ab-44d3-aa10-d62381ef7a97" />



## References

1. [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
2. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
3. TensorFlow & Keras Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)


