import streamlit as st
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image
from io import BytesIO
import requests
import time
import os
from torchvision.models import resnet18, ResNet18_Weights
import json

# Загрузка словаря классов из JSON файла
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Путь к файлу JSON с вашим словарем классов
file_path = 'class_names.json'
ind2class = load_class_names(file_path)

# Загрузка модели
class Myresnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, 11)
        for i in self.model.parameters():
            i.requires_grad = False
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True
    def forward(self, x):
        x = self.model(x)
        return x

device = 'cpu'
model = Myresnet18()
model.to(device)
model.load_state_dict(torch.load('resnet_weight_cpu.pt', map_location='cpu'))
model.to(device)
model.eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# Функция для загрузки изображения по URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

# Функция для предсказания
def predict_image(image):
    image = preprocess(image).unsqueeze(0).to(device)
    start_time = time.time()
    with torch.inference_mode():
        pred_class = model(image).argmax().item()
    prediction_time = time.time() - start_time
    return pred_class, prediction_time

# Функция для визуализации изображения
def show_image(image, title, prediction_time=None):
    st.image(image, caption=title, use_column_width=True)
    if prediction_time:
        st.write(f"Prediction time: {prediction_time:.4f} seconds")

# Функция для отображения графиков
def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(history['train_losses'], label='Train Loss')
    ax[0].plot(history['valid_losses'], label='Valid Loss')
    ax[0].set_title('Loss over Epochs')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(history['train_accs'], label='Train Accuracy')
    ax[1].plot(history['valid_accs'], label='Valid Accuracy')
    ax[1].set_title('Accuracy over Epochs')
    ax[1].grid(True)
    ax[1].legend()

    st.pyplot(fig)

def show_metrics(history, confusion_matrix):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    ax[0].plot(history['train_f1s'], label='Train F1 Score')
    ax[0].plot(history['valid_f1s'], label='Valid F1 Score')
    ax[0].set_title('F1 Score over Epochs')
    ax[0].grid(True)
    ax[0].legend()

    sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=ax[1])
    ax[1].set_title('Confusion Matrix')

    st.pyplot(fig)

# Основное приложение
st.title('Image Classifier with ResNet18')

page = st.sidebar.selectbox('Choose a page', ['Classify Images', 'Training Information'])

if page == 'Classify Images':
    st.header('Classify Images')
    
    option = st.selectbox('Choose how to upload image', ['From URL', 'From File'])
    
    if option == 'From URL':
        url = st.text_input('Image URL', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQaAs15lzVwOEZ9PZ2O2nl6g8gcSmDS7mjRg&s')
        if st.button('Classify'):
            image = load_image_from_url(url)
            pred_class, prediction_time = predict_image(image)
            class_name = ind2class[str(pred_class)]
            show_image(image, f"Predicted: {class_name}", prediction_time)

    elif option == 'From File':
        uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert('RGB')
                pred_class, prediction_time = predict_image(image)
                class_name = ind2class[str(pred_class)]
                show_image(image, f"Predicted: {class_name}", prediction_time)

elif page == 'Training Information':
    st.header('Training Information')
    
    st.subheader('Training History')
    # Загрузите историю обучения из файла или используйте логи
    history = json.load(open('training_history.json'))
    plot_history(history)

    st.subheader('Dataset Information')
    # Загрузите информацию о датасете
    dataset_info = {
        'Number of classes': 10,
        'Number of training images': 5496,
        'Number of validation images': 1390,
        'Training time, m': 8.57
    }
    st.write(dataset_info)
    
    st.subheader('F1 Scores and Confusion Matrix')
    # Загрузите confusion matrix из файла
    confusion_matrix = np.load('confusion_matrix.npy')
    show_metrics(history, confusion_matrix)
