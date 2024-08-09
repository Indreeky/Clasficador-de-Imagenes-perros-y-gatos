from utils import db_connect
engine = db_connect()

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Función para descargar y extraer datos de Kaggle
def download_and_extract_kaggle_data():
    api = KaggleApi()
    api.authenticate()

    data_dir = 'data/raw'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test1')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print("Descargando train.zip...")
    api.competition_download_file('dogs-vs-cats', 'train.zip', path=data_dir)
    print("Descargando test1.zip...")
    api.competition_download_file('dogs-vs-cats', 'test1.zip', path=data_dir)

    train_zip_path = os.path.join(data_dir, 'train.zip')
    test_zip_path = os.path.join(data_dir, 'test1.zip')

    # Verificar si los archivos ZIP existen
    if os.path.exists(train_zip_path) and os.path.exists(test_zip_path):
        print("Extrayendo train.zip...")
        try:
            with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                zip_ref.extractall(train_dir)
        except zipfile.BadZipFile as e:
            print(f"Error al extraer {train_zip_path}: {e}")

        print("Extrayendo test1.zip...")
        try:
            with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
                zip_ref.extractall(test_dir)
        except zipfile.BadZipFile as e:
            print(f"Error al extraer {test_zip_path}: {e}")
    else:
        print("La descarga de los archivos ZIP falló.")

# Llamar a la función de descarga y extracción
download_and_extract_kaggle_data()

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
import os
from PIL import Image

# Directorios
train_dir = '../data/raw/train'
test_dir = '../data/raw/test1'

# Configuración de los generadores de datos con aumento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),  # Tamaño reducido para entrenamiento más rápido
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Generador de datos de validación
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),  # Tamaño reducido para entrenamiento más rápido
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Generador de datos de prueba (sin etiquetas)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='../data/raw/',
    classes=['test1'],
    target_size=(100, 100),  # Tamaño reducido para entrenamiento más rápido
    batch_size=32,
    class_mode=None,  # No hay etiquetas
    shuffle=False
)

# Mostrar algunas imágenes de entrenamiento
def plot_images(images_arr, labels_arr=None, preds_arr=None):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for i, (img, ax) in enumerate(zip(images_arr, axes)):
        ax.imshow(img)
        ax.axis('off')
        if labels_arr is not None:
            title = 'Cat' if labels_arr[i] == 0 else 'Dog'
            if preds_arr is not None:
                pred = 'Cat' if preds_arr[i] == 0 else 'Dog'
                title += f'/{pred}'
            ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Obtener un lote de imágenes de entrenamiento
images, labels = next(train_generator)
plot_images(images[:10], labels[:10])

# Crear el modelo usando MobileNetV2 como base
base_model = MobileNetV2(input_shape=(100, 100, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar las capas del modelo base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Configurar callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('model_transfer.keras', save_best_only=True)

# Entrenar el modelo
history = model.fit(
    train_generator, 
    validation_data=validation_generator, 
    epochs=15,  # Entrenar durante 15 épocas
    callbacks=[early_stopping, model_checkpoint]
)

# Graficar pérdida y precisión
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history)

# Cargar el mejor modelo guardado
model.load_weights('model_transfer.keras')

# Realizar predicciones sobre el conjunto de pruebas
predictions = model.predict(test_generator)
predicted_classes = [1 if p > 0.5 else 0 for p in predictions]  # Ajustar según sea perro (1) o gato (0)

# Opcional: Mostrar algunas predicciones
test_images = next(test_generator)  # Solo obtenemos las imágenes sin etiquetas
plot_images(test_images[:10], preds_arr=predicted_classes[:10])
print(predicted_classes[:10])
