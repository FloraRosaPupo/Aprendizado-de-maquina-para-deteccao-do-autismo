"""
 ----------------- File: main.py ---------------------+
|			    DESCRICAO DO ARQUIVO			      |
|   Diagnóstico de Autismo através de Reconhecimento  |
|      e Padrões Faciais com Aprendizado de Máquina   |
|                                                     |
|   Implementado por Flora Rosa, Sabrina Guimaraes 	  |
|                     e Grazielle Stefane             |
+-----------------------------------------------------+ 
""" 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import random
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(os.listdir("input"))

filenames = os.listdir("input/AutismDataset/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'Autistic':
        categories.append('Autistic')
    else:
        categories.append('Non_Autistic')

train_df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
train_df['category'].value_counts().plot.bar()

test_filenames = os.listdir("input/AutismDataset/test")
categories = []
for filename in test_filenames:
    category = filename.split('.')[0]
    if category == 'Autistic':
        categories.append('Autistic')
    else:
        categories.append('Non_Autistic')

test_df = pd.DataFrame({
    'filename': test_filenames,
    'category': categories
})
test_df.head()

sample = random.choice(filenames)
image = load_img("input/AutismDataset/train/" + sample)
plt.imshow(image)

image_size = 224
input_shape = (image_size, image_size, 3)

# Hyperparameters
epochs = 30
batch_size = 64

pre_trained_model = VGG19(input_shape=input_shape, include_top=False, weights="imagenet")

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output

x = GlobalMaxPooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=1e-3, momentum=0.9),
              metrics=['accuracy'])

model.summary()

train_df, validate_df = train_test_split(train_df, test_size=0.1)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "input/AutismDataset/train",
    x_col='filename',
    y_col='category',
    class_mode='categorical',
    target_size=(image_size, image_size),
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "input/AutismDataset/train",
    x_col='filename',
    y_col='category',
    class_mode='categorical',
    target_size=(image_size, image_size),
    batch_size=batch_size
)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=total_train // batch_size,
    validation_steps=total_validate // batch_size
)

# Avaliação do modelo no conjunto de validação
y_true = validation_generator.classes
y_pred = np.argmax(model.predict(validation_generator), axis=1)

# Matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Acurácia
accuracy = accuracy_score(y_true, y_pred)
print("Acurácia:", accuracy)

# Report de classificação
print("Relatório de Classificação:")
print(classification_report(y_true, y_pred))

model.save('vgg19.h5')
