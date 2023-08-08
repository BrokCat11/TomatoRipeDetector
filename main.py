import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from PIL import Image
import numpy as np

# Définir les classes d'images
classes = ['Tomate en cours de maturation','Tomate mûre','Tomate pas mûre']

# Définir les chemins vers les dossiers d'images
train_dir = 'cnn_database/test'
test_dir = 'cnn_database/test'

# Préparer les données d'entrée pour le modèle
input_shape = (224, 224, 3)

# Utiliser des générateurs d'images pour charger les données d'entraînement et de test
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=input_shape[:2], 
                                                    batch_size=32, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=input_shape[:2], 
                                                  batch_size=32, class_mode='categorical')

# Créer le modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_generator, epochs=5, validation_data=test_generator)

# Charger une image à prédire
image = Image.open('imgs\IMG007.jpg')
image = image.resize(input_shape[:2])
image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Prédire la classe de l'image
predictions = model.predict(image)
class_idx = np.argmax(predictions[0])
predicted_class = classes[class_idx]

# Vérifier si la classe prédite appartient à l'une des classes spécifiées
if predicted_class in classes:
    # Afficher la classe prédite
    print('Predicted class:', predicted_class)
else:
    # La classe prédite n'appartient à aucune des classes spécifiées
    print('Aucune classe détectée')
