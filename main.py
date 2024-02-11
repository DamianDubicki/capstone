import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

#Update path to where your artwork folder is
images_directory = r"C:\artwork"
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)  # Adjusted validation_split value

train_generator = datagen.flow_from_directory(
    images_directory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    images_directory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#print("Class indices:", train_generator.class_indices)
#print("Number of classes:", train_generator.num_classes)
#print("Number of images in training set:", train_generator.samples)

num_classes = len(train_generator.class_indices)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Use num_classes here
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=validation_generator)

model.save("painting_recognition_model")

def predict_painting(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training data

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    class_name = list(train_generator.class_indices.keys())[predicted_class_index]

    return class_name

#Update path to where your Newfolder folder is
new_image_path = r"C:\Users\damia\Downloads\Newfolder\test.png"
predicted_painting = predict_painting(new_image_path)
print(f"The predicted painting is: {predicted_painting}")
