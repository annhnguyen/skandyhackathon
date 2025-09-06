import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# classes
classes = ['cardboard', 'food', 'glass','metal','paper', 'plastic', 'textiles','misc']
num_classes=len(classes)

# image size
my_img_size = (128, 128)
my_batch_size = 32

#training data - normalize data + augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

#get training data
train_data = train_datagen.flow_from_directory(
    "training_trash_data",
    target_size=my_img_size,
    batch_size=my_batch_size,
    class_mode="categorical"
)

#testing data -normalize
test_datagen = ImageDataGenerator(rescale=1.0/255)

#get testing data
test_data = test_datagen.flow_from_directory(
    "testing_trash_data",
    target_size=my_img_size,
    batch_size=my_batch_size,
    class_mode="categorical"
)

#building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# train model
model.fit(train_data, validation_data=test_data, epochs=10)

# save trained model
model.save("model.h5")

loss, accuracy = model.evaluate(test_data)
print("Test accuracy: ", accuracy)