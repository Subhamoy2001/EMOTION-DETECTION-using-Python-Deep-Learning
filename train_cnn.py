import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---- SETTINGS ----
original_train = "train"
original_test = "test"
fast_train = "train_fast"
fast_test = "test_fast"
img_size = (48,48)
batch_size = 16
epochs = 3    # Fast training
samples_per_class = 50
# -------------------

# Function to create a small subset
def create_fast_subset(original_dir, fast_dir):
    if os.path.exists(fast_dir):
        shutil.rmtree(fast_dir)
    os.makedirs(fast_dir)
    for class_name in os.listdir(original_dir):
        class_path = os.path.join(original_dir, class_name)
        fast_class_path = os.path.join(fast_dir, class_name)
        os.makedirs(fast_class_path)
        images = os.listdir(class_path)
        random.shuffle(images)
        for img_name in images[:samples_per_class]:
            shutil.copy(os.path.join(class_path,img_name), fast_class_path)
    print(f"Subset created at {fast_dir}")

# Create small subsets
create_fast_subset(original_train, fast_train)
create_fast_subset(original_test, fast_test)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    fast_train,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    fast_test,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1],1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=test_data, epochs=epochs)

# Save model
model.save("emotion_model_fast.h5")
print("✅ Fast model saved as emotion_model_fast.h5")

