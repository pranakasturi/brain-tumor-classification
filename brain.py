# Brain Tumor Classification: Custom CNN and MobileNetV2

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Set paths
data_dir = r'C:\Users\Vinit\labmentix\brainTumor\Tumour'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'test')

# Parameters
img_height, img_width = 224, 224
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# Callbacks
checkpoint_custom = ModelCheckpoint('models/custom_cnn_best.h5', monitor='val_loss', save_best_only=True, verbose=1)
checkpoint_mobilenet = ModelCheckpoint('models/mobilenetv2_best.h5', monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ----------------------
# Custom CNN Model
# ----------------------
custom_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

custom_model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Train Custom CNN
history_custom = custom_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=[early_stop, checkpoint_custom],
    verbose=1
)

# ----------------------
# MobileNetV2 Model
# ----------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

mobilenet_model = Model(inputs=base_model.input, outputs=predictions)

mobilenet_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# Train MobileNetV2
history_mobilenet = mobilenet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=[early_stop, checkpoint_mobilenet],
    verbose=1
)

# ----------------------
# Evaluation
# ----------------------
def evaluate_model(model, name):
    val_labels = val_generator.classes
    val_class_names = list(val_generator.class_indices.keys())

    preds = np.argmax(model.predict(val_generator), axis=1)
    print(f"\n{name} Classification Report:")
    print(classification_report(val_labels, preds, target_names=val_class_names))

    cm = confusion_matrix(val_labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=val_class_names, yticklabels=val_class_names, cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Load best models
best_custom = load_model('models/custom_cnn_best.h5')
best_mobilenet = load_model('models/mobilenetv2_best.h5')

evaluate_model(best_custom, "Custom CNN")
evaluate_model(best_mobilenet, "MobileNetV2")
