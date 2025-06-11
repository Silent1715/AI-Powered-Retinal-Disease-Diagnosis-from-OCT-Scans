import os
os.environ['TF_LOGGING'] = '0'  # Reduce TensorFlow logging

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import io

# Configuration
num_classes = 4
img_rows, img_cols = 224, 224
batch_size = 16
train_data_dir = r'oct2017\OCT2017\train'
validation_data_dir = r'oct2017\OCT2017\test'
nb_train_samples = 83484  # Matches your dataset
nb_validation_samples = 968  # Matches your dataset
epochs = 10
model_path = 'retinal_cnn.keras'  # Keras format

# Check if model exists, otherwise train it
if not os.path.exists(model_path):
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Define the CNN model
    def cnn():
        model = Sequential([
            Input(shape=(img_rows, img_cols, 3)),
            Conv2D(64, (3, 3), activation="relu", padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dense(num_classes, activation="softmax"),
        ])
        return model

    # Build and compile model
    model = cnn()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.00001)
    callbacks = [earlystop, checkpoint, reduce_lr]

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
    )

    # Save the model
    model.save(model_path)
    print("Model saved as", model_path)
else:
    print("Model already exists at", model_path)

# Load the model
model = tf.keras.models.load_model(model_path)

# Get class labels
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

# Compute metrics
def compute_metrics():
    validation_generator.reset()
    y_pred = model.predict(validation_generator, steps=len(validation_generator))
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    cm = confusion_matrix(y_true, y_pred_label)
    cr = classification_report(y_true, y_pred_label, target_names=classes, output_dict=False)
    return cm, cr

confusion_matrix_result, classification_report_result = compute_metrics()

# Create interactive UI with ipywidgets
uploader = widgets.FileUpload(accept='.jpg,.png', multiple=False)
output = widgets.Output()
predict_button = widgets.Button(description="Classify Image")
metrics_button = widgets.Button(description="Show Metrics")

def on_predict_button_clicked(b):
    with output:
        clear_output()
        if uploader.value:
            # Get the uploaded image
            uploaded_file = uploader.value[0]  # Updated for ipywidgets 8.x
            img = Image.open(io.BytesIO(uploaded_file['content']))
            img = np.array(img)
            if len(img.shape) == 2:  # Convert grayscale to RGB if needed
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (img_rows, img_cols))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            prediction = model.predict(img)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_class = class_labels[predicted_class_idx]

            # Display result
            plt.imshow(img[0])
            plt.title(f"Predicted Class: {predicted_class}")
            plt.axis('off')
            plt.show()
            print(f"Predicted Class: {predicted_class}")
        else:
            print("Please upload an image.")

def on_metrics_button_clicked(b):
    with output:
        clear_output()
        print("Confusion Matrix:\n", confusion_matrix_result)
        print("\nClassification Report:\n", classification_report_result)

predict_button.on_click(on_predict_button_clicked)
metrics_button.on_click(on_metrics_button_clicked)

# Display widgets
display(uploader)
display(predict_button)
display(metrics_button)
display(output)