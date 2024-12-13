import requests
import zipfile
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# Set global seeds
np.random.seed(42)
tf.random.set_seed(42)

# Download the file
# url = 'https://surfdrive.surf.nl/files/index.php/s/GsH5DxUdBgDR64B/download' # 75 x 75 pixels
url = 'https://surfdrive.surf.nl/files/index.php/s/B8emtQRGUeAaqmz/download' # 150 x 150 pixels
response = requests.get(url)

# Save the file locally
zip_filename = 'data.zip'
with open(zip_filename, 'wb') as f:
    f.write(response.content)

# Unzip the file
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('unzipped_data')

# Get the names of the extracted files
extracted_files = os.listdir('unzipped_data')

# Load the NPY files containing the images and labels
image_file_path = os.path.join('unzipped_data', 'images.npy')
label_file_path = os.path.join('unzipped_data', 'labels.npy')
images = np.load(image_file_path)
labels = np.load(label_file_path)

# Define a function to display an image
def display_image(image_data):
    plt.imshow(image_data, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

# Pick out three random sample indices
sample1 = np.random.randint(images.shape[0])
sample2 = np.random.randint(images.shape[0])
sample3 = np.random.randint(images.shape[0])

# Display the three sample images and their labels
display_image(images[sample1])
print(labels[sample1])
display_image(images[sample2])
print(labels[sample2])
display_image(images[sample3])
print(labels[sample3])

# Image dimensions
img_rows, img_cols = images.shape[1:]

# Adjust the input shape based on the image data format ('channels_first' or 'channels_last')
if keras.backend.image_data_format() == 'channels_first':
    images = images.reshape(images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    images = images.reshape(images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert image array to float
images = images.astype('float32')

# Normalize the pixel intensities to [0, 1] range
images /= 255

# Generate a random permutation of indices based on the number of samples
shuffle_indices = np.random.permutation(images.shape[0])

# Shuffle both images and labels using the same indices
shuffled_images = images[shuffle_indices]
shuffled_labels = labels[shuffle_indices]

n_intervals = 60  # Set number of intervals
intervals = np.linspace(0, 60, n_intervals + 1)  # Divide range [0, 60] into equal intervals

# Create array with time classes
classes = [(h, min) for h in range(12) for min in intervals[:-1]]
classes = np.array(classes)
classes = classes.astype('int64')

# Number of classes
n_classes = classes.shape[0]

# Extract array with hour values only (for subsequent mapping)
classes_h = [c[0] for c in classes]

# Create class mapping
shuffled_labels_classes = [
    classes_h.index(h) + np.digitize(min, intervals) - 1
    for h, min in shuffled_labels
]
shuffled_labels_classes = np.array(shuffled_labels_classes)

# Combine class labels and actual times
y_data = np.hstack([
    shuffled_labels_classes.reshape(-1, 1),  # Class labels
    shuffled_labels  # Actual time values (hours, minutes)
])

# Split 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(shuffled_images, y_data, test_size=0.2, random_state=42)

# Split the remaining 20% into 50% validation, 50% test (i.e., 10% each of the original data)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the CNN model architecture
input_ = keras.layers.Input(shape=input_shape, name='input')
conv1 = keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', name='conv1')(input_)
conv2 = keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu', name='conv2')(conv1)
maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool1')(conv2)
conv3 = keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', name='conv3')(maxpool1)
conv4 = keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', name='conv4')(conv3)
maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2')(conv4)
conv5 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='conv5')(maxpool2)
conv6 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='conv6')(conv5)
maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool3')(conv6)
flatten = keras.layers.Flatten(name='flatten')(maxpool3)
dense1 = keras.layers.Dense(128, activation='relu', name='dense1')(flatten)
dropout1 = keras.layers.Dropout(0.3, name='dropout1')(dense1)
dense2 = keras.layers.Dense(64, activation='relu', name='dense2')(dropout1)
dropout2 = keras.layers.Dropout(0.2, name='dropout2')(dense2)
output = keras.layers.Dense(n_classes, activation='softmax', name='output')(dropout2)
model = keras.Model(inputs=[input_], outputs=[output])

model.summary()

def custom_loss(y_true, y_pred):

    # Extract class labels
    y_true_labels = tf.cast(y_true[:, 0], tf.int32)

    # Compute sparse categorical cross-entropy loss
    loss = keras.losses.sparse_categorical_crossentropy(y_true_labels, y_pred)

    return loss

# Convert the 'classes' array to a TensorFlow constant (this format is needed in the following function)
classes = tf.constant(classes, dtype=tf.int32)

# Define custom metric function using TensorFlow operations
def common_sense_error(y_true, y_pred_probs):

    # Extract actual times
    y_true_times = tf.cast(y_true[:, 1:], tf.int32)  # Hours and minutes

    # Map predicted class indices to time values
    y_pred_class = tf.argmax(y_pred_probs, axis=1, output_type=tf.int32)
    y_pred_times = tf.gather(classes, y_pred_class)

    # Extract hours and minutes for true and predicted times
    y_true_h, y_true_min = y_true_times[:, 0], y_true_times[:, 1]
    y_pred_h, y_pred_min = y_pred_times[:, 0], y_pred_times[:, 1]

    # Compute deltas for hours and minutes
    h_delta = y_true_h - y_pred_h
    min_delta = y_true_min - y_pred_min

    # Determine t1 and t2 based on h_delta and min_delta
    cond = tf.logical_or(h_delta > 0, tf.logical_and(h_delta == 0, min_delta >= 0))
    t1_h, t1_min = tf.where(cond, y_pred_h, y_true_h), tf.where(cond, y_pred_min, y_true_min)
    t2_h, t2_min = tf.where(cond, y_true_h, y_pred_h), tf.where(cond, y_true_min, y_pred_min)

    # Calculate hour distances and select the minimum distance
    h_dist1 = t2_h - t1_h
    h_dist2 = (12 - t2_h) + t1_h
    use_h_dist1 = h_dist1 <= h_dist2
    h_dist = tf.where(use_h_dist1, h_dist1, h_dist2)

    # Calculate minute distance
    min_dist = tf.where(
        use_h_dist1,
        tf.where(t1_min > t2_min, (60 - t1_min) + t2_min, t2_min - t1_min),
        tf.where(t2_min > t1_min, (60 - t2_min) + t1_min, t1_min - t2_min)
    )

    # Adjust hour distance if needed
    cond = tf.logical_or(tf.logical_and(use_h_dist1, t1_min > t2_min), tf.logical_and(~use_h_dist1, t2_min > t1_min))
    h_dist = tf.where(cond, h_dist - 1, h_dist)

    # Calculate total distance in minutes
    total_dist_min = h_dist * 60 + min_dist

    # Return the mean distance for the batch
    return tf.reduce_mean(total_dist_min)

# Compile the model
model.compile(
    loss=custom_loss,
    optimizer=keras.optimizers.Adam(learning_rate=1e-4, weight_decay=4e-4),
    metrics=[common_sense_error]
)

# Define the checkpoint callback to save the best model weights based on validation loss
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'tell_the_time_CNN_classification.keras',
    save_best_only=True
)

# Define early stopping callback to stop training when validation loss stops improving
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=20, # patience=10 for 24 classes; patience=20 for 240/720 classes
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=600,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Convert the training history to a dataframe
df = pd.DataFrame(history.history)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 'loss' and 'val_loss' on the first subplot
axes[0].plot(df['loss'], label='Training')
axes[0].plot(df['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Plot 'common_sense_error' and 'val_common_sense_error' on the second subplot
axes[1].plot(df['common_sense_error'], label='Training')
axes[1].plot(df['val_common_sense_error'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Common Sense Error (min)')
axes[1].legend()
axes[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show both plots
plt.show()

# Evaluate model performance on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test common sense error:', score[1])

# Save the model
model.save('tell_the_time_CNN_classification.keras')