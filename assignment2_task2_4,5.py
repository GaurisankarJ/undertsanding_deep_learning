#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: right">   </div>
# 
# 
# Introduction to Deep Learning (2024) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| &nbsp;
# -------|-------------------
# **Assignment 2 - Sequence processing using RNNs** | <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/UniversiteitLeidenLogo.svg/1280px-UniversiteitLeidenLogo.svg.png" width="300">
# 
# 
# 
# # Introduction
# 
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# 
# ### Description
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 

# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. We will use these operand signs together with the MNIST dataset to represent the digits.

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose


# In[16]:


from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates
        if sign == '*':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            # Rotate 45 degrees
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)
            blank_images[i] = rotate(blank_images[i], -50, reshape=False)
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA)

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))


# In[17]:


def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# # Creating our data
# 
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
# 
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
# 
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
# 
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]

# In[18]:


# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])


# ## Helper functions
# 
# The functions below will help with input/output of the data.

# In[39]:


# Split data into train and test sets
from sklearn.model_selection import train_test_split

# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=1)
    predicted = ''.join([unique_characters[i] for i in pred])

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)

X_text_onehot_train, X_text_onehot_test, y_text_onehot_train, y_text_onehot_test = train_test_split(
    X_text_onehot, y_text_onehot, test_size=0.1, random_state=42
)
print("X_text_onehot_train shape:", X_text_onehot_train.shape)
print("y_text_onehot_train shape:", y_text_onehot_train.shape)
print("X_text_onehot_test shape:", X_text_onehot_test.shape)
print("y_text_onehot_test shape:", y_text_onehot_test.shape)

X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
    X_img, y_img, test_size=0.1, random_state=42
)
print("X_img_train shape:", X_img_train.shape)
print("y_img_train shape:", y_img_train.shape)
print("X_img_test shape:", X_img_test.shape)
print("y_img_test shape:", y_img_test.shape)


# ---
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 2. TimeDistributed
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 

# In[20]:


def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text


# In[28]:


# Build the model
model_0 = build_text2text_model()

# Train the model
history_0 = model_0.fit(
    X_text_onehot_train, y_text_onehot_train,
    validation_data=(X_text_onehot_test, y_text_onehot_test),
    epochs=40, batch_size=64, verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model_0.evaluate(X_text_onehot_test, y_text_onehot_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[29]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_0.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_0.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 0 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_0.history['loss'], label='Training Loss', marker='o')
plt.plot(history_0.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 0 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[30]:


# Predict on the test set
y_pred = model_0.predict(X_text_onehot_test)

# Decode true and predicted labels into text
decoded_true = [decode_labels(y) for y in y_text_onehot_test]  # True labels
decoded_pred = [decode_labels(y) for y in y_pred]  # Predicted labels
decoded_test = [decode_labels(y) for y in X_text_onehot_test]

# Identify misclassified samples
misclassified_indices = [
    i for i, (true, pred) in enumerate(zip(decoded_true, decoded_pred))
    if true != pred
]

# Display results for a random selection of misclassified examples
for i in random.sample(misclassified_indices, min(10, len(misclassified_indices))):
    plt.figure(figsize=(8, 4))
    plt.suptitle(f"Sample {i} | True vs Predicted Text Comparison")

    # Display the query text
    plt.subplot(1, 2, 1)
    plt.title("Query")
    plt.text(0.5, 0.5, decoded_test[i], fontsize=12, ha='center', va='center', wrap=True)
    plt.axis('off')

    # Display the expected and predicted text
    plt.subplot(1, 2, 2)
    plt.title("Results")
    plt.text(0.5, 0.6, f"True: {decoded_true[i]}", fontsize=12, ha='center', va='center', wrap=True, color='green')
    plt.text(0.5, 0.4, f"Pred: {decoded_pred[i]}", fontsize=12, ha='center', va='center', wrap=True, color='red')
    plt.axis('off')

    plt.show()


# In[31]:


def build_text2text_model_1():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters)), return_sequences=True))
    
    text2text.add(LSTM(256))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text


# In[32]:


# Build the model
model_1 = build_text2text_model_1()

# Train the model
history_1 = model_1.fit(
    X_text_onehot_train, y_text_onehot_train,
    validation_data=(X_text_onehot_test, y_text_onehot_test),
    epochs=40, batch_size=64, verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model_1.evaluate(X_text_onehot_test, y_text_onehot_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[42]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_1.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_1.history['loss'], label='Training Loss', marker='o')
plt.plot(history_1.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 1 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[36]:


# Predict on the test set
y_pred = model_1.predict(X_text_onehot_test)

# Decode true and predicted labels into text
decoded_true = [decode_labels(y) for y in y_text_onehot_test]  # True labels
decoded_pred = [decode_labels(y) for y in y_pred]  # Predicted labels
decoded_test = [decode_labels(y) for y in X_text_onehot_test]

# Identify misclassified samples
misclassified_indices = [
    i for i, (true, pred) in enumerate(zip(decoded_true, decoded_pred))
    if true != pred
]

# Display results for a random selection of misclassified examples
for i in random.sample(misclassified_indices, min(10, len(misclassified_indices))):
    plt.figure(figsize=(8, 4))
    plt.suptitle(f"Sample {i} | True vs Predicted Text Comparison")

    # Display the query text
    plt.subplot(1, 2, 1)
    plt.title("Query")
    plt.text(0.5, 0.5, decoded_test[i], fontsize=12, ha='center', va='center', wrap=True)
    plt.axis('off')

    # Display the expected and predicted text
    plt.subplot(1, 2, 2)
    plt.title("Results")
    plt.text(0.5, 0.6, f"True: {decoded_true[i]}", fontsize=12, ha='center', va='center', wrap=True, color='green')
    plt.text(0.5, 0.4, f"Pred: {decoded_pred[i]}", fontsize=12, ha='center', va='center', wrap=True, color='red')
    plt.axis('off')

    plt.show()


# In[35]:


import matplotlib.pyplot as plt

# Create a 2x2 figure
plt.figure(figsize=(12, 8))

# Model 0 Accuracy
plt.subplot(2, 2, 1)
plt.plot(history_0.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_0.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 0 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Model 0 Loss
plt.subplot(2, 2, 2)
plt.plot(history_0.history['loss'], label='Training Loss', marker='o')
plt.plot(history_0.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 0 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Model 1 Accuracy
plt.subplot(2, 2, 3)
plt.plot(history_1.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Model 1 Loss
plt.subplot(2, 2, 4)
plt.plot(history_1.history['loss'], label='Training Loss', marker='o')
plt.plot(history_1.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 1 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 
# ---
# ---
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as input vectors) or recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.

# In[37]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def build_image2text_model():
    # Image Input (sequence of 5 grayscale images)
    image_input = Input(shape=(5, 28, 28, 1), name="Image_Input")
    
    # CNN Encoder for Images
    cnn_output = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(image_input)
    cnn_output = TimeDistributed(MaxPooling2D((2, 2)))(cnn_output)
    cnn_output = TimeDistributed(Flatten())(cnn_output)
    
    # LSTM Encoder
    encoded = LSTM(256)(cnn_output)

    # Repeat Vector for Decoder
    repeated_vector = RepeatVector(max_answer_length)(encoded)

    # LSTM Decoder
    decoded = LSTM(256, return_sequences=True)(repeated_vector)

    # Output Layer
    output = TimeDistributed(Dense(len(unique_characters), activation="softmax"))(decoded)

    # Define the Model
    model = Model(inputs=image_input, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


# In[41]:


# Build the model
model_0 = build_image2text_model()

# Train the model
history_0 = model_0.fit(
    X_img_train, y_text_onehot_train,  # Use image inputs and text outputs
    validation_data=(X_img_test, y_text_onehot_test),
    epochs=25, batch_size=64, verbose=1
)


# Evaluate on test set
test_loss, test_accuracy = model_0.evaluate(X_img_test, y_text_onehot_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[43]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_0.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_0.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 0 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_0.history['loss'], label='Training Loss', marker='o')
plt.plot(history_0.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 0 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[46]:


# Predict on the test set to get the predicted labels
y_pred = model_0.predict(X_img_test)

# Assuming `decode_labels` is a function to decode the labels into text
decoded_true = [decode_labels(y) for y in y_text_onehot_test]  # True labels
decoded_pred = [decode_labels(y) for y in y_pred]  # Predicted labels

# Identify misclassified samples
misclassified_indices = [
    i for i, (true, pred) in enumerate(zip(decoded_true, decoded_pred))
    if true != pred
]

# Improved plotting
for i in random.sample(misclassified_indices, min(5, len(misclassified_indices))):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"True: {decoded_true[i]} | Pred: {decoded_pred[i]}", fontsize=16, y=0.92)

    # Plot the image sequence
    axs[0].imshow(
        np.hstack([X_img_test[i][j].squeeze() for j in range(X_img_test.shape[1])]),
        cmap='gray',
    )
    axs[0].set_title("Query Image Sequence", fontsize=14)
    axs[0].axis('off')

    # Display the expected answer (text)
    axs[1].text(
        0.5,
        0.5,
        f"Expected Answer: {decoded_true[i]}",
        fontsize=14,
        ha='center',
        va='center',
        wrap=True,
    )
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# In[49]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def build_image2text_model_1():
    # Image Input (sequence of 5 grayscale images)
    image_input = Input(shape=(5, 28, 28, 1), name="Image_Input")
    
    # CNN Encoder for Images
    cnn_output = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(image_input)
    cnn_output = TimeDistributed(MaxPooling2D((2, 2)))(cnn_output)
    cnn_output = TimeDistributed(Flatten())(cnn_output)
    
    # LSTM Encoder
    encoded_1 = LSTM(256, return_sequences=True)(cnn_output)
    encoded_2 = LSTM(256)(encoded_1)

    # Repeat Vector for Decoder
    repeated_vector = RepeatVector(max_answer_length)(encoded_2)

    # LSTM Decoder
    decoded = LSTM(256, return_sequences=True)(repeated_vector)

    # Output Layer
    output = TimeDistributed(Dense(len(unique_characters), activation="softmax"))(decoded)

    # Define the Model
    model = Model(inputs=image_input, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


# In[50]:


# Build the model
model_1 = build_image2text_model_1()

# Train the model
history_1 = model_1.fit(
    X_img_train, y_text_onehot_train,  # Use image inputs and text outputs
    validation_data=(X_img_test, y_text_onehot_test),
    epochs=25, batch_size=64, verbose=1
)


# Evaluate on test set
test_loss, test_accuracy = model_1.evaluate(X_img_test, y_text_onehot_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[51]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_1.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 0 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_1.history['loss'], label='Training Loss', marker='o')
plt.plot(history_1.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 0 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[52]:


# Predict on the test set to get the predicted labels
y_pred = model_1.predict(X_img_test)

# Assuming `decode_labels` is a function to decode the labels into text
decoded_true = [decode_labels(y) for y in y_text_onehot_test]  # True labels
decoded_pred = [decode_labels(y) for y in y_pred]  # Predicted labels

# Identify misclassified samples
misclassified_indices = [
    i for i, (true, pred) in enumerate(zip(decoded_true, decoded_pred))
    if true != pred
]

# Improved plotting
for i in random.sample(misclassified_indices, min(5, len(misclassified_indices))):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"True: {decoded_true[i]} | Pred: {decoded_pred[i]}", fontsize=16, y=0.92)

    # Plot the image sequence
    axs[0].imshow(
        np.hstack([X_img_test[i][j].squeeze() for j in range(X_img_test.shape[1])]),
        cmap='gray',
    )
    axs[0].set_title("Query Image Sequence", fontsize=14)
    axs[0].axis('off')

    # Display the expected answer (text)
    axs[1].text(
        0.5,
        0.5,
        f"Expected Answer: {decoded_true[i]}",
        fontsize=14,
        ha='center',
        va='center',
        wrap=True,
    )
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# In[53]:


import matplotlib.pyplot as plt

# Create a 2x2 figure
plt.figure(figsize=(12, 8))

# Model 0 Accuracy
plt.subplot(2, 2, 1)
plt.plot(history_0.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_0.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 0 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Model 0 Loss
plt.subplot(2, 2, 2)
plt.plot(history_0.history['loss'], label='Training Loss', marker='o')
plt.plot(history_0.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 0 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Model 1 Accuracy
plt.subplot(2, 2, 3)
plt.plot(history_1.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history_1.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model 1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Model 1 Loss
plt.subplot(2, 2, 4)
plt.plot(history_1.history['loss'], label='Training Loss', marker='o')
plt.plot(history_1.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model 1 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ---
# ---
# 
# ## III. Text to image RNN Model
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.

# In[18]:


# Your code
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Conv2DTranspose, Reshape, Flatten
from tensorflow.keras.models import Model

def build_text2image_model():
    # Input: Text query (batch_size, max_query_length, num_characters)
    text_input = Input(shape=(max_query_length, len(unique_characters)), name="Text_Input")

    # RNN for text encoding
    text_encoder = LSTM(256)(text_input)
    latent_representation = Dense(128, activation="relu")(text_encoder)

    # Repeat latent representation for each output image
    repeated_latent = RepeatVector(max_answer_length)(latent_representation)

    # RNN for sequence decoding
    image_decoder = LSTM(256, return_sequences=True)(repeated_latent)
    dense_decoder = TimeDistributed(Dense(7 * 7 * 128, activation="relu"))(image_decoder)
    
    # CNN decoder to generate images
    cnn_decoder = TimeDistributed(
        tf.keras.Sequential(
            [
                Reshape((7, 7, 128)),
                Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same"),
                Conv2DTranspose(1, (3, 3), strides=(2, 2), activation="sigmoid", padding="same"),
            ]
        )
    )(dense_decoder)

    # Define the model
    model = Model(inputs=text_input, outputs=cnn_decoder)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()

    return model


# Build the text-to-image model
text2image_model = build_text2image_model()


# In[19]:


# Split data into training and testing sets
X_text_train, X_text_test, y_img_train, y_img_test = train_test_split(
    X_text_onehot, y_img, test_size=0.1, random_state=42
)

# Train the model
history = text2image_model.fit(
    X_text_train,
    y_img_train,
    validation_data=(X_text_test, y_img_test),
    epochs=50,
    batch_size=64,
)


# In[20]:


# Evaluate the model
test_loss, test_accuracy = text2image_model.evaluate(X_text_test, y_img_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predict on test samples
y_pred = text2image_model.predict(X_text_test)

# Visualize generated images
def visualize_generated_images(n=5):
    for i in range(n):
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Query: {decode_labels(X_text_test[i])}")
        for j in range(max_answer_length):
            plt.subplot(2, max_answer_length, j + 1)
            plt.imshow(y_img_test[i][j], cmap="gray")
            plt.axis("off")
            plt.title(f"True {j+1}")

            plt.subplot(2, max_answer_length, max_answer_length + j + 1)
            plt.imshow(y_pred[i][j], cmap="gray")
            plt.axis("off")
            plt.title(f"Pred {j+1}")
        plt.show()

# Visualize a few examples
visualize_generated_images(n=10)


# In[21]:


from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Conv2DTranspose, Reshape, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf

def build_text2image_model_2():
    text_input = Input(shape=(max_query_length, len(unique_characters)), name="Text_Input")

    # RNN Encoder with stacked LSTMs
    # First LSTM layer returns sequences so the second LSTM can process them
    text_encoder_1 = LSTM(256, return_sequences=True)(text_input)
    text_encoder_2 = LSTM(256, return_sequences=True)(text_encoder_1)  # Second LSTM consumes the sequence
    text_encoder_3 = LSTM(256)(text_encoder_2)  # Third LSTM consumes the sequence

    latent_representation = Dense(128, activation="relu")(text_encoder_3)

    # Repeat latent representation for each output image
    repeated_latent = RepeatVector(max_answer_length)(latent_representation)

    # RNN Decoder
    image_decoder = LSTM(256, return_sequences=True)(repeated_latent)
    dense_decoder = TimeDistributed(Dense(7 * 7 * 128, activation="relu"))(image_decoder)
    
    # CNN decoder to generate images
    cnn_decoder = TimeDistributed(
        tf.keras.Sequential(
            [
                Reshape((7, 7, 128)),
                Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same"),
                Conv2DTranspose(1, (3, 3), strides=(2, 2), activation="sigmoid", padding="same"),
            ]
        )
    )(dense_decoder)

    # Define the model
    model = Model(inputs=text_input, outputs=cnn_decoder)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()

    return model


# Build the modified text-to-image model with deeper encoder
text2image_model_2 = build_text2image_model_2()


# In[22]:


# Split data into training and testing sets
X_text_train, X_text_test, y_img_train, y_img_test = train_test_split(
    X_text_onehot, y_img, test_size=0.1, random_state=42
)

# Train the model
history = text2image_model_2.fit(
    X_text_train,
    y_img_train,
    validation_data=(X_text_test, y_img_test),
    epochs=50,
    batch_size=64,
)


# In[23]:


# Evaluate the model
test_loss, test_accuracy = text2image_model_2.evaluate(X_text_test, y_img_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predict on test samples
y_pred = text2image_model_2.predict(X_text_test)

# Visualize generated images
def visualize_generated_images(n=5):
    for i in range(n):
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Query: {decode_labels(X_text_test[i])}")
        for j in range(max_answer_length):
            plt.subplot(2, max_answer_length, j + 1)
            plt.imshow(y_img_test[i][j], cmap="gray")
            plt.axis("off")
            plt.title(f"True {j+1}")

            plt.subplot(2, max_answer_length, max_answer_length + j + 1)
            plt.imshow(y_pred[i][j], cmap="gray")
            plt.axis("off")
            plt.title(f"Pred {j+1}")
        plt.show()

# Visualize a few examples
visualize_generated_images(n=10)


# In[ ]:




