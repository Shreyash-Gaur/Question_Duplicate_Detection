
---

# Question Duplicate Detection


Welcome to the Question Duplicate Detection project! This project utilizes a Siamese network model to identify duplicate questions using the Quora question dataset, a common challenge in natural language processing (NLP). By leveraging the Quora question dataset, this project aims to detect pairs of questions with the same intent or meaning, even if they are phrased differently.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Part 1: Importing the Data](#dataset)
4. [Part 2: Defining the Siamese Model](#model)
5. [Part 3: Model Architecture](#model-architecture)
6. [Part 4: Training](#training)
7. [Part 5: Evaluation](#evaluation)
8. [Part 6: Testing with Custom Questions](#testing)
9. [Usage](#usage)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

Duplicate question detection is essential for maintaining the quality and efficiency of Q&A platforms like Quora. By identifying and merging duplicate questions, these platforms can provide better search results and avoid redundant information.

This project implements a Siamese network, which is particularly well-suited for tasks where we need to measure the similarity between two inputs. The model is trained using triplet loss, ensuring that duplicate questions are close in the embedding space, while non-duplicate questions are far apart.
- **Learning Objectives:**
  - Understand Siamese networks and triplet loss
  - Evaluate model accuracy using cosine similarity
  - Use data generators for batching questions
  - Build a classifier to identify duplicate questions

## Installation

To run this project, you'll need Python and several libraries. Follow these steps to set up your environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shreyash-Gaur/Question_Duplicate_Detection.git
   cd Question_Duplicate_Detection
   ```

2. **Install the required packages:**
   ```bash
   pip install os numpy pandas random tensorflow keras
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Naive Machine Translation and LSH.ipynb"
   ```

## Part 1: Importing the Data
**Loading the Data:**
  - The project uses the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset. This dataset contains pairs of questions along with a label indicating whether they are duplicates. Ensure you have this dataset downloaded and placed in the appropriate directory.
  - Preprocess and load the dataset into a pandas DataFrame
  - Split the data into training and testing sets
  - Filter out non-duplicate questions to focus on duplicates only
  - Split the training set further into training and validation sets

**Learning Question Encoding:**
  - Use Keras `TextVectorization` to create a vocabulary from the training data
  - Convert questions into integer-encoded vectors

## Part 2: Defining the Siamese Model
**Understanding the Siamese Network:**
  - A Siamese network uses the same weights for two input vectors to compute comparable output vectors
  - Use triplet loss to minimize the distance between duplicate questions and maximize the distance between non-duplicates

**Hard Negative Mining:**
  - Improve training by focusing on the most challenging non-duplicate examples

## Part 3: Model Architecture
The core of this project is a Siamese network that uses the following architecture:

- **Embedding Layer:** Transforms input text into dense vectors.
- **LSTM Layer:** Captures sequential dependencies in the text.
- **Global Average Pooling:** Reduces the output dimension by averaging over all time steps.
- **Lambda Layer:** Normalizes the output to create unit vectors.

### Model Definition

```python
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, GlobalAveragePooling1D, Lambda, Input, Concatenate

def Siamese(text_vectorizer, vocab_size=36224, d_feature=128):
    branch = tf.keras.models.Sequential(name='sequential')
    branch.add(text_vectorizer)
    branch.add(Embedding(vocab_size, d_feature, name="embedding"))
    branch.add(LSTM(d_feature, return_sequences=True, name="LSTM"))
    branch.add(GlobalAveragePooling1D(name="mean"))
    branch.add(Lambda(lambda x: tf.math.l2_normalize(x), name="out"))
    
    input1 = Input(shape=(1,), dtype=tf.string, name='input_1')
    input2 = Input(shape=(1,), dtype=tf.string, name='input_2')
    
    branch1 = branch(input1)
    branch2 = branch(input2)
    
    conc = Concatenate(axis=1, name='conc')([branch1, branch2])
    
    return tf.keras.models.Model(inputs=[input1, input2], outputs=conc)

model = Siamese(text_vectorizer)
```

## Part 4: Training:
**Training the Model:**
The model is trained using triplet loss, which ensures that duplicate questions are close in the embedding space while non-duplicate questions are far apart. The training process involves:

1. Loading and preprocessing the dataset.
2. Defining the Siamese network.
3. Training the model using a suitable optimizer and loss function.

### Training Example

```python
# Define optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=10, validation_data=val_data)
```

## Part 5: Evaluation
**Evaluating the Siamese Network:**
  The model's performance is evaluated using cosine similarity to measure how closely the question pairs are in the embedding space. The evaluation involves:

- Calculating the cosine similarity between the question pairs.
- Thresholding the similarity scores to classify pairs as duplicates or non-duplicates.

## Part 6: Testing with Custom Questions
**Testing with Custom Questions:**
  - Use the trained model to predict whether custom pairs of questions are duplicates or not

## Usage

To use the model with your own question pairs, follow these steps:

1. **Load the model:**
   ```python
   model = tf.keras.models.load_model('path_to_saved_model')
   ```

2. **Prepare your questions:**
   ```python
   question1 = "Do they enjoy eating the dessert?"
   question2 = "Do they like hiking in the desert?"
   ```

3. **Predict duplicates:**
   ```python
   predict(question1 , question2, threshold, model, verbose=True)
    ''' Args:
             question1 (str): First question.
             question2 (str): Second question.
             threshold (float): Desired threshold.
             model (tensorflow.keras.Model): The Siamese model'''
   ```

## Results

The trained model achieves competitive accuracy of 75% on the Quora question dataset, demonstrating its effectiveness in identifying duplicate questions. Detailed confusion metrics can be found in the notebook.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request. Please ensure your contributions adhere to the project's coding standards and include relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
