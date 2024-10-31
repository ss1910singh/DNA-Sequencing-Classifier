# DNA-Sequencing-Classifier

DNA-Sequencing-Classifier is a machine learning project that classifies DNA sequences from humans, chimpanzees, and dogs. It uses k-mer encoding to convert DNA sequences into features and leverages a Naive Bayes classifier to distinguish between the species based on their sequence patterns.

## Project Overview

This project reads DNA sequence data from three species (human, chimpanzee, and dog), preprocesses the data by converting sequences into k-mers (hexamer words), and transforms these words into features using a count-based vectorization method. A Naive Bayes classifier is then trained to classify DNA sequences, and the model’s performance is evaluated on key metrics.

### Dataset

The project uses three datasets:
- `human_data.txt`: DNA sequence data for humans
- `chimp_data.txt`: DNA sequence data for chimpanzees
- `dog_data.txt`: DNA sequence data for dogs

Each dataset contains a `sequence` column representing DNA sequences.

## Setup

1. Clone this repository.
2. Install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the dataset files (`human_data.txt`, `chimp_data.txt`, and `dog_data.txt`) are in the root directory.

## Code Description

### Importing Libraries

We start by importing essential libraries, including `numpy`, `pandas`, and `matplotlib`, as well as Scikit-Learn for machine learning tasks.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

### Data Loading

DNA sequence data from humans, chimpanzees, and dogs is loaded using pandas:

```python
human_data = pd.read_table('human_data.txt')
chimp_data = pd.read_table('chimp_data.txt')
dog_data = pd.read_table('dog_data.txt')
```

### Preprocessing

DNA sequences are converted to k-mer words (default size 6) using a helper function:

```python
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
```

### Vectorization

After preprocessing, sequences are converted into feature vectors using `CountVectorizer` with a 4-gram range:

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)
```

### Model Training

The processed data is split into training and testing sets, with an 80-20 ratio:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=42)
```

A Naive Bayes classifier is trained on the training set:

```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
```

### Evaluation

The model is evaluated using metrics including accuracy, precision, recall, and F1 score:

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
```

## Results

The classifier's performance is displayed with a confusion matrix, and accuracy, precision, recall, and F1 scores are reported.

## Requirements

Please refer to the `requirements.txt` file for necessary packages.
