# DNA-Sequencing-Classifier

DNA-Sequencing-Classifier is a machine learning project designed to classify DNA sequences from humans, chimpanzees, and dogs. By using k-mer encoding, this project translates DNA sequences into identifiable features and applies a Naive Bayes classifier to accurately distinguish between DNA from different species.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
5. [Code Structure](#code-structure)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Future Improvements](#future-improvements)

---

## Project Overview

DNA sequences consist of a series of nucleotide bases (A, T, C, G) that encode biological information. By analyzing the unique patterns in these sequences, we can classify DNA data by species. In this project, we convert DNA sequences into k-mer words (subsequences of fixed length) and utilize a Naive Bayes classifier to predict the species of unknown DNA samples.

### Key Features:
- **Species Classification**: Determines whether a given DNA sequence belongs to a human, chimpanzee, or dog.
- **Feature Extraction**: Converts DNA sequences into feature vectors using k-mer encoding (nucleotide subsequences).
- **Model Evaluation**: Measures performance with accuracy, precision, recall, and F1 scores.

## Installation

To set up and run this project, please follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ss1910singh/DNA-Sequencing-Classifier.git
   cd DNA-Sequencing-Classifier
   ```

2. **Install dependencies**:

   Ensure you have Python installed (version 3.6+ is recommended) and run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**:

   Make sure the dataset files (`human_data.txt`, `chimp_data.txt`, and `dog_data.txt`) are placed in the root directory of the project.

## Dataset Description

This project works with three datasets, each containing DNA sequences from a specific species:

- **human_data.txt**: DNA sequences labeled for humans
- **chimp_data.txt**: DNA sequences labeled for chimpanzees
- **dog_data.txt**: DNA sequences labeled for dogs

Each file contains a `sequence` column with DNA base sequences. The class label (species type) is also included for training and evaluation purposes.

## Methodology

### Step 1: Data Preprocessing

We begin by converting DNA sequences into k-mer words, which are fixed-length subsequences from each DNA sequence. In this project, we use a default k-mer size of 6 (hexamers).

```python
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
```

### Step 2: Feature Extraction

After converting sequences into k-mer words, we vectorize these k-mers using `CountVectorizer` from Scikit-Learn. The `CountVectorizer` creates a matrix of token counts, enabling us to analyze the frequency of 4-grams in the sequences.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)  # Vectorize human DNA sequences
```

### Step 3: Model Training

The dataset is split into training and testing sets (80-20 split), and we apply a Multinomial Naive Bayes classifier. The classifier is trained on the vectorized DNA sequence data, which learns the unique patterns associated with each species.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=42)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
```

## Code Structure

The main code structure is as follows:

1. **Data Loading**: Load DNA sequence data for human, chimpanzee, and dog species.
2. **Data Preprocessing**: Convert sequences to k-mers and vectorize them.
3. **Model Training**: Split the data and train the Naive Bayes classifier.
4. **Evaluation**: Evaluate the model using metrics such as accuracy, precision, recall, and F1 score.

## Evaluation

After training, we evaluate the model's performance using various metrics:

- **Accuracy**: Overall correctness of the classifier.
- **Precision**: Proportion of true positive predictions.
- **Recall**: Ability of the classifier to find all positive samples.
- **F1 Score**: Weighted average of precision and recall.

Evaluation is performed using a custom function:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

# Example Output
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
```

## Results

The model's predictions are analyzed using a confusion matrix and the key metrics. These metrics provide insight into the classifier's effectiveness at identifying species from DNA sequences:

- **Confusion Matrix**: Displays the true positives, false positives, false negatives, and true negatives for each class.
- **Metrics**: Detailed performance values for accuracy, precision, recall, and F1 score.

## Future Improvements

Future enhancements may include:
1. **Expanding Data Variety**: Adding more species or larger datasets for increased robustness.
2. **Parameter Tuning**: Exploring hyperparameter optimization for improved accuracy.
3. **Alternative Models**: Testing other classifiers such as SVMs or neural networks to compare performance.
