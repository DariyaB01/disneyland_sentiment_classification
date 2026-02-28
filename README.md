# Disneyland Review Sentiment Classification (TF-IDF + PyTorch ANN)

A research-oriented NLP project exploring sentiment classification on Disneyland reviews using classical TF-IDF features and comparing a linear baseline with a shallow neural network.

---

## Project Overview

This project investigates how classical machine learning models and shallow neural networks perform on sparse high-dimensional text data.

The study includes:

- TF-IDF feature extraction (10,000 features, 1–2 ngrams)
- Logistic Regression baseline (balanced class weights)
- PyTorch ANN implementation
- Class imbalance handling (~83% positive class)
- Overfitting analysis
- Quantitative and qualitative evaluation

---

## Dataset

Source: HuggingFace  
Dataset: `dariadaria/disneyland_reviews`

Original labels:
- 0 → Negative
- 1 → Neutral
- 2 → Positive

Preprocessing:
- Neutral class removed
- Binary mapping applied (0 → Negative, 2 → Positive)
- Stratified train/validation split (80/20)

Final class distribution:
- ~83% Positive
- ~17% Negative

This dataset is highly imbalanced, which significantly impacts model behavior.

---

## Methodology

### 1. Feature Engineering
TF-IDF vectorization:
- max_features = 10000
- min_df = 3
- ngram_range = (1,2)

### 2. Baseline Model
Logistic Regression:
- class_weight="balanced"
- max_iter=300

### 3. Neural Network Model
Architecture:

Input → 64 → 16 → Output  
ReLU activation  
Dropout (0.5) regularization  

Loss function:
- BCEWithLogitsLoss
- Class weighting applied to mitigate imbalance

Optimizer:
- Adam (lr=1e-3)

---

## Results

| Model | Accuracy | F1-score |
|--------|----------|----------|
| Logistic Regression | (insert value) | (insert value) |
| PyTorch ANN | (insert value) | (insert value) |

---

## Observations

- TF-IDF features show strong linear separability.
- Logistic Regression performs competitively.
- The ANN demonstrates mild overfitting.
- Class imbalance significantly affects recall for the minority class.
- Some misclassifications reflect mixed-sentiment reviews or potential label noise.

---

## Error Analysis

Manual inspection of misclassified examples shows:

- Mixed polarity reviews
- Strong positive tokens influencing predictions
- Sensitivity to class imbalance
- Limitations of bag-of-words representations in capturing contextual nuance

---

## Key Takeaways

- Classical linear models remain strong baselines for sparse NLP tasks.
- Neural networks can overfit when feature space is high-dimensional.
- Proper handling of class imbalance is critical.
- Evaluation beyond accuracy is necessary in imbalanced datasets.

---

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- HuggingFace Datasets
- Matplotlib
- NumPy / Pandas

---

## How to Run

```bash
pip install -r requirements.txt
```

Run the notebook inside `/notebooks/`.

---

## Future Improvements

- Transformer-based models (DistilBERT, RoBERTa)
- Multi-class sentiment classification
- Threshold optimization
- Advanced imbalance techniques (SMOTE, focal loss)
