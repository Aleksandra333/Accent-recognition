# Accent-recognition

Project Overview

This project focuses on classifying accents based on speech recordings using machine learning techniques. The dataset comes from the Speech Accent Archive, and we apply Random Forest Classification with MFCC feature extraction for training and evaluation.
Dataset

The dataset is obtained from Kaggle:
🔗 Speech Accent Archive Dataset
Preprocessing Steps:

    Load the dataset and perform exploratory data analysis (EDA).
    Remove missing files from the dataset.
    Filter out specific native languages for classification.
    Extract features from audio files using Mel-frequency cepstral coefficients (MFCCs).
    Perform resampling to handle class imbalance.

Feature Extraction

We extract MFCCs from speech recordings:

    Compute Fourier Transform of the signal.
    Convert to the Mel scale using triangular overlapping windows.
    Take the log of power values at each Mel frequency.
    Apply Discrete Cosine Transform (DCT) to obtain the MFCCs.
    Compute delta MFCCs to capture temporal variations.

📌 Library used: librosa
Model Training

    Algorithm: Random Forest Classifier (via pyspark.ml)
    Training framework: Apache Spark (for scalability)
    Resampling method: RandomOverSampler (to handle class imbalance)
    Evaluation metric: Accuracy & Confusion Matrix

Pipeline:

    Convert extracted MFCC features into a structured dataset.
    Encode class labels using StringIndexer.
    Train a Random Forest Classifier.
    Evaluate performance using accuracy score.

Results & Performance

    Achieved ~65.12% accuracy with optimized MFCC coefficients.
    Accuracy varies based on the number of MFCC coefficients used.
    Confusion matrix is used to analyze misclassifications.

📊 Graphical Insights:

    Dependence of accuracy on the number of coefficients.
    Distribution of native languages & genders.
    Confusion matrix heatmap.