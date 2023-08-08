# STOCK-SENTIMENT-ANALYSIS-USING-NEWS-HEADLINES

## Overview
This project focuses on analyzing stock market sentiment using news headlines. The main goal is to predict whether the stock prices of a given company will go up or down based on the top headlines associated with the company. The project involves data preprocessing, text vectorization, and classification using machine learning algorithms.

## Introduction
In the world of finance, stock market sentiment analysis plays a crucial role in predicting the direction of stock prices. This project demonstrates how natural language processing (NLP) techniques and machine learning models can be used to analyze news headlines and predict whether a company's stock prices will go up or down.

## Dataset
The project uses a dataset containing historical stock news headlines and corresponding labels indicating whether the stock price increased (1) or decreased (0) on a given day. The dataset is provided in a CSV format and includes columns for the date, label, and the top 25 headlines for each day.

## Data Preprocessing
* Removed punctuation and converted text to lowercase to prepare the headlines for analysis.
* Divided the dataset into training and test sets based on the date.

## Text Vectorization
Two approaches for text vectorization were used:

* Bag of Words Model: Utilized the CountVectorizer to convert the text data into vectors.
* TF-IDF Vectorizer: Employed the TfidfVectorizer to convert the text data into vectors using the TF-IDF technique.

## Machine Learning Models
Two classification models were implemented:

### Random Forest Classifier
* Utilized the Random Forest Classifier with the Bag of Words and TF-IDF vectors for classification.
* Evaluated the model's performance using accuracy, confusion matrix, and classification report.

### Naive Bayes Classifier
* Used the Multinomial Naive Bayes Classifier with the Bag of Words vectors for classification.
* Evaluated the model's performance using accuracy, confusion matrix, and classification report.

## Evaluation
Both Random Forest and Naive Bayes classifiers were evaluated on the test dataset:

*Random Forest Classifier (Bag of Words):
  * Accuracy: 85.19%
  * Confusion Matrix: [[137, 49], [7, 185]]
  * Classification Report: Precision, recall, and F1-score metrics.


* Random Forest Classifier (TF-IDF):
  * Accuracy: 84.66%
  * Confusion Matrix: [[143, 43], [15, 177]]
  * Classification Report: Precision, recall, and F1-score metrics.

* Naive Bayes Classifier (Bag of Words):
  * Accuracy: 85.19%
  * Confusion Matrix: [[139, 47], [11, 181]]
  * Classification Report: Precision, recall, and F1-score metrics.
