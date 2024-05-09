# Recommendation System Model

## Introduction

This repository contains a recommendation system model designed to provide personalized movie recommendations to users utilizing collaborative filtering techniques and the MovieLens dataset.

## Model Functionality

### Data Preprocessing

- **Rating Thresholding**: preprocesses the raw MovieLens data by defining a rating threshold to separate positive (relevant) interactions from negative ones.
- **Negative Sample Generation**: balances the number of negative and positive samples by selecting dissimilar items (to the positive ones) based on metadata such as genre. This ensures a diverse set of negative samples for training.
- **Dataset Splitting**: divides the MovieLens dataset into training and testing subsets using either user-based or temporal splitting methods to ensure robust evaluation.

### Hyperparameter Tuning

- **Automated Search**: performs automated hyperparameter tuning using the Hyperopt library, optimizing the model's performance based on predefined search spaces and evaluation metrics. This includes tuning the architecture of the collaborative filtering model (dot product, concatenated, deep learning, and hybrid models).

### Evaluation

- **Mean Average Precision at k (MAP@k)**: evaluates the model's performance using MAP@k, a metric that measures the average precision of the top-k recommendations provided to users.
- **ROC Curve, Precision-Recall Curve, and Confusion Matrix**: analyzes the ROC curve, precision-recall curve, and confusion matrix to assess the model's ability to discriminate between positive and negative items.
- **Cosine Similarity**: computes the cosine similarity between item embeddings to measure the similarity between movies. This metric helps assess how well the model captures the relationships between similar movies.
- **Embedding Visualization**: the model visualizes item embeddings using t-SNE (t-distributed stochastic neighbor embedding), which allows to explore the distribution and relationships between items in the embedding space. 
