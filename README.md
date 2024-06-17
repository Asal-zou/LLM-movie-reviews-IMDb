# LLM Project

## Project Overview
This project aimed to develop a sentiment analysis tool using a Large Language Model (LLM). The primary objective was to determine the sentiment (positive or negative) of movie reviews from the IMDb dataset. This tool can help media companies and marketers understand consumer reactions and preferences, enabling better-targeted marketing strategies and content customization to enhance user engagement and satisfaction.

## Dataset
**Source:** The dataset used in this project is the IMDb reviews dataset. It's widely employed for binary sentiment classification and consists of a balanced collection of movie reviews tagged as positive or negative. This makes it an excellent resource for both training and testing our sentiment analysis model.

## Pre-trained Model
**Model:** The `distilbert-base-uncased` model from the Hugging Face `transformers` library is utilized in this project. DistilBERT is a streamlined version of the BERT model, designed to retain much of BERT's accuracy while reducing the number of parameters and increasing processing speed—making it ideal for sentiment analysis tasks.

BERT (Bidirectional Encoder Representations from Transformers) is a pivotal model in natural language processing (NLP). It pre-trains deep bidirectional representations from unlabeled text, considering both left and right context in all layers. Consequently, a pre-trained BERT model can be fine-tuned with just one additional output layer to craft state-of-the-art models for various NLP tasks, including question answering, sentiment analysis, and language inference, with minimal task-specific modifications.

## Performance Metrics
**Metrics Evaluated:**
- **Accuracy:** Measures the proportion of correctly predicted sentiments against the total predictions.
- **Precision:** Indicates the accuracy of positive sentiment predictions.
- **Recall:** Reflects the ability of the model to correctly identify all relevant positive cases.
- **F1 Score:** Balances precision and recall, especially useful in scenarios with imbalanced class distributions.

## Hyperparameters
**Important Hyperparameters:**
- **Learning Rate:** Set at 2e-5, crucial for how the model weights adjust relative to the error rate at each update.
- **Batch Sizes:** Configured at 16 for both training and evaluation phases to optimize computational efficiency and manage memory usage effectively.
- **Number of Epochs:** Limited to 1 to streamline the training process.

## Results
**DistilBERT Evaluation Results:**
- **Loss:** 0.1966
— Represents the model's error rate, indicating how well the model's predictions match the actual labels. Lower loss indicates better performance.
- **Accuracy:** 92.632%
— Shows that the model correctly predicts the sentiment of reviews with high reliability.
- **F1 Score:** 92.719%
— Demonstrates that the model effectively balances precision and recall, crucial for maintaining robust performance across different parts of the dataset.
- **Precision:** 91.633%
— Reflects the model's accuracy in identifying positive reviews, ensuring minimal false positives.
- **Recall:** 93.832%
  — Indicates that the model is excellent at identifying most of the true positive cases, essential for comprehensive sentiment analysis.
- **Runtime:** 411.6614 seconds
  — The time taken for the model to complete the evaluation, giving an idea of how the model performs under operational conditions.
- **Samples per Second:** 60.73
  — Reflects processing efficiency, indicating how many samples the model can handle per second.
- **Steps per Second:** 3.797
   — Provides insight into the model's operational speed in terms of optimization steps taken per second.
- **Epoch:** 1
  — Signifies that these results were achieved after a single round of training through the dataset, highlighting the model's efficiency in learning.


## Comparison with Linear Regression
To contextualize the performance of the DistilBERT model, it was compared to a baseline linear regression model:

- **Linear Regression Accuracy:**
  - **Training Data:** 89.28%
  - **Testing Data:** 87.048%

The comparison highlights the enhanced capability of the DistilBERT model, which significantly outperforms the linear regression in terms of accuracy, precision, and recall on the testing data, showcasing its effectiveness in handling complex NLP tasks like sentiment analysis.
