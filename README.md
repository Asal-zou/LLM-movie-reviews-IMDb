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
- **Accuracy:** Measures the proportion of correctly predicted sentiments against the total predictions. This metric gives a general sense of the model’s performance by showing the overall correctness of predictions, which is particularly useful when the class distribution is balanced.
- 
- **Precision:** Indicates the accuracy of positive sentiment predictions. Precision is crucial in situations where the cost of false positives is high. It ensures that when the model predicts a positive sentiment, it is likely correct, which is important for applications like customer feedback analysis where false positives could lead to misguided actions.
  
- **Recall:** Reflects the ability of the model to correctly identify all relevant positive cases. High recall ensures that most positive sentiments are detected, which is critical for understanding overall customer satisfaction or identifying critical feedback.
- 
- **F1 Score:** Balances precision and recall, providing a single metric that accounts for both false positives and false negatives. This is particularly useful in scenarios with imbalanced class distributions, where accuracy alone might be misleading. The F1 score helps ensure that the model performs well in detecting positive sentiments while also maintaining a low false positive rate.

## Hyperparameters
**Important Hyperparameters:**
- **Learning Rate:** Set at 2e-5, crucial for how the model weights adjust relative to the error rate at each update. 2e-5 was suggested to use in Hugging Face and worked the best with my model.
- 
- **Batch Sizes:** Configured at 16 for both training and evaluation phases to optimize computational efficiency and manage memory usage effectively. Smaller batch sizes can lead to more stable updates and allow for more frequent model updates, striking a balance between efficient computation and effective learning.
  
- **Number of Epochs:** Limited to 1 to streamline the training process. This decision is often based on prior experiments or resource constraints. Reducing the number of epochs helps in quicker model iterations and can prevent overfitting, especially in preliminary model evaluations.


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
