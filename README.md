# LLM Project

## Project Task
Develop a sentiment analysis tool using a Large Language Model (LLM) to undrestand if a rating is positive or negative on an IMDB dataset. 

## Dataset
**Source:** The dataset utilized in this project is the IMDb reviews dataset, which is commonly used for binary sentiment classification. This dataset consists of a balanced collection of movie reviews labeled as positive or negative, making it a good resource for training and testing our sentiment analysis model.

## Pre-trained Model
**Model:** This project employs the `distilbert-base-uncased` model from the Hugging Face `transformers` library. DistilBERT is a streamlined version of the BERT model that maintains most of BERT's accuracy but with fewer parameters and faster processing, ideal for deployment in sentiment analysis tasks.

*BERT (Bidirectional Encoder Representations from Transformers)* is a model in the field of natural language processing (NLP). It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks, such as question answering, sentiment analysis, and language inference, without substantial task-specific architecture modifications.


## Performance Metrics
**Metrics Evaluated:**
- **Accuracy:** This metric measures the proportion of correctly predicted sentiments against the total predictions.
- **Precision:** Indicates the accuracy of positive sentiment predictions.
- **Recall:** Reflects the model’s ability to correctly identify all relevant positives.
- **F1 Score:** Balances precision and recall, particularly valuable in scenarios where class distributions are imbalanced.

## Hyperparameters
**Important Hyperparameters:**
- **Learning Rate:** Set at 2e-5, crucial for controlling how the model weights are adjusted with respect to the error rate at each update.
- **Batch Sizes:** Configured at 16 for both training and evaluation phases to optimize computational efficiency and manage memory usage effectively.
- **Number of Epochs:** Limited to 1 to streamline the training process
