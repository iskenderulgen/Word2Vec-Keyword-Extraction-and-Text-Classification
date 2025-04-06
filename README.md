# Apache Spark Keyword Extraction and Text Classification

This project demonstrates a simplified or go-to approach to text classification by leveraging unsupervised learning to generate pseudo-labels for supervised (or semi-supervised) training. In the early stages of the project, labels are not available, so we rely on a Word2Vec model to extract keywords from a Reddit comments dataset. These keywords then serve as a basis to train a Logistic Regression classifier for final text classification. This project is available for research and educational purposes.

## Project Overview

The project is split into two main phases:

1. **Word2Vec Training**
   - We train a Word2Vec model on a dataset of Reddit comments.
   - The model learns word embeddings and uses these to extract keywords (synonyms) for specific topics such as music, gaming, politics, programming, and science.
   - This phase produces both the trained Word2Vec model and the extracted keywords, which are saved for later use.

2. **Text Classification**
   - Using the keywords from the first phase, the project trains a Logistic Regression classifier.
   - The classification process involves tokenizing the text, vectorizing it with the pretrained Word2Vec model, and then predicting categories for new comments.
   - The evaluation is done using multiple Spark metrics, including accuracy, precision, recall, and F-measure.

## Dataset

- **Source:** Reddit comments dataset available for research purposes  
  [Reddit Comments Dataset](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/)
- **Time Period:** July 2010
- **Volume:** Approximately 65 million words
- **Training Time:** Approximately 11 minutes on a 16-core processor

## Technology Stack

- **Language:** Python (with Apache Spark)
- **Framework:** Apache Spark 3.5.5
- **Machine Learning Models:**
  - *Word2Vec* for word embeddings and keyword extraction.
  - *Logistic Regression* for text classification.

## Project Structure

- `word2vec_train.ipynb`:  
  Contains code for reading, cleaning, tokenizing the dataset and training the Word2Vec model. It then extracts keywords by finding synonyms based on the learned word embeddings.
  
- `text_classification.ipynb`:  
  Contains the pipeline for loading the precomputed keywords and Word2Vec model, tokenizing new Reddit comments, vectorizing them, and finally classifying the comments using a Logistic Regression model.

## Installation and Setup

1. **Clone the Repository:**

2. **Install Required Packages:**

   ```bash
   pip install pyspark==3.5.5 pandas
   ```

3. **Dataset:**

   - Download the Reddit comments dataset from the source mentioned above.
   - Place the dataset into the `data/` directory (e.g., `data/RC_2010-07`).

## Running the Project

1. **Train the Word2Vec Model:**

   - Open the `word2vec_train.ipynb` notebook.
   - Execute the cells to process the data, train the Word2Vec model, and extract keywords.
   - The model and keywords are saved in the `data/` directory.

2. **Run Text Classification:**

   - Open the `text_classification.ipynb` notebook.
   - Execute the cells to load the trained model and keywords, process new Reddit comments, and classify them.

## Project Motivation and Future Work

The project addresses the challenge of sparse labeling in text classification. By generating pseudo-labels through keyword extraction, it creates a pathway for applying supervised learning in contexts where labels are initially unavailable.  
Future work may include:
- Incorporating advanced embedding models such as BERT or GloVe.
- Extending the approach to cover more diverse topics and larger datasets.
- Experimenting with additional classification methods and hyperparameter tuning for improved performance.

# References

This work is a simplified version of the following research

```
@inproceedings{8806496,
  author={Ogul, Iskender Ulgen and Ozcan, Caner and Hakdagli, Ozlem},
  booktitle={2019 27th Signal Processing and Communications Applications Conference (SIU)}, 
  title={Keyword Extraction Based on word Synonyms Using WORD2VEC}, 
  year={2019},
  volume={},
  number={},
  pages={1-4},
  keywords={Sparks;Automobiles;Spark;Word2Vec;Word Embedding;Keyword Extraction;Text Mining},
  doi={10.1109/SIU.2019.8806496}}
```
