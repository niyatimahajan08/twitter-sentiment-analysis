# Twitter Sentiment Analysis

This project performs sentiment analysis on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), which contains 1.6 million tweets. The goal is to classify tweets as having either a positive or negative sentiment. This is achieved through a pipeline of extensive text preprocessing, feature extraction using TF-IDF, and a comparative evaluation of five different machine learning models.

## 🌟 Project Overview

This repository provides a complete workflow for a classic NLP classification task. The core of the project involves training machine learning models to understand and predict the sentiment of textual data from Twitter.

### Key Features:

  * **Comprehensive Preprocessing:** Implements a robust text cleaning pipeline to handle noise typical in tweet data, including URLs, mentions, hashtags, emojis, and stop words.
  * **Efficient Vectorization:** Uses the TF-IDF (Term Frequency-Inverse Document Frequency) technique with n-grams to convert text into meaningful numerical features.
  * **Model Comparison:** Trains and evaluates five distinct classification models:
      * Bernoulli Naive Bayes
      * Linear Support Vector Machine (SVC)
      * Logistic Regression
      * Random Forest
      * XGBoost
  * **Optimized Performance:** Leverages parallel processing with `joblib` to speed up the data cleaning phase.

## 💾 Dataset

The project uses the **Sentiment140 dataset**.

  * **Source:** [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
  * **Size:** It consists of 1,600,000 tweets extracted using the Twitter API.
  * **Labels:** The tweets are automatically annotated, with polarity labels:
      * `0` = negative
      * `2` = neutral (This project removes neutral tweets to focus on a binary classification problem)
      * `4` = positive
  * For this project, the polarity values were mapped to `0` for negative and `1` for positive, creating a balanced dataset of 800,000 samples for each class.

## ⚙️ Methodology

The project follows a standard machine learning pipeline for NLP tasks.

### 1\. Data Preprocessing

The `clean_text` function performs several key steps to standardize the raw tweet text:

1.  **Lowercasing:** Converts all text to lowercase.
2.  **Emoji Handling:** Converts emojis into their text representations (e.g., `🙂` -\> `:slightly_smiling_face:`).
3.  **Token Replacement:** Uses regex to identify and replace special entities:
      * User mentions (`@username`) are replaced with `<USER>`.
      * Hashtags (`#topic`) are replaced with `<HASHTAG>`.
      * URLs (`http://...` or `www...`) are replaced with `<URL>`.
4.  **Negation Handling:** Preserves negation context by joining negation words with the following word (e.g., `not good` becomes `not_good`).
5.  **Punctuation Removal:** Efficiently strips all punctuation.
6.  **Tokenization:** Splits the text into individual words (tokens).
7.  **Stop Word Removal:** Removes common English stop words (e.g., 'a', 'the', 'in').
8.  **Lemmatization & Stemming:** Reduces words to their root forms using both `WordNetLemmatizer` and `PorterStemmer` for normalization.

### 2\. Feature Engineering

  * **TF-IDF Vectorization:** The cleaned text is converted into a matrix of TF-IDF features using `TfidfVectorizer`.
 
### 3\. Model Training

The dataset was split into an 80% training set and a 20% testing set. The following models were trained on the TF-IDF features:

  * Bernoulli Naive Bayes
  * LinearSVC
  * Logistic Regression
  * Random Forest Classifier
  * XGBoost Classifier

## 📊 Results

The models were evaluated based on their accuracy on the test set. Logistic Regression performed the best among the tested models.

| Model                       | Accuracy Score |
| --------------------------- | :------------: |
| **Logistic Regression** |   **80.23%** |
| Linear SVC (SVM)            |     80.00%     |
| Bernoulli Naive Bayes       |     78.35%     |
| XGBoost                     |     76.58%     |
| Random Forest               |     72.99%     |

Detailed classification reports including precision, recall, and F1-score for each class can be found in the notebook's output.

## 🚀 Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/niyatimahajan08/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    
4.  **Install the required packages:**

    ```bash
    pip install pandas scikit-learn nltk emoji xgboost joblib
    ```

5.  **Download NLTK data:**
    Run the following commands in a Python interpreter:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    ```

## 🏃 Usage

1.  **Download the Dataset:** Obtain the `twitter_dataset.csv` from the Sentiment140 website and place it in the project's root directory or update the file path in the notebook.
2.  **Run the Jupyter Notebook:** Launch Jupyter Notebook or Jupyter Lab and open `Twitter Sentiment Analysis.ipynb`.
    ```bash
    jupyter notebook "Twitter Sentiment Analysis.ipynb"
    ```
3.  **Execute the cells:** Run the notebook cells sequentially to perform data loading, preprocessing, model training, and evaluation.
      * **Note:** The text preprocessing step is computationally intensive and may take a significant amount of time. After the first run, it saves the cleaned data to `twitter_cleaned_dataset.csv`, which can be loaded directly in subsequent runs to save time.
