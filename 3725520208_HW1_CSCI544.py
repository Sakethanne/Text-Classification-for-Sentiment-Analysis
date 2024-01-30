#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
import re            # Regular expressions for text processing
from bs4 import BeautifulSoup  # For HTML parsing
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import nltk          # Natural Language Toolkit for text processing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Download WordNet data
nltk.download('stopwords')   # Download StopWords data

import warnings      # To handle warnings
warnings.filterwarnings("ignore")  # Ignore warnings for the remainder of the code
warnings.filterwarnings("default")  # Set warnings back to default behavior


# In[2]:


# ! pip install bs4 # in case you don't have it installed
# ! pip install contractions # in case contractions are not already installed

# # Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[3]:


# Reading the data from the tsv (Amazon Kitchen dataset) file as a Pandas frame
full_data = pd.read_csv("./amazon_reviews_us_Office_Products_v1_00.tsv", delimiter='\t', encoding='utf-8', error_bad_lines=False)


# In[4]:


# Printing the data frame that contains the entire dataset from the tsv file
# print(full_data)


# ## Keep Reviews and Ratings

# In[5]:


# Keep only the Reviews and Ratings fields from the full data
data = full_data[['review_body', 'star_rating', 'review_headline']]

# Converting 'star_rating' to numeric values
data['star_rating'] = pd.to_numeric(data['star_rating'], errors='coerce')

# Displaying three sample reviews along with ratings
sample_reviews = data.sample(3)
# print("=========================Sample Reviews:===========================")
# print(sample_reviews)


#  ##  Form two classes and select 100000 reviews randomly from each class.

# In[6]:
# Reporting statistics of the ratings
ratings_statistics = data['star_rating'].value_counts().sort_index()
print("\n========================Ratings Statistics:============================")
print("Ratings Count:")
print(ratings_statistics)


# Creating binary labels for sentiment analysis
data['sentiment'] = data['star_rating'].apply(lambda x: 1 if x > 3 else 0 if x <= 2 else None)

# Discarding neutral reviews (rating 3)
data = data.dropna(subset=['sentiment'])

# Selecting 100,000 positive and 100,000 negative reviews
positive_reviews = data[data['sentiment'] == 1].sample(100000, random_state=42)
negative_reviews = data[data['sentiment'] == 0].sample(100000, random_state=42)

# Concatenating positive and negative reviews into a single data set for further test and train set split
selected_reviews = pd.concat([positive_reviews, negative_reviews])

# Printing the reviews that have been selected for further processing randomly
# print(selected_reviews)


#  ## Split the dataset into training and testing dataset
# 
# 

# In[8]:


# Splitting the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(selected_reviews['review_body'],
                                                    selected_reviews['sentiment'],
                                                    test_size=0.2,
                                                    random_state=42)


# In[9]:


# Printing the Features of the training set
# print(X_train)


# In[10]:


# Printing the Features of the testing set
# print(X_test)


# In[11]:


# Printing the Target(s) of the training set
# print(y_train)


# In[12]:


# Printing the Target(s) of the testing set
# print(y_test)


# # Data Cleaning
# 
# 

# In[13]:


# Define a contraction map
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "we're": "we are",
    "they're": "they are",
    "isn't": "is not",
    "aren't": "are not",
    "haven't": "have not",
    "hasn't": "has not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "won't've": "will not have",
    "can't've": "cannot have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "that'll": "that will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "it'd": "it would",
    "that'd": "that would",
    "we'd": "we would",
    "they'd": "they would",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "oughtn't": "ought not",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "it's": "it is",
    "let's": "let us"
}

# Function to expand contractions
def expand_contractions(text):
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(contraction, expansion, text)
    return text

# Preprocess the reviews
def preprocess_reviews(reviews):
    # Convert to lowercase and handle NaN values
    reviews = reviews.apply(lambda x: str(x).lower() if pd.notna(x) else '')
    
    # Remove HTML and URLs
    reviews = reviews.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    reviews = reviews.apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove non-alphabetical characters (excluding single quote)
    reviews = reviews.apply(lambda x: re.sub(r'[^a-zA-Z\s\']', '', x))

    # Remove extra spaces
    reviews = reviews.apply(lambda x: re.sub(' +', ' ', x))

    # Perform contractions
    reviews = reviews.apply(expand_contractions)

    # Return the processed text of the review
    return reviews

# Preprocess the training set
X_train_preprocessed = preprocess_reviews(X_train)

# Print average length of reviews before and after cleaning
avg_length_before = X_train.apply(lambda x: len(str(x))).mean()
avg_length_after = X_train_preprocessed.apply(len).mean()
print("\n===================Printing the Average lenght of Reviews Before and After Cleaning====================")
print(f"{int(avg_length_before)}, {int(avg_length_after)}")


# # Pre-processing
# 
# ### -- remove the stop words
# ### -- perform lemmatization

# In[14]:


# Initialize NLTK's stopwords and WordNet lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to remove stop words and perform lemmatization
def preprocess_nltk(review):
    if pd.notna(review):
        words = nltk.word_tokenize(str(review).lower())  # Convert to lowercase
        words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        return ' '.join(words)
    else:
        return ''

# Preprocess the training set using NLTK
X_train_nltk_preprocessed = X_train_preprocessed.apply(preprocess_nltk)

# Print three sample reviews before and after NLTK preprocessing
sample_reviews_indices = X_train_preprocessed.sample(3).index

# print("============ Printing Sample Reviews Before and After Pre-processing =============")
# for index in sample_reviews_indices:
    # print(f"\nSample Review {index} Before Pre-processing:")
    # print(X_train_preprocessed.loc[index])

    # print(f"\nSample Review {index} After NLTK Pre-processing:")
    # print(X_train_nltk_preprocessed.loc[index])

# Print average length of reviews before and after NLTK processing
avg_length_before_nltk = X_train_preprocessed.apply(len).mean()
avg_length_after_nltk = X_train_nltk_preprocessed.apply(len).mean()
print("\n=================Printing the Average lenght of Reviews Before and After Pre-processing==================")
print(f"{int(avg_length_before_nltk)}, {int(avg_length_after_nltk)}")


# # TF-IDF Feature Extraction

# In[15]:


# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000000)

# Fit and transform the training set
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_nltk_preprocessed)

# Transform the test set
X_test_tfidf = tfidf_vectorizer.transform(X_test.apply(preprocess_nltk))

# Print the shape of the TF-IDF matrices
# print(f"\nShape of X_train_tfidf: {X_train_tfidf.shape}")
# print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}")


# # Perceptron

# In[16]:


# Initialize the Perceptron model
perceptron_model = Perceptron(random_state=42)

# Train the Perceptron model on the TF-IDF features
perceptron_model.fit(X_train_tfidf, y_train)

# Predictions on the training set
y_train_pred = perceptron_model.predict(X_train_tfidf)

# Predictions on the test set
y_test_pred = perceptron_model.predict(X_test_tfidf)

# Calculate metrics for the training set
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

# Calculate metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

# Print the results
print(f"\n================== Training Set Metrics: (Perceptron) ===================")
print(f"{accuracy_train}, {precision_train}, {recall_train}, {f1_train}")

print(f"\n================== Testing Set Metrics: (Perceptron) ====================")
print(f"{accuracy_test}, {precision_test}, {recall_test}, {f1_test}")


# # SVM

# In[17]:


# Initialize the SVM model
svm_model = SVC(random_state=42)

# Train the SVM model on the TF-IDF features
svm_model.fit(X_train_tfidf, y_train)

# Predictions on the training set
y_train_pred_svm = svm_model.predict(X_train_tfidf)

# Predictions on the test set
y_test_pred_svm = svm_model.predict(X_test_tfidf)

# Calculate metrics for the training set
accuracy_train_svm = accuracy_score(y_train, y_train_pred_svm)
precision_train_svm = precision_score(y_train, y_train_pred_svm)
recall_train_svm = recall_score(y_train, y_train_pred_svm)
f1_train_svm = f1_score(y_train, y_train_pred_svm)

# Calculate metrics for the test set
accuracy_test_svm = accuracy_score(y_test, y_test_pred_svm)
precision_test_svm = precision_score(y_test, y_test_pred_svm)
recall_test_svm = recall_score(y_test, y_test_pred_svm)
f1_test_svm = f1_score(y_test, y_test_pred_svm)

# Print the results
print(f"\n================== Training Set Metrics: (SVM) ====================")
print(f"{accuracy_train_svm}, {precision_train_svm}, {recall_train_svm}, {f1_train_svm}")

print(f"\n================== Testing Set Metrics: (SVM) ====================")
print(f"{accuracy_test_svm}, {precision_test_svm}, {recall_test_svm}, {f1_test_svm}")

# # Logistic Regression

# In[18]:


# Initialize the Logistic Regression model
logreg_model = LogisticRegression(random_state=42)

# Train the Logistic Regression model on the TF-IDF features
logreg_model.fit(X_train_tfidf, y_train)

# Predictions on the training set
y_train_pred_logreg = logreg_model.predict(X_train_tfidf)

# Predictions on the test set
y_test_pred_logreg = logreg_model.predict(X_test_tfidf)

# Calculate metrics for the training set
accuracy_train_logreg = accuracy_score(y_train, y_train_pred_logreg)
precision_train_logreg = precision_score(y_train, y_train_pred_logreg)
recall_train_logreg = recall_score(y_train, y_train_pred_logreg)
f1_train_logreg = f1_score(y_train, y_train_pred_logreg)

# Calculate metrics for the test set
accuracy_test_logreg = accuracy_score(y_test, y_test_pred_logreg)
precision_test_logreg = precision_score(y_test, y_test_pred_logreg)
recall_test_logreg = recall_score(y_test, y_test_pred_logreg)
f1_test_logreg = f1_score(y_test, y_test_pred_logreg)

# Print the results
print(f"\n================== Training Set Metrics: (Logistic Regression) ====================")
print(f"{accuracy_train_logreg}, {precision_train_logreg}, {recall_train_logreg}, {f1_train_logreg}")

print(f"\n================== Testing Set Metrics: (Logistic Regression) ====================")
print(f"{accuracy_test_logreg}, {precision_test_logreg}, {recall_test_logreg}, {f1_test_logreg}")


# # Naive Bayes

# In[19]:


# Initialize the Multinomial Naive Bayes model
nb_model = MultinomialNB()

# Train the Multinomial Naive Bayes model on the TF-IDF features
nb_model.fit(X_train_tfidf, y_train)

# Predictions on the training set
y_train_pred_nb = nb_model.predict(X_train_tfidf)

# Predictions on the test set
y_test_pred_nb = nb_model.predict(X_test_tfidf)

# Calculate metrics for the training set
accuracy_train_nb = accuracy_score(y_train, y_train_pred_nb)
precision_train_nb = precision_score(y_train, y_train_pred_nb)
recall_train_nb = recall_score(y_train, y_train_pred_nb)
f1_train_nb = f1_score(y_train, y_train_pred_nb)

# Calculate metrics for the test set
accuracy_test_nb = accuracy_score(y_test, y_test_pred_nb)
precision_test_nb = precision_score(y_test, y_test_pred_nb)
recall_test_nb = recall_score(y_test, y_test_pred_nb)
f1_test_nb = f1_score(y_test, y_test_pred_nb)

# Print the results
print(f"\n================== Training Set Metrics: (Multinomial Naive Bayes) ====================")
print(f"{accuracy_train_nb}, {precision_train_nb}, {recall_train_nb}, {f1_train_nb}")

print(f"\n================== Testing Set Metrics: (Multinomial Naive Bayes) ====================")
print(f"{accuracy_test_nb}, {precision_test_nb}, {recall_test_nb}, {f1_test_nb}")