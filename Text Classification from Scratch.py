# Importing the necessary modules
import nltk
import re
import string
import pandas as pd
import numpy as np

nltk.download('twitter_samples')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Load the train and test data
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

positive_train_tweets = positive_tweets[:4000]
positive_test_tweets = positive_tweets[4000:]
negative_train_tweets = negative_tweets[:4000]
negative_test_tweets = negative_tweets[4000:]

# Assign label 1 for positive tweets and 0 for negative tweets
train_positive_df = pd.DataFrame(positive_train_tweets)
train_positive_df['label'] = 1
train_negative_df = pd.DataFrame(negative_train_tweets)
train_negative_df['label'] = 0

# Create the dataset
df = pd.concat([train_positive_df, train_negative_df], axis=0)
df = df.reset_index(drop=True)
df = df.rename(columns={0: 'text'})

# Load the stopwords
stop_words = set(stopwords.words('english'))


# Define the functions for prepocessing text, creating sequences and padding
class textPreprocessing():
    """
    Prepossesses and cleans the input text by removing urls, 
    punctuations, and stopwords.
    """

    def remove_url(text):
        """
        :param text: input string
        :return:     string without url
        """
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r"", text)

    def remove_punct(text):
        """
        :param text: input string
        :return:     string without punctuations
        """
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def remove_stopwords(text):
        """
        :param text: input string
        :return:     string without stopwords
        """
        filtered_words = [word.lower() for word in text.split() if word.lower()\
                          not in stop_words]
        return " ".join(filtered_words)


class textConversion():
    """
    Counts the occurrence of each word in the text, creates an index 
    dictionary for each unique word, converts text to sequences, and applies
    padding so all sequences have the same length.
    """

    def count_frequency(df, text):
        """
        :param text: input string
        :return:     counts the frequency of each word
        """
        freq_dict = {}
        for row in df[text]:
            for word in row.split():  # tokenize
                if word not in freq_dict:
                    freq_dict[word] = 1
                else:
                    freq_dict[word] += 1
        return freq_dict

    def find_unique_words(df, text):
        """
        :param text: input string
        :return:     list of unique words
        """
        unique_words = []
        for row in df[text]:
            for word in row.split():
                if word not in unique_words:
                    unique_words.append(word)
        unique_words = sorted(list(set(unique_words)))
        return unique_words

    def index_assignment(unique_words):
        """
        :return: assigned index to each unique word
        """
        index_dict = {}
        count = 0
        for word in unique_words:
            count += 1
            index_dict[word] = count
        return index_dict

    def text_to_sequences(df, text, index_dict):  # convert text to sequence
        """
        :param text:        input string
        :param index_dict:  dictionary with the unique index for each word
        :return:            the index for each word present in the input string
        """
        text_sequences = []
        for row in df[text]:
            empty = []
            for word in row.split():
                empty.append(index_dict.get(word))
            text_sequences.append(empty)
        return text_sequences

    def padded_sequences(text_sequences, max_len, pad_val):
        """
        :param max_len: max length of the padded sequences
        :param pad_val: value to fill with padding
        :return:        applies padding so all text sequences have the same length
        """
        padded_sequences = []
        for text_sequence in text_sequences:
            length = len(text_sequence)
            i = 0
            while length + i < max_len:
                text_sequence.insert(i, pad_val)
                i += 1
            padded_sequences.append(text_sequence)
        return padded_sequences


# Apply the functions in the training data 
df['text'] = df['text'].apply(lambda x: textPreprocessing.remove_url(x))
df['text'] = df['text'].apply(lambda x: textPreprocessing.remove_punct(x))
df['text'] = df['text'].apply(lambda x: textPreprocessing.remove_stopwords(x))

# Limit the sentences to 30 words max  
max_len = 30

frequency_dict = textConversion.count_frequency(df, 'text')
unique_words = textConversion.find_unique_words(df, 'text')
index_dictionary = textConversion.index_assignment(unique_words)
train_sequences = textConversion.text_to_sequences(df, 'text', index_dictionary)
train_padded = textConversion.padded_sequences(train_sequences, max_len=max_len,\
                                               pad_val=0)
# Create X_train and y_train datasets 
X_train = train_padded 
y_train = df['label'].tolist() 

# Classify text with Binary Labels using Logistic Regression
class LogisticRegression():
  def __init__(self,lr:float, n_iters:int):
    self.weights = None 
    self.bias = None 
    # Hyperparameters
    self.lr = lr 
    self.n_iters = n_iters

  @ staticmethod
  def accuracy(y_train,predictions):
    counter = 0 
    for label, prediction in zip(y_train,predictions):
      if label == prediction:
        counter +=1 
    return print(f'Accuracy is {counter / len(y_train) * 100}%.')

  def fit(self,X_train,y_train):
    '''
    Fit the Logistic Regression Model to X_train and y_train 
    ''' 
    m = 1 / len(X_train)
    # Initialize the weights and bias with random values 
    self.weights = [0.5 for x in range(len(X_train[0]))]
    self.bias = 0.5 

    # Training loop 
    for _ in range(self.n_iters):
      predictions = []
      for features, label in zip(X_train,y_train):
          derivatives = []
          prediction = 0
          actual = label 
          for weight, feature in zip(self.weights,features):
              prediction += feature * weight
          prediction += self.bias
          sigmoid = 1 / (1 + np.exp(-prediction))
          predictions.append(round(sigmoid))

          for i in range(len(self.weights)):
              # Calculate derivatives
              dw = (1 / m) * (features[i] * (sigmoid - actual)) * self.lr
              derivatives.append(dw)

          for i, derivative in enumerate(derivatives):
              # Update weights
              self.weights[i] -= derivative

          # Calculate bias
          db = (1 / m) * (sigmoid - actual) * self.lr
          # Update bias
          self.bias -= db

    return self.accuracy(y_train,predictions) 
  
  def predict(self,X_test):
    '''
    Make predctions for new data 
    '''
    predictions = []
    for features in X_test:
      prediction = 0 
      for weight,feature in zip(self.weights, features):
        prediction += feature * weight
      prediction += self.bias
      sigmoid = 1 / (1 + np.exp(-prediction))
      predictions.append(round(sigmoid))
    return predictions 
