#!pip3 install yfinance
#pip install python-dateutil

import urllib.request  
import bs4 as bs
import time

import yfinance as yf
import time
from datetime import date
import datetime as dt
import pandas as pd
import numpy as np
import string

from dateutil import parser as date_parser

#The function gathers together all the recent articles corresponding to the specified ticker, extracting the headlines for each article and returns a pandas dataframe including headline and the date as strings.

def get_titles(ticker):
    
    url = 'https://news.google.com/rss/search?hl=en-US&q=' + ticker + '&gl=US&ceid=US:en'
    time.sleep(15)
    
    doc = urllib.request.urlopen(url).read()
    parsed_doc = bs.BeautifulSoup(doc,'lxml')
  
    text = parsed_doc.find_all('title')[1:100]
    dates = parsed_doc.find_all('pubdate')[1:100]
    
    dates_cut = [str(x)[9:-10] for x in dates]
    new_date = [str(date_parser.parse(y).date()) for y in dates_cut]
    
    text_cut = [str(z)for z in text]
    new_text = [cleanhtml(a)for a in text_cut]
    
    df = pd.DataFrame()
    df['date'] = new_date
    df['headline'] = new_text
      
    return df
    
   #The function cleans the HTML style text from tags
   
   import re

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

#The function obtains the prices of the specified ticker, according to input arguments, calculates the daily returns and labels them accordingly.

def get_label(ticker='TSLA',start_date = '2020-01-01',end_date=date.today(),period='1d'):

    share_prices = yf.Ticker(ticker)

    #get the historical prices 
    train_data = share_prices.history(period=period, start=start_date, end=end_date)
    
    #label the data
    train_data['date'] = train_data.index.astype(str)
    train_data['daily_change'] = train_data.Close / train_data.Close.shift(1) - 1
    train_data.loc[train_data['daily_change'] <= -0.03, 'Label'] = 'Negative' 
    train_data.loc[train_data['daily_change'] > 0.03, 'Label'] = 'Positive' 
    
    return train_data[1:]
    
    #The function joins the data from the two previous functions in order to return a dataset containing headlines and labels without empty fields, allowing for multiple headlines per day.

def create_dataset(ticker):
    
    a = get_label(ticker)
    b = get_titles(ticker)
    
    merged_data = pd.merge(a, b, how = 'left', on=["date", "date"])
    
    return merged_data[['Label','headline','date']].dropna()
    
    #The function loops over create_dataset for a list of tickers given as arguments in order to return a unified dataset containing multiple stocks with corresponding headlines and labels.
    
   def merge_datasets(ticker):
    
    data = create_dataset(ticker[0])
    
    if len(ticker) != 1:
    
        for i in range(1,len(ticker)):

            temp = create_dataset(ticker[0+i])
            data = pd.concat([data, temp])
            
    else:
        
        data = data
        
    return data 
    
tickerlist = ['aapl','msft','amzn','goog','tsla','nvda','pypl','nflx','csco','avgo',"abb","apd",'gme','amc', "are", 'cop', 'cp', 'csx', 'eog', 'epam' ,'orcl','qcom','c','jpm','ba','a','ivz','pltr','vgt','jnj','fb','nflx','mrna','pfe','azn','rkt']
    
input_data = merge_datasets(tickerlist)

headlines = list(input_data['headline'])
labels = list(input_data["Label"])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(headlines, labels, test_size=0.2)

#Aiming to convert the text data into a vector in order to interact with a classifier.

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_x)

#This could be used to filter the most frequent words appearing in a minimum number of headlines.

vectorizer = CountVectorizer(min_df = 1 , stop_words='english')
vectorizer.fit(train_x)

print("vocabulary size: "+str(len(vectorizer.get_feature_names())))

 #Generates the bag-of-word representation for each headline present in the train set.
 
train_x_vector = vectorizer.transform(train_x)

vocabulary = vectorizer.get_feature_names()

from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression().fit(train_x_vector, train_y)

#Testing the model on some sample headlines.

# Obtaining some random headlines in order to test how well our predictions would work.
headlines_sample = []
headlines_sample.append('Why Tesla Stock Is Roaring Back on Monday')
headlines_sample.append('Apple (AAPL) Down 11.7% Since Last Earnings Report: Can It Rebound?')
headlines_sample.append('Apple stock has biggest day since Oct. 12 after Buffett endorsement, stores reopen')
headlines_sample.append('Tesla (TSLA) Extends Breakdown Amid Valuation Concerns')
headlines_sample.append('Why Apple\'s iPhone 13 Could Be A \'Game Changer\' With 1TB Storage Option, Lidar')
headlines_sample.append('JPMorgan (JPM) Breaks Out to All-Time High')
headlines_sample.append('Barron\'s Latest Picks And Pans: Berkshire Hathaway, Citigroup, Dow, Twitter And More Warren Buffett\'s favorite stock market indicator still screams sell')

# Result of model's prediction
transformed_headlines = vectorizer.transform(headlines_sample)
predictions = logit_model.predict(transformed_headlines)
print(predictions)

# Probabilities of predictions
predicted_probabilities = logit_model.predict_proba(transformed_headlines)
print(predicted_probabilities)

#The parameters of the model must be analysed order to investigate the performance of the model produced.

model_params = [(vocabulary[j],logit_model.coef_[0][j]) for j in range(len(vocab))]
model_params

#Testing the model predictions on the test set.

test_x_vector = vectorizer.transform(test_x)
pred_y = logit_model.predict(test_x_vector)

#Producing an accuracy score of the model, and plotting the corresponding confusion matrix.

# Calculating accuracy of model
from sklearn.metrics import accuracy_score
print('accuracy: '+ str(accuracy_score(pred_y, test_y)))

# Plot the confusion matrix
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(logit_model, test_x_vector, test_y, values_format='d')  
plt.show()  

#Importing Google News' Word2Vec embedding model.

import gensim.downloader as api
model_google = api.load("word2vec-google-news-300")

#Next step is to convert the headlines into an embedding representation.
#It is achieved by computing the sum (or average) of the embedding vectors of the words in the headline.
#First converting everything in lowercase and then tokenise it, then obtaining the embedding vectors for the tokens situated in the embedding vocabular, lastly vectorizing such embeddings either using the sum or average method, as previously mentioned.

regex = '['+string.punctuation+']'

def vectorize(docs, embedding_model = model_google, useSum=True):
    vectors = np.zeros((len(docs),300))
  
    for i in range(len(docs)):
        
    #Convert to lower-case and tokenise
    tokens = re.sub(regex,'',docs[i].lower()).split()
    embeddings = [embedding_model.get_vector(token) for token in tokens if token in embedding_model.vocab]
    
    if (len(embeddings) > 0):
        
        # sum of embeddings
        if (useSum): 
        vectors[i] = sum(embeddings)
        
        # average of embeddings
        else:
        vectors[i] = np.mean(embeddings,axis=0)
  
    return vectors
    
    #Split the dataset dividing it into train, validation and test set.
    
    from sklearn.model_selection import train_test_split

temp_x, test_x, temp_y, test_y = train_test_split(headlines, labels, test_size=0.2)
train_x, valid_x, train_y, valid_y = train_test_split(temp_x, temp_y, test_size=0.2)

#Vecotrize the train set.
train_x_vector = vectorize(train_x)

from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression(max_iter=1000).fit(train_x_vector, train_y)

headlines_sample = []
headlines_sample.append('Why Tesla Stock Is Roaring Back on Monday')
headlines_sample.append('Apple (AAPL) Down 11.7% Since Last Earnings Report: Can It Rebound?')
headlines_sample.append('Apple stock has biggest day since Oct. 12 after Buffett endorsement, stores reopen')
headlines_sample.append('Tesla (TSLA) Extends Breakdown Amid Valuation Concerns')
headlines_sample.append('Why Apple\'s iPhone 13 Could Be A \'Game Changer\' With 1TB Storage Option, Lidar')
headlines_sample.append('JPMorgan (JPM) Breaks Out to All-Time High')
headlines_sample.append('Barron\'s Latest Picks And Pans: Berkshire Hathaway, Citigroup, Dow, Twitter And More Warren Buffett\'s favorite stock market indicator still screams sell')

transformed_headlines = vectorize(headlines_sample)
predictions = logit_model.predict(transformed_headlines)
predicted_probabilities = logit_model.predict_proba(transformed_headlines)

for i in range(len(headlines)):
    print('headline: ',headlines[i])
    print('prediction: ',predictions[i])
    print('confidence: ',predicted_probabilities[i])
    print()


#Producing an accuracy score of the model, and plotting the corresponding confusion matrix for the validation set.

valid_x_vector = vectorize(valid_x)
pred_y = logit_model.predict(valid_x_vector)

# Calculating accuracy
from sklearn.metrics import accuracy_score
print('accuracy: '+ str(accuracy_score(pred_y, valid_y)))

# Plot the confusion matrix
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(logit_model, valid_x_vector, valid_y, values_format='d')  
plt.show() 

#Producing an accuracy score of the model, and plotting the corresponding confusion matrix for the test set.

test_x_vector = vectorize(test_x)
pred_y = logit_model.predict(test_x_vector)

from sklearn.metrics import accuracy_score
print('accuracy: '+ str(accuracy_score(pred_y, test_y)))

import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(logit_model, test_x_vector, test_y, values_format='d')  
plt.show() 

#Plotting different accuracy results, based on different chosen thresholds.
import matplotlib.pyplot as plt

import pandas as pd


data = {'0.0': {"Embedding":0.535,"Bag of words":0.549}, '0.01': {"Embedding":0.545, "Bag of words":0.584}, '0.02': {"Embedding":0.683,"Bag of words":0.690 }, '0.03': {"Embedding":0.74,"Bag of words":0.80}, '0.04': {"Embedding":0.809,"Bag of words":0.823}, '0.05': {"Embedding":0.848,"Bag of words":0.830}, '0.06': {"Embedding":0.866,"Bag of words":0.905 }, '0.1': {"Embedding":0.927,"Bag of words":0.920 }}



df = pd.DataFrame(data)



df.plot(kind='bar')



plt.show()



