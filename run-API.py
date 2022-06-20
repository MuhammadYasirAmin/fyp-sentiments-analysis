import requests
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pickle
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
from flask import Flask , request


app =Flask(__name__)
def data():
    url="https://optimal.solar/Sentiments-Analysis/fetchReviews.php"
    headers = {'Content-Type': 'application/json' }
    response = requests.get(url=url,headers=headers)
    df = pd.read_json(response.text)
    reviews = df.Customer_review.tolist()
    if df.empty:
        pass
    else:
        return reviews

def ML_Predict(text):
    model = pickle.load(open('model.pkl', 'rb'))
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    #text1 = vectorizer.transform([review]).toarray()
    if model.predict([review]) == 0:
        prediction =False 
    elif model.predict([review]) == 1: 
        prediction = True
    return prediction
def data_perprocessing(text):
    #clean
    text = re.sub('[^A-Za-z]+', ' ', text)
    # POS tagger dictionary
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    #lemma
    lemma_rew = " "
    wordnet_lemmatizer = WordNetLemmatizer()
    for word, pos in newlist:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew
def getPolarity(review):
    mean_list = []
    #polarity by textblob
    text_polarity = TextBlob(review).sentiment.polarity
    mean_list.append(text_polarity)
    #polarity by VADER
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(review)
    mean_list.append(vs['compound'])
    #mean calculate
    mean = statistics.mean(mean_list)
    return mean
def analysis(score):
    if score < 0:
        return False
    elif score == 0:
        return None
    else:
        return True
def predict(text):
    process_text = data_perprocessing(text)
    polarity_value = getPolarity(process_text)
    prediction =  analysis(polarity_value)
    return prediction

def results(review):
    if str(type(review)) == "<class 'list'>":
        Ans = []
        for i in review:
            RB = predict(i)
            ML = ML_Predict(i)
            if RB == True and ML == True:
                Ans.append(True)
            elif RB == False and ML == False:
                Ans.append(False)
            else:
                Ans.append(None)
        return Ans
    elif str(type(review)) == "<class 'str'>":
        RB = predict(review)
        ML = ML_Predict(review)
        if RB == True and ML == True:
            return True
        elif RB == False and ML == False:
            return False
        else:
            return None 


@app.route('/', methods = ['GET'])
def index():
    review = data()
    ans = results(review[0])
    return str(ans)
if __name__ == '__main__':
    app.run(debug = False)   