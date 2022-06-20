from flask import Flask, render_template, request, jsonify
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
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/', methods=['GET'])
def home():
    return render_template('sentiment_index.html')

def ML_Predict(text):
    model = pickle.load(open('model.pkl', 'rb'))
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    if model.predict([review]) == 0:
        prediction ='NEGATIVE' 
    elif model.predict([review]) == 1: 
        prediction ='POSITIVE'
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
        return 'NEGATIVE'
    else:
        return 'POSITIVE'
def predict(text):
    process_text = data_perprocessing(text)
    polarity_value = getPolarity(process_text)
    prediction =  analysis(polarity_value)
    return prediction
@app.route('/', methods=['POST'])

def webapp():
    text = request.form['text']
    prediction1 = predict(text)
    prediction2 = ML_Predict(text)
    prediction = [prediction1,prediction2]
    return render_template('sentiment_index.html', text=text, result=prediction)
    
@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)
if __name__ == "__main__":
    app.run()