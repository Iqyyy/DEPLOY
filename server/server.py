import matplotlib
matplotlib.use('Agg')
import os
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import nltk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

nltk.download('punkt')

file_path_plain = 'DATA_BARU/processed_tni5.csv'
file_path_nb = 'DATA_BARU/predict_model_nb_tni5.csv'
file_path_dt= 'DATA_BARU/predict_model_dt_tni5.csv'

def process_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace('[^\w\s]', '')
    sentence = sentence.strip()
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    sentence = stopword_remover.remove(sentence)
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    sentence = stemmer.stem(sentence)
    return [sentence]

def readCSV(file_path) :
    df = pd.read_csv(file_path)

    return df

def predict_class(model_path, tfidf_vectorizer_path, sentence):
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

    text_tf = tfidf_vectorizer.transform([sentence])

    predicted_class = model.predict(text_tf)
    return predicted_class[0]

@app.route('/check-sentence', methods=['POST'])
def check_sentence():
    data = request.get_json()
    sentence = data['sentence']

    # Process sentence
    processed_sentence = process_sentence(sentence)

    # Predict using Naive Bayes model
    nb_model_path = 'DATA_BARU/naive_bayes_model_tni5.pkl'
    nb_tfidf_vectorizer_path = 'DATA_BARU/tfidf_vectorizer_nb.pkl'
    nb_predicted_class = predict_class(nb_model_path, nb_tfidf_vectorizer_path, processed_sentence)

    # Predict using Decision Tree model
    dt_model_path = 'DATA_BARU/decision_tree_model_tni5.pkl'
    dt_tfidf_vectorizer_path = 'DATA_BARU/tfidf_vectorizer_dt.pkl'
    dt_predicted_class = predict_class(dt_model_path, dt_tfidf_vectorizer_path, processed_sentence)

    return jsonify({
        "Naive Bayes": nb_predicted_class,
        "Decision Tree": dt_predicted_class
    })

@app.route('/accuracynbdt', methods=['GET'])
def get_accuracynbdt():

    image_path = 'GAMBAR/accuracy_nbdt.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/kFoldNaiveBayes', methods=['GET'])
def get_kFoldNaiveBayes():

    image_path = 'kfold_images/kfold_nb.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/kFoldDecisionTree', methods=['GET'])
def get_kFoldDecisionTree():

    image_path = 'kfold_images/kfold_dt.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/cmNaiveBayes', methods=['GET'])
def get_cmNaiveBayes():

    image_path = 'GAMBAR/cm_nb.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/cmDecisionTree', methods=['GET'])
def get_cmDecisionTree():

    image_path = 'GAMBAR/cm_dt.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/pcNaiveBayes', methods=['GET'])
def get_pcNaiveBayes():

    image_path = 'GAMBAR/piechart_nb.png'
    return send_file(image_path, mimetype='image/png')

@app.route('/pcDecisionTree', methods=['GET'])
def get_pcDecisionTree():

    image_path = 'GAMBAR/piechart_dt.png'
    return send_file(image_path, mimetype='image/png')


if __name__ == '__main__':
    # Printing message when the server has successfully started
    print("Server is running!")
    app.run(debug=True)
