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

# Load NLTK resources
nltk.download('punkt')

file_path_plain = 'DATA_BARU/processed_tni5.csv'
file_path_nb = 'DATA_BARU/predict_model_nb_tni5.csv'
file_path_dt= 'DATA_BARU/predict_model_dt_tni5.csv'


# def process_data(file_path):
#     df = pd.read_csv(file_path)

#     to_drop = ["id_str", "quote_count", "reply_count", "retweet_count", "favorite_count",
#                "lang", "user_id_str", "conversation_id_str", "PUSPEN/POLRI/DLL", "LINK"]
#     df.drop(to_drop, inplace=True, axis=1)

#     df["full_text"] = df["full_text"].str.lower()
#     df["full_text"] = df["full_text"].str.replace('[^\w\s]', '')
#     df["full_text"] = df["full_text"].str.strip()

#     stopword_factory = StopWordRemoverFactory()
#     stopword_remover = stopword_factory.create_stop_word_remover()
#     df["full_text"] = df["full_text"].apply(stopword_remover.remove)

#     stemmer_factory = StemmerFactory()
#     stemmer = stemmer_factory.create_stemmer()
#     df["full_text"] = df["full_text"].apply(stemmer.stem)

#     df["full_text"] = df["full_text"].apply(nltk.word_tokenize)
#     df["value"] = df["value"].astype('category')
#     df["full_text"] = df["full_text"].astype('string')

#     return df

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


def train_and_evaluate(classifier, X_train, X_test, y_train, y_test):
    print("Training and evaluating model...")
    if classifier == 'MultinomialNB':
        clf = MultinomialNB()
    elif classifier == 'DecisionTree':
        clf = DecisionTreeClassifier()
    else:
        return jsonify({"error": "Invalid classifier. Supported classifiers are 'MultinomialNB' and 'DecisionTree'."}), 400

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)
    confusion = confusion_matrix(y_test, predicted).tolist()
    report = classification_report(y_test, predicted, zero_division=0, output_dict=True)

    cm = confusion_matrix(y_test, predicted, labels=clf.classes_).tolist()

    classes = clf.classes_.tolist()
    return {"accuracy": accuracy, "confusion_matrix": confusion, "classification_report": report, "cm":cm, "classes" :classes}


@app.route('/naivebayes', methods=['GET'])
def get_naivebayes_result():
    file_path = file_path_nb
    if not file_path:
        return jsonify({"error": "CSV file not found."}), 404
    processed_df = readCSV(file_path)
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(processed_df['full_text'].astype('U'))
    X_train, X_test, y_train, y_test = train_test_split(text_tf, processed_df['value'], test_size=0.2, random_state=42)
    result = train_and_evaluate('MultinomialNB', X_train, X_test, y_train, y_test)
    print("Successfully loaded data and trained MultinomialNB model.")
    return jsonify(result)


@app.route('/decisiontree', methods=['GET'])
def get_decisiontree_result():
    file_path = file_path_dt
    if not file_path:
        return jsonify({"error": "CSV file not found."}), 404
    processed_df = readCSV(file_path)
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(processed_df['full_text'].astype('U'))
    X_train, X_test, y_train, y_test = train_test_split(text_tf, processed_df['value'], test_size=0.2, random_state=42)
    result = train_and_evaluate('DecisionTree', X_train, X_test, y_train, y_test)
    print("Successfully loaded data and trained DecisionTree model.")
    return jsonify(result)


@app.route('/kfoldNb', methods=['GET'])
def get_kfoldNb_result():
    file_path = file_path_nb
    if not file_path:
        return jsonify({"error": "CSV file not found."}), 404
    processed_df = readCSV(file_path)
    X = processed_df['full_text'].astype('U')
    y = processed_df['value']
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    validation_results = []

    tfidf_vectorizer = TfidfVectorizer()  # Buat objek TfidfVectorizer
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        X_train = tfidf_vectorizer.fit_transform(X_train_fold)  # Gunakan objek tfidf_vectorizer yang baru
        X_val = tfidf_vectorizer.transform(X_val_fold)  # Gunakan objek yang sama untuk transformasi

        result = train_and_evaluate('MultinomialNB', X_train, X_val, y_train_fold, y_val_fold)
        validation_results.append(result)

    print("Successfully loaded data and performed K-Fold validation.")
    return jsonify(validation_results)

@app.route('/kfoldDt', methods=['GET'])
def get_kfoldDt_result():
    file_path = file_path_dt
    if not file_path:
        return jsonify({"error": "CSV file not found."}), 404
    processed_df = readCSV(file_path)
    X = processed_df['full_text'].astype('U')
    y = processed_df['value']
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    validation_results = []

    tfidf_vectorizer = TfidfVectorizer()  # Buat objek TfidfVectorizer
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        X_train = tfidf_vectorizer.fit_transform(X_train_fold)  # Gunakan objek tfidf_vectorizer yang baru
        X_val = tfidf_vectorizer.transform(X_val_fold)  # Gunakan objek yang sama untuk transformasi

        result = train_and_evaluate('DecisionTree', X_train, X_val, y_train_fold, y_val_fold)
        validation_results.append(result)

    print("Successfully loaded data and performed K-Fold validation.")
    return jsonify(validation_results)

@app.route('/cm/<classifier>', methods=['GET'])
def get_cm(classifier):
    if classifier == 'MultinomialNB':
        file_path = file_path_nb
    elif classifier == 'DecisionTree':
        file_path = file_path_dt
    else:
        return jsonify({"error": "Invalid classifier. Supported classifiers are 'MultinomialNB' and 'DecisionTree'."}), 400

    if not file_path:
        return jsonify({"error": "CSV file not found."}), 404
    processed_df = readCSV(file_path)
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(processed_df['full_text'].astype('U'))
    X_train, X_test, y_train, y_test = train_test_split(text_tf, processed_df['value'], test_size=0.2, random_state=42)
    result = train_and_evaluate(classifier, X_train, X_test, y_train, y_test)
    
    # Retrieve confusion matrix data
    cm_data = result['cm']
    classes = result['classes']
    
    # Convert confusion matrix to NumPy array
    cm_data_np = np.array(cm_data)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_data_np, display_labels=classes)
    disp.plot()
    
    # Determine the filename based on the classifier type
    filename = f'confusion_matrix_{classifier}.png'
    
    # Save the plot as an image
    plt.savefig(filename)
    
    # Return the path to the saved image
    image_path = os.path.abspath(filename)
    # return jsonify({"image_path": image_path})
    return send_file(image_path, mimetype='image/png')

# @app.route('/kfold/<classifier>', methods=['GET'])
# def get_kfold(classifier):
#     if classifier == 'MultinomialNB':
#         clf = MultinomialNB()
#         file_path = file_path_nb
#     elif classifier == 'DecisionTree':
#         clf = DecisionTreeClassifier()
#         file_path = file_path_dt
#     else:
#         return jsonify({"error": "Invalid classifier. Supported classifiers are 'MultinomialNB' and 'DecisionTree'."}), 400
    
#     if not file_path:
#         return jsonify({"error": "CSV file not found."}), 404
#     processed_df = readCSV(file_path)

#     X = processed_df['full_text'].astype('U')
#     y = processed_df['value']
#     k = 5
#     kf = KFold(n_splits=k)
#     validation_results = []

#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         tfidf_vectorizer = TfidfVectorizer()
#         X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
#         X_test_tfidf = tfidf_vectorizer.transform(X_test)

#         clf.fit(X_train_tfidf, y_train)
#         score = clf.score(X_test_tfidf, y_test)
#         validation_results.append(score)

#     # Plot the bar chart
#     plt.barh(range(k), validation_results, color='skyblue')
#     plt.axvline(x=np.mean(validation_results), color='red', linestyle='--', linewidth=1)
#     plt.xlabel('Accuracy')
#     plt.ylabel('Fold')
#     plt.title('k-fold Cross Validation Performance (k=5)')
#     plt.yticks(range(k), [f'Fold {i+1}' for i in range(k)])
#     plt.xlim(0, 1)
#     plt.gca().invert_yaxis()

#     # Determine the filename based on the classifier type
#     filename = f'kfold_validation_{classifier}.png'

#     # Save the plot as an image
#     plt.savefig(filename)

#     # Return the path to the saved image
#     image_path = os.path.abspath(filename)
#     return send_file(image_path, mimetype='image/png')

@app.route('/kfold/MultinomialNB', methods=['GET'])
def get_kfoldNB():

    clf = MultinomialNB()
    file_path = file_path_nb
    
    if not file_path:
        return jsonify({"error": "CSV file not found."}), 404
    processed_df = readCSV(file_path)

    X = processed_df['full_text'].astype('U')
    y = processed_df['value']
    k = 5
    kf = KFold(n_splits=k)
    validation_results = []


    tfidf_vectorizer = TfidfVectorizer()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        clf.fit(X_train_tfidf, y_train)
        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=k)
        print("NB : ", scores)

    # Plot the bar chart
    plt.barh(range(k), scores, color='skyblue')
    plt.axvline(x=np.mean(scores), color='red', linestyle='--', linewidth=1)
    plt.xlabel('Accuracy')
    plt.ylabel('Fold')
    plt.title('k-fold Cross Validation Performance (k=5)')
    plt.yticks(range(k), [f'Fold {i+1}' for i in range(k)])
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()  # Invert y-axis to have fold 1 at the top

    # Determine the filename based on the classifier type
    filename = f'kfold_validation_MultinomialNB.png'

    # Save the plot as an image
    plt.savefig(filename)

    # Return the path to the saved image
    image_path = os.path.abspath(filename)
    return send_file(image_path, mimetype='image/png')

@app.route('/kfold/DecisionTree', methods=['GET'])
def get_kfoldDT():

    clf = DecisionTreeClassifier()
    file_path = file_path_dt
    
    if not file_path:
        return jsonify({"error": "CSV file not found."}), 404
    processed_df = readCSV(file_path)

    X = processed_df['full_text'].astype('U')
    y = processed_df['value']
    k = 5
    kf = KFold(n_splits=k)
    validation_results = []

    tfidf_vectorizer = TfidfVectorizer()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        clf.fit(X_train_tfidf, y_train)
        scores = cross_val_score(clf, X_test_tfidf, y_test, cv=k)
        print("DT : ", scores)

    # Plot the bar chart
    plt.barh(range(k), scores, color='skyblue')
    plt.axvline(x=np.mean(scores), color='red', linestyle='--', linewidth=1)
    plt.xlabel('Accuracy')
    plt.ylabel('Fold')
    plt.title('k-fold Cross Validation Performance (k=5)')
    plt.yticks(range(k), [f'Fold {i+1}' for i in range(k)])
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()  # Invert y-axis to have fold 1 at the top

    # Determine the filename based on the classifier type
    filename = f'kfold_validation_DecisionTree.png'

    # Save the plot as an image
    plt.savefig(filename)

    # Return the path to the saved image
    image_path = os.path.abspath(filename)
    return send_file(image_path, mimetype='image/png')




@app.route('/check-sentence', methods=['POST'])
def check_sentence():
    data = request.get_json()
    sentence = data['sentence']

    # Process sentence
    processed_sentence = process_sentence(sentence)

    # Predict using Naive Bayes model
    nb_model_path = 'DATA_BARU/naive_bayes_model_tni5.pkl'
    nb_tfidf_vectorizer_path = 'DATA_BARU/tfidf_vectorizer_tnifix.pkl'
    nb_predicted_class = predict_class(nb_model_path, nb_tfidf_vectorizer_path, processed_sentence)

    # Predict using Decision Tree model
    dt_model_path = 'DATA_BARU/decision_tree_model_tni5.pkl'
    dt_tfidf_vectorizer_path = 'DATA_BARU/tfidf_vectorizer_tnifix.pkl'
    dt_predicted_class = predict_class(dt_model_path, dt_tfidf_vectorizer_path, processed_sentence)

    return jsonify({
        "Naive Bayes": nb_predicted_class,
        "Decision Tree": dt_predicted_class
    })


if __name__ == '__main__':
    # Printing message when the server has successfully started
    print("Server is running!")
    app.run(debug=True)
