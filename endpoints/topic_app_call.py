import pickle
import string

import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from langdetect import detect
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import hstack
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, jaccard_score, hamming_loss
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from tqdm import tqdm
from unidecode import unidecode

tqdm.pandas()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp_gr = spacy.load("el_core_news_sm")
stop_words_gr = set(nltk.corpus.stopwords.words('greek'))
stop_words_en = set(stopwords.words('english'))


def preprocess_text(topics_df, fe_type):




    df_clean = topics_df

    df_clean['title'] = df_clean['title'].str.replace('«', '').str.replace('»', '')
    df_clean['content'] = df_clean['content'].str.replace('«', '').str.replace('»', '')

    df_clean['title'] = df_clean['title'].str.replace('‘', '').str.replace('’', '')
    df_clean['content'] = df_clean['content'].str.replace('‘', '').str.replace('’', '')

    print('Removing Html tags')
    df_clean['content'] = df_clean['content'].progress_apply(remove_html_tags)

    print('Converting to Lowercase')
    df_clean['title'] = df_clean['title'].progress_apply(lowercase_text)
    df_clean['content'] = df_clean['content'].progress_apply(lowercase_text)

    print('Removing Punctuation\n')
    df_clean['title'] = df_clean['title'].progress_apply(remove_punctuation)
    df_clean['content'] = df_clean['content'].progress_apply(remove_punctuation)



    print('Removing Stopwords')
    df_clean['title'] = df_clean['title'].progress_apply(remove_stopwords_gr)
    df_clean['content'] = df_clean['content'].progress_apply(remove_stopwords_gr)

    df_clean = df_clean[df_clean['content'].notna() & (df_clean['content'] != '')]

    print("Applying Lemmatization")
    df_clean['title'] = df_clean['title'].progress_apply(lemmatize_text)
    df_clean['content'] = df_clean['content'].progress_apply(lemmatize_text)

    print("Applying Tokenization")
    df_clean['title_tokenized'] = df_clean['title'].progress_apply(tokenize_text_gr)
    df_clean['content_tokenized'] = df_clean['content'].progress_apply(tokenize_text_gr)



    #df_clean.to_csv("data/merged_data_tokenized2.csv")

    if fe_type == 'train':
        x_title, x_content = train_feature_extraction(df_clean[df_clean['is_train'] == 1])
    elif fe_type == 'test':
        x_title, x_content = test_feature_extraction(df_clean)

    return df_clean, x_title, x_content


def label_binarazation(y):
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    y_classes = label_binarizer.classes_
    y = np.argmax(y, axis=1)
    return y, y_classes


def train_evaluate_model():

    train_data = pd.read_csv("data/train_data.csv")
    test_data = pd.read_csv("data/unseen_test_data.csv")

    # Add a flag column to identify train and test data
    train_data['is_train'] = 1
    test_data['is_train'] = 0

    # Merge train and test data
    topics_df = pd.concat([train_data, test_data], ignore_index=True)

    print(f"Num of train data: {topics_df[topics_df['is_train'] == 1].shape[0]}")
    print(f"Num of test data: {topics_df[topics_df['is_train'] == 0].shape[0]}")

    df, x_title, x_content = preprocess_text(topics_df, fe_type='train')


    df_train = df[df['is_train'] == 1]


    X_train = hstack([x_title, x_content])
    y_train = df_train['label']
    y_train, y_classes = label_binarazation(y_train)

    df_test = df[df['is_train'] == 0]
    x_content_test, x_title_test = test_feature_extraction(df_test)
    X_test = hstack([x_title_test, x_content_test])
    y_test = df_test['label']
    y_test, _ = label_binarazation(y_test)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifiers = {
        "Linear SVM": LinearSVC(),
        "SGD": SGDClassifier(n_jobs=-1)
    }
    resp_lst = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        resp = evaluate_model(y_test, y_pred, name, y_classes)
        filename = 'models/' + name + '.sav'
        joblib.dump(clf, filename)
        resp_lst.append(resp)

    return resp_lst, 200


def load_model(model_name):
    loaded_model = joblib.load("models/"+model_name+'.sav')
    df = pd.read_csv("data/unseen_test_data.csv")

    df, x_title_test, x_content_test = preprocess_text(df, fe_type='test')


    X_test = hstack([x_title_test, x_content_test])
    y_test = df['label']
    y_test, y_classes = label_binarazation(y_test)

    y_pred = loaded_model.predict(X_test)
    resp = evaluate_model(y_test, y_pred, model_name, y_classes)

    return resp, 200

def train_feature_extraction(df):

    title_tokenized = [' '.join(tokens) for tokens in df["title_tokenized"]]
    content_tokenized = [' '.join(tokens) for tokens in df["content_tokenized"]]

    content_tfidf_vectorizer = TfidfVectorizer(max_features=100000)
    title_tfidf_vectorizer = TfidfVectorizer(max_features=100000)
    x_title = title_tfidf_vectorizer.fit_transform(title_tokenized)
    x_content = content_tfidf_vectorizer.fit_transform(content_tokenized)
    #print(x_content, x_title)

    joblib.dump(content_tfidf_vectorizer, "fe_data/tf-idf-vect-content-1.pkl")
    joblib.dump(title_tfidf_vectorizer, "fe_data/tf-idf-vect-title-1.pkl")

    # Save the TF-IDF matrix to a file
    with open(f"fe_data/tfidf_X_title.pkl", "wb") as f:
        pickle.dump(x_title, f)
    with open(f"fe_data/tfidf_X_content.pkl", "wb") as f:
        pickle.dump(x_content, f)

    return x_title, x_content


def test_feature_extraction(df):
    loaded_content_vector = joblib.load('fe_data/tf-idf-vect-content-1.pkl')
    loaded_title_vector = joblib.load('fe_data/tf-idf-vect-title-1.pkl')

    x_content_test = [' '.join(tokens) for tokens in df["title_tokenized"]]
    x_title_test = [' '.join(tokens) for tokens in df["content_tokenized"]]

    x_content_test = loaded_content_vector.transform(x_content_test)
    x_title_test = loaded_title_vector.transform(x_title_test)

    return x_content_test, x_title_test


def tokenize_text_gr(text):
    doc = nlp_gr(text)
    tokenized_text = [token.text for token in doc]
    return tokenized_text


def lemmatize_text_gr(text):
    doc = nlp_gr(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text


def remove_stopwords_gr(words):
    words = word_tokenize(words)
    filtered_words = [word for word in words if word.lower() not in stop_words_gr]
    return ' '.join(filtered_words)


def lowercase_text(text):
    return text.lower()

# def remove_diacritics(text):
#     return unidecode(text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def lemmatize_text_en(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Tokenize text into words
    words = nltk.word_tokenize(text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in words])
    return lemmatized_text


def lemmatize_text(text):
    language = detect(text)
    lemmatized_words = text
    if language == 'en':
        lemmatized_words = lemmatize_text_en(text)
    elif language == 'el':
        lemmatized_words = lemmatize_text_gr(text)

    return lemmatized_words


def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def evaluate_model(y_test, y_pred, model_name, y_classes):
    precision, recall, fscore, support = score(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    jacc_sc = jaccard_score(y_test, y_pred, average='weighted')

    print("Accuracy: ", acc)
    print("Classifier Used:", str(model_name))
    print(f'Jaccard Score: {jacc_sc:.4f}')
    print("\n")

    metric_df = pd.DataFrame(data=[precision, recall, fscore, support],
                             index=["Precision", "Recall", "F-1 score", "True Count"],
                             columns=y_classes)
    metric_df.to_csv("models/models_results/" + str(model_name) + "_metrics.xlsx")
    print(metric_df)
    resp = {"Accuracy": acc, "Jaccard Score": jacc_sc, "Classifier": str(model_name)}
    return resp