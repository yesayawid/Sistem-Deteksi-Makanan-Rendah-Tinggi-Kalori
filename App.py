from flask import Flask, render_template, request, jsonify
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from bs4 import BeautifulSoup
from flask_restful import Resource, Api
import requests
import random
import pickle
from flask_cors import CORS
import json
from nltk.stem import WordNetLemmatizer
import nltk
import pymysql
from Stremlitpage import is_table_empty, read_mysql_table

app = Flask(__name__)
api = Api(app)
CORS(app)

# Load Chatbot model
class Chatbot:
    def __init__(self, model_path):
        # Pindahkan model loading ke dalam __init__
        self.model = load_model(model_path)
        self.intents = json.loads(open("./model/intents.json").read())
        self.words = pickle.load(open('./model/words.pkl', 'rb'))
        self.classes = pickle.load(open('./model/classes.pkl', 'rb'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag[:117])

    def predict_class(self, sentence):
        p = self.bow(sentence, self.words, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        error = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.predict_class(text)
        res = self.getResponse(ints)
        return res

# Buat instance Chatbot
chatbot_instance = Chatbot('./model/chatbot_model.h5')

# Load Image Classification model
model = load_model('model/model_calori.h5')
labels = {0: 'ayam_goreng', 1: 'ayam_pop', 2: 'daging_rendang', 3: 'dendeng_batokok', 4: 'gulai_ikan', 5: 'gulai_tambusu', 6: 'gulai_tunjang',
          7: 'telur_balado', 8: 'telur_dadar'}

rendah_kalori = ['ayam_pop', 'telur_dadar']
tinggi_kalori = ['ayam_goreng', 'daging_rendang', 'dendeng_batokok', 'telur_balado', 'gulai_ikan', 'gulai_tambusu', 'gulai_tunjang']

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction 
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        print("Can't able to fetch the Calories")
        print(e)

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

@app.route("/")
def hello_world():
    return render_template('index.html');

@app.route("/post")
def post():
    return render_template('post.html');

# Update the Flask route to include prediction result
@app.route('/predict', methods=['GET', 'POST'])
def main_page():
    prediction_result = None

    if request.method == 'POST':
        # Your existing code to process the uploaded image
        img_file = request.files['file']
        img = Image.open(img_file).resize((250, 250))
        save_image_path = './upload_images/' + img_file.filename
        img.save(save_image_path)

        result = processed_img(save_image_path)
        category = 'Rendah Kalori' if result in tinggi_kalori else 'Tinggi Kalori'
        success_message = f"Predicted: {result}\nCategory: {category}"

        cal = fetch_calories(result)
        if cal:
            success_message += f"\nCalories (100 grams): {cal}"

        # Update prediction_result with the relevant information
        prediction_result = {
            'result': result,
            'category': category,
            'calories': cal if cal else None
        }

    return render_template('uploud.html', prediction_result=prediction_result)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data.get('message')
    response = chatbot_instance.chatbot_response(message)
    return jsonify({'response': response})

from flask import render_template

@app.route('/chatbot-interface', methods=['GET'])
def chatbot_interface():
    return render_template('chatbot.html')




# Database configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'review',
}

# Function to insert data into MySQL
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_review as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Insert data into the table
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            cursor.execute(sql_insert, (data['nama'], data['tanggal'], data['review']))
        connection.commit()
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Enable CORS for all routes
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.after_request(add_cors_headers)


@app.route('/submit', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    data_to_insert = request.get_json()
    insert_data_to_mysql(data_to_insert)
    return jsonify({'status': 'success'})

@app.route('/review', methods=['GET'])
def reviews():
    reviews_data = {
        'review': [
            {'author': 'John Doe', 'content': 'Great service!'},
            {'author': 'Jane Smith', 'content': 'Awesome experience!'},
            # Add more review data as needed
        ]
    }
    return render_template('review.html', reviews_data=reviews_data)

import nltk
import re
import pickle
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import pymysql


@app.route('/tambah_review', methods=['POST'])
def tambah_review():
    data = request.json
    try:
        insert_data_to_mysql(data)
        return jsonify({'status': 'sukses', 'pesan': 'Data berhasil ditambahkan ke database'}), 201
    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)}), 500
    
@app.route('/sentimen')
def index():
    table_name = 'input_review'
    if not is_table_empty(table_name):
        df = read_mysql_table(table_name)
        data_content = df['review']
        # casefolding
        data_casefolding = data_content.str.lower()
        data_casefolding.head()
        #url
        filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", str(tweet)) for tweet in data_casefolding]
        #cont
        filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
        #punctuatuion
        filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]
        #  hapus #tagger
        filtering_tagger = [re.sub(r'#([^\s]+)', '', tweet) for tweet in filtering_punctuation]
        #numeric
        filtering_numeric = [re.sub(r'\d+', ' ', tweet) for tweet in filtering_tagger]
        # # filtering RT , @ dan #
        # fungsi_clen_rt = lambda x: re.compile('\#').sub('', re.compile('rt @').sub('@', x, count=1).strip())
        # clean = [fungsi_clen_rt for tweet in filtering_numeric]
        data_filtering = pd.Series(filtering_numeric)
        # #tokenize
        tknzr = TweetTokenizer()
        data_tokenize = [tknzr.tokenize(tweet) for tweet in data_filtering]
        data_tokenize
        #slang word
        path_dataslang = open("data/kamus kata baku-clear.csv")
        dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")
        def replaceSlang(word):
            if word in list(dataslang[0]):
                indexslang = list(dataslang[0]).index(word)
                return dataslang[1][indexslang]
            else:
                return word
            
        data_formal = []
        for data in data_tokenize:
            data_clean = [replaceSlang(word) for word in data]
            data_formal.append(data_clean)
        len_data_formal = len(data_formal)
        # print(data_formal)
        # len_data_formal

        nltk.download('stopwords')
        default_stop_words = nltk.corpus.stopwords.words('indonesian')
        stopwords = set(default_stop_words)

        def removeStopWords(line, stopwords):
            words = []
            for word in line:  
                word=str(word)
                word = word.strip()
                if word not in stopwords and word != "" and word != "&":
                    words.append(word)

            return words
        reviews = [removeStopWords(line,stopwords) for line in data_formal]

        # Specify the file path of the pickle file
        file_path = 'model/training.pickle'

        # Read the pickle file
        with open(file_path, 'rb') as file:
            data_train = pickle.load(file)
            
        # pembuatan vector kata
        vectorizer = TfidfVectorizer()
        train_vector = vectorizer.fit_transform(data_train)
        reviews2 = [" ".join(r) for r in reviews]

        load_model = pickle.load(open('model/tfidf_Model Naive Bayes_nvb.pkl','rb'))

        result = []

        for test in reviews2:
            test_data = [str(test)]
            test_vector = vectorizer.transform(test_data)
            pred = load_model.predict(test_vector)
            result.append(pred[0])
            
        unique_labels(result)

        df['label'] = result

        def delete_all_data_from_table(table, host='localhost', user='root', password='', database='review'):
            # Establish a connection to the MySQL database
            connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()
            
            # Delete all data from the specified table
            query = f"DELETE FROM {table}"
            cursor.execute(query)
            
            # Commit the changes
            connection.commit()
            
            # Close the cursor and the database connection
            cursor.close()
            connection.close()

        delete_all_data_from_table('input_review')

        def insert_df_into_hasil_model(df, host='localhost', user='root', password='', database='review'):
        # Establish a connection to the MySQL database
            connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )

        # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

        # Iterate through each row in the DataFrame and insert it into the 'hasil_model' table
            for index, row in df.iterrows():
                query = "INSERT INTO hasil_model (id_review, nama, tanggal, review, label) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(query, (row['id_review'], row['nama'], row['tanggal'], row['review'], row['label']))

        # Commit the changes
            connection.commit()

        # Close the cursor and the database connection
            cursor.close()
            connection.close()

        insert_df_into_hasil_model(df)

        table_name = 'hasil_model'
        hasil_df = read_mysql_table(table_name)
        hasil_df.to_csv('data/hasil_model.csv')
        data = pd.read_csv('data/hasil_model.csv')
    else:
        # Membaca data dari file CSV
        data = pd.read_csv('data/hasil_model.csv')

    data = data[['review', 'label']]
    # Menghitung jumlah data dengan label positif, negatif, dan netral
    jumlah_positif = len(data[data['label'] == 1])
    jumlah_negatif = len(data[data['label'] == -1])
    jumlah_netral = len(data[data['label'] == 0])

    return render_template('sentimen.html', jumlah_positif=jumlah_positif, jumlah_negatif=jumlah_negatif, jumlah_netral=jumlah_netral)

if __name__ == '__main__':
    app.run(debug=True)
