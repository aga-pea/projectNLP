#!/usr/bin/env python

#### NLP ENGINE ####

import pandas as pd
import numpy as np

import re

from nltk import DecisionTreeClassifier
from nltk import MaxentClassifier
from nltk import NaiveBayesClassifier
from nltk import pos_tag
from nltk import word_tokenize

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import pickle
from nltk.corpus import brown
from nltk.tag import UnigramTagger

DIR_PICKLE = "./train.pickle"
dict_preprocessing = {
    "casefolding" : 1, 
    "occurences" : 0, 
    "stemming" : 1, 
    "stopword" : 0,
    "tweet" : 1
}

'''
kelas yang dapat save dan open pickle
-pickle adalah sebuah object yang disimpan dalam sebuah file
sehingga ketika program mati dan suatu saat object itu diperlukan langsung dapat di load
'''
class PickleManager():
    @staticmethod
    def save_pickle(object, path):
        pickle.dump(object, open(path, "wb"))
        
    
    @staticmethod
    def open_pickle(path):
        return pickle.load(open(path, "rb"))

'''
menyimpan 1 buah document, seperti struct
'''
class DataElement():
    
    def __init__(self, document, index, label = None):
        self.raw_text = document
        self.label = label
        self.clean_text = Preprocessing.do_preprocessing(self.raw_text)
        self.index = index
        
    def get_data_str(self):
        print("=="*30)
        print(self.index)
        print("raw \t:", self.raw_text)
        print("clean \t:", self.clean_text)
        print("label \t:",self.label)
    
'''
class preprocessing untuk melakukan normalisasi raw input
'''

# stemmer = PorterStemmer()
# gstopwords = stopwords.words('english')

class Preprocessing():
    
    @staticmethod
    def do_preprocessing(document):
        document = document.strip()
        if dict_preprocessing["casefolding"] is 1:
            document = document.lower()
        
        if dict_preprocessing["occurences"] is 1:
            document = re.sub(r'([a-z])\1+', r'\1', document)
        
        if dict_preprocessing["tweet"] is 1 :
            document = Preprocessing.tweet_preprocessing(document)
        
        if dict_preprocessing["stopword"] is 1 :
            document = " ".join([word for word in document.split(" ") if word not in gstopwords ])

            
        return document
        
        #stop word - cari stopword kamus dlu
        # stemming - cari lib dulu
    
    @staticmethod
    def tweet_preprocessing(sent):
        sent = re.sub("@\\w+|#\\w+|\\bRT\\b", "", sent)
        sent = re.sub("https?://\\S+\\s?", "<LINK>", sent)
#         sent = re.sub(r"(.)\1+",r"\1",sent)
        sent = re.sub("[ ]+", " ", sent);
#         sent = " ".join([ stemmer.stem(kw) for kw in sent.split(" ")])
        return sent
            
'''
kelas yang akan digunakan untuk mengextract feature
bisa kita modifikasi fitur yang akan digunakan
'''
class FeaturesetExtractor():    
    
    def __init__(self):
        self.neg_words = [line.rstrip('\n') for line in open(NEG_WORD)]
        self.pos_words = [line.rstrip('\n') for line in open(POS_WORD)]
        self.happy_words = [line.rstrip('\n') for line in open(HAPPY_WORD)]
        self.sad_words = [line.rstrip('\n') for line in open(SAD_WORD)]
        self.fear_words = [line.rstrip('\n') for line in open(FEAR_WORD)]
        self.anger_words = [line.rstrip('\n') for line in open(ANGER_WORD)]
        self.tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])

    def get_featureset(self, data_element):
        mapFeatureset = {}

        size = len(data_element.clean_text)
        word = data_element.clean_text
        list_word = word.split(" ")
        raw = data_element.raw_text
        list_word_raw = raw.split(" ")
         
        total_neg = len(set(list_word) & set(self.neg_words))
        total_pos = len(set(list_word) & set(self.pos_words))
#         total_happy = len(set(list_word) & set(self.happy_words))
#         total_sad = len(set(list_word) & set(self.sad_words))
#         total_fear = len(set(list_word) & set(self.fear_words))
#         total_anger = len(set(list_word) & set(self.anger_words))
        
        list_happy = tuple(set(list_word) & set(self.happy_words))
        list_sad = tuple(set(list_word) & set(self.sad_words))
        list_fear = tuple(set(list_word) & set(self.fear_words))
        list_anger = tuple(set(list_word) & set(self.anger_words))
        
        exclamation_count = raw.count("!")
        quetion_count = raw.count("?")
        uppercase_count = sum(1 for c in raw if c.isupper())
        
        mapFeatureset["bias"] = 1
        mapFeatureset["word"] = tuple(list_word)
#         mapFeatureset["size"] = size
        mapFeatureset["neg_words"] = total_neg
        mapFeatureset["pos_words"] = total_pos
#         mapFeatureset["total_happy"] = total_happy
#         mapFeatureset["total_sad"] = total_sad
#         mapFeatureset["total_fear"] = total_fear
#         mapFeatureset["total_anger"] = total_anger
        
        mapFeatureset["exclamation_count"] = exclamation_count
        mapFeatureset["quetion_count"] = quetion_count
#         mapFeatureset["uppercase_count"] = uppercase_count
        
        mapFeatureset["list_happy"] = list_happy
        mapFeatureset["list_sad"] = list_sad
        mapFeatureset["list_fear"] = list_fear
        mapFeatureset["list_anger"] = list_anger
        
#         mapFeatureset["pos_tag"] = tuple(pos_tag(word_tokenize(word)))
        pos_tag_temp = self.tagger.tag((word).split(" "))
        list_pos_tag = []
        for element in pos_tag_temp:
            list_pos_tag.append(element[1])
        mapFeatureset["pos_tag"] = tuple(list_pos_tag)
        
#         print(mapFeatureset)
        return mapFeatureset

'''
class untuk machine learningnya
1.input awal : memilih tipe classifier dengan default tipe maxent (sebagai input constructor)
-meminta input list_data_element (sebagai input constructor)
-class feature_extractor

2. memanggil fungsi build_model(), dimana fungsi build model adalah proses train 
dengan urutan seperti berikut:
    2a. melakukan ekstraksi fitur untuk setiap data_element yang berada dalam list_data_element
    2b. melakukan proses training
3. fungsi build_model() akan menghasilkan self.model yang siap diguanakan

untuk melakukan proses prediksi panggil function get_classify()
'''
class TextCategorization():
    
    def __init__(self, list_data_element, feature_extractor, classifier = "maxent"):
        self.classifier = classifier
        self.model = self.get_model()
        self.featuresets = []
        self.list_data_element = list_data_element
        self.feature_extractor = feature_extractor
    
    '''
    menentukan algoritma apa
    '''
    def get_model(self):
        model = None
        self.classifier = ""
        if self.classifier == "decision_tree":
            model = DecisionTreeClassifier
        elif self.classifier == "maxent":
            model = MaxentClassifier
        else:
            model = NaiveBayesClassifier
        return model
        
    def build_model(self, featuresets_input=None):
        #        featuresets = self.get_raw_data()
        print("Build Model...")
        self.get_featuresets()
        self.classifier = self.model.train(self.featuresets)
#         self.classifier.show_most_informative_features(5)
    
        
    def get_classify(self, data_element):
        featureset = self.feature_extractor.get_featureset(data_element)
        result = self.classifier.classify(featureset)
        return result
    
    #normal classification word classification
    def get_featuresets(self):
        print("Get Featureset...")
        for data_element in self.list_data_element :
            featureset = self.feature_extractor.get_featureset(data_element)
            self.featuresets.append((featureset, data_element.label))

def do_test_from_input(hasil):
#     text_input = input("input text?")
    textCat = PickleManager.open_pickle(DIR_PICKLE)
    data_element = DataElement(hasil, 0)
    result = textCat.get_classify(data_element)
    data_element.get_data_str()
    return result
#     print("::RESULT::",result)
#     while("exit" not in text_input):
#         data_element = DataElement(text_input, 0)
#         result = textCat.get_classify(data_element)
#         data_element.get_data_str()
#         print("::RESULT::",result)
#         text_input = input("input text?")


import json
import os
import requests
import datetime
from datetime import timedelta

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import auth

from googletrans import Translator

from flask import Flask
from flask import request
from flask import make_response

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

# firebase
cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://treat-me-22bff.firebaseio.com/'
})


# Flask app should start in global layout
app = Flask(__name__)

line_bot_api = LineBotApi('QiRriK22eidlQYKXbPseOKC9VEoRnR4/Jvo1GMxQMZXkzYoI+wtql1HchBjEdAcwSBrkj9RNBrixAyV9C0Rx1/6AXu/DqNwnVOaZ7b+ouBHvLZUM3NNntPFAz4V6O3gjyDElT/8FslyCkuRJVQd3wAdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('66102e73c1b74719168a8873e307430b')

@app.route('/call', methods=['GET'])


def call():
    translator = Translator()
    database = db.reference()
    user = database.child("user")

    #membaca apakah ada data pada firebase

    snapshot = user.order_by_key().get()
    #key = userId Line
    d=""
    for key, val in snapshot.items():
        try:
            if str(val["stat"])=="2":
                lMessage= user.child(str(val["connect"])).child("lastMessage").get()
                lName= user.child(str(val["connect"])).child("name").get()
                #push message jika User memiliki lastMessage
                if lMessage!=None:
                    hasil = str(translator.translate(str(lMessage), src="id",dest="en").text)
                    hasil_klasifikasi = str(do_test_from_input(hasil))
                    line_bot_api.push_message(key, TextSendMessage(text="Emosi pasien "+str(lName)+" selama 5 menit terakhir terdeteksi sebagai : "+str(hasil_klasifikasi)+"\n\nSumber : "+str(hasil)))
#                     return do_test_from_input(hasil)
                    #untuk reset
                    reset = user.child(str(val['connect']))
                    reset.update({
                        "lastMessage" : None
                    })
        except Exception as res:
            d = d+"\n"+str(res)
    return d



        
    
    
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 4040))

    print ("Starting app on port %d" %(port))

    app.run(debug=False, port=port, host='0.0.0.0')
