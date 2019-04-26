#!/usr/bin/env python


import json
import os
import requests
import datetime
from datetime import timedelta

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import auth

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
    database = db.reference()
    user = database.child("user")

    #membaca apakah ada data pada firebase

    snapshot = user.order_by_key().get()
    #key = userId Line
    for key, val in snapshot.items():
        try:
            lMessage= val["lastMessage"]
            #push message jika User memiliki lastMessage
            if lMessage!=None:
                lMessage = str(lMessage).split(" ")
                line_bot_api.push_message(key, TextSendMessage(text="Jumlah Kata dalam 5 Menit terakhir : "+str(len(lMessage))))
                #untuk reset
                reset = user.child(key)
                reset.update({
                    "lastMessage" : None
                })
        except Exception as res:
            print("Error")
    return "Success"



        
    
    
    
if __name__ == '__main__':
    port = int(os.getenv('PORT', 4040))

    print ("Starting app on port %d" %(port))

    app.run(debug=False, port=port, host='0.0.0.0')
