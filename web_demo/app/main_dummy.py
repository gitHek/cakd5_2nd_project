# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os

app = Flask(__name__)

app.static_folder = 'static'
# app.template_folder = 'templates'
run_with_ngrok(app)
@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip() # 사용자가 입력한 문장
    if userText == "와인 추천해줘":
        return "어떤 와인으로 추천해드릴까요?" # 챗봇이 이용자에게 하는 말을 return
    elif userText == "달지 않은 와인으로 추천해줘":
        return "달지 않은 와인으로 추천해드릴게요."

