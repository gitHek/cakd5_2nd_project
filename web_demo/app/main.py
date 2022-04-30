# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os
import re
import sys
import pickle
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image
sys.path.append('/content/drive/MyDrive/Colab_Notebooks/2nd_project/web_demo/app')
from recommend import recommend

# enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    allow_soft_placement=True,
    device_count={"GPU": 0},
)
sess = tf.compat.v1.Session(config=config)
graph = tf.compat.v1.get_default_graph()

############################### TODO : 경로 수정하기 ##########################################
sys.path.append(os.path.join(os.getcwd(), "/content/drive/MyDrive/Colab_Notebooks/2nd_project/bert_slot_kor"))
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer

# pretrained model path
# bert_model_hub_path = os.path.join(os.getcwd(), "bert-module")
# colaboratory에서 실행 시
bert_model_hub_path = '/content/drive/MyDrive/Colab_Notebooks/2nd_project/dataset/model'

# fine-tuned model path
# load_folder_path = os.path.join(os.getcwd(), "model")
# colaboratory에서 실행 시
load_folder_path = "/content/drive/MyDrive/Colab_Notebooks/2nd_project/dataset/saved_model"

# tokenizer vocab file path
vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
bert_to_array = BERTToArray(vocab_file)
###########################################################################################


# 슬롯태깅 모델과 토크나이저 불러오기
tags_to_array_path = os.path.join(load_folder_path, "tags_to_array.pkl")
with open(tags_to_array_path, "rb") as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)

model = BertSlotModel.load(load_folder_path, sess)

tokenizer = FullTokenizer(vocab_file=vocab_file)


################# TODO : 슬롯 및 선택지 채워넣기 #################
sweetness = ['안 달', '달지 않은', '달지 않고', '드라이', '안 단','안달고','달지않고','달지않은','달달한','달달하고', '달달하지만','달짝지근','단','달콤한', '스위트','많이 단']
body = ['가벼운', '라이트', '가볍', '상쾌한','청량한', '가볍지만','가볍지 않은', '미디엄','진한', '진하고',  '무거운','무겁고', '헤비', '풀', '풀바디', '끈적한', '무겁지만']
sourness = ['안 신', '안 시고',  '시지 않은', '시지않고','새콤한','상큼한', '시지만', '조금 시큼한','시고','신', '시큼한','시큼하고']
wine_type = ['레드', '화이트', '스파클링', '샴페인', '로제']
price = ['1만정도', '1만원정도', '1만원쯤', '1만원?', '1만?', '1만 이하', '1만원 이하', '1만원이하', '1만이하', '2만정도', '2만원정도', '2만원쯤', '2만원?', '2만?', '2만 이하', '2만원 이하',
 '2만원이하', '2만이하', '3만정도', '3만원정도', '3만원쯤', '3만원?', '3만?', '3만 이하', '3만원 이하', '3만원이하', '3만이하', '4만정도', '4만원정도', '4만원쯤', '4만원?', '4만?', '4만 이하',
 '4만원 이하', '4만원이하', '4만이하', '5만정도', '5만원정도', '5만원쯤', '5만원?', '5만?', '5만 이하', '5만원 이하', '5만원이하', '5만이하', '6만정도', '6만원정도', '6만원쯤', '6만원?', '6만?',
 '6만 이하', '6만원 이하', '6만원이하', '6만이하', '7만정도', '7만원정도', '7만원쯤', '7만원?', '7만?', '7만 이하', '7만원 이하', '7만원이하', '7만이하', '8만정도', '8만원정도', '8만원쯤',
 '8만원?', '8만?', '8만 이하', '8만원 이하', '8만원이하', '8만이하', '9만정도', '9만원정도', '9만원쯤', '9만원?', '9만?', '9만 이하', '9만원 이하', '9만원이하', '9만이하', '10만정도', '10만원정도',
 '10만원쯤', '10만원?', '10만?', '10만 이하', '10만원 이하', '10만원이하', '10만이하', '20만정도', '20만원정도', '20만원쯤', '20만원?', '20만?', '20만 이하', '20만원 이하', '20만원이하', '20만이하',
 '30만정도', '30만원정도', '30만원쯤', '30만원?', '30만?', '30만 이하', '30만원 이하', '30만원이하', '30만이하', '40만정도', '40만원정도', '40만원쯤', '40만원?', '40만?', '40만 이하', '40만원 이하',
 '40만원이하', '40만이하', '50만정도', '50만원정도', '50만원쯤', '50만원?', '50만?', '50만 이하', '50만원 이하', '50만원이하', '50만이하', '60만정도', '60만원정도', '60만원쯤', '60만원?', '60만?',
 '60만 이하', '60만원 이하', '60만원이하', '60만이하', '70만정도', '70만원정도', '70만원쯤', '70만원?', '70만?', '70만 이하', '70만원 이하', '70만원이하', '70만이하', '80만정도', '80만원정도',
 '80만원쯤', '80만원?', '80만?', '80만 이하', '80만원 이하', '80만원이하', '80만이하', '90만정도', '90만원정도', '90만원쯤', '90만원?', '90만?', '90만 이하', '90만원 이하', '90만원이하', '90만이하',
 '100만정도', '100만원정도', '100만원쯤', '100만원?', '100만?', '100만 이하', '100만원 이하', '100만원이하', '100만이하']
 ##############################################################

# 슬롯 사전
dic = {
    "당도" : sweetness,
    "바디감" : body,
    "산미": sourness,
    "종류" : wine_type,
    "금액" : price
}

# 슬롯 변수명 - 슬롯 이름 pairs
slots = {
    "당도": "당도",
    "바디감": "바디감",
    "산미": "산미",
    "종류": "종류",
    "금액": "가격"
}

# 명령어 설정 ( 챗봇 사용자가 문장 앞에 !를 붙이면 명령어로 인식 )
cmds = {
    "명령어" : ["명령어", "당도", "바디감", "산미", "종류", "가격"],
    "당도" : sweetness,
    "바디감" : body,
    "산미" : sourness,
    "종류" : wine_type,
    "금액" : price
}


# 슬롯이라고 인식한 토큰을 slot_text에 저장하기
def catch_slot(i, inferred_tags, text_arr, slot_text):
    if not inferred_tags[0][i] == "O": # 0(숫자)이아리나 O(영어)이다
        word_piece = re.sub("_", "", text_arr[i])
        slot_text[inferred_tags[0][i]] += word_piece


# 슬롯이 다 채워지면 챗봇 유저에게 확인 메세지 보내기
def check_order_msg(app, slots):
    order = []
    for slot, option in app.slot_dict.items():
        order.append(f"{slots[slot]}: {option}")
    # br 태그는 html에서 줄바꿈을 의미함
    order = "<br />\n".join(set(order))

    message = f"""
        {order} <br />
        위의 와인으로 추천해드릴까요? (예 or 아니오)
        """
    return message


# 슬롯 초기화 함수
def init_app(app):
    app.slot_dict = {
        "당도": "",
        "바디감": "",
        "산미": "",
        "종류": "",
        "금액": ""
    }
    app.confirm = False


app = Flask(__name__)

# colaboratory에서 실행 시
run_with_ngrok(app)

app.static_folder = 'static'

@app.route("/")
def home():
# 사용자가 입력한 슬롯을 저장할 슬롯 사전
    app.slot_dict = {
        "당도": "",
        "바디감": "",
        "산미": "",
        "종류": "",
        "금액": ""
    }
############ TODO : 슬롯이라고 인식할 점수 및 대화에 필요한 변수 설정하기 ############
    # 슬롯으로 인식할 점수 설정하기
    app.score_limit = 0.9
    # 대화에 필요한 변수 설정
    app.confirm = 0
    # (필요시) 대화에 필요한 변수 추가하기
##########################################################################
    return render_template("index.html")
 

# 챗봇 사용자가 특정 메세지를 입력했을 때 실행   
@app.route("/get")
def get_bot_response():
    # 사용자가 입력한 문장
    userText = request.args.get('msg').strip() 
    # 명령어 인식 - 사용자가 입력한 문장이 느낌표(!)로 시작할 때
    if userText[0] == "!":
        try:
            li = cmds[userText[1:]]
            message = "<br />\n".join(li)
        except:
            message = "입력한 명령어가 존재하지 않습니다."
        return message


    # 사용자가 입력한 문장을 토큰화
    text_arr = tokenizer.tokenize(userText)
    input_ids, input_mask, segment_ids = bert_to_array.transform([" ".join(text_arr)])

    # 훈련한 슬롯태깅 모델을 사용하여 슬롯 예측
    with graph.as_default():
        with sess.as_default():
            inferred_tags, slots_score = model.predict_slots(
                [input_ids, input_mask, segment_ids], tags_to_array
            )

    # 결과 체크
    print("text_arr:", text_arr)
    print("inferred_tags:", inferred_tags[0])
    print("slots_score:", slots_score[0])

    # inference 결과로 나온 토큰을 저장하는 사전
    slot_text = {k: "" for k in app.slot_dict}
    # 슬롯태깅 실시
    for i in range(0, len(inferred_tags[0])):
        if slots_score[0][i] >= app.score_limit: # 설정한 점수보다 슬롯 점수가 높을 시
            catch_slot(i, inferred_tags, text_arr, slot_text) # 슬롯을 저장하는 함수 실행
        else:
            print("슬롯 인식 중 에러가 발생했습니다.")
############ TODO : 토큰이 알맞은 슬롯으로 인식되게끔 코드 수정 ############
    # 슬롯 사전에 있는 단어와 일치하는지 검증 후 슬롯 사전에 최종으로 저장
    for slot in slot_text:
        if slot_text[slot] in dic[slot]:
            app.slot_dict[slot] = slot_text[slot]
############################################################################
    print(app.slot_dict)
    print(app.confirm)
    print(userText,type(userText))
    
    # 채워지지 않은 슬롯들을 한국어 슬롯 이름으로 변환
    empty_slot = [slots[slot] for slot in app.slot_dict if app.slot_dict[slot] == "" ]


##### TODO : 추출된 슬롯 정보를 가지고 추천 와인까지 출력하는 대화 완성하기 (recommend 함수 적용) #####
    if empty_slot:
        message = ", ".join(empty_slot) + "가 아직 선택되지 않았습니다."
    elif app.confirm == 0:
        message = check_order_msg(app, slots)
        app.confirm += 1
    else:
        if userText.startswith("예"):
            wine_list = recommend(app.slot_dict)
            name = wine_list.iloc[0]['이름']
            img = wine_list.iloc[0]['이미지주소']
            page = wine_list.iloc[0]['주소']
            res = requests.get(img)
            request_get_img = Image.open(BytesIO(res.content))
            init_app(app)
            return name
        elif userText.startswith("아니오"):
            message = "다시 주문해주세요."
            init_app(app)
    return message
########################################################################################



