# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os
import re
import sys
import pickle
import tensorflow as tf

# enable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    allow_soft_placement=True,
    device_count={"GPU": 0},
)
sess = tf.compat.v1.Session(config=config)


############################### TODO : 경로 수정하기 ##########################################
sys.path.append(os.path.join(os.getcwd(), "../bert_slot_kor"))
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer

# pretrained model path
bert_model_hub_path = os.path.join(os.getcwd(), "bert-module")
# colaboratory에서 실행 시
# bert_model_hub_path = '/content/drive/MyDrive/bert-module'

# fine-tuned model path
load_folder_path = os.path.join(os.getcwd(), "model")
# colaboratory에서 실행 시
# load_folder_path = "/content/drive/MyDrive/model"

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
sweetness = ["달지 않은", "드라이", "달달한"]
body = []
sourness = []
wine_type = []
price = []
##############################################################

# 슬롯 사전
dic = {
    "sweetness" : sweetness,
    "body" : body,
    "sourness": sourness,
    "wine_type" : wine_type,
    "price" : price
}

# 슬롯 변수명 - 슬롯 이름 pairs
slots = {
    "sweetness": "당도",
    "body": "바디감",
    "sourness": "산미",
    "wine_type": "종류",
    "price": "가격"
}

# 명령어 설정 ( 챗봇 사용자가 문장 앞에 !를 붙이면 명령어로 인식 )
cmds = {
    "명령어" : ["명령어", "당도", "바디감", "산미", "종류", "가격"],
    "당도" : sweetness,
    "바디감" : body,
    "산미" : sourness,
    "종류" : wine_type,
    "가격" : price
}


# 슬롯이라고 인식한 토큰을 slot_text에 저장하기
def catch_slot(i, inferred_tags, text_arr, slot_text):
    if not inferred_tags[0][i] == "0":
        word_piece = re.sub("_", " ", text_arr[i])
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
        "sweetness": "",
        "body": "",
        "sourness": "",
        "wine_type": "",
        "price": ""
    }
    app.confirm = False


app = Flask(__name__)

# colaboratory에서 실행 시
# run_with_ngrok(app) 

app.static_folder = 'static'

@app.route("/")
def home():
# 사용자가 입력한 슬롯을 저장할 슬롯 사전
    app.slot_dict = {
        "sweetness": "",
        "body": "",
        "sourness": "",
        "wine_type": "",
        "price": ""
    }
############ TODO : 슬롯이라고 인식할 점수 및 대화에 필요한 변수 설정하기 ############
    # 슬롯으로 인식할 점수 설정하기
    app.score_limit = 0.7
    # 대화에 필요한 변수 설정
    app.confirm = False
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

    # 채워지지 않은 슬롯들을 한국어 슬롯 이름으로 변환
    empty_slot = [slots[slot] for slot in app.slot_dict if app.slot_dict[slot] == "" ]


##### TODO : 추출된 슬롯 정보를 가지고 추천 와인까지 출력하는 대화 완성하기 (recommend 함수 적용) #####
    if empty_slot:
        message = ", ".join(empty_slot) + "가 아직 선택되지 않았습니다."
    elif app.confirm == False:
        message = check_order_msg(app, slots)
        app.confirm == True
    else:
        if userText.strip().startswith("예"):
            message = "와인 추천을 진행하겠습니다."
        elif userText.strip().startswith("아니오"):
            message = "다시 주문해주세요."
            init_app(app)
    return message
########################################################################################



