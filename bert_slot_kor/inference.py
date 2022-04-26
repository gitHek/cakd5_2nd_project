# -*- coding: utf-8 -*-

import os
import pickle
import argparse

############################### TODO ##########################################
# 필요한 모듈 불러오기
###############################################################################

import tensorflow as tf


if __name__ == "__main__":
    # Reads command-line parameters
    parser = argparse.ArgumentParser("Evaluating the BERT NLU model")
    parser.add_argument("--model", "-m",
                        help="Path to BERT NLU model",
                        type=str,
                        required=True)
    
    args = parser.parse_args()
    load_folder_path = args.model
    

    # this line is to disable gpu
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=True,
                            device_count = {"CPU": 1})
    sess = tf.compat.v1.Session(config=config)

    ################################ TODO 경로 고치기 ##################
    bert_model_hub_path = "/content/drive/MyDrive/bert-module"
    ####################################################################

    ############################### TODO ###############################
    # 모델과 기타 필요한 것들 불러오기
    ####################################################################
    
    
    while True:
        print("\nEnter your sentence: ")
        try:
            input_text = input().strip()
        except:
            continue
            
        if input_text == "quit":
            break
    
    ############################### TODO ###############################
    # 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
    ####################################################################
    
    tf.compat.v1.reset_default_graph()
    
