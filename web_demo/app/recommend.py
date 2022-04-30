import pandas as pd
import re

data_path = '/content/drive/MyDrive/Colab_Notebooks/2nd_project/dataset/wine_df_all_save.csv' ##### 경로 수정 #####
df = pd.read_csv(data_path) 

def recommend(slot_dict):
    # 예시) "달지 않은 와인 추천해줘"
    # 1. 맵핑이 필요한 변수 사전 만들기 
    mapping_dict = {"당도": {'안 달' : 0, '달지않은': 0, '달지않고': 0, '드라이': 0, '안 단': 0,'안달고': 0,'달지않고': 0,'달지않은': 0,
                              '달달한': 1,'달달하고': 1, '달달하지만': 1,'달짝지근': 1,
                              '단': 2,'달콤한': 2, '스위트': 2,'많이 단': 2},
                    "바디감": {'가벼운': 0, '라이트': 0, '가볍': 0, '상쾌한': 0,'청량한': 0, '가볍지만': 0,
                              '가볍지 않은': 1, '미디엄': 1,
                              '진한': 2, '진하고': 2,  '무거운': 2,'무겁고': 2, '헤비': 2, '풀': 2, '풀바디': 2, '끈적한': 2, '무겁지만': 2},
                    "산미": {'안 신': 0, '안 시고': 0,  '시지 않은': 0, '시지 않고': 0,
                            '새콤한': 1,'상큼한': 1, '시지만': 1, '조금 시큼한': 1,
                            '시고': 2,'신': 2, '시큼한': 2,'시큼하고': 2}}
    
    # 2. slot_dict의 변수 불러오기 
    tmp_df = df 
    filter_order = ['종류', '금액', '당도', '바디감', '산미'] ##### 필터링 조건 우선순위 수정 #####
    for k, v in sorted(slot_dict.items(), key = lambda i: filter_order.index(i[0])):
        # 예시) k : sweetness, v: 달지 않은

        # 필터링할 와인이 5개 이하면 필터링 중단
        if len(tmp_df) <= 5: 
            break 

        if v:
            # 당도/바디감/산미와 같은 맵핑이 필요한 변수의 경우
            if k in mapping_dict:
                if v in mapping_dict[k]:
                    v = mapping_dict[k][v]
                else: # 맵핑 사전에 없는 슬롯이 출력된 경우, 디폴트값을 2로 지정 ex) 달달구리
                    v = 1


            # 3. df에서 조건 걸어서 찾기 - 필터링 후 tmp_df 변수에 저장
            ##### 필터링 방식 수정 : 여러가지 경우로 생각해보세요 #####
            if k == '종류':
                tmp_df = tmp_df[tmp_df[k] == v]
            elif k == '금액':
                price = re.search('\d+', v).group()
                if '이하' in v:
                    tmp_df = tmp_df[tmp_df[k] <= int(price)*10000]
                ###### elif 정도 in v ######
                else:
                  tmp_df=tmp_df[(tmp_df[k] <= int(price + 1)*10000)&(tmp_df[k] >= int(price - 1)*10000)]
            elif k == '당도':
                if v == 0:
                  tmp_df = tmp_df[tmp_df[k] ==0] # 당도 3으로 exact search를 할 수도 있지만, 3 이상으로 조건을 줄 수도 있습니다
                elif v == 1:
                  tmp_df = tmp_df[(tmp_df[k] >= 1)&(tmp_df[k] <= 2)]
                elif v == 2:
                  tmp_df = tmp_df[tmp_df[k] >=3]
            elif k == '바디감':
                if v == 0:
                  tmp_df = tmp_df[tmp_df[k] <= 2]
                elif v == 1:
                  tmp_df = tmp_df[tmp_df[k] >= 3]
                elif v == 2:
                  tmp_df = tmp_df[tmp_df[k] >= 3]
            elif k == '산미':
                if v == 0:
                  tmp_df = tmp_df[tmp_df[k] <= 2]
                elif v == 1:
                  tmp_df = tmp_df[tmp_df[k] == 3]
                elif v == 2:
                  tmp_df = tmp_df[tmp_df[k] >= 4]
        
    # 4. 걸러진 와인 리스트 추천순위 기준 정렬
    wine_list = tmp_df.sort_values('평점지수')

    ##### 현재 코드는 필터링된 와인 중 추천 순위 1위 와인 출력, 필요한 요소들을 리턴하도록 수정 #####
    return wine_list




if __name__ == "__main__":
    example = {
            "sweetness": "달지 않은",
            "body": "",
            "sourness": "",
            "wine_type": "",
            "price": ""
        }
    print(recommend(example))