import pandas as pd
import re

data_path = 'wine_df_all.csv' ##### 경로 수정 #####
df = pd.read_csv(data_path) 
df = df.rename(columns = {'당도':'sweetness', '바디감':'body', '산도': 'sourness', '종류': 'wine_type', '가격': 'price'})


def recommend(slot_dict):
    # 예시) "달지 않은 와인 추천해줘"
    # 1. 맵핑이 필요한 변수 사전 만들기 
    mapping_dict = {"sweetness": {'달지 않은': 0, '달달한': 2},
                    "body": {},
                    "sourness": {}}
    
    # 2. slot_dict의 변수 불러오기 
    tmp_df = df 
    filter_order = ['wine_type', 'price', 'sweetness', 'body', 'sourness'] ##### 필터링 조건 우선순위 수정 #####
    for k, v in sorted(slot_dict.items(), key = lambda i: filter_order.index(i[0])):
        # 예시) k : sweetness, v: 달지 않은

        # 필터링할 와인이 5개 이하면 필터링 중단
        if len(tmp_df) <= 5: 
            break 

        if v:
            # 당도/바디감/산도와 같은 맵핑이 필요한 변수의 경우
            if k in mapping_dict:
                if v in mapping_dict[k]:
                    v = mapping_dict[k][v]
                else: # 맵핑 사전에 없는 슬롯이 출력된 경우, 디폴트값을 1로 지정 ex) 달달구리
                    v = 1 


            # 3. df에서 조건 걸어서 찾기 - 필터링 후 tmp_df 변수에 저장
            ##### 필터링 방식 수정 : 여러가지 경우로 생각해보세요 #####
            if k == 'wine_type':
                tmp_df = tmp_df[tmp_df[k] == v]
            elif k == 'price':
                price = re.search('\d+', v).group()
                if '이하' in v:
                    tmp_df = tmp_df[tmp_df[k] <= price]
                ###### elif 정도 in v ######
            elif k == 'sweetness':
                tmp_df = tmp_df[tmp_df[k] == v] # 당도 3으로 exact search를 할 수도 있지만, 3 이상으로 조건을 줄 수도 있습니다
            elif k == 'body':
                tmp_df = tmp_df[tmp_df[k] == v]
            elif k == 'sourness':
                tmp_df = tmp_df[tmp_df[k] == v]
        
    # 4. 걸러진 와인 리스트 추천순위 기준 정렬
    wine_list = tmp_df.sort_values('추천순위')

    ##### 현재 코드는 필터링된 와인 중 추천 순위 1위 와인 출력, 필요한 요소들을 리턴하도록 수정 #####
    return wine_list.iloc[0]['이름']




if __name__ == "__main__":
    example = {
            "sweetness": "달지 않은",
            "body": "",
            "sourness": "",
            "wine_type": "",
            "price": ""
        }
    print(recommend(example))