from app.model.request import *
from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError

from app.config.database import engine#, engine_mariadb
from app.config.config import DATABASE_URL2  # DB 설정 파일에서 불러오기
import pandas as pd
import numpy as np
import joblib
import os

import xml.etree.ElementTree as ET
import requests

model_path = '/app/pkl/LightGBM_250226.pkl'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # scheduler.py의 디렉터리
MODEL_PATH = os.path.join(BASE_DIR, '..', 'pkl', 'LightGBM_tunning_250819.pkl')
#MODEL_PATH = os.path.join(BASE_DIR, '..', 'pkl', 'LightGBM_SMOTE.pkl')
snow = "null"
def get_control():
    import warnings
    warnings.filterwarnings('ignore')
    
    global snow

    check_query = text("select count(*) from info_control_model where reg_date = (select reg_date from info_rwis where latitude = '33.38483693' order by reg_date desc limit 1)")
    query = text("select * from info_rwis where reg_date is not null and latitude = '33.38483693' order by reg_date desc limit 1")   
    # senser data 적설량 추가
    snowcover_query = text("select snowcover from sensor_data_snowcover order by create_time desc limit 1")
    # snowcover_query = text("select snowcover from sensor_data_jeju order by create_time desc limit 1")
    # snow = snow if snow == "null" else 0 
    engine_mariadb = create_engine(DATABASE_URL2, pool_pre_ping=True)
    with engine_mariadb.begin() as conn:
        snow = conn.execute(snowcover_query).scalar()        
        # snow = snow if snow == "null" else 0
        if snow is None or snow == "null":
        # if snow == "null":
            snow = 0
        else:
            snow = int(snow)        
        

    with engine.begin() as conn:
        check = conn.execute(check_query).scalar()
        if check == 0:
            result = conn.execute(query)
            data = [dict(row) for row in result.mappings()]  # 결과를 딕셔너리로 변환        
            freezing, controlL, controlS = get_add_data()
            reg_date = data[0]["reg_date"]
            data_sample = {
                "date": [data[0]["reg_date"].strftime("%Y-%m-%d")],
                "hour": [data[0]["reg_date"].hour],      
                "visibility": [float(data[0]["visibility"])],
                #"snow": [data[0]["snow"]],
                "road_temp": [float(data[0]["road_temp"])],
                "water_film": [float(data[0]["water_film"])],
                "friction": [float(data[0]["friction"])],                                
                "control_l": [controlL],
                "control_s": [controlS],
                "snow": [snow],
                "frozen": [freezing]
                }                

        
            data_process = pd.DataFrame(data_sample)    

            #result_data = preprocessing(data_rwis, data_road)
            # 모델 로드
            model = joblib.load(MODEL_PATH)
        
            # 모델 예측 및 필요 결과값 출력
            pred = model.predict(data_process.drop(['date', 'hour'], axis=1))
            pred_proba = model.predict_proba(data_process.drop(['date', 'hour'], axis=1))      
            pred_proba_np = np.array(pred_proba)

            flattened_proba = pred_proba_np.flatten()
            
            probability_columns = [f'probability{group}_{index}h'
                     for index in range(1, 4)
                     for group in range(0, 3)]
            
            data_process[probability_columns] = flattened_proba            
            data_process[['pred_1h', 'pred_2h', 'pred_3h']] = pred[0]
            data_process['tp'] = data_process[['control_s', 'control_l']].max(axis=1)            
            data_process['min'] = data[0]["reg_date"].minute
            #"min": [data[0]["reg_date"].minute],

            insert_control_data(conn, data_process)
            #print(data_process)


def get_add_data():
    # RSS 데이터 가져오기
    response = requests.get("https://www.jjpolice.go.kr/jjpolice/notice/traffic.htm?act=rss")
    response.raise_for_status()  # 요청 에러 체크

    # XML 파싱
    root = ET.fromstring(response.content)

    # 'item' 태그 찾기
    items = root.findall(".//item")

    # 원하는 데이터 찾기
    for item in items:
        title = item.find("title").text if item.find("title") is not None else ""
        
        if "5.16도로(1131)" in title:  # 제목 필터링
            freezing = item.find("freezing").text if item.find("freezing").text is not None else 0
            controlL = item.find("controlL").text if item.find("controlL") is not None else 0
            controlS = item.find("controlS").text if item.find("controlS") is not None else 0

            return freezing, controlL, controlS  # 데이터 반환
    
    return 0, 0, 0  # 해당 title이 없을 경우

def preprocessing(data_rwis, data_road):
    # 1. 위경도 및 도로명 필터링
    '''
    성판악 위경도
    = (33.38483693, 126.62063386)

    첨단과학기술단지입구교차로교차로 위경도
    = (33.45622129, 126.5512213)

    서성로입구 위경도
    = (33.31792112, 126.59859289)
    '''
    # data_rwis = data_rwis[(data_rwis['위도'] == 33.38483693) & (data_rwis['경도'] == 126.62063386)]
 #   data_road = data_road[data_road['도로명'] == '5.16도로']

    # 3. 텍스트 변환
    data_road['결빙량'] = data_road['결빙량'].apply(
        lambda x: float(str(x).split('~')[-1]) if x != -1 and '부분' not in str(x) else 0)

    # 4. 날짜/시/분 형식 맞추기
    data_road['hour'] = data_road['시간'].apply(lambda x: int(str(x).split(':')[0]))
    data_road['min'] = data_road['시간'].apply(lambda x: int(str(x).split(':')[1]))
    data_rwis['date'] = data_rwis['reg_date'].dt.strftime("%Y-%m-%d")
    data_rwis['hour'] = data_rwis['reg_date'].dt.hour
    data_rwis['min'] = data_rwis['reg_date'].dt.minute
    # data_rwis['날짜'] = data_rwis['reg_date'].apply(lambda x: x.split(' ')[0])
    # data_rwis['시'] = data_rwis['reg_date'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))
    # data_rwis['분'] = data_rwis['reg_date'].apply(lambda x: int(x.split(' ')[1].split(':')[1]))

    # 5. 날짜/시 기준으로 데이터 병합
    #data_merge = pd.merge(data_rwis, data_road.drop('적설량', axis=1), on=['날짜', '시'], how='outer').sort_values(['날짜', '시'])
    data_merge = pd.concat([data_rwis, data_road.drop('적설량', axis=1)], ignore_index=True)

    # 6. 날짜/시 기준 그룹화하여 집계 함수 적용
    agg_cols = ['날짜', '시', '노면온도', '마찰계수', '적설량', '결빙량', '가시거리', '수막두께', '소형_통제', '대형_통제']
    data_agg = data_merge.groupby(['날짜', '시'])[agg_cols].agg({
        '노면온도': 'max',
        '마찰계수': 'max',
        '적설량': 'max',
        '결빙량': 'max',
        '가시거리': 'max',
        '수막두께': 'max',
        '소형_통제': 'max',
        '대형_통제': 'max'
    }).reset_index()

    # 6. 결측치 처리
    X_test = data_agg.ffill()

    return X_test

def insert_control_data(conn, df):    
    try:
        df.to_sql('info_control_pred',con=conn, if_exists='append', index=False, method='multi')
    except SQLAlchemyError as e:
        print(f"❌ 데이터 삽입 오류 발생: {e}")