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
MODEL_PATH2 = os.path.join(BASE_DIR, '..', 'pkl', 'LightGBM_tunning_251204.pkl') # 2025.09.05 신규모델
MODEL_PATH = os.path.join(BASE_DIR, '..', 'pkl', 'LightGBM_SMOTE.pkl') # 기존모델

MODEL_FROZEN = os.path.join(BASE_DIR, '..', 'pkl', 'frozen_model.pkl')
MODEL_1H = os.path.join(BASE_DIR, '..', 'pkl','road_control_model_target_1h.pkl')
MODEL_2H = os.path.join(BASE_DIR, '..', 'pkl', 'road_control_model_target_2h.pkl')
MODEL_3H = os.path.join(BASE_DIR, '..', 'pkl','road_control_model_target_3h.pkl')

snow = "null"


def kalman_filter_v3(series, Q=0.01, R=1.0):
    if len(series) == 0: return []
    x_est = series.iloc[0]
    P = 1.0
    x_estimates = []
    for z in series:
        x_pred = x_est
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x_est = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        x_estimates.append(x_est)
    return x_estimates


def get_control():
    import warnings
    warnings.filterwarnings('ignore')
    
    global snow

    check_query = text("select count(*) from info_control_model where reg_date = (select reg_date from info_rwis where latitude = '33.38483693' order by reg_date desc limit 1)")
    query = text("select * from info_rwis where reg_date is not null and latitude = '33.38483693' order by reg_date desc limit 1")   
    # senser data 적설량 추가
    snowcover_query = text("select snowcover from sensor_data_snowcover order by create_time desc limit 1")
    
    control_6h_query = text("SELECT reg_date, frozen, control_l, control_s FROM info_control_pred WHERE reg_date >= NOW() - INTERVAL '6 HOUR' ORDER BY reg_date ASC")
    rwis_6h_query = text("""
    SELECT reg_date,
           CAST(NULLIF(visibility, '') AS FLOAT) as visibility,
           CAST(NULLIF(road_temp, '') AS FLOAT) as road_temp,
           CAST(NULLIF(water_film, '') AS FLOAT) as water_film,
           CAST(NULLIF(friction, '') AS FLOAT) as friction
    FROM info_rwis 
    WHERE reg_date >= NOW() - INTERVAL '6 HOUR' 
      AND latitude = '33.38483693'
    ORDER BY reg_date ASC
""")

    snow_6h_query = text("""
        SELECT create_time as reg_date, CAST(snowcover AS FLOAT) as snow
        FROM sensor_data_snowcover
        WHERE create_time >= DATE_SUB(NOW(), INTERVAL 6 HOUR)
        ORDER BY create_time ASC
    """)

    

    engine_mariadb = create_engine(DATABASE_URL2, pool_pre_ping=True)

    with engine_mariadb.begin() as conn_maria:
        snow = conn_maria.execute(snowcover_query).scalar()    
        if snow is None or snow == "null":
            snow = 0
        else:
            snow = int(snow)      
        res_snow_6h = conn_maria.execute(snow_6h_query)
        df_snow_6h = pd.DataFrame(res_snow_6h.fetchall(), columns=res_snow_6h.keys())

    with engine.begin() as conn:
        check = conn.execute(check_query).scalar()
        res_rwis = conn.execute(rwis_6h_query)
        df_rwis_6h = pd.DataFrame(res_rwis.fetchall(), columns=res_rwis.keys())
        
        res_ctrl = conn.execute(control_6h_query)
        df_control_6h = pd.DataFrame(res_ctrl.fetchall(), columns=res_ctrl.keys())

        if check == 0:
            result = conn.execute(query)
            data = [dict(row) for row in result.mappings()]  # 결과를 딕셔너리로 변환        
            freezing, controlL, controlS = get_add_data()
            reg_date = data[0]["reg_date"]
            data_sample = {
                "date": [data[0]["reg_date"].strftime("%Y-%m-%d")],
                "hour": [data[0]["reg_date"].hour],    
                "road_temp": [float(data[0]["road_temp"])],  
                "friction": [float(data[0]["friction"])],      
                "snow": [float(snow)],
                "visibility": [float(data[0]["visibility"])],
                "water_film": [float(data[0]["water_film"])],                  
                # "frozen": [freezing],
                #"snow": [data[0]["snow"]],        
                "frozen": [float(freezing)], # 명시적으로 float()으로 한번 더 감싸서 확실히 float로 처리
                "control_l": [float(controlL)], # 명시적으로 float()으로 한번 더 감싸서 확실히 float로 처리
                "control_s": [float(controlS)]
                }                

            

            data_process = pd.DataFrame(data_sample)    
            data_process2 = pd.DataFrame(data_sample)   
            df_current = pd.DataFrame(data_sample)
            df_current['reg_date'] = reg_date 

            df_raw = pd.concat([df_rwis_6h, df_snow_6h, df_control_6h, df_current], axis=0, ignore_index=True)
            df_raw = df_raw.drop_duplicates('reg_date').sort_values('reg_date').reset_index(drop=True)

            # 모델 로드
            try:
                print(f"⏳ 모델 로드 시도: {MODEL_PATH}")
                model = joblib.load(MODEL_PATH)
                print("✅ 모델 로드 성공.")
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}")
                # 모델 로드에 실패하면 여기서 함수를 종료할 수 있습니다.
                return
            
            
            # 모델 예측 및 필요 결과값 출력
            pred = model.predict(data_process.drop(['date', 'hour'], axis=1))
            pred_proba = model.predict_proba(data_process.drop(['date', 'hour'], axis=1))      
            pred_proba_np = np.array(pred_proba)

            flattened_proba = pred_proba_np.flatten()
        
            probability_columns = [f'probability{group}_{index}h'
                     for index in range(1,2)
                     for group in range(0, 3)]
            
            data_process[probability_columns] = flattened_proba            
            data_process[['pred_1h']] = pred[0]
            data_process['tp'] = data_process[['control_s', 'control_l']].max(axis=1)            
            data_process['min'] = data[0]["reg_date"].minute
            #"min": [data[0]["reg_date"].minute],

            #
            try:
                print(f"⏳ 신규모델 로드 시도: {MODEL_PATH2}")
                model2 = joblib.load(MODEL_PATH2)
                print("✅ 신규모델 로드 성공.")
            except Exception as e:
                print(f"❌ 신규모델 로드 실패: {e}")
                # 모델 로드에 실패하면 여기서 함수를 종료할 수 있습니다.
                return
            
            
            # 신규모델 예측 및 필요 결과값 출력
            pred2 = model2.predict(data_process2.drop(['date', 'hour'], axis=1))
            pred_proba2 = model2.predict_proba(data_process2.drop(['date', 'hour'], axis=1))      
            pred_proba_np2 = np.array(pred_proba2)

            flattened_proba2 = pred_proba_np2.flatten()
            
            probability_columns2 = [f'probability{group}_{index}h'
                     for index in range(1, 4)
                     for group in range(0, 3)]
            
            data_process2[probability_columns2] = flattened_proba2           
            data_process2[['pred_1h', 'pred_2h', 'pred_3h']] = pred2[0]
            data_process2['tp'] = data_process2[['control_s', 'control_l']].max(axis=1)            
            data_process2['min'] = data[0]["reg_date"].minute


            # --- [frozen 결측 보정: v3 전용] ---
            frozen_val = float(freezing)
            if frozen_val == -1 or pd.isna(frozen_val):
                try:
                    frozen_model = joblib.load(MODEL_FROZEN)
                    frozen_input = pd.DataFrame([{
                        "visibility": float(data[0]["visibility"]),
                        "snow": float(snow),
                        "friction": float(data[0]["friction"]),
                        "road_temp": float(data[0]["road_temp"]),
                        "control_l": float(controlL),
                        "control_s": float(controlS)
                    }])
                    frozen_val = float(frozen_model.predict(frozen_input)[0])
                    print(f"✅ frozen_model 예측값 사용: {frozen_val}")
                except Exception as e:
                    print(f"❌ frozen_model 예측 실패, 기존값 사용: {e}")

            # --- [고도화 프로세스 시작] ---
            window_size = 60 # 최근 60분

            # 마찰계수 보정 (Friction >= 1.0 일 때 최근 60분 평균으로 대체)
            f_rolling_mean = df_raw['friction'].where(df_raw['friction'] < 1).rolling(window=window_size, min_periods=1).mean()
            df_raw.loc[df_raw['friction'] >= 1, 'friction'] = f_rolling_mean.fillna(0.82) # 데이터 없으면 기본값

            # 가시거리 보정 (Visibility >= 20000 일 때 최근 60분 평균으로 대체)
            v_rolling_mean = df_raw['visibility'].where(df_raw['visibility'] < 20000).rolling(window=window_size, min_periods=1).mean()
            df_raw.loc[df_raw['visibility'] >= 20000, 'visibility'] = v_rolling_mean.fillna(20000.0)

            # 적설량 보정 (Snow >= 400 일 때 최근 60분 중앙값으로 대체)
            s_rolling_median = df_raw['snow'].where(df_raw['snow'] < 400).rolling(window=window_size, min_periods=1).median()
            df_raw.loc[df_raw['snow'] >= 400, 'snow'] = s_rolling_median.fillna(0.0)

            # 전처리 및 시계열 보간 (1분 단위 정규화)
            df_all = df_raw.set_index('reg_date').resample('1min').last().ffill().fillna(0)

            # 파생 변수 생성
            # 칼만 필터 적용
            for col in ['snow', 'friction', 'road_temp', 'frozen']:
                df_all[f'{col}_km'] = kalman_filter_v3(df_all[col])

            # 시간 관련 파생변수
            df_all['year'], df_all['month'], df_all['day'] = df_all.index.year, df_all.index.month, df_all.index.day
            df_all['hour_sin'] = np.sin(2 * np.pi * df_all.index.hour / 24)
            df_all['hour_cos'] = np.cos(2 * np.pi * df_all.index.hour / 24)
            df_all['minute_sin'] = np.sin(2 * np.pi * df_all.index.minute / 60)
            df_all['minute_cos'] = np.cos(2 * np.pi * df_all.index.minute / 60)
            df_all['visibility_log'] = np.log1p(df_all['visibility'])

            # 롤링 통계량 (현재 시점 기준 과거 1h, 3h 계산)
            df_all['snow_diff_1h'] = df_all['snow'] - df_all['snow'].shift(60)
            df_all['snow_std_3h'] = df_all['snow'].rolling(180, min_periods=1).std()
            df_all['snow_hours_3h'] = (df_all['snow'] > 0).rolling(180, min_periods=1).sum()
            
            # 마찰계수 및 온도 플래그
            f_q05 = df_all['friction'].quantile(0.05)
            df_all['low_friction_flag'] = (df_all['friction'] <= f_q05).astype(int)
            df_all['low_friction_hours_3h'] = df_all['low_friction_flag'].rolling(180, min_periods=1).sum()
            
            df_all['freezing_flag'] = (df_all['road_temp'] <= 0).astype(int)
            df_all['extreme_cold_flag'] = (df_all['road_temp'] <= -5).astype(int)
            df_all['freezing_hours_3h'] = df_all['freezing_flag'].rolling(180, min_periods=1).sum()
            df_all['extreme_cold_hours_3h'] = df_all['extreme_cold_flag'].rolling(180, min_periods=1).sum()
            
            # 결빙 위험 지수
            df_all["intensity"] = pd.cut(df_all["frozen"], bins=[-0.001, 0, 1, np.inf], labels=[0, 1, 2]).fillna(0).astype(int)
            df_all["frozen_risk"] = df_all["frozen"] * (0 - df_all["road_temp"]) * (df_all["snow"] + 1)
            df_all.fillna(0, inplace=True)

            # 최종 모델 예측
            # [1h 예측] 
            m_1h = joblib.load(MODEL_1H)
            cols_1h = ['road_temp', 'friction', 'snow', 'frozen', 'visibility', 'water_film', 'control_l', 'control_s']
            current_x_1h = df_all[cols_1h].iloc[[-1]]
            
            pred_1h = m_1h.predict(current_x_1h)[0]
            prob_1h = m_1h.predict_proba(current_x_1h)[0]

            # [2h, 3h 예측] - 최근 10분 시퀀스 사용
            features_seq = [
                'control_l', 'control_s', 'year', 'month', 'day', 'hour_sin', 'hour_cos', 
                'minute_sin', 'minute_cos', 'visibility_log', 'snow_km', 'friction_km', 
                'road_temp_km', 'frozen_km', 'snow_diff_1h', 'snow_std_3h', 'snow_hours_3h', 
                'low_friction_flag', 'low_friction_hours_3h', 'freezing_flag', 'extreme_cold_flag', 
                'freezing_hours_3h', 'extreme_cold_hours_3h', 'intensity', 'frozen_risk'
            ]
            X_seq = df_all[features_seq].tail(10).values.reshape(1, -1)
            
            m_2h = joblib.load(MODEL_2H)
            m_3h = joblib.load(MODEL_3H)
            
            pred_2h = m_2h.predict(X_seq)[0]
            prob_2h = m_2h.predict_proba(X_seq)[0]
            
            pred_3h = m_3h.predict(X_seq)[0]
            prob_3h = m_3h.predict_proba(X_seq)[0]

            # 결과 데이터 매핑 
            data_process3 = pd.DataFrame([{
                "date": data[0]["reg_date"].replace(hour=0, minute=0, second=0, microsecond=0), # timestamp용 날짜만
                "hour": int(data[0]["reg_date"].hour),
                "min": int(data[0]["reg_date"].minute),
                "road_temp": str(data_sample["road_temp"][0]), # varchar(30) 대응
                "friction": str(round(df_all['friction'].iloc[-1], 2)), # varchar(30) 대응
                "snow": str(round(df_all['snow'].iloc[-1], 2)),         # varchar(30) 대응
                "frozen": frozen_val,             # float8 대응 (frozen_model 예측값 적용)
                "visibility": str(int(df_all['visibility'].iloc[-1])), # varchar(30) 대응
                "water_film": str(data_sample["water_film"][0]),       # varchar(30) 대응
                "control_l": int(data_sample["control_l"][0]),
                "control_s": int(data_sample["control_s"][0]),
                "probability0_1h": float(prob_1h[0]), 
                "probability1_1h": float(prob_1h[1]), 
                "probability2_1h": float(prob_1h[2]),
                "pred_1h": int(pred_1h),
                "tp": int(max(data_sample["control_l"][0], data_sample["control_s"][0])),
                "probability0_2h": float(prob_2h[0]), 
                "probability1_2h": float(prob_2h[1]), 
                "probability2_2h": float(prob_2h[2]),
                "probability0_3h": float(prob_3h[0]), 
                "probability1_3h": float(prob_3h[1]), 
                "probability2_3h": float(prob_3h[2]),
                "pred_2h": int(pred_2h),
                "pred_3h": int(pred_3h)
            }])
                
            insert_control_data(conn, data_process, data_process2, data_process3)


def get_add_data():

    # 2025.12.03(pjh) 기존 int 형으로 들어오던 데이터가 새로운 text 형식으로 적용되어 controlL, controlS 해당 txt->int형변환 하드코딩
    def convert_control_code(code):
        if code == "TCS002":
            return 1 # 체인 이상 (Level 1)
        elif code == "TCS003":
            return 2 # 통제 (전면 통제) (Level 2)
        # TCS001은 정상(Normal) 상태로 추정, 0으로 처리
        elif code == "TCS001":
             return 0
        else:
            return 0 # None이나 알 수 없는 코드도 0으로 처리 (Level 0)
        
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
        
        freezing_text = item.find("freezing").text
            
        if freezing_text is not None and freezing_text.strip().isdigit():
            try:
                freezing = int(freezing_text)
            except ValueError:
                freezing = 0 # 숫자로 변환 불가능하면 0으로 처리
        else:
            freezing = 0 # 텍스트가 없거나(None) 비어있으면 0으로 처리

        controlL_element = item.find("contolL")
        controlS_element = item.find("contolS")

        # .text를 안전하게 가져오고 공백 제거
        controlL_text = controlL_element.text.strip() if controlL_element is not None and controlL_element.text is not None else None
        controlS_text = controlS_element.text.strip() if controlS_element is not None and controlS_element.text is not None else None
        
        # 변환 함수를 통해 최종 int 값 할당
        controlL = convert_control_code(controlL_text) # int (0, 1, 2 중 하나)
        controlS = convert_control_code(controlS_text) # int (0, 1, 2 중 하나)
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


def insert_control_data(conn, df, df2, df3):  
    v3_columns = [
            'date', 'hour', 'min', 'road_temp', 'friction', 'snow', 'frozen', 
            'visibility', 'water_film', 'control_l', 'control_s', 
            'probability0_1h', 'probability1_1h', 'probability2_1h', 'pred_1h', 'tp',
            'probability0_2h', 'probability1_2h', 'probability2_2h',
            'probability0_3h', 'probability1_3h', 'probability2_3h',
            'pred_2h', 'pred_3h'
        ]
    try:
        
        df.to_sql('info_control_pred',con=conn, if_exists='append', index=False, method='multi')
        df2.to_sql('info_control_pred_v2',con=conn, if_exists='append', index=False, method='multi')

        df3_final = df3.reindex(columns=v3_columns).iloc[[-1]].replace({np.nan: None})
        
        int_cols = ['hour', 'min', 'control_l', 'control_s', 'pred_1h', 'pred_2h', 'pred_3h', 'tp']
        for col in int_cols:
            df3_final[col] = pd.to_numeric(df3_final[col], errors='coerce').fillna(0).astype(int)
        with engine.begin() as connection:
            
            col_names = ", ".join([f'"{c}"' if c in ['date', 'hour', 'min'] else c for c in v3_columns])
            placeholders = ", ".join([f":{c}" for c in v3_columns])
            
            sql_str = f"INSERT INTO info_control_pred_v3 ({col_names}) VALUES ({placeholders})"
            
            # dict 형태로 데이터 변환 후 실행
            insert_values = df3_final.to_dict(orient='records')
            
            connection.execute(text(sql_str), insert_values)

        print("✅ 모든 테이블(v1, v2, v3) 적재 완료")
    except SQLAlchemyError as e:
        print(f"❌ 데이터 삽입 오류 발생: {e}")
    except Exception as e:
        print(f"❌ v3 적재 실패: {e}")