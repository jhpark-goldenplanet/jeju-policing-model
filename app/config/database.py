from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config.config import DATABASE_URL, DATABASE_URL2  # DB 설정 파일에서 불러오기

# 데이터베이스 엔진 생성 (서버 시작 시 연결 유지)
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
engine_mariadb = create_engine(DATABASE_URL2, pool_size=5, max_overflow=10, pool_recycle=3500)

# 세션 팩토리 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SessionLocal_mariadb = sessionmaker(autocommit=False, autoflush=False, bind=engine_mariadb)

# Base 모델 설정
Base = declarative_base()

# 전역 DB 세션 변수
db_session = None
db_session_mariadb = None

def connect_db():
    """서버 시작 시 DB 연결"""
    global db_session, db_session_mariadb
    if db_session is None:
        db_session = SessionLocal()      
    if db_session_mariadb is None:
        db_session_mariadb = SessionLocal_mariadb()      

def disconnect_db():
    """서버 종료 시 DB 연결 해제"""
    global db_session, db_session_mariadb
    if db_session:
        db_session.close()
        db_session = None        
    if db_session_mariadb:
        db_session_mariadb.close()
        db_session_mariadb = None    