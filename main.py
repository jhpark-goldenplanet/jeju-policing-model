import uvicorn
from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
import time

from app.config.database import engine, Base, connect_db, disconnect_db
from app.api.router import router
from app.scheduler.control import get_control
from app.scheduler.risk import get_risk
from app.scheduler.risk_m import get_risk_m

# 스케줄러 객체 생성 (전역 변수)
scheduler = BackgroundScheduler()


def create_app() -> FastAPI:
    """ app 변수 생성 및 초기값 설정 """
    _app = FastAPI(
        title="policing_model",
        description="policing_model",
        version="1.0.0",
    )
    _app.include_router(router, prefix='/api')

    # 서버 시작 시 DB 연결 및 스케줄러 실행
    @_app.on_event("startup")
    def startup():
        connect_db()
        scheduler.add_job(get_control, 'cron', minute='1-59/5')  # 매시간 1분부터 5분주기로 실행      
        scheduler.add_job(get_risk, 'cron', minute=40)  # 매시간 40분에 실행
        scheduler.add_job(get_risk_m, 'cron', minute='4-59/5')  # 매시간 4분부터 5분주기로 실행          
        # print("🔍 테스트: 함수 직접 실행 시작")
        # get_control() 
        # print("🔍 테스트: 함수 직접 실행 종료")
        scheduler.start()                                        
        print("✅ 스케줄러 시작!")

    # 서버 종료 시 DB 연결 해제 및 스케줄러 종료
    @_app.on_event("shutdown")
    def shutdown():
        disconnect_db()
        scheduler.shutdown()
        print("🛑 스케줄러 종료!")

    return _app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8015, reload=False)
