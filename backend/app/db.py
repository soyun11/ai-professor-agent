from sqlmodel import SQLModel, Session, create_engine

DB_URL = "sqlite:///./app.db"
engine = create_engine(DB_URL, echo=False)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session():
    """ DB와 안전하게 대화하는 통로를 만들어줌"""
    with Session(engine) as session: # DB 연결 시작
        yield session # 함수에 session 전달
        session.close() # 함수 끝나면 자동으로 여기로 들어오는데, DB 연결 종료(메모리 누수 방지)
        
