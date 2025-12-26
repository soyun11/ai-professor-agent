from sqlmodel import SQLModel, Session, create_engine

DB_URL = "sqlite:///./app.db"
engine = create_engine(DB_URL, echo=False)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
