from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db() -> None:
    # создаем таблицы, если их нет
    Base.metadata.create_all(bind=engine)


class RequestHistory(Base):
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(String, nullable=False)

    endpoint = Column(String, nullable=False)
    status_code = Column(Integer, nullable=False)

    user_id = Column(Integer)
    requested_k = Column(Integer)

    is_cold = Column(Integer)
    processing_ms = Column(Float, nullable=False)
    inference_ms = Column(Float)
    recs_count = Column(Integer)
    db_write_ms = Column(Float)
    db_error = Column(Integer)
    error_type = Column(String)
    error = Column(String)
