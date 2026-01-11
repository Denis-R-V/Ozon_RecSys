from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


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
    request_bytes = Column(Integer)
    token_count = Column(Integer)

    error = Column(String)
