from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import Base

DB_PATH = Path(__file__).resolve().parents[2]/'results.db'

engine = create_engine(f"sqlite:///{DB_PATH}")

SessionLocal = sessionmaker(bind=engine)

def create_tables():
    Base.metadata.create_all(engine)
