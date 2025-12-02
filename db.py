# db.py  (SQLite version – no XAMPP needed)

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

# ==============================
# DATABASE SETTINGS (SQLite)
# ==============================
# This will create a file called "phishing_history.db" in your project folder.
DATABASE_URL = "sqlite:///./phishing_history.db"

# For SQLite, we need "check_same_thread=False" so SQLAlchemy can reuse the connection
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # set True if you want to see SQL logs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==============================
# TABLE MODEL
# ==============================
class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(500))
    label = Column(Integer)          # 1 for phishing, 0 for safe
    prediction = Column(String(50))
    confidence = Column(Float)
    explanation = Column(Text)       # Store as a JSON string
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the table if it doesn't exist
Base.metadata.create_all(bind=engine)

# Dependency used in main.py
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()