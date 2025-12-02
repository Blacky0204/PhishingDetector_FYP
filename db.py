# db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Connect to MySQL (XAMPP default configuration)
DATABASE_URL = "mysql+pymysql://root:@127.0.0.1:3306/phishingdb"

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define SearchHistory model
class SearchHistory(Base):
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(500))
    label = Column(Integer)  # 1 for phishing, 0 for safe
    prediction = Column(String(50))
    confidence = Column(Float)
    explanation = Column(Text)  # Store as a JSON string
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables (if not already created)
Base.metadata.create_all(bind=engine)

# Create a session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()