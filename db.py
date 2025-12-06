# db.py - UPDATED VERSION
import sqlite3
from sqlite3 import Error
from typing import List, Tuple, Optional
import os

class SQLiteDB:
    """SQLite database operations"""
    def __init__(self, db_file: str = "phishing_history.db"):
        self.db_file = db_file
        self.conn = None
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.conn.row_factory = sqlite3.Row  # Return dictionaries
            return self.conn
        except Error as e:
            print(f"Database connection error: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def execute_query(self, query: str, params: tuple = ()):
        """Execute a query"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Query error: {e}")
            return None
    
    def fetch_all(self, query: str, params: tuple = ()) -> List[dict]:
        """Fetch all rows"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Error as e:
            print(f"Fetch error: {e}")
            return []
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        """Fetch single row"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        except Error as e:
            print(f"Fetch error: {e}")
            return None

# ============= SPECIFIC DATABASE OPERATIONS =============

def init_database():
    """Initialize database tables if they don't exist"""
    db = SQLiteDB()
    
    # Create users table
    users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(50) NOT NULL UNIQUE,
        email VARCHAR(100) UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create search_history table (updated with user_id)
    search_table = """
    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER DEFAULT NULL,
        url VARCHAR(500),
        label INTEGER,
        prediction VARCHAR(50),
        confidence FLOAT,
        explanation TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    
    db.execute_query(users_table)
    db.execute_query(search_table)
    print("Database initialized")
    db.close()

# User operations
def create_user(username: str, email: str, password_hash: str) -> Optional[int]:
    """Create a new user"""
    db = SQLiteDB()
    query = """
    INSERT INTO users (username, email, password_hash)
    VALUES (?, ?, ?)
    """
    user_id = db.execute_query(query, (username, email, password_hash))
    db.close()
    return user_id

def get_user_by_username(username: str) -> Optional[dict]:
    """Get user by username"""
    db = SQLiteDB()
    query = "SELECT * FROM users WHERE username = ?"
    user = db.fetch_one(query, (username,))
    db.close()
    return user

def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email"""
    db = SQLiteDB()
    query = "SELECT * FROM users WHERE email = ?"
    user = db.fetch_one(query, (email,))
    db.close()
    return user

def get_user_by_id(user_id: int) -> Optional[dict]:
    """Get user by ID"""
    db = SQLiteDB()
    query = "SELECT * FROM users WHERE id = ?"
    user = db.fetch_one(query, (user_id,))
    db.close()
    return user

# Search history operations
def add_search_history(url: str, user_id: Optional[int] = None, 
                       prediction: str = None, confidence: float = None, 
                       explanation: str = None) -> Optional[int]:
    """Add search history record"""
    db = SQLiteDB()
    query = """
    INSERT INTO search_history (url, user_id, prediction, confidence, explanation)
    VALUES (?, ?, ?, ?, ?)
    """
    record_id = db.execute_query(query, (url, user_id, prediction, confidence, explanation))
    db.close()
    return record_id

def get_user_search_history(user_id: int, limit: int = 50) -> List[dict]:
    """Get search history for a user"""
    db = SQLiteDB()
    query = """
    SELECT * FROM search_history 
    WHERE user_id = ? 
    ORDER BY created_at DESC 
    LIMIT ?
    """
    history = db.fetch_all(query, (user_id, limit))
    db.close()
    return history

# Initialize database when module loads
init_database()