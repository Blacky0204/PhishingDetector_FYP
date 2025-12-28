# db.py - Render-safe (SQLite)
import sqlite3
from sqlite3 import Error
from typing import List, Optional
import os

class SQLiteDB:
    """SQLite database operations."""
    def __init__(self, db_file: str = None):
        # Use absolute path so cloud working directory doesn't break it
        if db_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_file = os.path.join(base_dir, "phishing_history.db")
        self.db_file = db_file
        self.conn = None

    def connect(self):
        """Connect to SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            return self.conn
        except Error as e:
            print(f"[DB] Connection error: {e}")
            return None

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def execute_query(self, query: str, params: tuple = ()):
        """Execute INSERT/UPDATE/DELETE query."""
        try:
            conn = self.connect()
            if conn is None:
                return None
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            last_id = cursor.lastrowid
            return last_id
        except Error as e:
            print(f"[DB] Query error: {e}")
            return None
        finally:
            self.close()

    def fetch_all(self, query: str, params: tuple = ()) -> List[dict]:
        """Fetch all rows from a SELECT query."""
        try:
            conn = self.connect()
            if conn is None:
                return []
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Error as e:
            print(f"[DB] Fetch error: {e}")
            return []
        finally:
            self.close()

    def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        """Fetch one row from a SELECT query."""
        try:
            conn = self.connect()
            if conn is None:
                return None
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        except Error as e:
            print(f"[DB] Fetch error: {e}")
            return None
        finally:
            self.close()

# ---------------- Database Initialization ----------------
def init_database():
    """Create tables if they don't exist."""
    db = SQLiteDB()

    users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(50) NOT NULL UNIQUE,
        email VARCHAR(100) UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """

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
    print("[DB] Database initialized")

# ---------------- User Operations ----------------
def create_user(username: str, email: str, password_hash: str) -> Optional[int]:
    db = SQLiteDB()
    query = """
    INSERT INTO users (username, email, password_hash)
    VALUES (?, ?, ?)
    """
    return db.execute_query(query, (username, email, password_hash))

def get_user_by_username(username: str) -> Optional[dict]:
    db = SQLiteDB()
    query = "SELECT * FROM users WHERE username = ?"
    return db.fetch_one(query, (username,))

def get_user_by_email(email: str) -> Optional[dict]:
    db = SQLiteDB()
    query = "SELECT * FROM users WHERE email = ?"
    return db.fetch_one(query, (email,))

def get_user_by_id(user_id: int) -> Optional[dict]:
    db = SQLiteDB()
    query = "SELECT * FROM users WHERE id = ?"
    return db.fetch_one(query, (user_id,))

# ---------------- Search History Operations ----------------
def add_search_history(
    url: str,
    user_id: Optional[int] = None,
    prediction: str = None,
    confidence: float = None,
    explanation: str = None
) -> Optional[int]:
    db = SQLiteDB()
    query = """
    INSERT INTO search_history (url, user_id, prediction, confidence, explanation)
    VALUES (?, ?, ?, ?, ?)
    """
    return db.execute_query(query, (url, user_id, prediction, confidence, explanation))

def get_user_search_history(user_id: int, limit: int = 50) -> List[dict]:
    db = SQLiteDB()
    query = """
    SELECT * FROM search_history
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT ?
    """
    return db.fetch_all(query, (user_id, limit))
