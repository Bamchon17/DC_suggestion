import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# โหลด .env ที่ root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Database configuration จาก .env
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")  # default

# กำหนด DSN เป็น dictionary
DSN = {
    'user': DB_USER,
    'password': DB_PASSWORD,
    'host': DB_HOST,
    'port': DB_PORT,
    'database': DB_NAME
}

# ตัวแปร global สำหรับ connection และ engine (เริ่มต้นเป็น None)
_conn = None
_engine = None

def get_connection():
    """Create and return a psycopg2 connection object."""
    global _conn
    if _conn is None or _conn.closed:
        try:
            _conn = psycopg2.connect(
                **DSN,
                sslmode="require"
            )
        except Exception as e:
            print(f"Connection failed: {e}")
            raise
    return _conn

def get_engine():
    """Return or create a SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
        )
    return _engine

def fetch_query(query, columns=None):
    """Execute a query and return results as a pandas DataFrame."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame(rows)
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

# เปิดใช้งาน connection และ engine เมื่อเรียกใช้เท่านั้น
conn = get_connection()
engine = get_engine()

# เปิดเผย DSN สำหรับการใช้งานในโมดูลอื่น
dsn = DSN