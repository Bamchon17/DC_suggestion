import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# โหลด .env ที่ root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Database configuration จาก .env
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

# DSN เป็น dictionary
DSN = {
    'user': DB_USER,
    'password': DB_PASSWORD,
    'host': DB_HOST,
    'port': DB_PORT,
    'database': DB_NAME
}

_conn = None
_engine = None

def get_connection():
    global _conn
    if _conn is None or _conn.closed:
        try:
            _conn = psycopg2.connect(**DSN, sslmode="require", connect_timeout=10)
            logger.info("Database connection established successfully.")
        except psycopg2.Error as e:
            logger.error(f"Connection failed: {e}")
            raise
    return _conn

def get_engine():
    global _engine
    if _engine is None:
        try:
            _engine = create_engine(
                f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require",
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_pre_ping=True
            )
            logger.info("SQLAlchemy engine created successfully.")
        except Exception as e:
            logger.error(f"Engine creation failed: {e}")
            raise
    return _engine

def fetch_query(query, columns=None):
    """Execute query and return DataFrame. Auto detect columns if not provided."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                if columns is None and rows:
                    columns = [desc[0] for desc in cur.description]
                df = pd.DataFrame(rows, columns=columns)
                return df
    except psycopg2.Error as e:
        logger.error(f"Error executing query: {e}")
        return pd.DataFrame()
