#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_SSL_MODE
)

def main():
    # Load environment variables from .env
    load_dotenv()
    # Establish connection
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        sslmode=POSTGRES_SSL_MODE
    )
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM votes;")
            rows = cursor.fetchall()
            if not rows:
                print("No rows found in votes table.")
            else:
                for row in rows:
                    print(row)
    except Exception as e:
        print(f"Error querying votes table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
