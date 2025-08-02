import psycopg2
from psycopg2.extras import RealDictCursor
from db_manager import DatabaseManager

def get_table_schema(table_name):
    conn = None
    try:
        conn = DatabaseManager.get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
                """,
                (table_name,)
            )
            columns = cursor.fetchall()
            return columns
    except Exception as e:
        print(f"Error fetching schema for table {table_name}: {e}")
    finally:
        if conn:
            conn.close()

def get_sample_rows(table_name, limit=5):
    conn = None
    try:
        conn = DatabaseManager.get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT %s;", (limit,))
            rows = cursor.fetchall()
            return rows
    except Exception as e:
        print(f"Error fetching sample rows from table {table_name}: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    table = "votes"
    print(f"Schema for table '{table}':")
    schema = get_table_schema(table)
    for col in schema:
        print(f"  - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")

    print(f"\nSample rows from '{table}':")
    samples = get_sample_rows(table)
    for row in samples:
        print(row)