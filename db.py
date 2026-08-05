import sqlite3
import datetime

DB_NAME = "audit_logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            user_email TEXT,
            action TEXT,
            question TEXT,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_query(user_email, question, answer):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO audit_logs (timestamp, user_email, action, question, answer) VALUES (?, ?, ?, ?, ?)",
        (timestamp, user_email, "QUERY", question, answer)
    )
    conn.commit()
    conn.close()

def log_action(user_email, action):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO audit_logs (timestamp, user_email, action, question, answer) VALUES (?, ?, ?, ?, ?)",
        (timestamp, user_email, action, None, None)
    )
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    
    # Convert sqlite3.Row to dict for easy use in pandas/streamlit
    return [dict(row) for row in rows]

# Initialize db on module load
init_db()
