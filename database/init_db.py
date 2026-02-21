import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'candidates.db')
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    is_admin INTEGER
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT,
    experience REAL,
    skills REAL,
    quiz REAL,
    selected INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')

# Create admin user
c.execute("INSERT INTO users (username, password, is_admin) VALUES ('admin', 'admin', 1)")

conn.commit()
conn.close()
print("DB cleared and recreated successfully with admin user!")