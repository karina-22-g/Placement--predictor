import sqlite3
import random

# -----------------------------
# CONNECT DATABASE
# -----------------------------
conn = sqlite3.connect("university.db")
cursor = conn.cursor()

# -----------------------------
# CREATE TABLE
# -----------------------------
cursor.execute("DROP TABLE IF EXISTS students")

cursor.execute("""
CREATE TABLE students(
    student_id INTEGER PRIMARY KEY,
    cgpa REAL,
    backlogs INTEGER,
    internships INTEGER,
    projects INTEGER,
    aptitude_score INTEGER,
    coding_score INTEGER,
    communication INTEGER,
    placed INTEGER
)
""")

# -----------------------------
# GENERATE DATA
# -----------------------------
data = []

for i in range(1000):

    cgpa = round(random.uniform(5.0, 9.8), 2)
    backlogs = random.randint(0, 10)
    internships = random.randint(0, 10)
    projects = random.randint(0, 10)
    aptitude = random.randint(40, 100)
    coding = random.randint(30, 100)
    communication = random.randint(40, 100)

    # -----------------------------
    # REALISTIC SCORING
    # -----------------------------
    score = (
        cgpa * 10
        - backlogs * 12
        + internships * 8
        + projects * 6
        + aptitude * 0.3
        + coding * 0.5
        + communication * 0.2
    )

    # add randomness
    score += random.uniform(-20, 20)

    data.append((i, cgpa, backlogs, internships, projects,
                 aptitude, coding, communication, score))

# -----------------------------
# SORT BY SCORE (IMPORTANT)
# -----------------------------
data.sort(key=lambda x: x[-1], reverse=True)

# -----------------------------
# ASSIGN 70% PLACED
# -----------------------------
final_data = []

cutoff = int(0.7 * len(data))

for i, row in enumerate(data):
    placed = 1 if i < cutoff else 0
    final_data.append(row[:-1] + (placed,))

# -----------------------------
# INSERT INTO DB
# -----------------------------
cursor.executemany("""
INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", final_data)

conn.commit()
conn.close()

print("✅ Dataset created with 70% placed / 30% not placed")
