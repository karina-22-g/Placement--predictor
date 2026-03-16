import sqlite3
import random

# SQLite automatically closes connection using "with"
with sqlite3.connect("university.db", timeout=30) as conn:

    cursor = conn.cursor()

    # Drop table if it exists (prevents locking problems)
    cursor.execute("DROP TABLE IF EXISTS students")

    # Create table
    cursor.execute("""
    CREATE TABLE students(
        student_id INTEGER PRIMARY KEY,
        attendance INTEGER,
        study_hours INTEGER,
        assignments INTEGER,
        internal_marks INTEGER,
        pass INTEGER
    )
    """)

    # Generate synthetic dataset
    for i in range(1000):

        attendance = random.randint(40,100)
        study_hours = random.randint(0,6)
        assignments = random.randint(0,5)
        internal_marks = random.randint(30,90)

        score = attendance*0.3 + study_hours*5 + assignments*5 + internal_marks*0.4

        result = 1 if score > 80 else 0

        cursor.execute(
            "INSERT INTO students VALUES (?,?,?,?,?,?)",
            (i, attendance, study_hours, assignments, internal_marks, result)
        )

    conn.commit()

print("Dataset generated successfully in university.db")



from google.colab import files
files.download("university.db")
