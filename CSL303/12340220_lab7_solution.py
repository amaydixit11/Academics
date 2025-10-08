import sqlite3, time

DB_FILE = "lab7-1.db"
QUERY = "SELECT * FROM Students WHERE major = 'Computer Science';"
ITERATIONS = 100

con = sqlite3.connect(DB_FILE)
cur = con.cursor()

# Drop index if exists
# try:
#     cur.execute("DROP INDEX idx_students_major")
#     print("Index dropped.")
# except sqlite3.OperationalError:
#     print("Index did not exist.")

total_time = 0
for _ in range(ITERATIONS):
    start = time.time()
    cur.execute(QUERY).fetchall()
    end = time.time()
    total_time += (end - start)

print(f"Avg. time with index: {total_time / ITERATIONS:.6f} seconds")

# Drop index if exists
try:
    cur.execute("DROP INDEX idx_students_major")
    print("Index dropped.")

    total_time = 0
    for _ in range(ITERATIONS):
        start = time.time()
        cur.execute(QUERY).fetchall()
        end = time.time()
        total_time += (end - start)

    print(f"Avg. time without index: {total_time / ITERATIONS:.6f} seconds")
except sqlite3.OperationalError:
    print("Index did not exist.")


con.close()

print("\n\n==============================\n\n")

import random

con = sqlite3.connect(DB_FILE)
cur = con.cursor()

ITERATIONS = 500
students = [random.randint(1,2000) for _ in range(ITERATIONS)]
courses = [random.randint(1,100) for _ in range(ITERATIONS)]
grades = [round(random.uniform(0,10), 2) for _ in range(ITERATIONS)]

start = time.time()
for i in range(ITERATIONS):
    cur.execute("INSERT INTO Enrollments(student_id, course_id, grade) VALUES (?, ?, ?)", 
                (students[i], courses[i], grades[i]))
con.commit()
end = time.time()
print(f"Avg. insert time with indexes: {(end-start)/ITERATIONS:.6f} seconds")

try:
    cur.execute("DROP INDEX idx_enrollments_student_id")
    cur.execute("DROP INDEX idx_enrollments_course_id")
    print("Index dropped.")
    start = time.time()
    for i in range(ITERATIONS):
        cur.execute("INSERT INTO Enrollments(student_id, course_id, grade) VALUES (?, ?, ?)", 
                    (students[i], courses[i], grades[i]))
    con.commit()
    end = time.time()
    print(f"Avg. insert time without indexes: {(end-start)/ITERATIONS:.6f} seconds")
except sqlite3.OperationalError:
    print("Index did not exist.")

con.close()



