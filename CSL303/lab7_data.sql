-- This script populates the database with a large amount of random data.
-- It is designed to be run from the command line:
-- sqlite3 university.db < lab7_data.sql

PRAGMA a;
PRAGMA foreign_keys = OFF;
PRAGMA journal_mode = MEMORY;
PRAGMA synchronous = OFF;

BEGIN TRANSACTION;

-- Generate 2000 Students
WITH RECURSIVE 
  cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt LIMIT 2000),
  random_majors(id, major) AS (
    VALUES (0, 'Computer Science'), (1, 'Physics'), (2, 'Chemistry'), (3, 'Mathematics'), (4, 'Humanities'), (5, 'Engineering')
  )
INSERT INTO Students (first_name, last_name, email, major, enrollment_date)
SELECT
  'FirstName' || x,
  'LastName' || x,
  'student' || x || '@university.edu',
  (SELECT major FROM random_majors WHERE id = ABS(RANDOM()) % 6),
  DATE('2020-01-01', '+' || (ABS(RANDOM()) % 1825) || ' days')
FROM cnt;


-- Generate 100 Courses
WITH RECURSIVE
  cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt LIMIT 100),
  random_depts(id, dept) AS (
    VALUES (0, 'Science'), (1, 'Humanities'), (2, 'Engineering'), (3, 'Arts'), (4, 'Business')
  )
INSERT INTO Courses (course_name, department, credits)
SELECT
  'Course' || x || ' ' || (SELECT dept FROM random_depts WHERE id = ABS(RANDOM()) % 5),
  (SELECT dept FROM random_depts WHERE id = ABS(RANDOM()) % 5),
  (ABS(RANDOM()) % 3) + 2 -- Credits between 2 and 4
FROM cnt;


-- Generate 10000 Enrollments
WITH RECURSIVE
  cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt LIMIT 10000)
INSERT INTO Enrollments (student_id, course_id, grade)
SELECT
  (ABS(RANDOM()) % 2000) + 1, -- student_id between 1 and 2000
  (ABS(RANDOM()) % 100) + 1,  -- course_id between 1 and 100
  ROUND((ABS(RANDOM()) % 2.0) + 2.0, 2) -- grade between 2.0 and 4.0
FROM cnt;

COMMIT;

PRAGMA synchronous = FULL;
PRAGMA journal_mode = DELETE;
PRAGMA foreign_keys = ON;
VACUUM;
