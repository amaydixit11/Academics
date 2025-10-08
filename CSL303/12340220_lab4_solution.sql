-- lab4
-- 1.
CREATE TABLE enrolled (
    student_id INTEGER,
    course_id INTEGER,
    grade TEXT,
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

-- 2.
INSERT INTO enrolled (student_id, course_id, grade)
SELECT student_id, course_id, grade
FROM enrollments;

-- 3.
UPDATE Students
SET department = 'Philosophy'
WHERE name LIKE '%i%';

-- 4.
ALTER TABLE students
ADD COLUMN email TEXT;

-- 5.
UPDATE Students
SET email = LOWER(name) || '@iitbhilai.ac.in';

-- 6.
SELECT s.name
FROM students
WHERE department='Computer Science';

-- 7.
SELECT s.name
FROM students s
JOIN courses c
on s.name=c.course_name;

-- 8.
SELECT s.name, c.course_name
FROM students s
JOIN enrollments e
ON s.student_id=e.student_id
JOIN courses c
ON c.course_id=e.course_id
ORDER BY c.course_name;

-- 9.
SELECT s.name, c.course_name
FROM students s
LEFT JOIN enrollments e
ON s.student_id=e.student_id
LEFT JOIN courses c
ON c.course_id=e.course_id;

-- 10.
SELECT s.name
FROM students s
WHERE s.name LIKE 'A%';

-- 11.
SELECT DISTINCT s.name
FROM students s
JOIN enrollments e
ON s.student_id=e.student_id
JOIN courses c
ON c.course_id=e.course_id
WHERE c.credits>3;

-- 12.
SELECT s.name
FROM students s
LEFT JOIN enrollments e
ON s.student_id=e.student_id
LEFT JOIN courses c
ON c.course_id=e.course_id
WHERE c.course_id IS NULL;
