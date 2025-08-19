-- CREATE FACULTY TABLE
CREATE TABLE faculty (
facultyID int primary key,
firstName text not null,
lastName text not null,
department text);

-- CREATE STUDENTS TABLE
CREATE TABLE student (
studentID int primary key,
firstName text not null,
lastName text not null,
discipline text);

--INSERTING VALUES INTO STUDENTS TABLES
INSERT INTO student VALUES(1,'saurav','dixit','cse');
INSERT INTO student VALUES(2,'amay','dixit','dsai');
INSERT INTO student VALUES(3,'amay','amazing','dsai');

--INSERTING VALUES INTO FACULTY TABLES
INSERT INTO faculty VALUES(1,'rohit','raghu','cse');
INSERT INTO faculty VALUES(2,'chetan','raghu','dsai');
INSERT INTO faculty VALUES(3,'chetan','rathod','dsai');

--LISTING ALL STUDENTS OF CSE DISCIPLINE
select * from student where discipline="cse";

--LISTING ALL FACULTIES OF CSE DEPARTMENT
select * from faculty where department="cse";

--LIST THE FIRST AND LAST NAMES OF ALL THE STUDENTS
select firstName, lastName from student;

--LIST LAST NAME AND DEPARTMENT OF ALL THE FACULTIES
select lastName, department from faculty;

--LIST ALL UNIQUE FIRST NAME OF BOTH STUDENTS AND FACULTY
select firstName from student union select firstName from faculty;

--LIST ALL UNIQUE LAST NAMES OF BOTH STUDENTS AND FACULTY
select lastName from student union select lastName from faculty;

--LIST ALL FIRST NAME COMMON TO BOTH STUDENTS AND FACULTY
select firstName from student intersect select firstName from faculty;

--LIST ALL LAST NAME COMMON TO BOTH STUDENTS AND FACULTY
select lastName from student intersect select lastName from faculty;
