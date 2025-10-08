-- Drop tables in reverse order of dependency to avoid foreign key constraints
DROP TABLE IF EXISTS Assignments;
DROP TABLE IF EXISTS Employees;
DROP TABLE IF EXISTS Projects;
DROP TABLE IF EXISTS Departments;

-- Create the database schema
CREATE TABLE Departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL UNIQUE,
    hod TEXT -- Head of Department
);

CREATE TABLE Projects (
    proj_id TEXT PRIMARY KEY,
    proj_name TEXT NOT NULL,
    budget REAL
);

CREATE TABLE Employees (
    emp_id INTEGER PRIMARY KEY,
    emp_name TEXT NOT NULL,
    dept_id INTEGER,
    salary REAL,
    hire_date DATE,
    manager_id INTEGER, -- Self-referencing key
    FOREIGN KEY (dept_id) REFERENCES Departments(dept_id),
    FOREIGN KEY (manager_id) REFERENCES Employees(emp_id)
);

CREATE TABLE Assignments (
    emp_id INTEGER,
    proj_id TEXT,
    PRIMARY KEY (emp_id, proj_id),
    FOREIGN KEY (emp_id) REFERENCES Employees(emp_id) ON DELETE CASCADE,
    FOREIGN KEY (proj_id) REFERENCES Projects(proj_id) ON DELETE CASCADE
);

-- Insert data
INSERT INTO Departments (dept_id, dept_name, hod) VALUES
(1, 'Engineering', 'Dr. Evelyn Reed'),
(2, 'Marketing', 'Mr. Frank Black'),
(3, 'Human Resources', 'Ms. Grace Hall'),
(4, 'Research & Development', 'Dr. Henry Shaw');

INSERT INTO Projects (proj_id, proj_name, budget) VALUES
('P101', 'Project Phoenix', 500000),
('P102', 'Project Neptune', 750000),
('P103', 'Project Gemini', 1200000),
('P104', 'Marketing Campaign Q3', 250000),
('P105', 'R&D Initiative Alpha', 900000);

INSERT INTO Employees (emp_id, emp_name, dept_id, salary, hire_date, manager_id) VALUES
(1, 'Alice Johnson', 1, 90000, '2021-06-15', NULL),
(2, 'Bob Williams', 1, 80000, '2022-01-20', 1),
(3, 'Charlie Brown', 1, 82000, '2021-11-30', 1),
(4, 'David Smith', 2, 70000, '2023-03-10', NULL),
(5, 'Eve Davis', 2, 65000, '2023-05-22', 4),
(6, 'Frank Miller', 4, 110000, '2020-09-01', NULL),
(7, 'Grace Wilson', 4, 105000, '2021-02-18', 6),
(8, 'Henry Moore', 4, 95000, '2022-08-12', 6),
(9, 'Ivy Taylor', 3, 60000, '2023-01-05', NULL),
(10, 'Jack Anderson', 1, 78000, '2023-07-19', 1),
(11, 'Karen Thomas', 4, 98000, '2023-09-01', 6),
(12, 'Leo Martin', 2, 68000, '2024-01-15', 4),
(13, 'Mia Jackson', NULL, 55000, '2024-02-20', 1); -- Employee with no department

INSERT INTO Assignments (emp_id, proj_id) VALUES
(1, 'P101'), (1, 'P102'),
(2, 'P101'),
(3, 'P102'),
(4, 'P104'),
(5, 'P104'),
(6, 'P103'), (6, 'P105'),
(7, 'P103'),
(8, 'P105'),
(10, 'P101'),
(11, 'P105'),
(12, 'P104');
