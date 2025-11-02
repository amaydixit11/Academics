# Student Database Management Program

This project is part of Lab Sheet 8 for Programming Language with DBMS.  
The objective is to connect a programming language with PostgreSQL running in Docker and perform DB operations such as table creation, data manipulation, and queries through a menu-driven Python program.

---

## 📌 Features

✔ Connects to PostgreSQL using psycopg2  
✔ Creates table dynamically using user input (DDL)  
✔ Inserts, updates, and deletes student records (DML)  
✔ Menu-based user interaction  
✔ Query operations include:
- Show all students
- Show by department
- Show average age by department (GROUP BY)
- Pattern matching using LIKE

✔ Proper DB connection closing and error handling

---

## 🛠️ Requirements

- Python 3.x installed
- Docker installed on the system
- PostgreSQL running in Docker container

Install Python dependency:
```bash
pip install -r requirements.txt
````

---

## 🐳 Docker Setup

1️⃣ Pull PostgreSQL image

```bash
docker pull postgres
```

2️⃣ Run PostgreSQL container

```bash
docker run --name pg_lab -e POSTGRES_PASSWORD=admin123 -p 5432:5432 -d postgres
```

3️⃣ Create database `studentdb`

```bash
docker exec -it pg_lab psql -U postgres
CREATE DATABASE studentdb;
\q
```

---

## ▶️ Execution Steps

1️⃣ Navigate to the project directory

```bash
cd <project-folder>
```

2️⃣ Run the Python program

```bash
python main.py
```

3️⃣ Use the menu options displayed on screen:

| Option | Action                           |
| ------ | -------------------------------- |
| 1      | Create a table                   |
| 2      | Insert student data              |
| 3      | Update a student’s department    |
| 4      | Delete a student record          |
| 5      | Perform queries and view results |
| 6      | Exit the program                 |

Example table creation input:

```
Table name: Students
Columns: id INT PRIMARY KEY, name VARCHAR(50), age INT, dept VARCHAR(20)
```

---

## 📂 Project Structure

```
├── main.py
├── requirements.txt
└── README.md
```

---

## ✅ Program Completion

Before exiting, ensure:
✅ All operations tested successfully
✅ Database connection closes properly
✅ Container remains running until lab/testing finishes

---

## 📝 Notes

* Make sure port **5432** is free before running PostgreSQL
* If container stops, restart using:

```bash
docker start pg_lab
```

---

## ✨ Author

**Name:** Amay
**Course:** Programming Language with DBMS Lab
**Institute:** IIT Bhilai