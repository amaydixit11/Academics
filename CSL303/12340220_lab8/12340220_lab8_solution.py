import psycopg2

def create_connection():
    return psycopg2.connect(
        host="localhost",
        database="studentdb",
        user="postgres",
        password="password",
        port="5432"
    )

def create_table():
    conn = create_connection()
    cur = conn.cursor()

    table_name = input("Enter table name: ")
    columns = input("Enter columns (e.g. id INT PRIMARY KEY, name VARCHAR(50), age INT, dept VARCHAR(20)): ")

    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
    cur.execute(query)
    conn.commit()

    print(f"Table {table_name} created successfully")

    cur.close()
    conn.close()

def insert_students():
    conn = create_connection()
    cur = conn.cursor()

    while True:
        student_id = input("Enter Student ID: ")
        name = input("Enter Student Name: ")
        age = input("Enter Age: ")
        dept = input("Enter Department: ")

        cur.execute("INSERT INTO students (id, name, age, dept) VALUES (%s, %s, %s, %s)",
                    (student_id, name, age, dept))
        conn.commit()

        ch = input("Add another student? (y/n): ")
        if ch.lower() != 'y':
            break

    print("Data inserted successfully")

    cur.close()
    conn.close()

def update_student():
    conn = create_connection()
    cur = conn.cursor()

    name = input("Enter the student name to update department: ")
    new_dept = input("Enter new department: ")

    cur.execute("UPDATE students SET dept=%s WHERE name=%s", (new_dept, name))
    conn.commit()

    print("Student updated")

    cur.close()
    conn.close()

def delete_student():
    conn = create_connection()
    cur = conn.cursor()

    student_id = input("Enter Student ID to delete: ")
    cur.execute("DELETE FROM students WHERE ID=%s", (student_id,))
    conn.commit()

    print("🗑️ Student deleted")

    cur.close()
    conn.close()

def query_data():
    conn = create_connection()
    cur = conn.cursor()

    print("\nQuery Options:")
    print("1. Show all students")
    print("2. Show students by department")
    print("3. Average age by department")
    print("4. Students name starts with letter")

    ch = input("Enter your choice: ")

    if ch == '1':
        cur.execute("SELECT * FROM students;")
        print(cur.fetchall())

    elif ch == '2':
        dept = input("Enter department: ")
        cur.execute("SELECT * FROM students WHERE dept=%s", (dept,))
        print(cur.fetchall())

    elif ch == '3':
        cur.execute("SELECT dept, AVG(age) FROM students GROUP BY dept;")
        print(cur.fetchall())

    elif ch == '4':
        letter = input("Enter starting letter: ")
        cur.execute("SELECT * FROM students WHERE name LIKE %s", (letter + '%',))
        print(cur.fetchall())

    conn.commit()
    cur.close()
    conn.close()

def main():
    while True:
        print("\n===== STUDENT DB MENU =====")
        print("1. Create Table")
        print("2. Insert Student")
        print("3. Update Student")
        print("4. Delete Student")
        print("5. Query Data")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            create_table()
        elif choice == '2':
            insert_students()
        elif choice == '3':
            update_student()
        elif choice == '4':
            delete_student()
        elif choice == '5':
            query_data()
        elif choice == '6':
            print("Exiting... Closing DB Connection.")
            break
        else:
            print("Invalid choice Try again.")

if __name__ == "__main__":
    main()
