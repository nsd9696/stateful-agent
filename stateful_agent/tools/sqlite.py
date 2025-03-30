import os
import sqlite3

from dotenv import load_dotenv
from hyperpocket.tool import function_tool

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)


@function_tool
def insert_user_data(
    user_name: str,
    user_date_of_birth: str,
    user_age: int,
    user_email: str,
    user_phone_number: str,
    user_address: str,
    user_city: str,
    user_state: str,
    user_zip_code: str,
    user_country: str,
):
    """
    Insert user data into the database.
    """

    user_name = user_name.lower()
    conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH"))
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS users
             (user_name text PRIMARY KEY, user_date_of_birth text, user_age integer, user_email text, user_phone_number text, user_address text, user_city text, user_state text, user_zip_code text, user_country text)"""
    )
    try:
        cursor.execute(
            "INSERT INTO users (user_name, user_date_of_birth, user_age, user_email, user_phone_number, user_address, user_city, user_state, user_zip_code, user_country) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_name,
                user_date_of_birth,
                user_age,
                user_email,
                user_phone_number,
                user_address,
                user_city,
                user_state,
                user_zip_code,
                user_country,
            ),
        )
        conn.commit()
        conn.close()
        return "User data inserted successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return f"User '{user_name}' already exists"


@function_tool
def get_user_data(user_name: str):
    """
    Get user data from the database.
    """
    user_name = user_name.lower()
    conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH"))
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_name = ?", (user_name,))
    result = cursor.fetchall()
    conn.commit()
    conn.close()
    return result
