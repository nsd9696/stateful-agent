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


@function_tool
def create_lab(
    lab_name: str,
    institution: str,
    leader: str,
    members: list,
    research_areas: list,
    website: str = "",
    description: str = "",
):
    """
    Create a new research lab in the database with its information.
    """
    lab_name = lab_name.lower()
    conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH"))
    cursor = conn.cursor()

    # Create labs table if it doesn't exist
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS labs
             (lab_name text PRIMARY KEY, institution text, leader text, website text, description text)"""
    )

    # Create lab_members table if it doesn't exist
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS lab_members
             (lab_name text, member_name text, scholar_url text, PRIMARY KEY (lab_name, member_name))"""
    )

    # Create lab_research_areas table if it doesn't exist
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS lab_research_areas
             (lab_name text, research_area text, PRIMARY KEY (lab_name, research_area))"""
    )

    try:
        # Insert lab basic information
        cursor.execute(
            "INSERT INTO labs (lab_name, institution, leader, website, description) VALUES (?, ?, ?, ?, ?)",
            (lab_name, institution, leader, website, description),
        )

        # Insert lab members
        for member in members:
            name = member.get("name", "")
            scholar_url = member.get("scholar_url", "")
            cursor.execute(
                "INSERT INTO lab_members (lab_name, member_name, scholar_url) VALUES (?, ?, ?)",
                (lab_name, name, scholar_url),
            )

        # Insert research areas
        for area in research_areas:
            cursor.execute(
                "INSERT INTO lab_research_areas (lab_name, research_area) VALUES (?, ?)",
                (lab_name, area),
            )

        conn.commit()
        conn.close()
        return f"Lab '{lab_name}' created successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return f"Lab '{lab_name}' already exists"


@function_tool
def get_lab_info(lab_name: str):
    """
    Get lab information including its members and research areas.
    """
    lab_name = lab_name.lower()
    conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH"))
    cursor = conn.cursor()

    # Get lab basic information
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"

    # Get lab members
    cursor.execute(
        "SELECT member_name, scholar_url FROM lab_members WHERE lab_name = ?",
        (lab_name,),
    )
    members = cursor.fetchall()

    # Get research areas
    cursor.execute(
        "SELECT research_area FROM lab_research_areas WHERE lab_name = ?", (lab_name,)
    )
    research_areas = cursor.fetchall()

    conn.close()

    return {"lab_info": lab_info, "members": members, "research_areas": research_areas}


@function_tool
def add_lab_member(lab_name: str, member_name: str, scholar_url: str):
    """
    Add a new member to an existing lab.
    """
    lab_name = lab_name.lower()
    conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH"))
    cursor = conn.cursor()

    # Check if lab exists
    cursor.execute("SELECT * FROM labs WHERE lab_name = ?", (lab_name,))
    lab_info = cursor.fetchone()

    if not lab_info:
        conn.close()
        return f"Lab '{lab_name}' does not exist"

    try:
        cursor.execute(
            "INSERT INTO lab_members (lab_name, member_name, scholar_url) VALUES (?, ?, ?)",
            (lab_name, member_name, scholar_url),
        )
        conn.commit()
        conn.close()
        return f"Member '{member_name}' added to lab '{lab_name}' successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return f"Member '{member_name}' already exists in lab '{lab_name}'"


@function_tool
def get_all_labs():
    """
    Get a list of all labs in the database.
    """
    conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH"))
    cursor = conn.cursor()
    cursor.execute("SELECT lab_name FROM labs")
    labs = cursor.fetchall()
    conn.close()
    return labs
