import os
import sqlite3
import time
import google.generativeai as genai
from dotenv import load_dotenv
import gradio as gr
import json

# Load environment variables from a .env file
load_dotenv()

# Fetch API key for GEMINI from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the GemAI API with the fetched API key
genai.configure(api_key=GEMINI_API_KEY)

# Connect to the SQLite database and create a cursor
con = sqlite3.connect("database.db", check_same_thread=False)
cur = con.cursor()

# Database schema prompt
db_schema_prompt = '''
You are provided with a database schema that contains multiple tables, each with specific columns and properties. Here are the details of the tables:

Table: departments
Columns:
dept_no: type char(4), primary key
dept_name: type varchar(40)

Table: dept_emp
Columns:
emp_no: type INTEGER, primary key
dept_no: type char(4), primary key, foreign key referencing departments(dept_no)
from_date: type date
to_date: type date

Table: dept_manager
Columns:
dept_no: type char(4), primary key, foreign key referencing departments(dept_no)
emp_no: type INTEGER, primary key, foreign key referencing employees(emp_no)
from_date: type date
to_date: type date

Table: employees
Columns:
emp_no: type INTEGER, primary key
birth_date: type date
first_name: type varchar(14)
last_name: type varchar(16)
gender: type TEXT
hire_date: type date

Table: salaries
Columns:
emp_no: type INTEGER, primary key, foreign key referencing employees(emp_no)
salary: type INTEGER
from_date: type date, primary key
to_date: type date

Table: titles
Columns:
emp_no: type INTEGER, primary key, foreign key referencing employees(emp_no)
title: type varchar(50)
from_date: date, primary key
to_date: date, nullable

Generate only a SQL query based on the question. Return the response in this exact format:
{
    "sqlQuery": "YOUR_SQL_QUERY_HERE",
    "description": "BRIEF_DESCRIPTION_OF_QUERY"
}
'''

response_prompt = '''
You are a helpful assistant that generates natural language responses based on database query results.
Take the query results and create a clear, concise, and informative response.
Return the response in this exact format:
{
    "message": "YOUR_RESPONSE_HERE"
}
'''

# Initialize both models with different configurations
sql_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction=db_schema_prompt,
    generation_config={
        "temperature": 0.3,
    }
)

response_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=response_prompt,
    generation_config={
        "temperature": 0.7,
    }
)

# Start chat sessions for both models
sql_chat = sql_model.start_chat(history=[])
response_chat = response_model.start_chat(history=[])

def extract_json_from_response(text):
    """Extract JSON from model response, handling different formats."""
    try:
        # Try direct JSON parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Try to extract JSON from markdown code blocks
            if '```json' in text:
                json_text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                json_text = text.split('```')[1].split('```')[0]
            else:
                json_text = text
            # Clean up the text
            json_text = json_text.strip()
            return json.loads(json_text)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw text: {text}")
            raise

def ask(question):
    """
    Process the question using two different models:
    1. SQL Model (Gemini Pro 1.5) generates the SQL query
    2. Response Model (Gemini Flash 1.5) generates the user-friendly response
    """
    tries = 4
    start_time = time.time()

    for attempt in range(tries):
        try:
            # Step 1: Generate SQL query using Gemini Pro 1.5
            print(f"\nProcessing question: {question}")
            
            sql_result = sql_chat.send_message(
                f"Generate a SQL query for this question: {question}\nRespond only with a JSON object in the specified format."
            )
            print(f"\nSQL Model Response:\n{sql_result.text}")
            
            sql_json = extract_json_from_response(sql_result.text)
            query = sql_json["sqlQuery"]
            print(f"\nExtracted SQL Query: {query}")

            # Execute the generated SQL query
            cur.execute(query)
            data = cur.fetchall()
            dataText = json.dumps(data)
            print(f"\nQuery Results: {dataText}")

            # Step 2: Generate response using Gemini Flash 1.5
            response_result = response_chat.send_message(
                f"The database query returned these results: {dataText}\nCreate a natural language response in the specified JSON format."
            )
            print(f"\nResponse Model Output:\n{response_result.text}")
            
            response_json = extract_json_from_response(response_result.text)
            
            end_time = time.time()
            print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
            
            return response_json["message"], data

        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {str(e)}")
            if attempt < tries - 1:
                print("Retrying...")
                time.sleep(1)  # Add a small delay between retries
            else:
                print("All attempts failed.")
                return f"Sorry, I encountered an error: {str(e)}", None

# Create a Gradio interface
demo = gr.Interface(
    fn=ask,
    inputs=["text"],
    outputs=["text", "dataframe"],
    title="Two-Model Employee Database Chatbot",
    description="Ask questions about employee data using natural language. Powered by Gemini Pro 1.5 (SQL) and Gemini Flash 1.5 (Response).",
    examples=[
        "How many employees are in the company",
        "Who is the highest paid employee and what is his position?",
        "En yaşlı kişi kim?",
        "Şirkette kaç çalışan var?",
        "Do you know a person joined the company before 1998. Can you give me a name."
    ]
)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()