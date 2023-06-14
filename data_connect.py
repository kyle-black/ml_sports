import pandas as pd
from sqlalchemy import create_engine

# Connect to SQLite database (or create a new one if it doesn't exist)
database_path = 'sqlite:///my_database.db'
engine = create_engine(database_path)

# Read CSV file using pandas
csv_file_path = 'your_data.csv'
data = pd.read_csv(csv_file_path)

# Create a new table (if it doesn't exist) and insert data into the SQLite database
table_name = 'my_table'
data.to_sql(table_name, engine, if_exists='replace', index=False)

# Read data from the SQLite database
query = f'SELECT * FROM {table_name}'
database_data = pd.read_sql_query(query, engine)

print(database_data)
