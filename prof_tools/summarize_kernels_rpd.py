import sqlite3
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process rpd trace')
parser.add_argument('rpd', action='store', type=str, help='name of rpd trace file.')

args = parser.parse_args()

conn = sqlite3.connect(args.rpd)
# Execute SQL query to get the table names
df_top = pd.read_sql_query("SELECT * from top", conn)
conn.close()
print(df_top)
df_top.to_csv(args.rpd+".csv")
