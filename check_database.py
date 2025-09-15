import sqlite3

# Connect to your database
conn = sqlite3.connect('ramp_database.db')
cursor = conn.cursor()

# Check how many rows are in stock_data
cursor.execute("SELECT COUNT(*) FROM stock_data")
count = cursor.fetchone()[0]
print(f"Total rows in stock_data: {count}")

# Look at first 5 rows
cursor.execute("SELECT * FROM stock_data")
rows = cursor.fetchall()

print("\nFirst 5 rows:")
for row in rows:
    print(row)

# Check which stocks you have
cursor.execute("SELECT DISTINCT symbol FROM stock_data")
symbols = cursor.fetchall()
print(f"\nStocks in database: {[s[0] for s in symbols]}")

# Check date range
cursor.execute("SELECT MIN(date), MAX(date) FROM stock_data")
dates = cursor.fetchone()
print(f"\nDate range: {dates[0]} to {dates[1]}")

conn.close()