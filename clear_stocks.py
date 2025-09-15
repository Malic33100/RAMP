import sqlite3
conn = sqlite3.connect('ramp_database.db')
cursor = conn.cursor()
cursor.execute("DELETE FROM stock_data")
conn.commit()
print("Cleared stock_data table - ready for S&P 100!")
conn.close()