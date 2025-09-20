import sqlite3

def create_tables():
    # Connect to database
    conn = sqlite3.connect('ramp_database.db')
    cursor = conn.cursor()
    
    # Create fred_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fred_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            date DATE NOT NULL,
            value REAL
        )
    ''')
    
    # Create stock_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, 
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER NOT NULL
        )
    ''')
    
    # Create bls_data table  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bls_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            date DATE NOT NULL,
            value REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Tables created successfully!")

def insert_fred_data(series_id, date, value):
    conn = sqlite3.connect('ramp_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO fred_data (series_id, date, value)
        VALUES (?, ?, ?)
    ''', (series_id, date, value))
    
    conn.commit()
    conn.close()

def insert_stock_data(symbol, date, open, high, low, close, volume):
    conn = sqlite3.connect('ramp_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO stock_data (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, date, open, high, low, close, volume))
    
    conn.commit()
    conn.close()

def insert_bls_data(series_id, date, value):
    conn = sqlite3.connect('ramp_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO bls_data (series_id, date, value)
        VALUES (?, ?, ?)
    ''', (series_id, date, value))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()