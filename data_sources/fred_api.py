# fred_api.py
import requests
import json
import pandas as pd
import os
from dotenv import load_dotenv


def main():

# Load environment variables
load_dotenv()

# FRED API Configuration  
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
FRED_BASE_URL = 'https://api.stlouisfed.org/fred/'

# Create directories if they don't exist
os.makedirs('data/raw/fred', exist_ok=True)
os.makedirs('data/processed/fred', exist_ok=True)

def get_fred_data(series_id, start_date='2020-01-01', end_date='2024-12-31'):
    """
    Get economic data from FRED API
    
    Args:
        series_id (str): FRED series ID (e.g., 'GDP', 'UNRATE')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        dict: JSON response from FRED API
    """
    
    # Build the API URL
    url = f"{FRED_BASE_URL}series/observations"
    
    # Set up parameters
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Save raw JSON data
        raw_filename = f"data/raw/fred/{series_id}_raw.json"
        with open(raw_filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Raw data saved to {raw_filename}")
        
        return data
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def process_fred_data(data, series_id):
    """
    Process FRED API response and save to processed folder
    
    Args:
        data (dict): JSON response from FRED API
        series_id (str): FRED series ID for filename
    """
    
    if not data or 'observations' not in data:
        print(f"No data found for series {series_id}")
        return
    
    observations = data['observations']
    
    # Create a simple table format
    print(f"\n=== {series_id} Data ===")
    print(f"{'Date':<12} {'Value':<15}")
    print("-" * 30)
    
    # Prepare data for saving
    output_lines = [f"{series_id} Economic Data\n"]
    output_lines.append(f"{'Date':<12} {'Value':<15}\n")
    output_lines.append("-" * 30 + "\n")
    
    # Also create DataFrame for CSV export
    valid_data = []
    
    for obs in observations:
        date = obs['date']
        value = obs['value']
        
        # Skip missing values (marked as '.')
        if value != '.':
            print(f"{date:<12} {value:<15}")
            output_lines.append(f"{date:<12} {value:<15}\n")
            valid_data.append({'date': date, 'value': float(value)})
    
    # Save processed text file (overwrites existing)
    processed_txt_filename = f"data/processed/fred/{series_id}_data.txt"
    with open(processed_txt_filename, 'w') as f:
        f.writelines(output_lines)
    print(f"Processed data saved to {processed_txt_filename}")
    
    # Save as CSV for easier analysis (overwrites existing)
    if valid_data:
        df = pd.DataFrame(valid_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        processed_csv_filename = f"data/processed/fred/{series_id}_data.csv"
        df.to_csv(processed_csv_filename, index=False)
        print(f"CSV data saved to {processed_csv_filename}")

# Main execution
if __name__ == "__main__":
    # Key FRED economic indicators to test with
    fred_series = [
        'GDP',        # Gross Domestic Product
        'UNRATE',     # Unemployment Rate
        'CPIAUCSL',   # Consumer Price Index
        'FEDFUNDS',   # Federal Funds Rate
        'PAYEMS'      # Nonfarm Payrolls
    ]
    
    print("Fetching FRED economic data...")
    print("Files will be saved to data/raw/fred/ and data/processed/fred/")
    
    for series in fred_series:
        print(f"\nProcessing {series}...")
        data = get_fred_data(series)
        if data:
            process_fred_data(data, series)
        else:
            print(f"Failed to get data for {series}")
    
    print("\nFRED data collection complete!")