# data_sources/bls_api.py
import requests
import json
import pandas as pd
import os
from dotenv import load_dotenv
import sys
sys.path.append('.')
from src.data.database import insert_bls_data

def main():
    """
    Main function that collects BLS employment data
    """
    print("BLS Script Starting...")
    # Load environment variables
    load_dotenv()
    
    # Make these global so other functions can use them
    global BLS_API_KEY
    global BLS_BASE_URL
    
    BLS_API_KEY = os.getenv('BLS_API_KEY', '')
    print(f"API Key loaded: {'Yes' if BLS_API_KEY else 'No'}")  # Add this
    print(f"Key length: {len(BLS_API_KEY)}")
    BLS_BASE_URL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    
    # Create directories if they don't exist
    os.makedirs('data/raw/bls', exist_ok=True)
    os.makedirs('data/processed/bls', exist_ok=True)
    
    def get_bls_data(series_ids, start_year=2020, end_year=2024):
        """
        Get employment data from BLS API
        
        Args:
            series_ids (list): List of BLS series IDs
            start_year (int): Start year
            end_year (int): End year
        
        Returns:
            dict: JSON response from BLS API
        """
        
        # BLS uses POST request with JSON body (different from FRED!)
        headers = {'Content-type': 'application/json'}
        
        # Build the request data
        data = json.dumps({
            "seriesid": series_ids,  # List of all series we want
            "startyear": str(start_year),
            "endyear": str(end_year),
            "registrationkey": BLS_API_KEY
        })
        
        # Make the API request (POST, not GET like FRED)
        response = requests.post(BLS_BASE_URL, data=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    
    def process_bls_data(data):
        """
        Process BLS API response and save to processed folder
        
        Args:
            data (dict): JSON response from BLS API containing ALL series
        """
        
        if not data or 'Results' not in data:
            print("No data found in BLS response")
            return
        
        # Map series IDs to friendly names
        series_names = {
            'LNS14000000': 'Unemployment_Rate',
            'CES0000000001': 'Nonfarm_Payrolls',
            'CES0500000003': 'Avg_Hourly_Earnings',
            'LNS11300000': 'Labor_Force_Participation'
        }
        
        # BLS returns data for ALL series in one response
        # Loop through each series in the results
        for series in data['Results']['series']:
            series_id = series['seriesID']
            series_name = series_names.get(series_id, series_id)
            
            print(f"\n=== {series_name} ===")
            print(f"{'Date':<12} {'Value':<15}")
            print("-" * 30)
            
            valid_data = []
            
            # BLS data structure: each series has a 'data' array
            for item in series['data']:
                year = item['year']
                period = item['period']  # M01 = January, M02 = February, etc.
                value = item['value']
                
                # Convert period to month number (M01 -> 01)
                if period.startswith('M'):
                    month = period[1:]
                    date_str = f"{year}-{month}-01"
                    
                    print(f"{date_str:<12} {value:<15}")
                    valid_data.append({
                        'date': date_str,
                        'value': float(value)
                    })
            
            # Save to CSV
            if valid_data:
                df = pd.DataFrame(valid_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                csv_filename = f"data/processed/bls/{series_name}.csv"
                df.to_csv(csv_filename, index=False)
                print(f"Saved to {csv_filename}")

                # Insert data into database
                print(f"Saving {series_name} to database...")
                for _, row in df.iterrows():
                    try:
                        insert_bls_data(
                            series_id=series_id,
                            date=row['date'].strftime('%Y-%m-%d'),
                            value=row['value']
                        )
                    except Exception as e:
                        print(f"Error inserting: {e}")
                print(f"✓ {series_name} saved to database")
    
    # BLS economic indicators for RAMP
    bls_series = [
        'LNS14000000',    # Unemployment Rate
        'CES0000000001',  # Total Nonfarm Payrolls  
        'CES0500000003',  # Average Hourly Earnings
        'LNS11300000'     # Labor Force Participation Rate
    ]
    
    print("Fetching BLS employment data...")
    print("Files will be saved to data/raw/bls/ and data/processed/bls/")
    
    # Get ALL series in one API call (efficient!)
    data = get_bls_data(bls_series)
    
    if data:
        # Save raw JSON response
        with open('data/raw/bls/bls_raw.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("Raw data saved!")
        
        # Process and save as CSV
        process_bls_data(data)
    else:
        print("Failed to get BLS data")
    
    print("\nBLS data collection complete!")


# Main execution
if __name__ == "__main__":
    main()