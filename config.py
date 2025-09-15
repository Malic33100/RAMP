# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# BLS API Configuration
BLS_API_KEY = os.getenv('BLS_API_KEY', '')
BLS_API_URL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
BLS_HEADERS = {'Content-type': 'application/json'}

# FRED API Configuration  
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
FRED_BASE_URL = 'https://api.stlouisfed.org/fred/'

# Economic indicators list
ECONOMIC_INDICATORS = [
    'LNS14000000',     # Unemployment Rate
    'CES0000000001',   # Total Employment
    'CES0500000003',   # Average Hourly Earnings
    'CUUR0000SA0',     # CPI All Items
    'CUUR0000SA0L1E',  # Core CPI
    'PRS85006092'      # Productivity 
]