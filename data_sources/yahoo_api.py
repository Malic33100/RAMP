import yfinance as yf
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append('.')  # Add current directory to path
from src.data.database import insert_stock_data

# Create directories if they don't exist
os.makedirs('data/raw/yahoo', exist_ok=True)
os.makedirs('data/processed/yahoo', exist_ok=True)

def get_stock_data(ticker, period='2y'):
    """
    Get stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period (str): Period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        pandas.DataFrame: Stock data with OHLCV information
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data
        data = stock.history(period=period)
        
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return None
        
        # Save raw data to CSV (overwrites existing)
        raw_filename = f"data/raw/yahoo/{ticker}_raw.csv"
        data.to_csv(raw_filename)
        print(f"Raw data saved to {raw_filename}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_stock_info(ticker):
    """
    Get basic company information and save raw info
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Save raw info as JSON (overwrites existing)
        raw_info_filename = f"data/raw/yahoo/{ticker}_info_raw.json"
        with open(raw_info_filename, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        print(f"Raw company info saved to {raw_info_filename}")
        
        # Extract key information
        company_info = {
            'symbol': info.get('symbol', 'N/A'),
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'current_price': info.get('currentPrice', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
        
        return company_info
        
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return None

def process_stock_data(data, ticker):
    """
    Process and save stock data to processed folder
    
    Args:
        data (pandas.DataFrame): Stock data from yfinance
        ticker (str): Stock ticker for filename
    """
    if data is None or data.empty:
        print(f"No data to process for {ticker}")
        return
    
    # Display summary statistics
    print(f"\n=== {ticker} Stock Data Summary ===")
    print(f"Date Range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Number of trading days: {len(data)}")
    print(f"\nPrice Summary:")
    print(f"Current Close: ${data['Close'].iloc[-1]:.2f}")
    print(f"52-week High: ${data['High'].max():.2f}")
    print(f"52-week Low: ${data['Low'].min():.2f}")
    print(f"Average Volume: {data['Volume'].mean():,.0f}")
    
    # Calculate some basic metrics
    processed_data = data.copy()
    processed_data['Daily_Return'] = processed_data['Close'].pct_change() * 100
    processed_data['50_Day_MA'] = processed_data['Close'].rolling(window=50).mean()
    processed_data['200_Day_MA'] = processed_data['Close'].rolling(window=200).mean()
    
    # Calculate performance metrics
    total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
    avg_daily_return = processed_data['Daily_Return'].mean()
    volatility = processed_data['Daily_Return'].std()
    
    print(f"\nPerformance Metrics:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Average Daily Return: {avg_daily_return:.2f}%")
    print(f"Volatility (Daily): {volatility:.2f}%")
    
    # Save processed data to CSV (overwrites existing)
    processed_filename = f"data/processed/yahoo/{ticker}_processed.csv"
    processed_data.to_csv(processed_filename)
    print(f"Processed data saved to {processed_filename}")
    
    # Save processed data to CSV (overwrites existing)
    processed_filename = f"data/processed/yahoo/{ticker}_processed.csv"
    processed_data.to_csv(processed_filename)
    print(f"Processed data saved to {processed_filename}")
    
    # NEW: Save to database
    print(f"Saving {ticker} to database...")
    for date, row in processed_data.iterrows():
        try:
            insert_stock_data(
                symbol=ticker,
                date=date.strftime('%Y-%m-%d'),  # Convert date to string
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=int(row['Volume'])  # Convert to integer
            )
        except Exception as e:
            print(f"Error inserting {ticker} data for {date}: {e}")
    print(f"✓ {ticker} saved to database")

    # Save summary to text file (overwrites existing)
    summary_filename = f"data/processed/yahoo/{ticker}_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"{ticker} Stock Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Date Range: {data.index.min().date()} to {data.index.max().date()}\n")
        f.write(f"Number of trading days: {len(data)}\n\n")
        f.write(f"Current Close: ${data['Close'].iloc[-1]:.2f}\n")
        f.write(f"52-week High: ${data['High'].max():.2f}\n")
        f.write(f"52-week Low: ${data['Low'].min():.2f}\n")
        f.write(f"Average Volume: {data['Volume'].mean():,.0f}\n\n")
        f.write(f"Total Return: {total_return:.2f}%\n")
        f.write(f"Average Daily Return: {avg_daily_return:.2f}%\n")
        f.write(f"Volatility (Daily): {volatility:.2f}%\n")
    
    print(f"Summary saved to {summary_filename}")

def get_market_indices():
    """
    Get data for major market indices
    
    Returns:
        dict: Dictionary with index data
    """
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^RUT': 'Russell 2000'
    }
    
    index_data = {}
    
    for ticker, name in indices.items():
        print(f"Fetching {name} data...")
        data = get_stock_data(ticker, period='1y')
        if data is not None:
            # Process index data
            process_stock_data(data, ticker.replace('^', 'INDEX_'))
            
            index_data[ticker] = {
                'name': name,
                'data': data,
                'current_price': data['Close'].iloc[-1],
                'ytd_return': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            }
    
    return index_data

def main():
    """Main function to run Yahoo Finance data collection"""
    # S&p 100
    stock_tickers = [
    'AAPL',   # Apple - Technology
    'MSFT',   # Microsoft - Technology
    'AMZN',   # Amazon - Consumer Discretionary
    'NVDA',   # NVIDIA - Technology
    'GOOGL',  # Alphabet Class A - Communication Services
    'META',   # Meta Platforms - Communication Services
    'GOOG',   # Alphabet Class C - Communication Services
    'TSLA',   # Tesla - Consumer Discretionary
    'BRK.B',  # Berkshire Hathaway - Financials
    'JPM',    # JPMorgan Chase - Financials
    'JNJ',    # Johnson & Johnson - Healthcare
    'V',      # Visa - Financials
    'UNH',    # UnitedHealth - Healthcare
    'XOM',    # Exxon Mobil - Energy
    'WMT',    # Walmart - Consumer Staples
    'MA',     # Mastercard - Financials
    'PG',     # Procter & Gamble - Consumer Staples
    'HD',     # Home Depot - Consumer Discretionary
    'CVX',    # Chevron - Energy
    'ABBV',   # AbbVie - Healthcare
    'MRK',    # Merck - Healthcare
    'LLY',    # Eli Lilly - Healthcare
    'PEP',    # PepsiCo - Consumer Staples
    'AVGO',   # Broadcom - Technology
    'KO',     # Coca-Cola - Consumer Staples
    'PFE',    # Pfizer - Healthcare
    'ORCL',   # Oracle - Technology
    'COST',   # Costco - Consumer Staples
    'TMO',    # Thermo Fisher Scientific - Healthcare
    'BAC',    # Bank of America - Financials
    'ACN',    # Accenture - Technology
    'MCD',    # McDonald's - Consumer Discretionary
    'CSCO',   # Cisco - Technology
    'ABT',    # Abbott Laboratories - Healthcare
    'WFC',    # Wells Fargo - Financials
    'CRM',    # Salesforce - Technology
    'ADBE',   # Adobe - Technology
    'DIS',    # Disney - Communication Services
    'VZ',     # Verizon - Communication Services
    'TXN',    # Texas Instruments - Technology
    'DHR',    # Danaher - Healthcare
    'NEE',    # NextEra Energy - Utilities
    'BMY',    # Bristol-Myers Squibb - Healthcare
    'PM',     # Philip Morris - Consumer Staples
    'CMCSA',  # Comcast - Communication Services
    'RTX',    # Raytheon - Industrials
    'NFLX',   # Netflix - Communication Services
    'UPS',    # United Parcel Service - Industrials
    'T',      # AT&T - Communication Services
    'AMD',    # AMD - Technology
    'INTC',   # Intel - Technology
    'NKE',    # Nike - Consumer Discretionary
    'HON',    # Honeywell - Industrials
    'QCOM',   # Qualcomm - Technology
    'MS',     # Morgan Stanley - Financials
    'COP',    # ConocoPhillips - Energy
    'UNP',    # Union Pacific - Industrials
    'LIN',    # Linde - Materials
    'AMGN',   # Amgen - Healthcare
    'LOW',    # Lowe's - Consumer Discretionary
    'IBM',    # IBM - Technology
    'BA',     # Boeing - Industrials
    'GS',     # Goldman Sachs - Financials
    'CAT',    # Caterpillar - Industrials
    'SBUX',   # Starbucks - Consumer Discretionary
    'DE',     # Deere & Company - Industrials
    'ELV',    # Elevance Health - Healthcare
    'PLD',    # Prologis - Real Estate
    'GE',     # General Electric - Industrials
    'INTU',   # Intuit - Technology
    'SPGI',   # S&P Global - Financials
    'AMAT',   # Applied Materials - Technology
    'MDLZ',   # Mondelez - Consumer Staples
    'BLK',    # BlackRock - Financials
    'ISRG',   # Intuitive Surgical - Healthcare
    'SYK',    # Stryker - Healthcare
    'ADP',    # ADP - Technology
    'CVS',    # CVS Health - Healthcare
    'GILD',   # Gilead Sciences - Healthcare
    'TJX',    # TJX Companies - Consumer Discretionary
    'BKNG',   # Booking Holdings - Consumer Discretionary
    'ADI',    # Analog Devices - Technology
    'VRTX',   # Vertex Pharmaceuticals - Healthcare
    'AXP',    # American Express - Financials
    'C',      # Citigroup - Financials
    'REGN',   # Regeneron - Healthcare
    'NOW',    # ServiceNow - Technology
    'MO',     # Altria - Consumer Staples
    'TMUS',   # T-Mobile - Communication Services
    'LRCX',   # Lam Research - Technology
    'SCHW',   # Charles Schwab - Financials
    'MMC',    # Marsh & McLennan - Financials
    'CB',     # Chubb - Financials
    'ZTS',    # Zoetis - Healthcare
    'CI',     # Cigna - Healthcare
    'ETN',    # Eaton - Industrials
    'BDX',    # Becton Dickinson - Healthcare
    'SO',     # Southern Company - Utilities
    'FISV',   # Fiserv - Technology
    'EQIX'    # Equinix - Real Estate
]
    
    print("Fetching Yahoo Finance stock data...")
    print("Files will be saved to data/raw/yahoo/ and data/processed/yahoo/")
    print("=" * 50)
    
    # Process individual stocks
    for ticker in stock_tickers:
        print(f"\nProcessing {ticker}...")
        
        # Get company info
        info = get_stock_info(ticker)
        if info:
            print(f"Company: {info['name']}")
            print(f"Sector: {info['sector']}")
        
        # Get and process stock data
        stock_data = get_stock_data(ticker, period='2y')
        if stock_data is not None:
            process_stock_data(stock_data, ticker)
    
    print("\n" + "=" * 50)
    print("Market Indices Summary:")
    
    # Get market indices data
    indices_data = get_market_indices()
    
    for ticker, data in indices_data.items():
        print(f"\n{data['name']} ({ticker}):")
        print(f"Current Level: {data['current_price']:.2f}")
        print(f"YTD Return: {data['ytd_return']:.2f}%")
    
    print("\nYahoo Finance data collection complete!")
    print("Check data/raw/yahoo/ for raw data and data/processed/yahoo/ for analysis-ready files")

# Main execution
if __name__ == "__main__":
    main()