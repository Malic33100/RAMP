# main_collector.py
"""
Master Data Collector Script - Runs all our API scripts
This script calls each of our data collection scripts in order
"""

import time
from datetime import datetime

# Import our data collection scripts
from data_sources import fred_api
from data_sources import yahoo_api
from data_sources import bls_api


def main():
    """
    Main function that runs all our data collectors
    """
    # Print a nice header so we know it started
    print("\n" + "="*50)
    print("  STARTING DATA COLLECTION")
    print("  Time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*50 + "\n")
    
    # Keep track of what worked and what didn't
    results = {
        "FRED": False,
        "Yahoo": False,
        "BLS": False
    }
    
    # 1. Run FRED data collection
    print("\n--- Collecting FRED Data ---")
    try:
        fred_api.main()
        results["FRED"] = True
        print("✓ FRED data collected successfully")
    except Exception as e:
        print(f"✗ FRED failed: {e}")
    
    # Wait a bit between API calls (be nice to the servers)
    time.sleep(2)
    
    # 2. Run Yahoo Finance data collection
    print("\n--- Collecting Yahoo Finance Data ---")
    try:
        yahoo_api.main()
        results["Yahoo"] = True
        print("✓ Yahoo data collected successfully")
    except Exception as e:
        print(f"✗ Yahoo failed: {e}")
    
    time.sleep(2)
    
    # 3. Run BLS data collection
    print("\n--- Collecting BLS Data ---")
    try:
        bls_api.main()
        results["BLS"] = True
        print("✓ BLS data collected successfully")
    except Exception as e:
        print(f"✗ BLS failed: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("  COLLECTION COMPLETE")
    print("="*50)
    
    # Count successes
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"  Successful: {successful}/{total}")
    
    # Show what worked and what didn't
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print("  Time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*50 + "\n")
    
    # If everything worked, return 0 (success), otherwise 1 (had some failures)
    if successful == total:
        print("All data collected successfully!")
        return 0
    else:
        print(f"Warning: {total - successful} data source(s) failed")
        return 1


if __name__ == "__main__":
    # Run the main function
    exit_code = main()
    exit(exit_code)