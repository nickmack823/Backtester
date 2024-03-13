import os
import sys
from typing import Dict, List
import pandas as pd
from datetime import datetime, timedelta
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

from utilities import save_pickle, load_pickle, logger, load_settings

# Today (duh)
today = datetime.now()

# Set the number of data points to fetch for each timeframe
num_datapoints = 5000 
def num_datapoints_to_days(timeframe: str) -> int:
    # One day is one data point for the daily timeframe; 1 hour is one data point for the hourly timeframe, etc.
    conversion = {
        "D": 1,
        "H4": 6,
        "H2": 12,
        "H1": 24,
        "M30": 48,
        "M20": 72,
        "M15": 96,
        "M5": 288,
    }

    today = datetime.today()

    days_needed = round(num_datapoints / conversion[timeframe])
    
    while True:
        end_date = today - timedelta(days=days_needed)
        num_weekdays = len([d for d in pd.date_range(end_date, today, freq='B')])

        if num_weekdays * conversion[timeframe] >= num_datapoints:
            break
        else:
            days_needed += 2

    return days_needed

# timeframe_ranges = {
#     timeframe: (today - timedelta(days=num_datapoints_to_days(timeframe))).strftime("%Y-%m-%d") for timeframe in timeframes
# }

timeframe_ranges = {
    "M5": (today - timedelta(days=33)).strftime("%Y-%m-%d"),
    "M15": (today - timedelta(days=69)).strftime("%Y-%m-%d"),
    "M30": (today - timedelta(days=138)).strftime("%Y-%m-%d"),
    "H1": (today - timedelta(days=275)).strftime("%Y-%m-%d"),
    "H4": (today - timedelta(days=1500)).strftime("%Y-%m-%d"),
    "D": (today - timedelta(days=3000)).strftime("%Y-%m-%d"),
}

# Start dates of each timeframe (for single-test runs for verifying logic)
timeframe_ranges_single = {
    "M5": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
    "M15": (today - timedelta(days=60)).strftime("%Y-%m-%d"),
    "M30": (today - timedelta(days=120)).strftime("%Y-%m-%d"),
    "H1": (today - timedelta(days=240)).strftime("%Y-%m-%d"),
    "H4": (today - timedelta(days=480)).strftime("%Y-%m-%d"),
    "D": (today - timedelta(days=960)).strftime("%Y-%m-%d"),
}

# Convert granularity to number of days for timedelta
granularity_in_days = {
    "S5": 5 / 86400,
    "M1": 1 / 24 / 60,
    "M5": 5 / 24 / 60,
    "M15": 15 / 24 / 60,
    "M20": 20 / 24 / 60,
    "M30": 30 / 24 / 60,
    "H1": 1 / 24,
    "H2": 2 / 24,
    "H4": 4 / 24,
    "D": 1,
}

def fetch_historical_data(api: API, instrument: str, granularity: str, start_date: datetime, end_date: datetime):
    """ Fetches historical data from Oanda for the given instrument and timeframe over the given date range.

    Args:
        api (API): The Oanda API object
        instrument (str): The name of the instrument to fetch data for
        granularity (str): The timeframe of the data to fetch
        start_date (datetime): The start date of the data to fetch
        end_date (datetime): The end date of the data to fetch

    Returns:
        pd.DataFrame: A DataFrame of the fetched data
    """
    # This value is used to download data in batches
    max_count = 2500 if granularity not in ["S5", "M1", "M5", "M15", "M20", "M30"] else 500

    # For printing progress
    str_start_date = start_date.strftime("%Y-%m-%d")
    str_end_date = end_date.strftime("%Y-%m-%d")

    # Calculate the date range
    date_range = timedelta(days=(max_count - 1) * granularity_in_days[granularity])
    data = []

    # Calculate the total duration
    total_duration = end_date - start_date

    # If granularity is M20 or H2
    resample = False
    if granularity in ["M20", "H2"]:
        resample = "20T" if granularity == "M20" else "2H"
        granularity = "M5" if granularity == "M20" else "H1"

    # Fetch the data, iterating through each date range
    while start_date < end_date:
        to_date = min(start_date + date_range, end_date)

        # Set the parameters for the request
        params = {
            "from": start_date.isoformat(),
            "to": to_date.isoformat(),
            "granularity": granularity,
            "price": "M",
        }

        # Request the data
        candles_request = InstrumentsCandles(instrument=instrument, params=params)
        api.request(candles_request)
        candles = candles_request.response["candles"]
        
        # Iterate through each candle and append the data to the list
        for candle in candles:
            timestamp = pd.to_datetime(candle["time"])
            open_price = float(candle["mid"]["o"])
            high_price = float(candle["mid"]["h"])
            low_price = float(candle["mid"]["l"])
            close_price = float(candle["mid"]["c"])
            volume = int(candle["volume"])

            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        # Calculate progress and print
        progress = 1 - (end_date - start_date) / total_duration
        sys.stdout.write(f"\rFetching {instrument} {granularity} ({str_start_date} to {str_end_date}) data: {progress:.2%} completed")
        sys.stdout.flush()

        # Update the start date to move to the next date range
        start_date = to_date

    # Log the completion of the data fetching
    print("")
    logger.info("\nFetching completed.")

    # Return the data as a DataFrame
    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("Timestamp", inplace=True)

    # Check if we need to resample
    if resample is not False:
        # Resample the data to M20 (20-minute) bars in a single line
        df = df.resample(resample).agg(
            {'Open': 'first',
             'High': 'first',
             'Low': 'first',
             'Close': 'first',
             'Volume': 'first'
             }).dropna()
    
    return df

def fetch_and_save(pair: str, granularity: str, start_date: str, end_date: str, data_dir: str=None):
    """ Fetch historical data from OANDA and save it to a CSV file.

    Args:
        instrument (str): Symbol of the instrument to fetch data for.
        granularity (str): Timeframe of the data to fetch.
        start_date (str): Begining of the time range to fetch data for in format YYYY-MM-DD.
        end_date (str): End of the time range to fetch data for in format YYYY-MM-DD.
        data_dir (str, optional): Directory to save the data to. Defaults to None.
    """    

    if data_dir is None:
        # Set the data directory if one isn't passed in
        data_dir = f"data/{pair}/{granularity}"

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Create the file name
    file_name = f"{start_date}_to_{end_date}.csv"
    file_path = os.path.join(data_dir, file_name)

    # Check if the data already exists
    if os.path.exists(file_path):
        print(f"Data already exists for {pair} {granularity} {start_date} to {end_date}.")
        return

    # Replace YOUR_API_KEY with your actual OANDA API key
    api_key = "bff0cd19d5e667790948f2820fc7bd40-3cee64d31f55fda091e2c65885ce9e87"
    api = API(access_token=api_key)

    # Convert start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    # With this line to call the new function
    df = fetch_historical_data(api, pair, granularity, start_date, end_date)

    # Save the data to a CSV file
    df.to_csv(file_path)

    print(f"Data saved to {file_path}\n")

def load_all_assets_data(assets="testing_pairs"):
    """ Returns a dictionary of DataFrames for each pair and timeframe combination.

    Returns:
        dict: A dictionary of DataFrames for each pair and timeframe, of structure:
            {pair: {timeframe: DataFrame}}
    """    
    # Load the list of forex pairs and timeframes
    pairs = load_settings()['all_pairs']

    # Path to the pickle file containing the data for all assets
    assets_pickle = f"pickles/{assets}.pickle"

    # Load the data dictionary from a pickle file if it exists, otherwise create it
    if os.path.exists(assets_pickle):
        assets_data = load_pickle(assets_pickle)
    else:
        assets_data = {pair: load_asset_data(pair) for pair in pairs}
        save_pickle(assets_data, assets_pickle)
    
    return assets_data

def load_asset_data(asset: str, data_dir="data"):
    """ Gets the data for a single asset for all timeframes.

    Args:
        asset (str): The name of the asset to get data for

    Returns:
        dict: A dictionary of timeframes and their data, of the form:
            {timeframe (str): data (DataFrame)}
    """
    data = {}
    directory = f"{data_dir}/{asset}"
    testing_timeframes = load_settings()["testing_timeframes"]

    # Traverse the data directory and get all data for the given asset
    for dir_path, sub_dirs, files in os.walk(directory):
        # Go through each subdirectory for each timeframe
        for sub_dir in sub_dirs:
            # Skip timeframes not in timeframes list above
            if sub_dir not in testing_timeframes:
                continue

            # Get the full path to the directory
            full_directory = os.path.join(dir_path, sub_dir)

            # Get the list of files in the directory
            directory_files = os.listdir(full_directory)

            # Get the full file path for each file in the directory
            full_filenames = [os.path.join(full_directory, f) for f in directory_files]

            # Get the most recent file
            latest_file = max(full_filenames, key=os.path.getctime)

            # Load the data from the file
            if latest_file.endswith('.csv'):
                df = load_data(latest_file)

                # Remove the timezone from the index
                df.index = df.index.tz_localize(None)

                 # Extract date range from data
                start_date = df.index[0]
                end_date = df.index[-1]
                date_range = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"

                # Add metadata to DataFrame
                df.attrs['pair'] = asset
                df.attrs['timeframe'] = sub_dir
                df.attrs['date_range'] = date_range

                # Add the data to the dictionary
                data[sub_dir.split('/')[-1]] = df

    return data

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the test data from the given file path."""
    data = pd.read_csv(file_path, index_col="Timestamp", parse_dates=True)
    data.index = data.index.tz_localize(None)
    # data.index = pd.DatetimeIndex(data.index).tz_convert(pytz.timezone('US/Eastern'))

    return data

def download_all_data(pairs="all_pairs"):
    """ Download data for each pair and timeframe and save it to a pickle file.

    Args:
        pairs (str, optional): The pairs to download data for. Defaults to "all_pairs". Can be "all_pairs" or "testing_pairs"
    """
    # Load the list of forex pairs
    pairs_list = load_settings()['all_pairs']

    # Download data for each pair and timeframe
    for pair in pairs_list:
        for timeframe, start_date in timeframe_ranges.items():
            fetch_and_save(pair=pair, granularity=timeframe, 
                 start_date=start_date, end_date=today.strftime("%Y-%m-%d"))
    
    # Save the data to a pickle file
    assets_data = load_all_assets_data(assets=pairs)
    save_pickle(assets_data, f"pickles/{pairs}.pickle")

def download_single_test_data(pair: str):
    """ Download data for each pair and timeframe and save it to a pickle file.
    """
    # Download data for each pair and timeframe
    all_timeframes = load_settings()["testing_timeframes"]
    for timeframe in all_timeframes:
        fetch_and_save(pair=pair, granularity=timeframe, 
                start_date=timeframe_ranges[timeframe], end_date=today.strftime("%Y-%m-%d"),
                data_dir=f"data/singles/{pair}/{timeframe}")

def main():
    download_all_data()

if __name__ == "__main__":
    main()
