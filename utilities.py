import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import json
import logging
import math
import os
import pickle
import sqlite3
import sys
import traceback
from typing import Dict, List
import colorlog
import time
import pandas as pd
import filelock
import winsound
import pyttsx3

os.makedirs("indicators/preprocessed", exist_ok=True)

db_columns = ["id", "Timeframe", "DateRange", "Categories", "CompositeScore", "SQN", "Params", "ExposureTime", 
    "EquityFinal", "EquityPeak", "Return", "BuyHoldReturn", "ReturnAnn", "VolatilityAnn", 
    "SharpeRatio", "SortinoRatio", "CalmarRatio", "MaxDrawdown", "AvgDrawdown", 
    "MaxDrawdownDuration", "AvgDrawdownDuration", "NumTrades", "WinRate", 
    "BestTrade", "WorstTrade", "AvgTrade", "MaxTradeDuration", "AvgTradeDuration", 
    "ProfitFactor", "Expectancy", "Strategy"]
num_cols_to_insert = len(db_columns) - 1

results_row = """(
            Timeframe,
            DateRange,
            Categories,
            CompositeScore,
            SQN,
            Params,
            ExposureTime,
            EquityFinal,
            EquityPeak,
            Return,
            BuyHoldReturn,
            ReturnAnn,
            VolatilityAnn,
            SharpeRatio,
            SortinoRatio,
            CalmarRatio,
            MaxDrawdown,
            AvgDrawdown,
            MaxDrawdownDuration,
            AvgDrawdownDuration,
            NumTrades,
            WinRate,
            BestTrade,
            WorstTrade,
            AvgTrade,
            MaxTradeDuration,
            AvgTradeDuration,
            ProfitFactor,
            Expectancy,
            Strategy
)"""

def get_logger():
    """ Returns a logger with a custom format and color scheme. """
    # Set up colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    ))

    logger = colorlog.getLogger('data_logger')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger

logger = get_logger()

def convert_seconds(seconds):
    # Calculate the days, hours, minutes, and remaining seconds
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return round(days), round(hours), round(minutes), round(seconds)

def play_sound(type: str="exit"):
    """ Plays a sound.

    Args:
        sound (str, optional): The name of the sound file. Defaults to "beep.wav".
    """
    
    if type == "exit":
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
    elif type == "error":
        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)

def save_pickle(data, file_path: str):
    """ Saves data to a pickle file.

    Args:
        data (any): A Python object to be saved.
        file_path (_type_): The path to the pickle file.
    """
    create_directories(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path: str):
    """ Load a pickle file.

    Args:
        file_path (str): A path to a pickle file.

    Returns:
        any: The contents of the pickle file.
    """
    if not os.path.exists(file_path):
        # logger.warning(f"File '{file_path}' does not exist.")
        return None
    
    # Lock the file to prevent concurrent access
    lock = filelock.FileLock(file_path + ".lock")

    # Attempt to acquire the lock for 10 seconds
    with lock.acquire(timeout=10): 
        # Load the data from the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

    return data

# Wrapper function to time the execution of a function
def timer(func: callable):
    """ Returns a wrapper function that times the execution of a function.

    Args:
        func (func): The function to be timed.
    """    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        # Convert elapsed time to days, hours, minutes, seconds using words
        days, hours, minutes, seconds = convert_seconds(seconds)
        printable_args = [str(arg) + ", " + str(kwarg) for arg, kwarg in zip(args, kwargs)]
        log(f"{func.__name__} took {days} days, {hours} hours, {minutes} minutes and {seconds} seconds to execute.")

        return result
    
    return wrapper

def load_json(filename: str) -> dict:
    """ Load a JSON file into a dictionary.

    Args:
        filename (str): The name of the JSON file to load.

    Returns:
        dict: A dictionary containing the JSON file's contents.
    """
    with open(filename) as f:
        return json.load(f)
    
def create_directories(filename: str):
    """ Creates directories for a file if they don't already exist.

    Args:
        filename (str): A file path.
    """
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)

def get_newest_file(path: str):
    """ Gets the newest file in a directory.

    Args:
        path (str): The path to the directory.

    Returns:
        str: The path to the newest file.
    """    
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    
    return max(paths, key=os.path.getctime)

def get_files(dir: str, extension: str = None):
    """ Gets all files in a directory.

    Args:
        dir (str): The path to the directory.
        extension (str, optional): The file extension to filter by. Defaults to None.

    Returns:
        list: A list of file paths.
    """
    if not os.path.exists(dir):
        return []
    files = os.listdir(dir)
    paths = [os.path.join(dir, basename) for basename in files]
    
    if extension is not None:
        paths = [path for path in paths if path.endswith(extension)]

    return paths

def log(msg, type="info", include_timestamp=True, file_path: str=None):

    # Remove any previous FileHandlers
    logger.handlers = [item for item in logger.handlers if not isinstance(item, logging.FileHandler)]

    if file_path is not None:
        # Create a new FileHandler
        file_handler = logging.FileHandler(filename=file_path)
        file_handler.setFormatter(logging.Formatter(fmt="%(levelname)s - %(message)s"))
        logger.handlers.append(file_handler)

    if include_timestamp:
        msg = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    if type == "info":
        logger.info(msg)
    elif type == "debug":
        logger.debug(msg)
    elif type == "warning":
        logger.warning(msg)
    elif type == "error":
        logger.error(msg)
    elif type == "critical":
        logger.critical(msg)

def load_df_from_db(db_path: str):
    """ Loads a dataframe from a database file.

    Args:
        db_path (str): The path to the database file.

    Returns:
        pandas.DataFrame: The dataframe.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results WHERE NumTrades != 0 ORDER BY CompositeScore DESC", conn)
    conn.close()

    return df

def get_result_db_name(strategy_name, asset, timeframe, test_subfolder="optimization") -> str:
    """ Get the name of the result database file for a test.

    Args:
        asset (_type_): The name of the asset to test
        strategy_name (_type_): The name of the strategy to test
        # timeframe (_type_): The timeframe to test
        # date_range (_type_): The date range to test
        # categories (_type_): The categories of indicators to test
        test_subfolder (str, optional): The /tests subfolder of the file. Defaults to "optimization".

    Returns:
        str: The name of the result database file
    """
    # Designate the output file
    result_db = f"tests/{test_subfolder}/{strategy_name}/{asset}/{timeframe}.db"
    create_directories(result_db)

    return result_db

def init_results_database(db_name: str) -> sqlite3.Connection:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Create the results table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timeframe TEXT,
            DateRange TEXT,
            Categories TEXT,
            CompositeScore REAL,
            SQN REAL,
            Params TEXT,
            ExposureTime REAL,
            EquityFinal REAL,
            EquityPeak REAL,
            Return REAL,
            BuyHoldReturn REAL,
            ReturnAnn REAL,
            VolatilityAnn REAL,
            SharpeRatio REAL,
            SortinoRatio REAL,
            CalmarRatio REAL,
            MaxDrawdown REAL,
            AvgDrawdown REAL,
            MaxDrawdownDuration TEXT,
            AvgDrawdownDuration TEXT,
            NumTrades INTEGER,
            WinRate REAL,
            BestTrade REAL,
            WorstTrade REAL,
            AvgTrade REAL,
            MaxTradeDuration TEXT,
            AvgTradeDuration TEXT,
            ProfitFactor REAL,
            Expectancy REAL,
            Strategy TEXT,
            UNIQUE(Timeframe, DateRange, Categories, Params, Strategy)
        )
    """)

    # Commit the changes and close the cursor
    conn.commit()
    cursor.close()

    return conn

def get_stats_tuple(stats: pd.Series) -> tuple:

    # Round each value in params to 3 decimal places (if it's a float)
    # params_dict = dict(stats["Params"])
    # for key in params_dict:
    #     if isinstance(params_dict[key], float):
    #         params_dict[key] = round(params_dict[key], 3)
    # stats["Params"] = str(params_dict)

    stats_tuple = (stats["Timeframe"],stats["DateRange"],stats["Categories"],stats["CompositeScore"], stats["SQN"], stats["Params"],
                    stats["Exposure Time [%]"],stats["Equity Final [$]"],stats["Equity Peak [$]"],stats["Return [%]"],
                    stats["Buy & Hold Return [%]"],stats["Return (Ann.) [%]"],stats["Volatility (Ann.) [%]"],
                    stats["Sharpe Ratio"],stats["Sortino Ratio"],stats["Calmar Ratio"],stats["Max. Drawdown [%]"],
                    stats["Avg. Drawdown [%]"],stats["Max. Drawdown Duration"],stats["Avg. Drawdown Duration"],
                    stats["# Trades"],stats["Win Rate [%]"],stats["Best Trade [%]"],stats["Worst Trade [%]"],
                    stats["Avg. Trade [%]"],stats["Max. Trade Duration"],stats["Avg. Trade Duration"],
                    stats["Profit Factor"],stats["Expectancy [%]"],stats["Strategy"]) 

    return stats_tuple  

# FOR CALCUATING COMPOSITE SCORE
metrics = ["ReturnAnn", 'VolatilityAnn', 'SharpeRatio', 'SortinoRatio', 'CalmarRatio', 
           "MaxDrawdown", "WinRate", 'ProfitFactor']
# Define min-max values and weights for each metric
min_max_values = {
    'ReturnAnn': (-0.5, 0.5),
    "VolatilityAnn": (0.5, 2),
    'SharpeRatio': (-1, 3),
    'SortinoRatio': (-1, 4),
    'CalmarRatio': (-1, 3),
    'MaxDrawdown': (-50, 0),
    'WinRate': (0, 100),
    'ProfitFactor': (0, 5),
}
metric_weights = {
    'ReturnAnn': {
        'D': 0.15,
        'H4': 0.12,
        'H2': 0.10,
        'H1': 0.08,
        'M30': 0.05,
        'M20': 0.04,
        'M15': 0.03,
        'M5': 0.02
    },
    'VolatilityAnn': {
        'D': 0.10,
        'H4': 0.12,
        'H2': 0.14,
        'H1': 0.16,
        'M30': 0.18,
        'M20': 0.19,
        'M15': 0.20,
        'M5': 0.21
    },
    'SharpeRatio': {
        'D': 0.20,
        'H4': 0.20,
        'H2': 0.20,
        'H1': 0.20,
        'M30': 0.20,
        'M20': 0.20,
        'M15': 0.20,
        'M5': 0.20
    },
    'SortinoRatio': {
        'D': 0.20,
        'H4': 0.20,
        'H2': 0.20,
        'H1': 0.20,
        'M30': 0.20,
        'M20': 0.20,
        'M15': 0.20,
        'M5': 0.20
    },
    'CalmarRatio': {
        'D': 0.15,
        'H4': 0.15,
        'H2': 0.15,
        'H1': 0.15,
        'M30': 0.15,
        'M20': 0.15,
        'M15': 0.15,
        'M5': 0.15
    },
    'MaxDrawdown': {
        'D': 0.10,
        'H4': 0.10,
        'H2': 0.10,
        'H1': 0.10,
        'M30': 0.10,
        'M20': 0.10,
        'M15': 0.10,
        'M5': 0.10
    },
    'WinRate': {
        'D': 0.10,
        'H4': 0.10,
        'H2': 0.10,
        'H1': 0.10,
        'M30': 0.12,
        'M20': 0.12,
        'M15': 0.12,
        'M5': 0.12
    },
    'ProfitFactor': {
        'D': 0.10,
        'H4': 0.10,
        'H2': 0.10,
        'H1': 0.10,
        'M30': 0.12,
        'M20': 0.12,
        'M15': 0.12,
        'M5': 0.12
    },
}

def calculate_composite_score(stats: pd.Series, avg_num_trades: int, threshold_ratio: float=0.5) -> float:
    """ Calculates the composite score for a given set of stats

    Args:
        stats (pd.Series): A Pandas Series containing the stats from a backtest
        avg_num_trades (int): The average number of trades for the given timeframe
        threshold_ratio (float, optional): The ratio of the average number of trades to use as the threshold value. Defaults to 0.5.

    Returns:
        float: The composite score for the given stats
    """
    timeframe = stats['Timeframe']
    num_trades = stats['NumTrades']

    # Normalize the metrics
    normalized_metrics = {}
    for metric in metrics:
        min_val, max_val = min_max_values[metric]
        if isinstance(stats[metric], int) or isinstance(stats[metric], float):
            normalized_metrics[metric] = (stats[metric] - min_val) / (max_val - min_val)
        else:
            normalized_metrics[metric] = 0

    # Calculate the weighted sum using the metric_weights dictionary for the specific timeframe
    weighted_sum = sum(normalized_metrics[metric] * metric_weights[metric][timeframe] for metric in metrics)

    # Calculate the threshold value (tests with NumTrades less than this value will be penalized)
    threshold_value = avg_num_trades * threshold_ratio

    # Calculate the penalty
    if num_trades < threshold_value:
        # Subtract num trades from threshold to make penalty more severe the further away from the threshold the value is,
        # then divide by threshold to normalize the penalty. Then,
        penalty = ((threshold_value - num_trades) / threshold_value) * 2
    else:
        penalty = 0

    # Calculate the adjusted weighted sum (final score)
    adjusted_weighted_sum = weighted_sum * (1 - penalty)

    # Check if final score is NaN
    if not isinstance(adjusted_weighted_sum, int) and not isinstance(adjusted_weighted_sum, float):
        return -99999
    
    return round(adjusted_weighted_sum, 3)

def process_composite_scores(chunk: List[pd.Series], column_names: List[str], avg_num_trades: Dict[str, int], db: str) -> int:
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    
    # Convert each row of fromm chunk of results into a Pandas Series with the column names as index and append to a list
    list_of_series = [pd.Series(row, index=column_names) for row in chunk]
    # Calculate the composite score for each row
    for series in list_of_series:
        # Calculate the composite score
        series['CompositeScore'] = calculate_composite_score(series, avg_num_trades[series['Timeframe']])

    def execute():
        try:
            # Update the database with the new composite scores
            cursor.executemany("UPDATE results SET CompositeScore = ? WHERE id = ? AND CompositeScore != ?", 
                [(series['CompositeScore'], series['id'], series['CompositeScore']) for series in list_of_series])
        except Exception as e:
            log(f"Error updating composite scores: {e}")
            time.sleep(5)
            execute()
    
    execute()

    rows_updated = cursor.rowcount

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()

    return rows_updated

def update_composite_score_db(db: str, categories: List[str]) -> None:
    """ Updates the composite score for each test in the results databases

    Args:
        results_db (str): The results databases to update
        categories (List[str]): The categories to update
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # Compute average number of trades for each timeframe
    cursor.execute("SELECT Timeframe, AVG(NumTrades) as AvgNumTrades FROM results WHERE Categories = ? GROUP BY Timeframe", (str(categories),))
    avg_num_trades = {row[0]: row[1] for row in cursor.fetchall()}

    # Fetch results
    cursor.execute("SELECT * FROM results WHERE Categories = ?", (str(categories),))
    all_results = cursor.fetchall()

    # Fetch column names (for creating Pandas Series for each row)
    column_names = [description[0] for description in cursor.description]

    num_workers = load_settings()['num_processes']
    chunk_size = math.ceil(len(all_results) / num_workers)
    if chunk_size != 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Split results into number of chunks equal to the number of workers
            db_chunks = [all_results[i:i + chunk_size] for i in range(0, len(all_results), chunk_size)]

            future_to_params = {executor.submit(process_composite_scores, chunk, column_names, avg_num_trades, db): 
                                chunk for chunk in db_chunks}

            # Process the chunks
            for future in as_completed(future_to_params):
                try:
                    # Retrieve the result if any
                    result = future.result()
                    # log(f"{db} {categories} composite scores: {result}")
                except Exception as e:
                    # Log any errors that occurred during the task, along with traceback information
                    log(f"An error occurred:\n{e}\n{traceback.format_exc()}", type="error")
                    play_sound("error")
                    sys.exit()
    
    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()

def insert_stats_into_db(conn: sqlite3.Connection, stats_list: List[pd.Series]):
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Prepare the data for insertion
    data_to_insert = []
    for stats in stats_list:
        # Convert each datetime column to a string
        stats = stats.astype(str)
        data_to_insert.append(get_stats_tuple(stats))

    # Insert the results into the results table using executemany
    cursor.executemany(f"""
        INSERT OR IGNORE INTO results {results_row} 
        VALUES ({','.join(['?'] * num_cols_to_insert)})
    """, data_to_insert)

    # Check how many were ignored during insertion
    ignored = len(stats_list) - cursor.rowcount
    if ignored > 0:
        log(f"{ignored} results already exist in {get_dbpath_from_conn(conn)}, insertion ignored.", "warning")
    # log(f"Inserted {cursor.rowcount} results into {get_dbpath_from_conn(conn)}.")

    # Commit the changes and close the cursor
    conn.commit()
    cursor.close()

def get_dbpath_from_conn(conn: sqlite3.Connection) -> str:
    """ Get the path to the database file from a connection object.

    Args:
        conn (sqlite3.Connection): The connection object.

    Returns:
        str: The path to the database file.
    """
    for id_, name, filename in conn.execute('PRAGMA database_list'):
        if name == 'main' and filename is not None:
            return filename

def merge_dbs(merge_from: sqlite3.Connection, merge_to: sqlite3.Connection):
    """ Add all items from one database to another database

    Args:
        conn (sqlite3.Connection): The database to add to the other database
        other_conn (sqlite3.Connection): The database to add to
    """    
    # Create a cursor object to execute SQL commands
    cursor = merge_from.cursor()

    # Get columns to select
    select_columns = results_row.replace("(", "").replace(")", "").replace(" ", "")

    # Get the path to the temporary database
    merge_from_path = get_dbpath_from_conn(merge_from)
    
    # Get the results from the database
    try:
        results = cursor.execute(f"SELECT {select_columns} FROM results").fetchall()
    except sqlite3.OperationalError:
        # If the table doesn't exist, delete the database and return
        merge_from.close()
        os.remove(merge_from_path)
        log(f"Database {merge_from_path} does not contain a results table, database removed.", "warning")
        return

    # If there are no results, delete the database and return
    if len(results) == 0:
        try:
            merge_from.close()
            os.remove(merge_from_path)
        except PermissionError:
            time.sleep(2)
            merge_dbs(merge_from, merge_to)
        return

    # Create a cursor object to execute SQL commands
    other_cursor = merge_to.cursor()
    
    def try_insert():
        try:
            # Insert the results into the other database
            other_cursor.executemany(f"""
                INSERT OR IGNORE INTO results {results_row} 
                VALUES ({','.join(['?'] * num_cols_to_insert)})
            """, results)
        except Exception as e:
            log(f"Failed to insert results into {get_dbpath_from_conn(merge_to)}: {e}", "warning")
            play_sound("error")
            time.sleep(5)
            try_insert()

    try_insert()

    # Commit the changes and close the cursors
    merge_to.commit()
    merge_from.close()

    # log(f"New total number of results in {get_dbpath_from_conn(merge_to)}: {db_count} --> {new_db_count}")

    # Delete the "from" database
    os.remove(merge_from_path)

def merge_temp_dbs(db_dir: str):
    """Merge all temporary databases into a single database.

    Args:
        db_dir (str): The directory containing the databases.
    """
    # Get list of paths for temporary databases (any .db file with 'temp' in the name)
    temp_dbs = [os.path.join(db_dir, db) for db in os.listdir(db_dir) if "temp" in db]
    temp_db_journals = [db for db in temp_dbs if db.endswith("-journal")]
    temp_dbs = [db for db in temp_dbs if db.endswith(".db")]
    
    # Delete db-journals from the directory
    for db_journal in temp_db_journals:
        if db_journal.endswith("-journal"):
            os.remove(db_journal)
    
    # Get the main databases by removing everything after the first underscore
    main_dbs = [db.split("_temp")[0] + ".db" for db in temp_dbs]
    
    # Remove duplicates
    main_dbs = list(set(main_dbs))
    
    # Pair each main database with its temporary databases
    db_pairs = [(main_db, [db for db in temp_dbs if main_db.split(".")[0] in db]) for main_db in main_dbs]
    
    # Loop through each pair and merge the temporary databases into the main database
    for main_db, temp_dbs in db_pairs:
        # log(f"Merging {len(temp_dbs)} temporary databases into {main_db}...")
        
        # Create a connection to the main database
        conn = sqlite3.connect(main_db)
        
        # Loop through each temporary database and merge it into the main database
        for temp_db in temp_dbs:
            # Create a connection to the temporary database
            temp_conn = sqlite3.connect(temp_db)
            
            # Merge the temporary database into the main database
            merge_dbs(temp_conn, conn)
        
        # Close the connection to the main database
        conn.close()

def del_records_by_col(conn: sqlite3.Connection, table: str, col: str, value: str, method="="):
    """ Delete records from a table where a column matches a value.

    Args:
        conn (sqlite3.Connection): The connection to the database.
        table (str): The name of the table.
        col (str): The name of the column.
        value (str): The value to match.
        method (str, optional): The method to use to match the value. Defaults to "=". Can be "=" or "LIKE".
    """
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Get the records
    cursor.execute(f"DELETE FROM {table} WHERE {col} {method} ?", (f"%{value}%",))
    # Print records we're deleting
    log(f"Deleting {cursor.rowcount} records from {get_dbpath_from_conn(conn)} where {col} {method} {value}")
    # Commit the changes
    conn.commit()
    # Close the cursor
    cursor.close()

def print_db(conn: sqlite3.Connection):
    """ Print the contents of a database.

    Args:
        conn (sqlite3.Connection): The connection to the database.
    """
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    # Get the names of the tables
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    # Loop through each table and print the contents
    for table in tables:
        # Get the column names
        columns = cursor.execute(f"PRAGMA table_info({table[0]})").fetchall()
        # Get the data
        data = cursor.execute(f"SELECT * FROM {table[0]}").fetchall()
        # Print the table name
        log(table[0])
        # Print the column names
        log([col[1] for col in columns])
        # Print a blank line
        print()

def save_best_results_csv(results_dbs: List[str], asset: str, num_results: int=5):
    """Extracts the top results from the SQLite database and saves them to a CSV file.

    Args:
        results_dbs (List[str]): A list of paths to the SQLite databases.
        asset (str): The name of the asset.
        num_results (int, optional): The number of top results to extract from each DB. Defaults to 1.
    """
    # Initialize the list of DataFrames
    all_results = []
    settings = load_settings()
    eval_column = settings["eval_column"]

    # Loop through the databases and extract the top results
    for results_db in results_dbs:
        # Load the results from the database
        results = load_df_from_db(results_db)

        # If there are no results, delete the database and continue
        if len(results) == 0:
            # Delete the database
            os.remove(results_db)
            continue

        # Add new column for NumTrades / SQN ratio and Trade Score (rounding values)
        # results["NumTradesSQNRatio"] = (results["NumTrades"] / results["SQN"]).round()
        # results['TradeScore'] = (results['NumTrades'] * (results['WinRate'] / 100)).round()

        # Determine which indicators appear most often in the top results
        find_top_indicators(results, asset, eval_column)

        # Get the n best results
        best_results = results.groupby(["DateRange"]).apply(lambda x: x.nlargest(num_results, eval_column))

        # Order by eval_column
        best_results = best_results.sort_values(by=eval_column, ascending=False)

        # Put Params column at the end
        best_results = best_results[best_results.columns.drop("Params").tolist() + ["Params"]]

        # Append the results to the list
        all_results.append(best_results)

    if len(all_results) == 0:
        return
    
    # Concatenate the DataFrames
    all_results = pd.concat(all_results)

    # Set .csv output file name
    output_file = f"tests/best_results/{asset}_top_{num_results}_{eval_column}.csv"
    create_directories(output_file)

    # Delete the file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Get top N results and write to a .csv file
    all_results.to_csv(output_file, index=False)
    log(f"Saved top {num_results} results to {output_file}")

def get_best_results_df(pair: str, eval_column: str):
    """Get the best results.

    Args:
        strategy (str): The strategy.

    Returns:
        pd.DataFrame: The best results.
    """
    # Get the directory of the database
    csv_dir = f"tests/best_results"
    if not os.path.exists(csv_dir):
        return None

    # Get the .csv files
    csv_files = [file for file in os.listdir(csv_dir) if file.endswith(".csv") and pair in file and eval_column in file]
    if len(csv_files) == 0:
        return None
    
    csv_file = csv_files[0]

    # Load the .csv file(s) into a DataFrame
    df = pd.read_csv(f"{csv_dir}/{csv_file}")
    df.attrs["path"] = f"{csv_dir}/{csv_file}"
    
    return df
    
def set_best_params(pair: str, timeframes: List[str], strategy, categories: List[str]):
    """
    Set the best parameters for a given currency pair, strategy, and categories.

    :param pair: The currency pair being tested.
    :param timeframes: A list of timeframes to consider.
    :param strategy: The trading strategy being tested.
    :param categories: A list of categories being tested.
    :return: The best parameters found for the given inputs, or None if no suitable parameters were found.
    """

    # Load settings from the settings file
    settings = load_settings()
    eval_column = settings["eval_column"]
    testing_order = settings["testing_order"]

    # Get the DataFrame with the best results for the given pair and strategy
    results_df = get_best_results_df(pair, eval_column)
    if results_df is None:
        return None

    log(f"Loaded {results_df.attrs['path']} for {pair} best params (eval column: {eval_column}))")

    # Set the path for the best_params JSON file and load it if it exists
    best_params_json_path = f"tests/best_params/{pair}_best_params_{eval_column}.json"
    create_directories(best_params_json_path)
    best_params = json.load(open(best_params_json_path, "r")) if os.path.exists(best_params_json_path) else {}

    # Initialize an empty dictionary to store new parameters
    new_params = {}
    for timeframe in timeframes:
        # Filter the results DataFrame for the current timeframe and categories
        filtered_df = results_df[(results_df['Timeframe'] == timeframe) & (results_df['Categories'] == str(categories))]

        # Skip to the next iteration if no matching rows were found
        if filtered_df.empty:
            continue

        # Sort the DataFrame by the evaluation column and select the top row
        best_row = filtered_df.sort_values(by=eval_column, ascending=False, inplace=False).iloc[0]

        # Convert the Params value from a string to a dictionary
        params = ast.literal_eval(str(best_row["Params"]))

        # Store the parameters in the new_params dictionary, nested under the testing_order and categories
        new_params.setdefault(timeframe, {str(testing_order): dict()})
        new_params[timeframe][str(testing_order)][str(categories)] = params

    # Return None if no new parameters were found
    if not new_params:
        return None

    # Update the best_params dictionary with the new parameters found
    for timeframe, timeframe_new_params in new_params.items():
        best_params.setdefault(pair, {}).setdefault(timeframe, {}).setdefault(str(testing_order), {})
        best_params[pair][timeframe][str(testing_order)].update(timeframe_new_params[str(testing_order)])

    # Save the updated best_params dictionary to the JSON file
    with open(best_params_json_path, "w") as f:
        json.dump(best_params, f, indent=4)

    # Return the best_params dictionary
    return best_params

def load_best_params(pair: str=None, timeframe: str=None, testing_order: List[List[str]]=None):
    settings = load_settings()
    eval_column = settings["eval_column"]
    best_params_json_path = f"tests/best_params/{pair}_best_params_{eval_column}.json"

    if not os.path.exists(best_params_json_path):
        return None
    
    with open(best_params_json_path, "r") as f:
        best_params = json.load(f)
    
    # Convert each list value to a tuple
    for pair, timerame_dict in best_params.items():
        for pair_timeframe, params in timerame_dict.items():
            for param, value in params.items():
                if isinstance(value, list):
                    best_params[pair][pair_timeframe][param] = tuple(value)
    
    if pair is None:
        return best_params
    if timeframe is None:
        return best_params[pair] if pair in best_params.keys() else None
    if testing_order is None:
        return best_params[pair][timeframe] if pair in best_params.keys() and timeframe in best_params[pair].keys() else None
    else:
        return best_params[pair][timeframe][str(testing_order)] if pair in best_params.keys() and timeframe in best_params[pair].keys() and str(testing_order) in best_params[pair][timeframe].keys() else None

def load_settings():
    with open("settings.json", "r") as f:
        settings = json.load(f)
    return settings

def save_settings(settings: dict):
    with open("settings.json", "w") as f:
        json.dump(settings, f, indent=4)

def set_setting(key: str, value):
    settings = load_settings()
    settings[key] = value
    save_settings(settings)

def find_top_indicators(results_df: pd.DataFrame, asset: str, eval_column: str, n_perc: float=5):
    # Sort by eval_column
    results_df = results_df.sort_values(eval_column, ascending=False)

    # Get 1% of the results
    one_perc = int(len(results_df) * (n_perc / 100))

    # Get a list of the 'Params' column that are in the top 1%
    top_params = results_df["Params"].head(one_perc).tolist()

    # Convert each param to a dictionary
    top_params = [ast.literal_eval(param) for param in top_params]

    # Initialize an empty dictionary to store the count of values for each key
    value_counts = {}

    for d in top_params:
        for key, value in d.items():
            if key not in value_counts:
                value_counts[key] = Counter()

            # If the value is a tuple, extract the first item
            if isinstance(value, tuple) or isinstance(value, list):
                value = value[0]

            value_counts[key][value] += 1

    # Convert the value_counts dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(value_counts, orient='index').fillna(0)
    sorted_df = df.apply(lambda x: x.sort_values(ascending=False), axis=0)
    sorted_df = sorted_df.replace(0, "")
    sorted_df = sorted_df.applymap(lambda x: int(x) if isinstance(x, float) else x)

    # Convert the DataFrame to JSON and save it to a file
    results_dict = sorted_df.to_dict(orient='index')

    # Remove any value with ""
    new_dict = {}
    for param, indicators in results_dict.items():
        new_dict[param] = {}
        for indicator_name, num_occurences in indicators.items():
            if not isinstance(num_occurences, int) or indicator_name in ["", None, "nan"]:
                continue
            elif indicator_name not in new_dict:
                new_dict[param][indicator_name] = num_occurences
    
    # Sort the dictionary by the number of occurences
    for param, indicators in new_dict.items():
        new_dict[param] = dict(sorted(indicators.items(), key=lambda item: item[1], reverse=True))

    create_directories(f"tests/top_indicators/")
    with open(f"tests/top_indicators/{asset}_top_{n_perc}percent_({eval_column}).json", "w") as f:
        json.dump(new_dict, f, indent=4)

def read_tts(text: str):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.say(text)
    engine.runAndWait()

