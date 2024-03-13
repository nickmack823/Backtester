import math
import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Type
from backtesting import Backtest, Strategy
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import utilities
import data as data_module
from strategies import NNFX, get_NNFX_combos
np.seterr(over='ignore')

# The order in which NNFX algorithm pieces were revealed
nnfx_algo_order = [
    ["main_confirmation"],
    ["exit"],
    ["volume"],
    ["secondary_confirmation"],
    ["baseline"],
    ["atr", "tp_mult"]
]

@utilities.timer
def single_test(data: pd.DataFrame, strategy, output_file: str) -> pd.DataFrame:
    """ Runs a single test of a strategy on a given dataset.

    Args:
        data (DataFrame): A DataFrame of OHLCV data
        strategy (class): The strategy class to backtest

    Returns:
        DataFrame: A DataFrame of the results
    """    
    # Backtest the strategy
    bt = Backtest(data, strategy, cash=1000, commission=.002)
    stats = bt.run()
    bt.plot(resample=False)

    # Save the results to a CSV file
    stats.to_csv(output_file)

    # Get the trade records from the strategy instance
    # trade_records = stats['_strategy'].trades_list
    
    trades = stats["_trades"]
    trades = trades.drop(columns=['EntryBar', 'ExitBar'])
    trades = trades.assign(Direction=lambda x: np.where(x['Size'] > 0, 'Long', 'Short'))
    trades = trades.round(5)
    trades.to_csv(output_file.replace('.csv', '_trades.csv'))

    return stats

def filter_existing_combos(conn, strategy, combos: List[dict], timeframe: str, date_range: str, categories: List[str]) -> List[dict]:
    """ Filter a list of combos to include only those that don't already exist in the database.

    Args:
        conn (sqlite3.Connection): An SQLite connection object
        strategy (_type_): A backtesting.py strategy class
        combos (List[dict]): A list of dictionaries of parameters to test
        timeframe (str): The timeframe to optimize on
        date_range (str): The date range to optimize on
        categories (List[str]): The indicator categories to optimize on

    Returns:
        List[dict]: A list of filtered combos
    """    
    cursor = conn.cursor()
    filtered_combos = []

    # Normalize each combo dict to match keys of strategy base params dict (which is the structure of the record in
    # the results table)
    strat_params = strategy.params

    # Add each strat_params key not present in the combo to the combo
    for key in strat_params.keys():
        if key not in combos[0].keys(): # If it's in one combo, it's in all of them
            for combo in combos:
                combo[key] = strat_params[key]
    
    # If testing main/secondary confirmation, remove combos where main and secondary are identical
    if 'main_confirmation' in combos[0].keys() and combos[0]['main_confirmation'][0] is not None:
        combos = [combo for combo in combos if combo['main_confirmation'][0] != combo['secondary_confirmation'][0]]

    # Prepare SQL query and parameters
    query = """
        SELECT Params FROM results
        WHERE Strategy = ? AND Timeframe = ? AND DateRange = ? AND Categories = ?
    """
    params = (strategy.__name__, timeframe, date_range, str(categories))

    # Tested combos in the database
    tested_combos = cursor.execute(query, params).fetchall()
    tested_combos = set(ec[0] for ec in tested_combos)

    # Filter to include only combos that don't already exist in the database
    # (sort combo by keys to match database format)
    filtered_combos = [
        combo for combo in combos if str({k: v for k, v in sorted(combo.items())}) not in tested_combos
    ]

    # utilities.log(f"Process {os.getpid()}: Filtered {len(combos) - len(filtered_combos)} existing combos.")

    return filtered_combos

def add_best_params_to_combos(asset: str, timeframe: str, categories: List[str], combos: List[dict]) -> List[dict]:
    # Get current settings
    settings = utilities.load_settings()
    testing_order = settings['testing_order']
    
    # Load current best params for asset/timeframe/testing order and set those params in the combos
    timeframe_best_params = utilities.load_best_params(asset, timeframe, testing_order)

    if timeframe_best_params is None:
        return combos

    # Get index of categories in testing order list
    i = testing_order.index(categories)

    # Get all previous categories (as strings)
    prev_categories = [str(sublist) for sublist in testing_order[:i]]

    # This is the first category in the testing order, so no need to set best params
    if i == 0:
        return combos

    # Get the best params from the previous categories
    prev_best_params = {k: v for k, v in timeframe_best_params.items() if k in prev_categories}

    # For each category in prev_best_params, combine into a single dictionary
    best_params = {}
    for _, params in prev_best_params.items():
        best_params.update(params)

    # Finally, set the best params in the combos
    params_changed = {}
    for combo in combos:
        for best_param, best_value in best_params.items():
            # Skip None values for indicators (these are undetermined)
            if isinstance(best_value, tuple) or isinstance(best_value, list):
                if best_value[0] is None:
                    continue
            
            # Skip params that are going to be tested in the current set or future of the testing_order
            params_to_be_tested = [param for sublist in testing_order[i:] for param in sublist]
            if best_param in params_to_be_tested:
                continue

            combo[best_param], params_changed[best_param] = best_value, best_value

    # utilities.log(f"{asset} {timeframe} Params set to best for all combos: {params_changed}")

    return combos

def test_combo(data: pd.DataFrame, strategy, combo: dict, timeframe: str, date_range: str, categories: List[str]):
    """ Test a single combo of parameters on a strategy and save the results.
    - This function is used in optimize_timeframe

    Args:
        data (pd.DataFrame): The OHLCV data to backtest on
        strategy (_type_): The strategy class to backtest
        combo (dict): The parameters to test
        result_db (str): The SQLite database to save the results to
    """
    # Backtest the strategy
    start = time.time()
    bt = Backtest(data, strategy, cash=1000, commission=0)
    stats = bt.run(params=combo) # Combo changes from this point

    # Remove _equity_curve and _trades from the stats
    for to_pop in ['Start', 'End', 'Duration', '_equity_curve', '_trades']:
        stats.pop(to_pop)

    settings = utilities.load_settings()
    
    # Add timeframe, date_range, strategy name, categories, SQN, and params to the stats
    stats = pd.concat([pd.Series(
        {'SQN': stats.pop('SQN'),
        'Params': str({k: v for k, v in sorted(combo.items())}), # Sort the combo by key to ensure consistent order when reading/writing
        'Asset': settings['asset'],
        'Timeframe': timeframe, 
        'DateRange': date_range, 
        'Strategy': strategy.__name__, 
        'Categories': str(categories)
        }), stats])

    # Round each value to 3 decimal places, if value is roundable
    for key, value in stats.items():
        if isinstance(value, float):
            stats[key] = round(value, 3)

    # Set CompositeScore to 0 for now (will be calculated after database mergers)
    stats["CompositeScore"] = 0

    elapsed = time.time() - start
    # if elapsed > 5:
    #     utilities.log(f"Process {os.getpid()}: Tested combo {combo} in {elapsed:.2f} seconds.")

    return stats

def optimize_timeframe(task_params: Tuple[str, Type[Strategy], List, str, pd.DataFrame, List, Dict],
    shared_dict: Dict):
    """
    Optimize a strategy on a single asset and timeframe based on the given parameter ranges and
    save the results.
    (Used for multiprocessing tasks in optimize_asset)

    csharp
    Copy code
    Args:
        task_params (tuple): A tuple of the task parameters:
            asset (str): The name of the asset to optimize
            strategy (_type_): A backtesting.py strategy class
            combos (list): A list of dictionaries of parameters to test, of the form:
                [{'indicator_type_1': tuple('name', {'param1': v1, 'param2': v2, ...}), 'indicator_type_2': ...},
                'sl_mult': val, 'tp_mult': val}, ...]
            timeframe (str): The timeframe to optimize on
            data (pd.DataFrame): A DataFrame of OHLCV data
            categories (list): A list of the categories to optimize
        shared_lock (Lock): A shared multiprocessing.Lock object to prevent multiple processes from
            writing to the same file at the same time
        shared_counter (Value): A shared multiprocessing.Value object to keep track of the number of
            combos tests
        shared_total (Value): A shared multiprocessing.Value object to keep track of the total number
            of combos to test
    """
    # Unpack the task parameters
    asset, strategy, combos, timeframe, data, categories = task_params
    start_date = data.index[0].strftime('%m-%d-%Y')
    end_date = data.index[-1].strftime('%m-%d-%Y')
    date_range = f"{start_date}_{end_date}"

    # Unpack shared variables
    shared_lock = shared_dict['lock']
    shared_counter = shared_dict['counter']
    shared_total = shared_dict['total']
    shared_start_time = shared_dict['start_time']

    # Get the name of the output database
    with shared_lock:
        result_db = utilities.get_result_db_name(strategy.__name__, asset, timeframe, test_subfolder="optimization")
    
    # Create a connection to the final timeframe database and the temporary timeframe database
    final_db_conn = utilities.init_results_database(result_db)
    temp_db_conn = utilities.init_results_database(result_db.replace('.db', f'_temp{os.getpid()}.db'))

    # Length of combos before filtering out any that already exist in the database
    start_len_combos = len(combos)

    # Add best params to each combo
    combos = add_best_params_to_combos(asset, timeframe, categories, combos)

    # Filter out any combos that already exist in the database
    combos = filter_existing_combos(final_db_conn, strategy, combos, timeframe, date_range, categories)

    # Update the total number of combos to test
    with shared_lock:
        shared_total.value -= (start_len_combos - len(combos))

    # If there are no combos to test, close database connections and return
    if len(combos) == 0:
        final_db_conn.close()
        temp_db_conn.close()
        # utilities.log(f"Process {os.getpid()}: No new combos to test for {asset} {timeframe}")
        return
    
    # Loop through all the parameter combos and run the tests
    stats_list = []
    for combo in combos:
        # Run the test for the current combo
        stats = test_combo(data, strategy, combo, timeframe, date_range, categories)
        stats_list.append(stats)

        # If we've tested 'n' combos, insert them into the database
        if len(stats_list) >= 100:
            # Insert the stats into the temporary database
            utilities.insert_stats_into_db(temp_db_conn, stats_list)

            # Empty the stats list
            stats_list = []

        # Update the time estimate
        with shared_lock:
            shared_counter.value += 1
            perc_complete = 100 * shared_counter.value / shared_total.value

            # Use rate of incrementation to estimate time remaining
            elapsed = time.time() - shared_start_time.value
            rate = elapsed / shared_counter.value
            time_remaining = rate * (shared_total.value - shared_counter.value)
            days, hours, minutes, seconds = utilities.convert_seconds(time_remaining)

            # Log the progress and estimated time remaining
            print(f"Progress: {shared_counter.value}/{shared_total.value} ({perc_complete:.2f}%) | Estimated time remaining: {days}d {hours}h {minutes}m {seconds}s", end='\r')
    
    # Insert the remaining stats into the database
    if len(stats_list) > 0:
        utilities.insert_stats_into_db(temp_db_conn, stats_list)

    # Merge the temporary database into the final database
    utilities.merge_dbs(temp_db_conn, final_db_conn)
    final_db_conn.close()

@utilities.timer
def optimize_asset(asset: str, strategy, combos: List[dict], categories: List[str]):
    """ Optimizes a strategy on a single asset based on the given parameter ranges.
        Returns the full metrics of the best test and a heatmap of the results.

    Args:
        asset (str): The name of the asset to optimize
        strategy (_type_): A backtesting.py strategy class
        combos (list): A list of dictionaries of parameters to test, of the form:
            [{indicator_type_1: tuple('name', {'param1': v1, 'param2': v2, ...}), indicator_type_2: ...}, 'sl_mult': val, 'tp_mult': val}, ...]
        categories (list): A list of the categories to optimize
        use_params (dict, optional): A dictionary of parameters to use instead of optimizing. Defaults to None.
    """
    # Get the asset data
    asset_data = data_module.load_asset_data(asset)

    # Divide the parameter combinations into smaller chunks
    num_chunks = num_processes  # Choose an appropriate number based on the desired degree of parallelism
    chunk_size = math.ceil(len(combos) / num_chunks)  # The number of parameter combinations per chunk
    combo_chunks = [combos[i:i + chunk_size] for i in range(0, len(combos), chunk_size)]

    # Create a list of task parameters
    task_params_list = []
    for timeframe, data in asset_data.items():
        # Add a task for each chunk
        for chunk in combo_chunks:
            # Make a copy of the data to avoid sharing the same object between processes
            data_copy = data.copy(deep=True)
            task_params_list.append((asset, strategy, chunk, timeframe, data_copy, categories))

    total_tests = len(combos) * len(asset_data)
    utilities.log(f"Optimizing {categories} ({len(combos)} parameter combinations, {total_tests} tests to run. {num_chunks} chunks ({chunk_size} combos/chunk ({chunk_size*len(asset_data)} tests))) for {asset} | Timeframes: {list(asset_data.keys())}")
    
    # Create a Manager and shared variables to be used by each process
    manager = Manager()
    shared_lock = manager.Lock() # A lock to prevent problematic concurrent I/O operations
    shared_counter = manager.Value('i', 0) # A counter to track the number of completed tasks
    shared_total = manager.Value('i', total_tests) # The total number of combos to test
    shared_start_time = manager.Value('d', time.time()) # The total time taken to run the tests
    shared_dict = {"lock": shared_lock, "counter": shared_counter, "total": shared_total, "start_time": shared_start_time}

    # Shuffle task params list
    # random.shuffle(task_params_list)

    # Create a ProcessPoolExecutor to run the tasks in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create a dictionary of future tasks and their parameters
        future_to_params = {executor.submit(optimize_timeframe, task, shared_dict): task for task in task_params_list}

        # Iterate through tasks
        for future in as_completed(future_to_params):
            try:
                # Retrieve the result if any
                result = future.result()
            except Exception as e:
                # Log any errors that occurred during the task, along with traceback information
                utilities.log(f"An error occurred:\n{e}\n{traceback.format_exc()}", type="error")
                utilities.play_sound("error")
                sys.exit()

def run_single_test(asset: str, strategy, timeframe: str):
    # Create the directories for the test data and results (if they don't already exist)
    data_dir = f"data/singles/{asset}/{timeframe}/"
    test_dir = f"tests/singles/{asset}/{timeframe}/"
    utilities.create_directories(data_dir)
    utilities.create_directories(test_dir)

    # Check if data_dir is empty
    if len(os.listdir(data_dir)) == 0:
        data_module.download_single_test_data(asset)

    # Load the newest data file
    test_data_file = utilities.get_newest_file(data_dir)
    test_data = data_module.load_data(test_data_file)
    print(test_data)

    # Set metadata on the test DataFrame
    test_data.attrs["pair"] = asset
    test_data.attrs["timeframe"] = timeframe
    test_data.attrs["date_range"] = f"{test_data.index[0].strftime('%Y-%m-%d')}_to_{test_data.index[-1].strftime('%Y-%m-%d')}"

    # Run the test
    single_test(test_data, strategy, test_dir + "/single_test.csv")

def process_optimization_data(asset: str, strategy, categories: List[str]):
    utilities.log(f"Processing optimization data for {asset} | Categories: {categories}...")
    # Create the directories for the test data and results (if they don't already exist)
    db_dir = f"tests/optimization/{strategy.__name__}/{asset}/"
    utilities.create_directories(db_dir)

    # Merge the temp databases into their respective main databases
    utilities.log("Merging temp databases...")
    utilities.merge_temp_dbs(db_dir)

    # Get the list of results databases
    results_dbs = utilities.get_files(db_dir, ".db")

    # Filter out databases with file name not in current timeframes
    results_dbs = [db for db in results_dbs if db.split("/")[-1].split(".")[0] in timeframes]

    # Update each record's composite score in the results databases 
    # (composite score relies on avg. NumTrades, so we update it after tests are done)
    # Create a ProcessPoolExecutor to update composite scores for each db in parallel
    utilities.log("Updating composite scores...")
    if len(results_dbs) > 0:
        for db in results_dbs:
            try:
                # Retrieve the result if any
                utilities.update_composite_score_db(db, categories)
            except Exception as e:
                # Log any errors that occurred during the task, along with traceback information
                utilities.log(f"An error occurred while updating composite scores:\n{e}\n{traceback.format_exc()}", type="error")
                utilities.play_sound("error")
                sys.exit()

    # Save the best results to a CSV file
    utilities.log("Saving best results...")
    utilities.save_best_results_csv(results_dbs, asset)

    # Save the best parameters to a JSON file
    utilities.log("Saving best params...")
    utilities.set_best_params(asset, timeframes, strategy.__name__, categories)

    utilities.log(f"Finished processing optimization data for {asset} | Categories: {categories}.\n")

def run_optimize_asset(asset, strategy, categories, use_default_params: bool=False):
    # Process previous optimization data
    process_optimization_data(asset, strategy, categories)

    # Get the master list of parameter combinations to test based on the given categories
    combos = get_NNFX_combos(categories, use_default_params=use_default_params) 

    # Run the optimization
    optimize_asset(asset, strategy, combos, categories) 

    # Process the optimization data
    process_optimization_data(asset, strategy, categories)

def main(strategy=NNFX, run_type: str="single_test", asset: str=None, categories: list=None, 
         single_test_timeframe: str="H1", use_default_params: bool=False):
    """ The main function. Runs single tests, asset optimizations, or optimizations for all assets.

    Args:
        strategy (class, optional): The strategy to test. Defaults to NNFX.
        run_type (str, optional): The type of test(s) to run. Defaults to "single_test".
        asset (str, optional): The asset to optimize. Defaults to None.
        categories (list, optional): The indicator categories to optimize. Defaults to None (all of them).
        single_test_timeframe (str, optional): The timeframe to use for single tests. Defaults to "H1".
        use_default_params (bool, optional): Whether or not to use the default parameters for generating combos. Defaults to False.
    """
    # Single test for plotting, verifying strategy logic, etc.
    if run_type == "single_test":
        run_single_test(asset, strategy, single_test_timeframe)
    # Single optimization for testing parameter ranges on one 
    elif run_type == "optimize_asset":
        run_optimize_asset(asset, strategy, categories, use_default_params)

if __name__ == "__main__":
    # TODO: Refer to NNFX algorithm PDF on desktop
    # TODO: VectorBT ?
    # TODO: Revisit NNFX Algo Tester software to compare results, find some new indicators to test

    # TODO: Determine which indicators appear most frequently in the top n% of results
    # TODO: Add parameter for combo generation to exclude indicators not in passed list
    # # https://stonehillforex.com/indicator-library/

    # NOTE: Adjust start_hour and end_hour in results appropriately
    # Fall/Winter: EST (UTC-5) Summer/Spring: EDT (UTC-4)

    run_types = ["single_test", "optimize_asset"]
    all_params = ['atr', 'baseline', 'main_confirmation', 'secondary_confirmation', 
                  'volume', 'exit']
    testing_all = ["baseline", "main_confirmation", "secondary_confirmation", 
                     "volume", "exit"]
    
    # Manually change testing order to determine how the optimization proceeds.
    # Based on the order of the categories in this list, the optimization will determine the best params for each
    # tested category and carry those params over into future tests for subsequent categories.
    testing_order = [
        ["main_confirmation", "exit"], # Signals
        ["atr"], # Misc
        ["baseline"], # Baseline
        # ["volume", "secondary_confirmation"] # Filters
    ]

    # testing_order = nnfx_algo_order

    # testing_order = [
    #     ["atr", "baseline", "main_confirmation", "secondary_confirmation", "volume", "exit"]
    # ]
    
    # Load the settings file
    # THESE ARE GLOBAL
    settings = utilities.load_settings()
    num_processes = settings["num_processes"]
    timeframes = settings["testing_timeframes"]
    all_pairs = settings["all_pairs"]
    testing_pairs = settings["testing_pairs"]
    settings["testing_order"] = testing_order

    if str(testing_order) not in settings["completed_tests"]:
        settings["completed_tests"][str(testing_order)] = []

    utilities.save_settings(settings)

    # main(strategy=NNFX, run_type="single_test", asset="EUR_USD", single_test_timeframe="H4", categories=None)
    # sys.exit()

    for pair in testing_pairs:
        settings = utilities.load_settings()

        # # Get the list of results databases
        # db_dir = f"tests/optimization/NNFX/{pair}/"
        # results_dbs = utilities.get_files(db_dir, ".db")

        # # Filter out databases with file name not in current timeframes
        # results_dbs = [db for db in results_dbs if db.split("/")[-1].split(".")[0] in timeframes]
        # utilities.save_best_results_csv(results_dbs, pair)

        # If this pair has been tested before, skip it
        if pair in settings["completed_tests"][str(testing_order)]:
            utilities.log(f"{pair} has already been tested for {testing_order}. Skipping...")
            continue

        settings["asset"] = pair
        utilities.save_settings(settings)

        # Test each chunk of 1+ categories sequentially in the order specified in testing_order
        # Doing this accumulates the best params for each chunk and uses them for the following chunks
        for category in testing_order:
            main(strategy=NNFX, run_type="optimize_asset", asset=pair, 
                categories=category, use_default_params=False)
            
        # Mark pair as completed for this particular testing order
        settings["completed_tests"][str(testing_order)].append(pair)

        utilities.save_settings(settings)
        utilities.log("============================")

    # Save best params
    settings = utilities.load_settings()
    for pair in testing_pairs:
        # Get best params for this pair
        df = utilities.get_best_results_df(pair, "CompositeScore")
        best = df.iloc[0]
        settings["best_params_nnfx"][pair] = best.Params
    utilities.save_settings(settings)

    # Play a sound when the script is done
    utilities.play_sound()


