import itertools
import pickle
import random
import time
import numpy as np
import pandas as pd
import sqlite3
from backtesting import Strategy

from indicators.indicators import atrs, baselines, confirmations, volumes, exits, all_indicators, all_categories, generate_param_combinations
from utilities import create_directories

def PPF(preprocessed_indicator_arrays: tuple, *args, **kwargs):
    """ A function that takes preprocessed indicator arrays as input and returns,
    to be used with Strategy.I()

    Args:
        preprocessed_indicator_arrays (tuple): A tuple of preprocessed indicator arrays.
        *args: Variable length argument list. (used for passing parameters to plot)
        **kwargs: Arbitrary keyword arguments. (used for passing parameters to plot)

    Returns:
        tuple: The preprocessed indicator arrays.
    """
    return preprocessed_indicator_arrays

def get_indicator_signals(indicator_name: str, indicator) -> dict:
    signals = {"trend": None, "signal": None}
    if not indicator_name:
        return signals
    else:
        signals["trend"], signals["signal"] = 0, 0

    # KASE
    if indicator_name == "kase":
        # Determine trend
        buy_line, sell_line = indicator[0], indicator[1]

        # Check if buy line is above sell line
        if buy_line[-1] > sell_line[-1] and buy_line[-1] > 50:
            signals["trend"] = 1
        # Check if sell line is above buy line
        elif buy_line[-1] < sell_line[-1] and buy_line[-1] < 50:
            signals["trend"] = -1
    
        # Check if buy line crossed above sell line
        if buy_line[-1] > sell_line[-1] and buy_line[-2] < sell_line[-2]:
            signals["signal"] = 1
        # Check if sell line crossed above buy line
        elif buy_line[-1] < sell_line[-1] and buy_line[-2] > sell_line[-2]:
            signals["signal"] = -1
    # Zero-Lag MACD
    elif indicator_name == "macd_zl":
        macd_line, signal_line = indicator[0], indicator[1]
        signals["trend"] = 1 if macd_line[-1] > 0 else -1 if macd_line[-1] < 0 else 0
        
        if signal_line[-1] > 0 and signal_line[-2] < 0:
            signals["signal"] = 1
        elif signal_line[-1] < 0 and signal_line[-2] > 0:
            signals["signal"] = -1
    # Kalman Filter
    elif indicator_name == "kalman_filter":
        bearish, bullish = indicator[0], indicator[1]

        # If current bullish is not NaN and current bearish is NaN, then bullish
        if not np.isnan(bullish[-1]) and np.isnan(bearish[-1]):
            signals["trend"] = 1
        # If current bearish is not NaN and current bullish is NaN, then bearish
        elif np.isnan(bullish[-1]) and not np.isnan(bearish[-1]):
            signals["trend"] = -1

        # If previous bullish was NaN and current bullish is not NaN, then bullish
        if np.isnan(bullish[-2]) and not np.isnan(bullish[-1]):
            signals["signal"] = 1
        # If previous bearish was NaN and current bearish is not NaN, then bearish
        elif np.isnan(bearish[-2]) and not np.isnan(bearish[-1]):
            signals["signal"] = -1
    # Fisher
    elif indicator_name == "fisher":
        fisher = indicator[0]
        signals["trend"] = 1 if fisher[-1] > 0 else -1 if fisher[-1] < 0 else 0

        if fisher[-1] > 0 and fisher[-2] < 0:
            signals["signal"] = 1
        elif fisher[-1] < 0 and fisher[-2] > 0:
            signals["signal"] = -1
    # Bulls Bears Impulse
    elif indicator_name == "bulls_bears_impulse":
        bulls, bears = indicator[0], indicator[1]
        signals["trend"] = 1 if bulls[-1] > 0 else -1 if bears[-1] > 0 else 0
        
        if bulls[-1] > 0 and bulls[-2] < 0:
            signals["signal"] = 1
        elif bears[-1] > 0 and bears[-2] < 0:
            signals["signal"] = -1
    # 3rd Generation Moving Average
    elif indicator_name == "gen3_ma":
        g3ma, g3ma_signal = indicator[0], indicator[1]
        signals["trend"] = 1 if g3ma[-1] > g3ma_signal[-1] else -1 if g3ma[-1] < g3ma_signal[-1] else 0
        
        if g3ma[-1] > g3ma_signal[-1] and g3ma[-2] < g3ma_signal[-2]:
            signals["signal"] = 1
        elif g3ma[-1] < g3ma_signal[-1] and g3ma[-2] > g3ma_signal[-2]:
            signals["signal"] = -1
    # Aroon
    elif indicator_name == "aroon":
        aroon_up, aroon_down = indicator[0], indicator[1]
        signals["trend"] = 1 if aroon_up[-1] > aroon_down[-1] else -1 if aroon_up[-1] < aroon_down[-1] else 0

        if aroon_up[-1] > aroon_down[-1] and aroon_up[-2] < aroon_down[-2]:
            signals["signal"] = 1
        elif aroon_up[-1] < aroon_down[-1] and aroon_up[-2] > aroon_down[-2]:
            signals["signal"] = -1
    # Coral
    elif indicator_name == "coral":
        coral, up, down = indicator[0], indicator[1], indicator[2]
        # Check if Up is not NaN and Down is NaN
        if not np.isnan(up[-1]) and np.isnan(down[-1]):
            signals["trend"] = 1
        # Check if Down is not NaN and Up is NaN
        elif np.isnan(up[-1]) and not np.isnan(down[-1]):
            signals["trend"] = -1
        
        # Check if Up went from NaN to not NaN
        if np.isnan(up[-2]) and not np.isnan(up[-1]):
            signals["signal"] = 1
        # Check if Down went from NaN to not NaN
        elif np.isnan(down[-2]) and not np.isnan(down[-1]):
            signals["signal"] = -1
    # Ehler's Center of Gravity
    elif indicator_name == "center_of_gravity":
        cog = indicator[0]
        signals["trend"] = 1 if cog[-1] > 0 else -1 if cog[-1] < 0 else 0

        if cog[-1] > 0 and cog[-2] < 0:
            signals["signal"] = 1
        elif cog[-1] < 0 and cog[-2] > 0:
            signals["signal"] = -1
    # Grucha Percentage Index
    elif indicator_name == "grucha":
        grucha, grucha_ma = indicator[0], indicator[1]
        signals["trend"] = 1 if grucha[-1] > grucha_ma[-1] else -1 if grucha[-1] < grucha_ma[-1] else 0

        if grucha[-1] > grucha_ma[-1] and grucha[-2] < grucha_ma[-2]:
            signals["signal"] = 1
        elif grucha[-1] < grucha_ma[-1] and grucha[-2] > grucha_ma[-2]:
            signals["signal"] = -1
    # Half Trend
    elif indicator_name == "half_trend":
        up, down = indicator[0], indicator[1]
        signals["trend"] = 1 if up[-1] > 0 else -1 if down[-1] > 0 else 0

        if up[-1] > 0 and up[-2] == 0:
            signals["signal"] = 1
        elif down[-1] > 0 and down[-2] == 0:
            signals["signal"] = -1
    # J_TPO Velocity
    elif indicator_name == "j_tpo":
        j_tpo = indicator[0]
        signals["trend"] = 1 if j_tpo[-1] > 0 else -1 if j_tpo[-1] < 0 else 0

        if j_tpo[-1] > 0 and j_tpo[-2] < 0:
            signals["signal"] = 1
        elif j_tpo[-1] < 0 and j_tpo[-2] > 0:
            signals["signal"] = -1
    # Klinger Volume Oscillator (KVO)
    elif indicator_name == "kvo":
        kvo, kvo_signal = indicator[0], indicator[1]
        signals["trend"] = 1 if kvo_signal[-1] > 0 else -1 if kvo_signal[-1] < 0 else 0

        if kvo_signal[-1] > 0 and kvo_signal[-2] < 0:
            signals["signal"] = 1
        elif kvo_signal[-1] < 0 and kvo_signal[-2] > 0:
            signals["signal"] = -1
    # Larry Williams Proxy Index (LWPI)
    elif indicator_name == "lwpi":
        lwpi = indicator[0]
        signals["trend"] = 1 if lwpi[-1] < 50 else -1 if lwpi[-1] > 50 else 0

        if lwpi[-1] < 50 and lwpi[-2] > 50:
            signals["signal"] = 1
        elif lwpi[-1] > 50 and lwpi[-2] < 50:
            signals["signal"] = -1
    # SuperTrend
    elif indicator_name == "supertrend":
        supertrend, trend = indicator[0], indicator[1]
        signals["trend"] = trend[-1]

        if trend[-1] == 1 and trend[-2] == -1:
            signals["signal"] = 1
        elif trend[-1] == -1 and trend[-2] == 1:
            signals["signal"] = -1
    # Trend Continuation Factor (TCF)
    elif indicator_name == "tcf":
        pass
    # Trend Trigger Factor (TTF)
    elif indicator_name == "ttf":
        ttf, ttf_signal = indicator[0], indicator[1]
        signals["trend"] = 1 if ttf_signal[-1] > 0 else -1 if ttf_signal[-1] < 0 else 0

        if ttf_signal[-1] > 0 and ttf_signal[-2] < 0:
            signals["signal"] = 1
        elif ttf_signal[-1] < 0 and ttf_signal[-2] > 0:
            signals["signal"] = -1
    # Vortex Indicator
    elif indicator_name == "vortex":
        plus_vi, minus_vi = indicator[0], indicator[1]
        signals["trend"] = 1 if plus_vi[-1] > minus_vi[-1] else -1 if plus_vi[-1] < minus_vi[-1] else 0

        if plus_vi[-1] > minus_vi[-1] and plus_vi[-2] < minus_vi[-2]:
            signals["signal"] = 1
        elif plus_vi[-1] < minus_vi[-1] and plus_vi[-2] > minus_vi[-2]:
            signals["signal"] = -1
    # Braid Filter (Histogram)
    elif indicator_name == "braid_filter_hist":
        up_hist, down_hist = indicator[0], indicator[1]
        signals["trend"] = 1 if not np.isnan(up_hist[-1]) else -1 if not np.isnan(down_hist[-1]) else 0

        if not np.isnan(up_hist[-1]) and np.isnan(up_hist[-2]):
            signals["signal"] = 1
        elif not np.isnan(down_hist[-1]) and np.isnan(down_hist[-2]):
            signals["signal"] = -1
    # Braid Filter
    elif indicator_name == "braid_filter":
        up, down = indicator[0], indicator[1]
        signals["trend"] = 1 if up[-1] > 0 else -1 if down[-1] < 0 else 0

        if up[-1] > 0 and up[-2] == 0:
            signals["signal"] = 1
        elif down[-1] < 0 and down[-2] == 0:
            signals["signal"] = -1
    # Detrended Synthetic Price (DSP)
    elif indicator_name == "dsp":
        pass
    # Laguerre
    elif indicator_name == "laguerre":
        laguerre = indicator[0]
        signals["trend"] = 1 if laguerre[-1] > 0.5 else -1 if laguerre[-1] < 0.5 else 0

        if laguerre[-1] > 0.5 and laguerre[-2] < 0.5:
            signals["signal"] = 1
        elif laguerre[-1] < 0.5 and laguerre[-2] > 0.5:
            signals["signal"] = -1
    # Recursive Moving Average
    elif indicator_name == "recursive_ma":
        rema, rema_signal = indicator[0], indicator[1]
        signals["trend"] = 1 if rema_signal[-1] > rema[-1] else -1 if rema_signal[-1] < rema[-1] else 0

        if rema_signal[-1] > rema[-1] and rema_signal[-2] < rema[-2]:
            signals["signal"] = 1
        elif rema_signal[-1] < rema[-1] and rema_signal[-2] > rema[-2]:
            signals["signal"] = -1
    # Schaff Trend Cycle
    elif indicator_name == "schaff_trend_cycle":
        stc, stc_diff = indicator[0], indicator[1]
        signals["trend"] = 1 if stc[-1] > 50 else -1 if stc[-1] < 50 else 0

        if stc[-1] > stc[-2] and stc_diff[-1] > stc_diff[-2]:
            signals["signal"] = 1
        elif stc[-1] < stc[-2] and stc_diff[-1] < stc_diff[-2]:
            signals["signal"] = -1
    # SmoothStep
    elif indicator_name == "smooth_step":
        ss = indicator[0]
        signals["trend"] = 1 if ss[-1] > 0.5 else -1 if ss[-1] < 0.5 else 0

        if ss[-1] > 0.5 and ss[-2] < 0.5:
            signals["signal"] = 1
        elif ss[-1] < 0.5 and ss[-2] > 0.5:
            signals["signal"] = -1
    # TopTrend
    elif indicator_name == "top_trend":
        top_trend = indicator[0]
        # Signal: -1 if reached end of downtrend, 1 if reached end of uptrend
        # Find most recent nonzero value
        recent_line = 0
        for i in range(len(top_trend) - 1, -1, -1):
            if top_trend[i] != 0:
                recent_line = top_trend[i]
                break
        signals["trend"] = recent_line * -1

        # End of uptrend
        if top_trend[-1] == 1 and top_trend[-2] != 1:
            signals["signal"] = -1
        # End of downtrend
        elif top_trend[-1] == -1 and top_trend[-2] != -1:
            signals["signal"] = 1
    # Trend Lord
    elif indicator_name == "trend_lord":
        trend_lord_main, trend_lord_signal = indicator[0], indicator[1]
        signals["trend"] = 1 if trend_lord_signal[-1] > trend_lord_main[-1] else -1 if trend_lord_signal[-1] < trend_lord_main[-1] else 0

        if trend_lord_signal[-1] > trend_lord_main[-1] and trend_lord_signal[-2] < trend_lord_main[-2]:
            signals["signal"] = 1
        elif trend_lord_signal[-1] < trend_lord_main[-1] and trend_lord_signal[-2] > trend_lord_main[-2]:
            signals["signal"] = -1
    # Twigg's Money Flow
    elif indicator_name == "twiggs_mf":
        twiggs_mf = indicator[0]
        signals["trend"] = 1 if twiggs_mf[-1] > 0 else -1 if twiggs_mf[-1] < 0 else 0
        if twiggs_mf[-1] > 0 and twiggs_mf[-2] < 0:
            signals["signal"] = 1
        elif twiggs_mf[-1] < 0 and twiggs_mf[-2] > 0:
            signals["signal"] = -1
    # UF2018
    elif indicator_name == "uf2018":
        # Hmmmmmm
        uf2018_down, uf2018_up = indicator[0], indicator[1]
        signals["trend"] = 1 if uf2018_up[-1] == 1 else -1 if uf2018_down[-1] == 1 else 0

        if uf2018_up[-1] == 1 and uf2018_up[-2] == 0:
            signals["signal"] = 1
        elif uf2018_down[-1] == 1 and uf2018_down[-2] == 0:
            signals["signal"] = -1
    # SSL
    elif indicator_name == "ssl":
        ssl_up, ssl_down = indicator[0], indicator[1]
        signals["trend"] = 1 if ssl_up[-1] < ssl_down[-1] else -1 if ssl_up[-1] > ssl_down[-1] else 0

        if ssl_up[-1] < ssl_down[-1] and ssl_up[-2] > ssl_down[-2]:
            signals["signal"] = 1
        elif ssl_up[-1] > ssl_down[-1] and ssl_up[-2] < ssl_down[-2]:
            signals["signal"] = -1
        else:
            signals["signal"] = 0

    return signals


base_params = {
    "atr": ("atr", atrs["atr"]["default_params"]),
    "baseline": (None, None),
    "main_confirmation": (None, None),
    "secondary_confirmation": (None, None),
    "volume": (None, None),
    "exit": (None, None),
    "sl_mult": 1.5, # For both running and non-running trades
    "tp_mult": 1, # For the 1 of the 2 trades that is not the running trade
    'start_hour': 0, # Default time window to allowing trading at any hour
    'end_hour': 23
}

single_test = {'atr': ('atr', {'period': 10}), 'baseline': (None, None), 'end_hour': 23, 'exit': ['gen3_ma', {'period': 100, 'sampling_period': 40}], 'main_confirmation': ['twiggs_mf', {'period': 15}], 'secondary_confirmation': (None, None), 'sl_mult': 1.5, 'start_hour': 0, 'tp_mult': 1, 'volume': (None, None)}

class NNFX(Strategy):
    # Params (changed during optimization; this is what gets passed in: a dict of params)
    params = single_test
    # params = single_test

    def load_indicator(self, price_data: pd.DataFrame, metadata: dict, indicator: tuple,
                        plot: bool = False, overlay: bool = False) -> np.ndarray:
        """ Loads an indicator from a file if it exists, otherwise calculates it and saves it to a file.

        Args:
            price_data (pd.DataFrame): A DataFrame containing the price data.
            metadata (dict): A dictionary containing the metadata for the price data.
            indicator (tuple): A tuple containing the indicator name, func, and params.
                -ind_name (str): The name of the indicator.
                -ind_func (callable): The function to calculate the indicator.
                -ind_params (dict): The parameters to pass to the indicator function.
            plot (bool, optional): Whether or not to plot the indicator. Defaults to False.

        Returns:
            np.ndarray: The calculated indicator.
        """
        # Unpack indicator tuple
        ind_name, ind_func, ind_params = indicator
        params_str = str(ind_params).replace(":", "-").replace(",", "_").replace(" ", "")

        # Get DataFrame metadata
        pair, timeframe, date_range = metadata["pair"], metadata["timeframe"], metadata["date_range"]

        # Get DB path
        db_path = f"indicators/preprocessed/{ind_name}/{pair}/{timeframe}/{ind_name}_values.db"
        create_directories(db_path)

        # Establish connection to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator TEXT,
                params TEXT,
                pair TEXT,
                timeframe TEXT,
                date_range TEXT,
                data BLOB
            )
            """)

        # Check if indicator values exist in database
        cursor.execute("SELECT indicator, params, pair, timeframe, date_range, data FROM data WHERE indicator=? AND params=? AND pair=? AND timeframe=? AND date_range=?",
                        (ind_name, params_str, pair, timeframe, date_range))
        result = cursor.fetchone()

        # If indicator values exist in database, load them
        if result:
            serialized_indicator = result[-1]
            deserialized_indicator = pickle.loads(serialized_indicator)
            indicator = self.I(PPF, deserialized_indicator, price_data, plot=plot, overlay=overlay, **ind_params)
        # If indicator values don't exist in database, calculate and save to database
        else:
            indicator = self.I(ind_func, price_data, plot=plot, overlay=overlay, **ind_params)
            serialized_indicator = pickle.dumps(indicator)

            # Save indicator values to database
            cursor.execute("""
            INSERT INTO data (indicator, params, pair, timeframe, date_range, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (ind_name, params_str, pair, timeframe, date_range, sqlite3.Binary(serialized_indicator)))
            conn.commit()

        return indicator

    def init(self):
        # Add unused params to params dict
        for category in all_categories:
            if category not in self.params.keys():
                self.params[category] = (None, None) if category not in ["sl_mult", "tp_mult"] else 0
        
        # Initialize params
        self.atr = self.params["atr"]
        self.baseline = self.params["baseline"]
        self.main_confirmation = self.params["main_confirmation"]
        self.secondary_confirmation = self.params["secondary_confirmation"]
        self.volume = self.params["volume"]
        self.exit = self.params["exit"]
        self.sl_mult = self.params["sl_mult"]
        self.tp_mult = self.params["tp_mult"]
        # self.use_bridge_too_far = self.params["use_bridge_too_far"]
        self.start_hour = self.params["start_hour"]
        self.end_hour = self.params["end_hour"]

        # Stop loss for order 2 (the running one)
        self.order_tracking = {
            "order2": {
                "order": None,
                "direction": None,
                "sl": None,
                "tp": None,
                "entry": None,
                "entry_atr": None,
                "last_sl_move": None,
            }
        }

        self.bars_since = {
            "c1_signal_long": -1,
            "c1_signal_short": -1,
            "baseline_cross_long": -1,
            "baseline_cross_short": -1,
        }

        # Get DataFrame metadata
        metadata = self.data.df.attrs

        self.i = 0

        # Print indicator names and parameters
        printing = False
        if printing:
            print("Initializing NNFX with parameters:\n")
            print(f"ATR: {self.atr}")
            print(f"Baseline: {self.baseline}")
            print(f"Main confirmation: {self.main_confirmation}")
            print(f"Secondary confirmation: {self.secondary_confirmation}")
            print(f"Volume: {self.volume}")
            print(f"Exit: {self.exit}")
            print(f"SL/TP multipliers: {self.sl_mult}, {self.tp_mult}\n")

        # Make a DataFrame for indicator calculations
        # price_data = self.data.df # Don't do this, it includes the index which doubles the size
        # of output indicator arrays
        price_data = pd.DataFrame({
            'Open': self.data.Open,
            'High': self.data.High,
            'Low': self.data.Low,
            'Close': self.data.Close,
            'Volume': self.data.Volume
        })

        # Normalize incoming parameters
        for name, params in [self.atr, self.baseline, self.main_confirmation, self.secondary_confirmation, self.volume, self.exit]:
            if isinstance(params, tuple):
                params = {param: value for param, value in params}
            if name == "None":
                name = None

        # Unpack parameters
        self.atr_name, self.atr_params = self.atr
        self.baseline_name, self.baseline_params = self.baseline
        self.main_confirmation_name, self.main_confirmation_params = self.main_confirmation
        self.secondary_confirmation_name, self.secondary_confirmation_params = self.secondary_confirmation
        self.volume_name, self.volume_params = self.volume
        self.exit_name, self.exit_params = self.exit

        # Set variables to indicate which indicators we're using
        self.using_atr = True if self.atr_name is not None else False
        self.using_baseline = True if self.baseline_name is not None else False
        self.using_main_confirmation = True if self.main_confirmation_name is not None else False
        self.using_secondary_confirmation = True if self.secondary_confirmation_name is not None else False
        self.using_volume = True if self.volume_name is not None else False
        self.using_exit = True if self.exit_name is not None else False
        self.use_main_confirmation_as_exit = True if self.exit_name is None else False

        # Initialize indicators
        self.atr_indicator = None
        self.baseline_indicator = None
        self.main_confirmation_indicator, self.secondary_confirmation_indicator = None, None
        self.volume_indicator = None
        self.exit_indicator = None

        # Load indicators (if not None)
        if self.using_atr:
            self.atr_indicator = self.load_indicator(price_data, metadata, 
                (self.atr_name, atrs[self.atr_name]["function"], self.atr_params), plot=True)
        if self.using_baseline:
            self.baseline_indicator = self.load_indicator(price_data, metadata, 
                (self.baseline_name, baselines[self.baseline_name]["function"], self.baseline_params), plot=True, overlay=True)
        if self.using_main_confirmation:
            self.main_confirmation_indicator = self.load_indicator(price_data, metadata, 
                (self.main_confirmation_name, confirmations[self.main_confirmation_name]["function"], self.main_confirmation_params), plot=True, overlay=True)
        if self.using_secondary_confirmation:
            self.secondary_confirmation_indicator = self.load_indicator(price_data, metadata, 
                (self.secondary_confirmation_name, confirmations[self.secondary_confirmation_name]["function"], self.secondary_confirmation_params), plot=True)
        if self.using_volume:
            self.volume_indicator = self.load_indicator(price_data, metadata, 
                (self.volume_name, volumes[self.volume_name]["function"], self.volume_params), plot=True)
        if self.using_exit:
            self.exit_indicator = self.load_indicator(price_data, metadata, 
                (self.exit_name, exits[self.exit_name]["function"], self.exit_params), plot=True)

    def check_baseline(self) -> int:
        baseline = {"trend": None, "signal": None}

        # For checking if price within 1x ATR of baseline/commencing pullback
        self.within_atr, self.pullback = None, None

        if not self.using_baseline:
            return baseline

        # Determine trend
        if self.baseline_indicator[-1] < self.data.Close[-1]:
            baseline["trend"] = 1
        elif self.baseline_indicator[-1] > self.data.Close[-1]:
            baseline["trend"] = -1
        else:
            baseline["trend"] = 0
        
        # Check for signal
        curr_baseline, prev_baseline = self.baseline_indicator[-1], self.baseline_indicator[-2]
        price, prev_price = self.data.Close[-1], self.data.Close[-2]

        # Price crossed above baseline
        if prev_price < prev_baseline and price > curr_baseline:
           baseline["signal"] = 1
           self.bars_since['baseline_cross_long'] = 0
        # Price crossed below baseline
        elif prev_price > prev_baseline and price < curr_baseline:
            baseline["signal"] = -1
            self.bars_since['baseline_cross_short'] = 0
        else:
            baseline["signal"] = 0

        # Check if price is within 1x ATR of baseline
        if (self.close >= (curr_baseline - self.atr)) and (self.close <= (curr_baseline + self.atr)):
            self.within_atr = True
        else:
            self.within_atr = False

        # PULLBACK - Check if price was beyond 1x ATR of baseline on previous candle
        self.prev_atr = self.atr_indicator[-2]
        if (self.prev_close < (prev_baseline - self.prev_atr)) or (self.prev_close > (prev_baseline + self.prev_atr)):
            # Previous price was beyond 1x ATR of baseline, check if we're now within 1x ATR of baseline
            if self.within_atr:
                self.pullback = True
            else:
                self.pullback = False

        return baseline

    def check_confirmation(self) -> int:
        final_confirmation = {"trend": None, "signal": None}

        if not self.using_main_confirmation:
            return final_confirmation

        # Get main and secondary confirmation signals
        main_confirmation = get_indicator_signals(self.main_confirmation[0], self.main_confirmation_indicator)
        secondary_confirmation = get_indicator_signals(self.secondary_confirmation[0], self.secondary_confirmation_indicator)
        
        # Check if main and secondary confirmation agree
        main_trend, main_signal = main_confirmation["trend"], main_confirmation["signal"]
        secondary_trend, secondary_signal = secondary_confirmation["trend"], secondary_confirmation["signal"]

        # Set bars since C1 signal
        if main_signal == 1:
            self.bars_since["c1_signal_long"] = 0
        elif main_signal == -1:
            self.bars_since["c1_signal_short"] = 0

        # One Candle Rule - C1 signal lasts from the candle it was on to the next one (unless we get an opposite signal)
        if self.bars_since["c1_signal_long"] == 1 and main_signal == 0:
            main_signal = 1
        elif self.bars_since["c1_signal_short"] == 1 and main_signal == 0:
            main_signal = -1

        # Check if main and secondary confirmation agree
        if secondary_trend in [main_trend, None]:
            final_confirmation["trend"] = main_trend
        else:
            final_confirmation["trend"] = 0
        if secondary_signal in [main_signal, None]:
            final_confirmation["signal"] = main_signal
        else:
            final_confirmation["signal"] = 0

        return final_confirmation

    def check_volume(self) -> int:
        vol = {"sufficient_strength": None, "direction": None}

        if not self.using_volume:
            return vol

        # Trend Direction Force Index (TDFI)
        if self.volume[0] == "tdfi":
            tdfi = self.volume_indicator[0]
            # print(type(tdfi))
            # print(tdfi)
            if tdfi[-1] > 0.05:
                vol["sufficient_strength"], vol["direction"] = True, 1
            elif tdfi[-1] < -0.05:
                vol["sufficient_strength"], vol["direction"] = True, -1
            else:
                vol["sufficient_strength"], vol["direction"] = False, 0
        # Average Directional Index (ADX)
        elif self.volume[0] == "adx":
            adx, diplus, diminus = self.volume_indicator[0], self.volume_indicator[1], self.volume_indicator[2]
            if adx[-1] > 25:
                vol["sufficient_strength"] = True
            elif adx[-1] < 25:
                vol["sufficient_strength"] = False
        # Normalized Volume
        elif self.volume[0] == "normalized_volume":
            nv = self.volume_indicator[0]
            if nv[-1] > 100:
                vol["sufficient_strength"] = True
            elif nv[-1] < 100:
                vol["sufficient_strength"] = False
        # Volatility Ratio (UNUSED)
        elif self.volume[0] == "volatility_ratio":
            vr, up, down = self.volume_indicator[0], self.volume_indicator[1], self.volume_indicator[2]
            if vr[-1] > 0:
                vol["sufficient_strength"] = True
            elif vr[-1] < 0:
                vol["sufficient_strength"] = False
        # Waddah Attar Explosion
        elif self.volume[0] == "wae":
            trend, explosion, dead = self.volume_indicator[0], self.volume_indicator[1], self.volume_indicator[2]
            vol['direction'] = 1 if trend[-1] > 0 else -1 if trend[-1] < 0 else 0
            vol['sufficient_strength'] = True if explosion[-1] > 0 else False
            
        return vol

    def record_trade(self, timestamp, action: str, price: float, sl=None, tp=None):
        msg = f"{timestamp} - {action} at {price}"
        if sl:
            msg += f" SL={round(sl, 5)}"
        if tp:
            msg += f" TP={round(tp, 5)}"
        msg += "\n"

        trade = {
            "timestamp": timestamp,
            "price": price,
            "action": action,
            "sl": sl,
            "tp": tp,
            "message": msg
        }
        # print(msg)
        self.trades_list.append(trade)

    def check_signal(self, direction):

        go, do_not_enter = False, False

        baseline = self.curr_baseline
        confirmation = self.curr_confirmation
        volume = self.curr_volume

        signal_val = 1 if direction == "long" else -1
        c1_signal_key, baseline_cross_key = f'c1_signal_{direction}', f'baseline_cross_{direction}'

        # Check overall directions of baseline and confirmation
        baseline_direction, confirmation_direction = baseline["trend"], confirmation["trend"]

        # Check baseline for signal and C1 for agreement
        baseline_entry = False
        if self.using_baseline:
            # Baseline gives signal and confirmation agrees/isn't being used
            if baseline["signal"] == signal_val and (confirmation["trend"] in [None, signal_val]):
                # Check "Bridge Too Far" (C1 last cross was 7+ bars ago)
                if self.bars_since[c1_signal_key] >= 7: # and self.use_bridge_too_far:
                    baseline_entry = False
                else:
                    baseline_entry = True
    
        # Check confirmation
        confirmation_entry, continuation = False, False
        if self.using_main_confirmation:
            if confirmation["signal"] == signal_val and (baseline["trend"] in [None, signal_val]):
                confirmation_entry = True

                # Check for continuation trade (ex. price crossed baseline, C1 signaled, now C1 signals again)
                barse_since_c1, bars_since_baseline = self.bars_since[c1_signal_key], self.bars_since[baseline_cross_key]
                if barse_since_c1 < bars_since_baseline:
                    continuation = True

        # Check volume (disregard for continuation trades)
        if self.using_volume and not continuation:
            # Check if volume is strong enough and in the right direction
            if not volume["sufficient_strength"] or volume["direction"] == -signal_val:
                do_not_enter = True

        # Check if price within 1x ATR of baseline (if not, don't buy except for pullback/continuation)
        if self.using_baseline and not self.within_atr and not continuation:
            do_not_enter = True

        # Check for pullback (baseline signal + price BEYOND 1x ATR of baseline previously, now within and indis agree)
        pullback_entry = False
        if (self.pullback and self.bars_since[baseline_cross_key] == 1):
            # Check if baseline and confirmation agree with pullback
            if baseline_direction == confirmation_direction and baseline_direction == signal_val:
                pullback_entry = True

        # Determine if we should enter
        if (baseline_entry or confirmation_entry or pullback_entry) and not do_not_enter:
            go = True
        else:
            go = False

        return go

    def should_buy(self):
        return self.check_signal("long")
        
    def should_sell(self):
        return self.check_signal("short")

    def should_exit(self):
        should_exit = False
        exit = get_indicator_signals(self.exit_name, self.exit_indicator)
        position = self.position
        baseline = self.curr_baseline
        confirmation = self.curr_confirmation

        # For trailing stoploss
        close_order2 = self.trigger_order2_sl()

        if position.is_long:
            # Check if we got a sell signal
            if self.go_sell:
                should_exit = True

            # Check baseline and exit for short signal
            if any(ind["signal"] == -1 for ind in [baseline, exit]):
                should_exit = True

            # Check if confirmation is opposite of position
            if self.use_main_confirmation_as_exit and confirmation["signal"] == -1:
                should_exit = True
            
        elif position.is_short:
            # Check if we got a buy signal
            if self.go_buy:
                should_exit = True
            
            # Check baseline and exit for long signal
            if any(ind["signal"] == 1 for ind in [baseline, exit]):
                should_exit = True
            
            # Check if confirmation is opposite of position
            if self.use_main_confirmation_as_exit and confirmation["signal"] == 1:
                should_exit = True
            
        return (should_exit or close_order2)

    def adjust_order2_sl(self):
        order2_info = self.order_tracking["order2"]

        if order2_info is None:
            return

        direction, sl, entry, entry_atr = order2_info["direction"], order2_info["sl"], order2_info["entry"], order2_info["entry_atr"]
        curr_atr = self.atr
        last_sl_move = order2_info["last_sl_move"]

        if direction == 1:
            # Check if Order1 hit TP (price is 1x ATR away from entry)
            if self.close >= (entry + (self.tp_mult * entry_atr)):
                # Move SL to entry
                sl = entry
            # Else if price is 2x ATR away from entry
            if self.close >= (entry + (curr_atr * 2)):
                # Move SL to 1.5x ATR away from current price
                sl = self.close - (curr_atr * 1.5)
                # Set price we last moved SL at
                order2_info["last_sl_move"] = self.close
            # Else if each next step of 0.5x ATR
            if last_sl_move is not None:
                if self.close >= (last_sl_move + (0.5 * curr_atr)):
                    # Move SL to 1.5x ATR away from current price
                    sl = self.close - (curr_atr * 1.5)
                    # Set price we last moved SL at
                    order2_info["last_sl_move"] = self.close
        elif direction == -1:
            # Check if Order1 hit TP (price is 1x ATR away from entry)
            if self.close <= (entry - (self.tp_mult * entry_atr)):
                # Move SL to entry
                sl = entry
            # Else if price is 2x ATR away from entry
            if self.close <= (entry - (curr_atr * 2)):
                # Move SL to 1.5x ATR away from current price
                sl = self.close + (curr_atr * 1.5)
                # Set price we last moved SL at
                order2_info["last_sl_move"] = self.close
            # Else if each next step of 0.5x ATR
            if last_sl_move is not None:
                if self.close <= (last_sl_move - (0.5 * curr_atr)):
                    # Move SL to 1.5x ATR away from current price
                    sl = self.close + (curr_atr * 1.5)
                    # Set price we last moved SL at
                    order2_info["last_sl_move"] = self.close

        # Don't change SL if new one would be behind previous
        order2_info["sl"] = sl if ((direction == 1 and sl > order2_info["sl"]) or (direction == -1 and sl < order2_info["sl"])) else order2_info["sl"]
        self.order_tracking["order2"] = order2_info

    def trigger_order2_sl(self):
        
        if self.order_tracking["order2"] is None:
            return False

        order2_sl = self.order_tracking["order2"]["sl"]
        if self.position.is_long and self.close <= order2_sl:
            return True
        elif self.position.is_short and self.close >= order2_sl:
            return True
        else:
            return False

    def next(self):
        # Set core variables
        position = self.position
        timestamp = self.data.index[-1] 
        curr_hour = pd.to_datetime(timestamp).hour

        # Adjut to align plot with timestamp
        # timestamp = timestamp - pd.Timedelta(hours=11)

        # Update "bars since" counters
        if self.bars_since["c1_signal_long"] >= 0:
            self.bars_since["c1_signal_long"] += 1
        if self.bars_since["c1_signal_short"] >= 0:
            self.bars_since["c1_signal_short"] += 1
        if self.bars_since["baseline_cross_long"] >= 0:
            self.bars_since["baseline_cross_long"] += 1
        if self.bars_since["baseline_cross_short"] >= 0:
            self.bars_since["baseline_cross_short"] += 1

        # Set current variables (for checking indicators/entry and exit logic)
        self.close, self.prev_close = self.data.Close[-1], self.data.Close[-2]
        self.atr = self.atr_indicator[-1]

        # Set buy and sell SL/TP
        buy_sl, sell_sl = self.close - (self.sl_mult * self.atr), self.close + (self.sl_mult * self.atr)
        buy_tp, sell_tp = self.close + (self.tp_mult * self.atr), self.close - (self.tp_mult * self.atr)

        # Adjust Order 2 Trailing SL
        self.adjust_order2_sl()

        # Set indicator signals/trends dictionaries (for should_buy, should_sell, and should_exit)
        self.curr_baseline = self.check_baseline()
        self.curr_confirmation = self.check_confirmation()
        self.curr_volume = self.check_volume()

        # Set trigger variables
        self.go_buy, self.go_sell = self.should_buy(), self.should_sell()
        should_exit = self.should_exit()
        
        # Check if we should exit a position
        if should_exit:
            self.position.close()
            self.order_tracking['order2'] = None

        # Check if we're within start_hour and end_hour
        if self.start_hour <= curr_hour <= self.end_hour:
            pass # We're in the allowed time window
        else:
            return # We're not in the allowed time window, do not enter

        # Check if we should enter a position
        if self.go_buy and not position.is_long:
            order1 = self.buy(size=0.25, sl=buy_sl, tp=buy_tp, limit=self.close)
            order2 = self.buy(size=0.25, sl=buy_sl, limit=self.close)
            self.order_tracking['order2'] = {
                "status": "OPEN", "direction": 1, "sl": buy_sl, "entry": self.close, "entry_atr": self.atr, "last_sl_move": None
            }
        elif self.go_sell and not position.is_short:
            order1 = self.sell(size=0.25, sl=sell_sl, tp=sell_tp, limit=self.close)
            order2 = self.sell(size=0.25, sl=sell_sl, limit=self.close)
            self.order_tracking['order2'] = {
                "status": "OPEN", "direction": -1, "sl": sell_sl, "entry": self.close, "entry_atr": self.atr, "last_sl_move": None
            }

def get_NNFX_combos(categories: list=None, use_default_params: bool=False) -> tuple:
    """ Generates all possible combinations of parameters for a given strategy.

    Args:
        strategy (Strategy subclass, optional): The strategy to test. Defaults to NNFX.
        categories (list, optional): A list of strings to determine which categories of indicators to test. 
        Defaults to None.
        use_default_params (bool, optional): Whether to use the default parameters for each indicator.

    Returns:
        tuple: A tuple containing the strategy and a tuple of indicator combinations.
    """
    # Determine which categories of indicators combinate for
    categories = all_categories if categories is None else categories
    all_combos = {}
    print(f"Generating combinations for {categories}...")
    # For each category included, generate all possible combinations of parameters
    for category in categories:
        all_combos[category] = []
        category_data = all_indicators[category]
        # Generate combinations for each indicator in the category
        if isinstance(category_data, list):
            pass
        else:
            for indicator_name, indicator_params in category_data.items():

                if use_default_params:
                    default_params = indicator_params["default_params"]
                    all_combos[category].append((indicator_name, default_params))
                else:
                    indicator_combos = generate_param_combinations(indicator_params)
                    for combo in indicator_combos:
                        all_combos[category].append((indicator_name, combo))

    # Add params that have lists if they are included
    if "sl_mult" in categories:
        all_combos["sl_mult"] = all_indicators["sl_mult"]
    if "tp_mult" in categories:
        all_combos["tp_mult"] = all_indicators["tp_mult"]
    if "use_main_confirmation_as_exit" in categories:
        all_combos["use_main_confirmation_as_exit"] = all_indicators["use_main_confirmation_as_exit"]
    if "start_hour" in categories:
        all_combos["start_hour"] = all_indicators["start_hour"]
    if "end_hour" in categories:
        all_combos["end_hour"] = all_indicators["end_hour"]

    # Generate dictionaries for all possible combinations of parameters
    all_params_combinations = []

    num_combos = 0
    max_combos = 9999999
    for combo in itertools.product(*all_combos.values()):
        params = {}
        for i, category in enumerate(all_combos.keys()):
            params[category] = combo[i]
        all_params_combinations.append(params)
        num_combos += 1
        print(f"Generated {num_combos} combinations", end="\r")
        if num_combos > max_combos:
            break

    # If there are an absurd number of combinations, only use a subset
    if len(all_params_combinations) > max_combos:
        print(f"Too many combinations ({len(all_params_combinations)}), only using 500,000")
        all_params_combinations = random.sample(all_params_combinations, max_combos)

    # random.shuffle(all_params_combinations)

    return all_params_combinations

# print(get_NNFX_combos(categories=['atr', 'baseline', 'main_confirmation', 'secondary_confirmation', 
#                   'volume', 'exit'], use_indicator_defaults=True))