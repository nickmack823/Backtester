import numpy as np
import pandas as pd
from indicators.indicators import get_indicator
from utilities import log

def get_indicator_signals(indicator_name: str, indicator) -> dict:
    signals = {"trend": None, "signal": None}
    if not indicator_name:
        return signals
    else:
        signals["trend"], signals["signal"] = 0, 0

    if isinstance(indicator, pd.DataFrame):
        indicator = [indicator[col] for col in indicator]
    elif isinstance(indicator, pd.Series):
        indicator = [indicator]

    # FOR DEBUG
    main_ind = indicator[0][-5:].tolist()

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

class NNFXSignaler:

    def __init__(self, params):

        # Initialize params
        self.atr = params["atr"]
        self.baseline = params["baseline"]
        self.main_confirmation = params["main_confirmation"]
        self.secondary_confirmation = params["secondary_confirmation"]
        self.volume = params["volume"]
        self.exit = params["exit"]
        self.sl_mult = params["sl_mult"]
        self.tp_mult = params["tp_mult"]

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

        self.i = 0

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

        # Get indicator functions
        self.atr_func = get_indicator(self.atr_name)["function"] if self.using_atr else None
        self.baseline_func = get_indicator(self.baseline_name)["function"] if self.using_baseline else None
        self.main_confirmation_func = get_indicator(self.main_confirmation_name)["function"] if self.using_main_confirmation else None
        self.secondary_confirmation_func = get_indicator(self.secondary_confirmation_name)["function"] if self.using_secondary_confirmation else None
        self.volume_func = get_indicator(self.volume_name)["function"] if self.using_volume else None
        self.exit_func = get_indicator(self.exit_name)["function"] if self.using_exit else None

        # For indicators with more than one value, will be list of Series; else just a list of values
        self.atr_indicator = None
        self.baseline_indicator = None
        self.main_confirmation_indicator, self.secondary_confirmation_indicator = None, None
        self.volume_indicator = None
        self.exit_indicator = None

        log(f"Initialized NNFXSignaler with params: {params}")
    
    def get_atr(self) -> pd.Series:
        return self.atr_indicator

    def calculate_indicators(self):
        if self.using_atr:
            self.atr_indicator = self.atr_func(self.data, **self.atr_params)
            if isinstance(self.atr_indicator, pd.DataFrame):
                for col in self.atr_indicator.columns:
                    self.data[col] = self.atr_indicator[col]
            elif isinstance(self.atr_indicator, pd.Series):
                self.data[self.atr_name] = self.atr_indicator
        if self.using_baseline:
            self.baseline_indicator = self.baseline_func(self.data, **self.baseline_params)
            if isinstance(self.baseline_indicator, pd.DataFrame):
                for col in self.baseline_indicator.columns:
                    self.data[col] = self.baseline_indicator[col]
            elif isinstance(self.baseline_indicator, pd.Series):
                self.data[self.baseline_name] = self.baseline_indicator
        if self.using_main_confirmation:
            self.main_confirmation_indicator = self.main_confirmation_func(self.data, **self.main_confirmation_params)
            if isinstance(self.main_confirmation_indicator, pd.DataFrame):
                for col in self.main_confirmation_indicator.columns:
                    self.data[col] = self.main_confirmation_indicator[col]
            elif isinstance(self.main_confirmation_indicator, pd.Series):
                self.data[self.atr_name] = self.main_confirmation_indicator
        if self.using_secondary_confirmation:
            self.secondary_confirmation_indicator = self.secondary_confirmation_func(self.data, **self.secondary_confirmation_params)
            if isinstance(self.secondary_confirmation_indicator, pd.DataFrame):
                for col in self.secondary_confirmation_indicator.columns:
                    self.data[col] = self.secondary_confirmation_indicator[col]
            elif isinstance(self.secondary_confirmation_indicator, pd.Series):
                self.data[self.secondary_confirmation_name] = self.secondary_confirmation_indicator
        if self.using_volume:
            self.volume_indicator = self.volume_func(self.data, **self.volume_params)
            if isinstance(self.volume_indicator, pd.DataFrame):
                for col in self.volume_indicator.columns:
                    self.data[col] = self.volume_indicator[col]
            elif isinstance(self.volume_indicator, pd.Series):
                self.data[self.volume_name] = self.volume_indicator
        if self.using_exit:
            self.exit_indicator = self.exit_func(self.data, **self.exit_params)
            if isinstance(self.exit_indicator, pd.DataFrame):
                for col in self.exit_indicator.columns:
                    self.data[col] = self.exit_indicator[col]
            elif isinstance(self.exit_indicator, pd.Series):
                self.data[self.exit_name] = self.exit_indicator

    def update_data(self, data: pd.DataFrame):
        self.data = data
        self.calculate_indicators()

    def get_data(self) -> pd.DataFrame:
        return self.data

    def check_baseline(self) -> dict:
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
        
        if baseline["signal"] == 1:
            self.bars_since['baseline_cross_long'] += 1
        elif baseline["signal"] == -1:
            self.bars_since['baseline_cross_short'] += 1

        # Check if price is within 1x ATR of baseline
        if (self.close >= (curr_baseline - self.curr_atr)) and (self.close <= (curr_baseline + self.curr_atr)):
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

    def check_confirmation(self) -> dict:
        final_confirmation = {"trend": None, "signal": None}

        if not self.using_main_confirmation:
            return final_confirmation

        # Get main and secondary confirmation signals
        main_confirmation = get_indicator_signals(self.main_confirmation_name, self.main_confirmation_indicator)
        secondary_confirmation = get_indicator_signals(self.secondary_confirmation_name, self.secondary_confirmation_indicator)
        
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

        if main_signal == 1:
            self.bars_since["c1_signal_long"] = 1
        elif main_signal == -1:
            self.bars_since["c1_signal_short"] = 1

        return final_confirmation

    def check_volume(self) -> dict:
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

    def check_signal(self, direction: str) -> bool:

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

    def should_buy(self) -> bool:
        return self.check_signal("long")
        
    def should_sell(self) -> bool:
        return self.check_signal("short")

    def get_exit_signals(self) -> dict:
        exit_signals = {"exit_long": False, "exit_short": False}

        exit = get_indicator_signals(self.exit_name, self.exit_indicator)
        baseline = self.curr_baseline
        confirmation = self.curr_confirmation

        # Check baseline and exit for short signal
        if any(ind["signal"] == -1 for ind in [baseline, exit]):
            exit_signals["exit_long"] = True

        # Check if confirmation is short
        if self.use_main_confirmation_as_exit and confirmation["signal"] == -1:
            exit_signals["exit_long"] = True
            
        # Check baseline and exit for long signal
        if any(ind["signal"] == 1 for ind in [baseline, exit]):
            exit_signals["exit_short"] = True
        
        # Check if confirmation is long
        if self.use_main_confirmation_as_exit and confirmation["signal"] == 1:
            exit_signals["exit_short"] = True
            
        return exit_signals

    def get_signals(self):
        signals = {"buy": False, "sell": False}

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
        self.curr_atr = self.atr_indicator[-1]

        # Set buy and sell SL/TP
        buy_sl, sell_sl = self.close - (self.sl_mult * self.curr_atr), self.close + (self.sl_mult * self.curr_atr)
        # buy_tp, sell_tp = self.close + (self.tp_mult * self.curr_atr), self.close - (self.tp_mult * self.curr_atr)
        signals.update({"buy_sl": buy_sl, "sell_sl": sell_sl})

        # Set indicator signals/trends dictionaries (for should_buy, should_sell, and should_exit)
        self.curr_baseline = self.check_baseline()
        self.curr_confirmation = self.check_confirmation()
        self.curr_volume = self.check_volume()

        # Set trigger variables
        self.go_buy, self.go_sell = self.should_buy(), self.should_sell()

        # Check for exit signals
        signals.update(self.get_exit_signals())
        
        # Check for entry signals
        if self.go_buy:
            signals["buy"] = True
        elif self.go_sell:
            signals["sell"] = True

        return signals
