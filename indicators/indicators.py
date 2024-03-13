import copy
import itertools
import os
import numpy as np
from typing import List
from indicators.indicator_functions import *
os.makedirs("indicators", exist_ok=True)

all_categories = ['atr', 'baseline', 'main_confirmation', 'secondary_confirmation', 'volume', 'exit']

atrs = {
    "atr": {
        "function": ATR,
        "default_params": {
            "period": 14
        },
        "param_ranges": {
            "period": list(range(10, 22, 2))
        }
    },
    "filtered_atr": {
        "function": FilteredATR,
        "default_params": {
            "period": 14,
            "ma_period": 14
        },
        "param_ranges": {
            "period": list(range(10, 22, 2)),
            "ma_period": list(range(10, 22, 2))
        }
    },
}

baselines = {
    # "sma": {
    #     "function": SMA,
    #     "default_params": {
    #         "period": 14,
    #     },
    #     "param_ranges": {
    #         "period": list(range(12, 22, 2))
    #     }
    # },
    # "ema": {
    #     "function": EMA,
    #     "default_params": {
    #         "period": 14,
    #     },
    #     "param_ranges": {
    #         "period": list(range(12, 22, 2))
    #     }
    # },
    # "wma": {
    #     "function": WMA,
    #     "default_params": {
    #         "period": 14,
    #     },
    #     "param_ranges": {
    #         "period": list(range(12, 22, 2))
    #     }
    # },
    # "hma": {
    #     "function": HMA,
    #     "default_params": {
    #         "period": 14,
    #     },
    #     "param_ranges": {
    #         "period": list(range(12, 22, 2))
    #     }
    # },
    "vidya": {
        "function": VIDYA,
        "default_params": {
            "period": 14,
            "histper": 20
        },
        "param_ranges": {
            "period": list(range(12, 22, 2)),
            "histper": list(range(15, 50, 5))
        }
    },
    "kama": {
        "function": KAMA,
        "default_params": {
            "period": 10,
            "fast": 2,
            "slow": 30
        },
        "param_ranges": {
            "period": list(range(8, 22, 2)),
            "fast": list(range(2, 10, 2)),
            "slow": list(range(20, 50, 5))
        }
    },
    # "alma": {
    #     "function": ALMA,
    #     "default_params": {
    #         "period": 9,
    #         "sigma": 6,
    #         "offset": 0.85
    #     },
    #     "param_ranges": {
    #         "period": list(range(8, 22, 2)),
    #         "sigma": list(range(4, 10, 2)),
    #         "offset": list(np.arange(0.55, 1.25, 0.1))
    #     }
    # },
    "t3": {
        "function": T3,
        "default_params": {
            "period": 14,
            "vfactor": 0.7
        },
        "param_ranges": {
            "period": list(range(12, 22, 2)),
            "vfactor": list(np.arange(0.5, 1.1, 0.1)),
        }
    },
    # "fantailvma": {
    #     "function": FantailVMA,
    #     "default_params": {
    #         "adx_length": 2,
    #         "weighting": 2.0,
    #         "ma_length": 1
    #     },
    #     "param_ranges": {
    #         "adx_length": list(range(1, 4, 1)),
    #         "weighting": list(np.arange(1.0, 3.0, 0.5)),
    #         "ma_length": list(range(1, 4, 1))
    #     }
    # },
    # "ehlers": {
    #     "function": EHLERS,
    #     "default_params": {
    #         "period": 10,
    #     },
    #     "param_ranges": {
    #         "period": list(range(8, 22, 2))
    #     },
    # },
    "mdi": {
        "function": McGinleyDI,
        "default_params": {
            "period": 12,
            "mcg_constant": 5
        },
        "param_ranges": {
            "period": list(range(12, 22, 2)),
            "mcg_constant": list(range(3, 7, 1))
        }
    },
    # "dema": {
    #     "function": DEMA,
    #     "default_params": {
    #         "period": 14,
    #     },
    #     "param_ranges": {
    #         "period": list(range(12, 22, 2))
    #     },
    # },
    "tema": {
        "function": TEMA,
        "default_params": {
            "period": 14,
        },
        "param_ranges": {
            "period": list(range(12, 22, 2))
        },
    },
    "kijunsen": {
        "function": KijunSen,
        "default_params": {
            "period": 26,
            "shift": 9
        },
        "param_ranges": {
            "period": list(range(12, 30, 2)),
            "shift": list(range(5, 13, 2))
        },
    }
}

all_confirmations = {
    "kase": {
        "function": KASE,
        "default_params": {
            "pstLength": 9,
            "pstX": 5,
            "pstSmooth": 3,
            "smoothPeriod": 10
        },
        "param_ranges": {
            "pstLength": list(range(6, 18, 3)),
            "pstX": list(range(3, 7, 1)),
            "pstSmooth": list(range(2, 5, 1)),
            "smoothPeriod": list(range(5, 15, 2))
        }
    },
    "macd_zl": {
        "function": MACDZeroLag,
        "default_params": {
            "short_period": 12,
            "long_period": 26,
            "signal_period": 9
        },
        "param_ranges": {
            "short_period": list(range(12, 22, 2)),
            "long_period": list(range(20, 50, 5)),
            "signal_period": list(range(5, 15, 2))
        }
    },
    "kalman_filter": {
        "function": KalmanFilter,
        "default_params": {
            "k": 1,
            "sharpness": 1,
        },
        "param_ranges": {
            "k": list(np.arange(0.8, 1.3, 0.1)),
            "sharpness": list(np.arange(0.8, 1.3, 0.1)),
        }
    },
    "fisher": {
        "function": Fisher,
        "default_params": {
            "range_periods": 10,
            "price_smoothing": 0.3,
            "index_smoothing": 0.3,
        },
        "param_ranges": {
            "range_periods": list(range(6, 16, 2)),
            "price_smoothing": list(np.arange(0.2, 0.6, 0.1)),
            "index_smoothing": list(np.arange(0.2, 0.6, 0.1)),
        },
    },
    "bulls_bears_impulse": {
        "function": BullsBearsImpulse,
        "default_params": {
            "ma_period": 13,
        },
        "param_ranges": {
            "ma_period": list(range(11, 23, 2)),
        },
    },
    "gen3_ma": {
        "function": Gen3MA,
        "default_params": {
            "period": 220,
            "sampling_period": 50,
        },
        "param_ranges": {
            "period": list(range(100, 300, 20)),
            "sampling_period": list(range(30, 80, 10)),
        },
    },
    "aroon": {
        "function": Aroon,
        "default_params": {
            "period": 14,
        },
        "param_ranges": {
            "period": list(range(12, 22, 2)),
        },
    },
    "coral": {
        "function": Coral,
        "default_params": {
            "period": 34,
        },
        "param_ranges": {
            "period": list(range(26, 50, 4)),
        },
    },
    "center_of_gravity": {
        "function": CenterOfGravity,
        "default_params": {
            "period": 10,
        },
        "param_ranges": {
            "period": list(range(8, 22, 2)),
        },
    },
    "grucha": {
        "function": GruchaIndex,
        "default_params": {
            "period": 10,
            "ma_period": 10,
        },
        "param_ranges": {
            "period": list(range(8, 22, 2)),
            "ma_period": list(range(8, 22, 2)),
        },
    },
    "half_trend": {
        "function": HalfTrend,
        "default_params": {
            "amplitude": 2,
        },
        "param_ranges": {
            "amplitude": list(range(1, 4, 1)),
        },
    },
    "j_tpo": {
        "function": J_TPO,
        "default_params": {
            "period": 14,
        },
        "param_ranges": {
            "period": list(range(12, 22, 2)),
        },
    },
    "kvo": {
        "function": KVO,
        "default_params": {
            "fast_ema": 34,
            "slow_ema": 55,
            "signal_ema": 13,
        },
        "param_ranges": {
            "fast_ema": list(range(26, 50, 4)),
            "slow_ema": list(range(50, 100, 5)),
            "signal_ema": list(range(12, 22, 2)),
        },
    },
    "lwpi": {
        "function": LWPI,
        "default_params": {
            "period": 8,
        },
        "param_ranges": {
            "period": list(range(6, 16, 2)),
        },
    },
    "ttf": {
        "function": TTF,
        "default_params": {
            "period": 8,
            "top_line": 75,
            "bottom_line": -75,
            "t3_period": 3,
            "b": 0.7
        },
        "param_ranges": {
            "period": list(range(6, 16, 2)),
            # "top_line": list(range(50, 100, 5)),
            # "bottom_line": list(range(-100, -50, 5)),
            "t3_period": list(range(2, 5, 1)),
            "b": list(np.arange(0.5, 1.0, 0.1)),
        },
    },
    "vortex": {
        "function": Vortex,
        "default_params": {
            "period": 14,
        },
        "param_ranges": {
            "period": list(range(12, 22, 2)),
        },
    },
    "recursive_ma": {
        "function": RecursiveMA,
        "default_params": {
            "period": 2,
            "recursions": 20
        },
        "param_ranges": {
            "period": list(range(1, 6, 1)),
            "recursions": list(range(10, 45, 5)),
        },
    },
    "schaff_trend_cycle": {
        "function": SchaffTrendCycle,
        "default_params": {
            "period": 10,
            "fast_ma_period": 23,
            "slow_ma_period": 50,
            "signal_period": 3
        },
        "param_ranges": {
            "period": list(range(8, 22, 2)),
            "fast_ma_period": list(range(10, 30, 4)),
            "slow_ma_period": list(range(40, 64, 4)),
            "signal_period": list(range(2, 6, 1)),
        },
    },
    "smooth_step": {
        "function": SmoothStep,
        "default_params": {
            "period": 32,
        },
        "param_ranges": {
            "period": list(range(28, 52, 4)),
        },
    },
    "top_trend": {
        "function": TopTrend,
        "default_params": {
            "period": 20,
            "deviation": 2,
            "money_risk": 1.00
        },
        "param_ranges": {
            "period": list(range(10, 30, 4)),
            "deviation": list(range(1, 3, 1)),
            # "money_risk": list(np.arange(0.1, 1.0, 0.1)),
        },
    },
    "trend_lord": {
        "function": TrendLord,
        "default_params": {
            "period": 12
        },
        "param_ranges": {
            "period": list(range(8, 22, 2)),
        },
    },
    "twiggs_mf": {
        "function": TwiggsMF,
        "default_params": {
            "period": 21
        },
        "param_ranges": {
            "period": list(range(15, 31, 2)),
        },
    },
    "uf2018": {
        "function": UF2018,
        "default_params": {
            "period": 20,
        },
        "param_ranges": {
            "period": list(range(16, 56, 4)),
        },
    },
}

confirmations = {
    "kase": {
        "function": KASE,
        "default_params": {
            "pstLength": 9,
            "pstX": 5,
            "pstSmooth": 3,
            "smoothPeriod": 10
        },
        "param_ranges": {
            "pstLength": list(range(6, 18, 3)),
            "pstX": list(range(3, 7, 1)),
            "pstSmooth": list(range(2, 5, 1)),
            "smoothPeriod": list(range(5, 15, 2))
        }
    },
    # "macd_zl": {
    #     "function": MACDZeroLag,
    #     "default_params": {
    #         "short_period": 12,
    #         "long_period": 26,
    #         "signal_period": 9
    #     },
    #     "param_ranges": {
    #         "short_period": list(range(12, 22, 2)),
    #         "long_period": list(range(20, 50, 5)),
    #         "signal_period": list(range(5, 15, 2))
    #     }
    # },
    "kalman_filter": {
        "function": KalmanFilter,
        "default_params": {
            "k": 1,
            "sharpness": 1,
        },
        "param_ranges": {
            "k": list(np.arange(0.8, 1.3, 0.1)),
            "sharpness": list(np.arange(0.8, 1.3, 0.1)),
        }
    },
    "gen3_ma": {
        "function": Gen3MA,
        "default_params": {
            "period": 220,
            "sampling_period": 50,
        },
        "param_ranges": {
            "period": list(range(100, 300, 20)),
            "sampling_period": list(range(30, 80, 10)),
        },
    },
    "kvo": {
        "function": KVO,
        "default_params": {
            "fast_ema": 34,
            "slow_ema": 55,
            "signal_ema": 13,
        },
        "param_ranges": {
            "fast_ema": list(range(26, 50, 4)),
            "slow_ema": list(range(50, 100, 5)),
            "signal_ema": list(range(12, 22, 2)),
        },
    },
    # "recursive_ma": {
    #     "function": RecursiveMA,
    #     "default_params": {
    #         "period": 2,
    #         "recursions": 20
    #     },
    #     "param_ranges": {
    #         "period": list(range(1, 6, 1)),
    #         "recursions": list(range(10, 45, 5)),
    #     },
    # },
    "top_trend": {
        "function": TopTrend,
        "default_params": {
            "period": 20,
            "deviation": 2,
            "money_risk": 1.00
        },
        "param_ranges": {
            "period": list(range(10, 30, 4)),
            "deviation": list(range(1, 3, 1)),
            # "money_risk": list(np.arange(0.1, 1.0, 0.1)),
        },
    },
    "twiggs_mf": {
        "function": TwiggsMF,
        "default_params": {
            "period": 21
        },
        "param_ranges": {
            "period": list(range(15, 31, 2)),
        },
    },
    "uf2018": {
        "function": UF2018,
        "default_params": {
            "period": 20,
        },
        "param_ranges": {
            "period": list(range(16, 56, 4)),
        },
    },
    # "accelerator_lsma": {
    #     "function": AcceleratorLSMA,
    #     "default_params": {
    #         "long_period": 34,
    #         "short_period": 5
    #     },
    #     "param_ranges": {
    #         "long_period": list(range(30, 40, 2)),
    #         "short_period": list(range(3, 9, 2)),
    #     }
    # },
    "ssl": {
        "function": SSL,
        "default_params": {
            "period": 10
        },
        "param_ranges": {
            "period": list(range(6, 20, 2))
        }
    },
}

exits = copy.deepcopy(confirmations)

volumes = {
    # "adx": {
    #     "function": ADX,
    #     "default_params": {
    #         "period": 14
    #     },
    #     "param_ranges": {
    #         "period": list(range(10, 30, 2))
    #     }
    # },
    "tdfi": {
        "function": TDFI,
        "default_params": {
            "period": 20
        },
        "param_ranges": {
            "period": list(range(10, 30, 2))
        }
    },
    "wae": {
        "function": WAE,
        "default_params": {
            "minutes": 0,
            "sensitivity": 150,
            "dead_zone_pip": 15,
        },
        "param_ranges": {
            "sensitivity": list(range(100, 200, 25)),
        }
    },
    # "normalized_volume": {
    #     "function": NormalizedVolume,
    #     "default_params": {
    #         "period": 14
    #     },
    #     "param_ranges": {
    #         "period": list(range(10, 34, 4))
    #     }
    # },
    # "volatility_ratio": {
    #     "function": VolatilityRatio,
    #     "default_params": {
    #         "period": 25
    #     },
    #     "param_ranges": {
    #         "period": list(range(15, 60, 5))
    #     },
    # },
}

all_indicators = {
    "atr": atrs,
    "baseline": baselines,
    "main_confirmation": confirmations,
    "secondary_confirmation": confirmations,
    "volume": volumes,
    "exit": exits,
    "sl_mult": [1.5],
    "tp_mult": [1, 1.5, 2, 2.5, 3],
    "start_hour": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "end_hour": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
}

def get_indicator(indicator_name: str) -> dict:
    """ Get the indicator with the given name.

    Args:
        indicator_name (str): The name of the indicator to get.

    Returns:
        dict: A dictionary containing the indicator's function, default parameters, and parameter ranges.
    """
    for category in all_indicators.values():
        for indicator in category.keys():
            if indicator == indicator_name:
                return category[indicator]

    raise ValueError(f"Indicator '{indicator_name}' not found.")

def generate_param_combinations(indicator: dict) -> List[dict]:
    """ Generate all possible parameter combinations for the given indicator.

    Args:
        indicator (dict): A dictionary containing the indicator's function, default parameters, and parameter ranges.

    Returns:
        List[dict]: A list of dictionaries containing all possible parameter combinations for the given indicator.
    """
    param_ranges = indicator["param_ranges"]
    combinations = list(itertools.product(*param_ranges.values()))
    param_combinations = [dict(zip(param_ranges.keys(), combination)) for combination in combinations]

    return param_combinations
