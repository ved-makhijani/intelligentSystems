# File: IS1.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Additional features added:
# - Robust data loader that supports multiple features (requirements a–e).
# - Local caching of downloads and scaler objects (no need to re-download every run).
# - NaN handling (ffill/bfill/drop/interpolate).
# - Flexible splits: ratio/date/random (we use ratio by default as you asked).
# - Optional feature scaling with re-usable scalers (saved to cache).
# - Candlestick plotting function with n-day aggregation (n ≥ 1).
# - Moving-window boxplot function for exploratory analysis.
# - (C.4) General DL model factory to build LSTM/GRU/RNN stacks from parameters.
# - (v0.5) ARIMA/SARIMA + DL ensembles with weight auto-tuning on a validation split.
# - (v0.5) Optional RandomForest baseline and 3-way ensemble.

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web  # (kept for assignment consistency though not used)
import datetime as dt
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, GRU, SimpleRNN, Bidirectional  # C.4: added
# Keras layers:
# - LSTM: Long Short-Term Memory (handles longer dependencies via gates)
# - GRU:  Gated Recurrent Unit (lighter than LSTM, often similar accuracy)
# - SimpleRNN: basic Elman RNN (fast but can struggle with long dependencies)
# - Bidirectional: wraps any recurrent layer to read sequences forward and backward

# v0.5: classical + tree baselines for ensembling
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    ARIMA = None
    SARIMAX = None

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

try:
    import mplfinance as mpf
except Exception:
    mpf = None

# ------------------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------------------
COMPANY = 'TSLA'                  # Ticker symbol to download
pred_Time = 90                    # Your comment: markets think in last quarter (90 days)
TRAIN_START = '2020-01-01'        # Start date for training range
TRAIN_END   = '2023-08-01'        # End date for training range

# ==============================================================================
# Utilities to normalize yfinance output so downstream code is predictable
# ==============================================================================

def _flatten_columns_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    If yfinance returns MultiIndex columns (common with group_by="column" or multiple tickers),
    flatten to simple strings by picking the last non-empty level for each column.
    This ensures we can refer to columns as 'Open', 'Close', etc., consistently.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for tup in df.columns:
        chosen = None
        for part in reversed(tup):
            if part and not str(part).startswith("Unnamed"):
                chosen = str(part)
                break
        if chosen is None:
            chosen = str(tup[-1])
        new_cols.append(chosen)
    df = df.copy()
    df.columns = new_cols
    return df


def _normalize_ohlcv(df: pd.DataFrame, *, ticker: str = None, require_ohlc: bool = True) -> pd.DataFrame:
    """
    Normalize to canonical OHLCV column names for downstream processing/plotting.
    """
    if isinstance(df.columns, pd.MultiIndex):
        try:
            lvl0 = df.columns.get_level_values(0)
            if len(set(lvl0)) == 1:
                df = df.droplevel(0, axis=1)
            else:
                sym = ticker if (ticker is not None and ticker in set(lvl0)) else list(dict.fromkeys(lvl0))[0]
                df = df.xs(sym, axis=1, level=0)
        except Exception:
            df = _flatten_columns_to_strings(df)
    df = _flatten_columns_to_strings(df)

    def canon(name: str) -> str:
        return re.sub(r'[^a-z]', '', str(name).lower())

    aliases = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
        "adjustedclose": "Adj Close",
        "volume": "Volume",
    }

    mapped = {}
    for col in df.columns:
        c = canon(col)
        for key, target in aliases.items():
            if c.startswith(key):
                mapped[col] = target
                break
    df = df.rename(columns=mapped)

    casing_fix = {c: c.title() for c in df.columns}
    casing_fix.update({"Adj close": "Adj Close"})
    df = df.rename(columns=casing_fix)

    for col in df.columns:
        if df[col].dtype == "object":
            s = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
            if s.notna().mean() > 0.5:
                df[col] = s

    if "Close" not in df.columns and len(df.columns) == 1:
        only_col = df.columns[0]
        df = df.rename(columns={only_col: "Close"})

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.copy()
        df["Close"] = df["Adj Close"]

    if not require_ohlc and "Close" in df.columns:
        c = df["Close"]
        if "Open" not in df.columns: df["Open"] = c
        if "High" not in df.columns: df["High"] = c
        if "Low"  not in df.columns: df["Low"]  = c

    if require_ohlc:
        needed = ["Open", "High", "Low", "Close"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns after normalization: {missing}. Available: {list(df.columns)}")

    return df


# ==============================================================================
# Load & process multi-feature data (assignment requirements a–e)
# ==============================================================================
def load_process_multifeature(
    ticker,
    start_date,
    end_date,
    *,
    feature_columns=None,
    target_column="Close",
    lookback=60,
    handle_nan="ffill",
    split_method="ratio",
    train_ratio=0.8,
    split_date=None,
    scale_features=True,
    scaler_type="minmax",
    scale_range=(0.0, 1.0),
    cache_dir="data_cache",
    use_cache=True,
    save_cache=True,
    random_state=42,
):
    """
    (a–e) Central function for loading, cleaning, scaling, and splitting data.
    """
    import pickle
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{ticker}_{start_date}_{end_date}".replace("/", "-").replace(" ", "_")
    csv_path = os.path.join(cache_dir, f"{cache_key}.csv")
    scalers_path = os.path.join(cache_dir, f"{cache_key}_scalers.pkl")

    def _read_cached_csv(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        except Exception:
            df_tmp = pd.read_csv(path)
            for candidate in ("Date", "date", "Datetime", "datetime", "Unnamed: 0"):
                if candidate in df_tmp.columns:
                    df_tmp[candidate] = pd.to_datetime(df_tmp[candidate], errors="coerce")
                    df_tmp = df_tmp.set_index(candidate)
                    break
            if not isinstance(df_tmp.index, pd.DatetimeIndex):
                raise ValueError("Cached CSV missing a Date index; re-download required.")
            df_tmp.index.name = "Date"
            return df_tmp

    if use_cache and os.path.isfile(csv_path):
        try:
            df = _read_cached_csv(csv_path)
        except Exception:
            df = yf.download(ticker, start=start_date, end=end_date,
                             auto_adjust=False, group_by="column")
            df = _normalize_ohlcv(df, require_ohlc=False)
            df.index.name = "Date"
            if save_cache:
                df.to_csv(csv_path)
    else:
        df = yf.download(ticker, start=start_date, end=end_date,
                         auto_adjust=False, group_by="column")
        df = _normalize_ohlcv(df, require_ohlc=False)
        df.index.name = "Date"
        if save_cache:
            df.to_csv(csv_path)

    if ("Close" not in df.columns) and (len(df.columns) == 1):
        df = df.rename(columns={df.columns[0]: "Close"})
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"]  = df["Close"]

    if feature_columns is None:
        feature_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    feature_columns = [c for c in feature_columns if c in df.columns]
    if not feature_columns:
        if "Close" in df.columns:
            feature_columns = ["Close"]
        else:
            raise ValueError(f"No valid feature columns found in the dataframe. Available: {list(df.columns)}")

    if target_column not in df.columns:
        if target_column != "Close" and "Close" in df.columns:
            target_column = "Close"
        else:
            raise ValueError(f"target_column '{target_column}' not found. Available: {list(df.columns)}")

    if handle_nan == "ffill":
        df = df.ffill()
    elif handle_nan == "bfill":
        df = df.bfill()
    elif handle_nan == "drop":
        df = df.dropna()
    elif handle_nan == "interpolate":
        df = df.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError("handle_nan must be 'ffill','bfill','drop','interpolate'")

    if len(df) <= lookback + 1:
        raise ValueError("Not enough rows after NaN handling to form sequences.")

    def make_scaler():
        if scaler_type.lower() == "minmax":
            return MinMaxScaler(feature_range=scale_range)
        elif scaler_type.lower() == "standard":
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")

    if split_method not in ("date", "ratio", "random"):
        raise ValueError("split_method must be 'ratio', 'date', or 'random'")

    if split_method == "date":
        if split_date is None:
            split_idx = int(len(df) * train_ratio)
            split_dt = df.index[split_idx]
        else:
            split_dt = pd.to_datetime(split_date)
        train_rows_mask = df.index <= split_dt
    else:
        split_idx = int(len(df) * train_ratio)
        train_rows_mask = pd.Series(False, index=df.index)
        train_rows_mask.iloc[:split_idx] = True

    scalers = {
        "features": {},
        "target": None,
        "meta": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "lookback": lookback,
            "split_method": split_method,
            "train_ratio": train_ratio,
            "split_date": (split_date if split_method == "date" else None),
            "scaler_type": scaler_type,
            "scale_range": scale_range,
        },
    }

    if scale_features:
        df_scaled = pd.DataFrame(index=df.index)
        for col in feature_columns:
            sc = make_scaler()
            sc.fit(df.loc[train_rows_mask, col].values.reshape(-1, 1))
            df_scaled[col] = sc.transform(df[col].values.reshape(-1, 1)).ravel()
            scalers["features"][col] = sc

        tgt_scaler = make_scaler()
        tgt_scaler.fit(df.loc[train_rows_mask, target_column].values.reshape(-1, 1))
        df_scaled[target_column + "__TGT"] = tgt_scaler.transform(
            df[target_column].values.reshape(-1, 1)
        ).ravel()
        scalers["target"] = tgt_scaler
    else:
        df_scaled = df.copy()
        for col in feature_columns:
            scalers["features"][col] = None
        scalers["target"] = None
        df_scaled[target_column + "__TGT"] = df[target_column].values

    feat_arr  = df_scaled[feature_columns].values
    tgt_arr   = df_scaled[target_column + "__TGT"].values
    X_all, y_all, idx_all = [], [], []
    for t in range(lookback, len(df_scaled)):
        X_all.append(feat_arr[t - lookback:t, :])
        y_all.append(tgt_arr[t])
        idx_all.append(t)
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.float32)
    idx_all = np.asarray(idx_all, dtype=np.int64)
    seq_times = df_scaled.index[idx_all]

    if split_method == "date":
        train_mask_seq = seq_times <= split_dt
        X_train, y_train = X_all[train_mask_seq], y_all[train_mask_seq]
        X_test,  y_test  = X_all[~train_mask_seq], y_all[~train_mask_seq]
    elif split_method == "ratio":
        cut = int(len(X_all) * train_ratio)
        X_train, y_train = X_all[:cut], y_all[:cut]
        X_test,  y_test  = X_all[cut:], y_all[cut:]
    else:  # 'random'
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X_all))
        cut = int(len(X_all) * train_ratio)
        train_idx = perm[:cut]
        test_idx  = perm[cut:]
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

    return X_train, y_train, X_test, y_test, df, scalers

def load_process_multifeature_v04(
    ticker, start_date, end_date, *,
    feature_columns=("Open","High","Low","Close","Adj Close","Volume"),
    target_column="Close",
    lookback=60, horizon=1, start_offset=0,
    handle_nan="ffill", split_method="ratio", train_ratio=0.8, split_date=None,
    scale_features=True, scaler_type="minmax", scale_range=(0.0,1.0),
    cache_dir="data_cache", use_cache=True, save_cache=True, random_state=42,
):
    """
    Multistep wrapper over load_process_multifeature:
    returns Y with shape (samples, horizon) starting at t+start_offset.
    """
    import numpy as np
    import pandas as pd

    # Reuse your robust single-step pipeline to get df and scalers
    Xtr, ytr, Xte, yte, df, scalers = load_process_multifeature(
        ticker=ticker, start_date=start_date, end_date=end_date,
        feature_columns=list(feature_columns), target_column=target_column,
        lookback=lookback, handle_nan=handle_nan,
        split_method=split_method, train_ratio=train_ratio, split_date=split_date,
        scale_features=scale_features, scaler_type=scaler_type, scale_range=scale_range,
        cache_dir=cache_dir, use_cache=use_cache, save_cache=save_cache,
        random_state=random_state,
    )

    # Rebuild the scaled frame exactly as your loader did
    df_scaled = pd.DataFrame(index=df.index)
    feat_cols = scalers["meta"]["feature_columns"]
    for col in feat_cols:
        sc = scalers["features"][col]
        if sc is None:
            df_scaled[col] = df[col].values
        else:
            df_scaled[col] = sc.transform(df[col].values.reshape(-1,1)).ravel()
    tgt_col = scalers["meta"]["target_column"]
    tgt_scaler = scalers["target"]
    if tgt_scaler is None:
        df_scaled[tgt_col + "__TGT"] = df[tgt_col].values
    else:
        df_scaled[tgt_col + "__TGT"] = tgt_scaler.transform(df[tgt_col].values.reshape(-1,1)).ravel()

    # Build multistep sequences
    feat_arr = df_scaled[feat_cols].values.astype(np.float32)
    tgt_arr  = df_scaled[tgt_col + "__TGT"].values.astype(np.float32)

    def _make_sequences_multistep(feat, tgt, lookback, horizon, start_offset):
        T = len(tgt)
        last_t = T - (start_offset + horizon)
        if last_t < lookback:
            raise ValueError("Not enough rows for given lookback/horizon/start_offset")
        X, Y, idxs = [], [], []
        for t in range(lookback, last_t + 1):
            X.append(feat[t - lookback:t, :])
            Y.append(tgt[t + start_offset : t + start_offset + horizon])
            idxs.append(t)
        return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(idxs, np.int64)

    X_all, Y_all, idx_all = _make_sequences_multistep(
        feat_arr, tgt_arr, lookback=lookback, horizon=horizon, start_offset=start_offset
    )
    seq_times = df_scaled.index[idx_all]

    # Split on sequence indices/times (consistent with your split options)
    if split_method == "date":
        if split_date is None:
            cut = int(len(seq_times) * train_ratio)
            split_dt = seq_times[cut]
        else:
            split_dt = pd.to_datetime(split_date)
        mask = seq_times <= split_dt
        X_train, Y_train = X_all[mask], Y_all[mask]
        X_test,  Y_test  = X_all[~mask], Y_all[~mask]
    elif split_method == "ratio":
        cut = int(len(X_all) * train_ratio)
        X_train, Y_train = X_all[:cut], Y_all[:cut]
        X_test,  Y_test  = X_all[cut:], Y_all[cut:]
    else:  # random
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X_all))
        cut = int(len(X_all) * train_ratio)
        tr, te = perm[:cut], perm[cut:]
        X_train, Y_train = X_all[tr], Y_all[tr]
        X_test,  Y_test  = X_all[te], Y_all[te]

    return X_train, Y_train, X_test, Y_test, df, scalers


# ==============================================================================
# Candlestick plotting (n-day aggregation)
# ==============================================================================

def aggregate_ohlc_by_n_days(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """
    Combine consecutive trading days into n-day candles (Open/High/Low/Close/Volume).
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    base = _normalize_ohlcv(df, require_ohlc=True).sort_index()
    block_id = np.arange(len(base)) // n
    grp = base.groupby(block_id)

    agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in base.columns:
        agg_dict["Volume"] = "sum"

    df_agg = grp.agg(agg_dict)
    last_dates = base.index.to_series().groupby(block_id).last()
    df_agg.index = pd.DatetimeIndex(last_dates.values, name=base.index.name)
    return df_agg


def plot_candlestick(
    df: pd.DataFrame,
    n: int = 1,
    title: str = None,
    volume: bool = True,
    mav=(20,),
    style: str = "yahoo",
    figsize=(12, 6),
    save_path: str = None,
):
    """
    Plot candlesticks using mplfinance, with optional n-day aggregation.
    """
    if mpf is None:
        raise ImportError("mplfinance is required for candlestick charts. Install: pip install mplfinance")

    try:
        base = _normalize_ohlcv(df, require_ohlc=True)
    except Exception:
        base = _normalize_ohlcv(df, require_ohlc=False)
        if not all(k in base.columns for k in ["Open", "High", "Low", "Close"]):
            if "Close" not in base.columns:
                raise
            c = base["Close"]
            if "Open" not in base.columns: base["Open"] = c
            if "High" not in base.columns: base["High"] = c
            if "Low"  not in base.columns: base["Low"]  = c

    df_use = aggregate_ohlc_by_n_days(base, n=n) if n > 1 else base.copy()

    mpf.plot(
        df_use,
        type="candle",
        volume=volume and ("Volume" in df_use.columns),
        mav=mav,
        style=style,
        figsize=figsize,
        title=title or f"Candlestick ({n}-day candles)",
        savefig=save_path
    )


# ==============================================================================
# Moving-window boxplots
# ==============================================================================

def plot_boxplot_moving_window(
    df: pd.DataFrame,
    column: str = "Close",
    window: int = 20,
    step: int = 1,
    showfliers: bool = False,
    title: str = None,
    figsize=(12, 6),
    save_path: str = None,
):
    """
    Show distribution of a price series over rolling windows as a sequence of boxplots.
    """
    if window < 1 or step < 1:
        raise ValueError("`window` and `step` must be >= 1.")
    df = _normalize_ohlcv(df, require_ohlc=False)

    column_norm = "Adj Close" if column.lower().replace(" ", "") == "adjclose" else column.title()
    if column_norm not in df.columns:
        raise ValueError(f"Column '{column}' not found; available: {list(df.columns)}")

    s = df.sort_index()[column_norm].dropna()
    if len(s) < window:
        raise ValueError("Not enough rows to form at least one window.")

    data_windows, end_labels = [], []
    for start in range(0, len(s) - window + 1, step):
        segment = s.iloc[start:start + window]
        data_windows.append(segment.values)
        end_labels.append(segment.index[-1].strftime("%Y-%m-%d"))

    if not data_windows:
        raise ValueError("Parameters produced zero windows. Check `window` and `step`.")

    plt.figure(figsize=figsize)
    bp = plt.boxplot(data_windows, patch_artist=True, showfliers=showfliers)
    for box in bp['boxes']:
        box.set_alpha(0.6)

    positions = np.arange(1, len(data_windows) + 1)
    label_every = max(1, len(end_labels) // 10)
    sel = positions[::label_every]
    sel_labels = [end_labels[i - 1] for i in sel]

    plt.xticks(sel, sel_labels, rotation=45, ha="right")
    plt.xlabel(f"Window end date (window={window}, step={step})")
    plt.ylabel(column_norm)
    plt.title(title or f"Moving-Window Boxplots of {column_norm}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ------------------------------------------------------------------------------
# original training pipeline 
# ------------------------------------------------------------------------------

def loadData(COMPANY,pred_Time,scale = True , Shuffle= True,lookup_Step=3,split_By_Date=True,test_Size=0.3, feature_Columns=['adjclose','volume','open','high','low']):
    """
    Your original function name and signature are kept for assignment consistency.
    We only replaced the inner data-prep section with a call to `load_process_multifeature`
    so we satisfy a–e without changing your downstream training/testing/plotting flow.
    """
    data = yf.download(COMPANY, TRAIN_START, TRAIN_END, auto_adjust=False, group_by="column")
    data = _normalize_ohlcv(data, require_ohlc=False)

    PRICE_VALUE = "Close"

    PREDICTION_DAYS = 60
    X_train, y_train, X_test, y_test, raw_df, scalers = load_process_multifeature(
        ticker=COMPANY,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        feature_columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        target_column=PRICE_VALUE,
        lookback=PREDICTION_DAYS,
        handle_nan="ffill",
        split_method="ratio",
        train_ratio=1.0 - test_Size,
        scale_features=True,
        scaler_type="minmax",
        cache_dir="data_cache",
        use_cache=True,
        save_cache=True,
    )
    x_train, y_train = X_train, y_train
    scaler = scalers["target"]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=25, batch_size=32)

    TEST_START = '2023-08-02'
    TEST_END   = '2024-07-02'

    test_data = yf.download(COMPANY, TEST_START, TEST_END, auto_adjust=False, group_by="column")
    test_data = _normalize_ohlcv(test_data, require_ohlc=False)

    actual_prices = test_data[PRICE_VALUE].values

    total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png", dpi=150)
    plt.show()

    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")


# ==============================================================================
# C.4 — General DL model factory (LSTM / GRU / RNN)
# ==============================================================================

LAYER_MAP = {
    "lstm": LSTM,
    "gru": GRU,
    "rnn": SimpleRNN,
}

def build_dl_model(
    layer_type: str,
    layer_sizes: list[int],
    input_timesteps: int,
    input_features: int = 1,
    *,
    dropout: float = 0.2,
    bidirectional: bool = False,
    dense_units: int = 1,
    optimizer: str = "adam",
    loss: str = "mean_squared_error",
) -> Sequential:
    """
    Construct and compile a deep sequence model for univariate or multivariate forecasting.
    """
    lt = (layer_type or "").strip().lower()
    if lt not in LAYER_MAP:
        raise ValueError("layer_type must be one of: 'lstm', 'gru', 'rnn'")
    RNN = LAYER_MAP[lt]

    model = Sequential()

    for i, units in enumerate(layer_sizes):
        is_last = (i == len(layer_sizes) - 1)
        return_sequences = not is_last
        core = RNN(units, return_sequences=return_sequences)
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(core, input_shape=(input_timesteps, input_features)))
            else:
                model.add(RNN(units, return_sequences=return_sequences,
                              input_shape=(input_timesteps, input_features)))
        else:
            if bidirectional:
                model.add(Bidirectional(core))
            else:
                model.add(core)
        model.add(Dropout(dropout))

    model.add(Dense(dense_units))
    model.compile(optimizer=optimizer, loss=loss)
    return model

# ========================== EXPERIMENT HARNESS (v0.3) ==========================
import math, time, json
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras import backend as K

def _metrics(y_true, y_pred):
    """Compute common regression metrics on numpy arrays (original price scale)."""
    y_true = y_true.astype(float).ravel()
    y_pred = y_pred.astype(float).ravel()
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

def _train_val_split_time_series(X, y, val_ratio=0.1):
    """Chronological split: last val_ratio of TRAIN becomes validation."""
    n = len(X)
    cut = max(1, int(n * (1 - val_ratio)))
    return (X[:cut], y[:cut], X[cut:], y[cut:])

def run_experiments_v03(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    lookback: int = 60,
    feature_columns = ("Close",),
    target_column: str = "Close",
    split_method: str = "ratio",
    train_ratio: float = 0.8,
    handle_nan: str = "ffill",
    scaler_type: str = "minmax",
    scale_range = (0, 1),
    cache_dir: str = "data_cache",
    results_csv: str = "experiments_v03.csv",
    val_ratio: float = 0.1,
    seed: int = 42,
    configs: list | None = None,
    verbose: int = 0,
):
    """
    Run multiple DL model configs built via build_dl_model(...) and log results.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, raw_df, scalers = load_process_multifeature(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_columns=list(feature_columns),
        target_column=target_column,
        lookback=lookback,
        handle_nan=handle_nan,
        split_method=split_method,
        train_ratio=train_ratio,
        scale_features=True,
        scaler_type=scaler_type,
        scale_range=scale_range,
        cache_dir=cache_dir,
        use_cache=True,
        save_cache=True,
        random_state=seed,
    )

    X_tr, y_tr, X_val, y_val = _train_val_split_time_series(X_train, y_train, val_ratio=val_ratio)

    target_scaler = scalers["target"]
    n_features = X_train.shape[2]

    if not configs:
        configs = [
            {"tag": "LSTM_50x3", "layer_type": "lstm", "layer_sizes": [50, 50, 50], "dropout": 0.2, "bidirectional": False, "optimizer": "adam", "loss": "mean_squared_error", "epochs": 25, "batch_size": 32},
            {"tag": "GRU_64x2",  "layer_type": "gru",  "layer_sizes": [64, 64],     "dropout": 0.2, "bidirectional": False, "optimizer": "adam", "loss": "mean_squared_error", "epochs": 25, "batch_size": 32},
            {"tag": "RNN_64x3",  "layer_type": "rnn",  "layer_sizes": [64, 64, 32], "dropout": 0.2, "bidirectional": False, "optimizer": "adam", "loss": "mean_squared_error", "epochs": 25, "batch_size": 64},
            {"tag": "LSTM_bi",   "layer_type": "lstm", "layer_sizes": [64, 64],     "dropout": 0.2, "bidirectional": True,  "optimizer": "adam", "loss": "mean_squared_error", "epochs": 25, "batch_size": 32},
            {"tag": "GRU_deep",  "layer_type": "gru",  "layer_sizes": [128, 128, 64],"dropout": 0.3, "bidirectional": False, "optimizer": "adam", "loss": "mean_squared_error", "epochs": 30, "batch_size": 32},
        ]

    rows = []
    for cfg in configs:
        model = build_dl_model(
            layer_type   = cfg.get("layer_type", "lstm"),
            layer_sizes  = cfg.get("layer_sizes", [50, 50, 50]),
            input_timesteps = lookback,
            input_features  = n_features,
            dropout      = cfg.get("dropout", 0.2),
            bidirectional= cfg.get("bidirectional", False),
            dense_units  = 1,
            optimizer    = cfg.get("optimizer", "adam"),
            loss         = cfg.get("loss", "mean_squared_error"),
        )

        tf.random.set_seed(seed)
        np.random.seed(seed)

        start = time.perf_counter()
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg.get("epochs", 25),
            batch_size=cfg.get("batch_size", 32),
            verbose=verbose,
        )
        train_time = time.perf_counter() - start

        y_val_pred_scaled = model.predict(X_val, verbose=0)
        y_tst_pred_scaled = model.predict(X_test, verbose=0)

        if target_scaler is not None:
            y_val_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
            y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled).ravel()
            y_tst_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
            y_tst_pred = target_scaler.inverse_transform(y_tst_pred_scaled).ravel()
        else:
            y_val_true, y_val_pred = y_val, y_val_pred_scaled.ravel()
            y_tst_true, y_tst_pred = y_test, y_tst_pred_scaled.ravel()

        val_metrics = _metrics(y_val_true, y_val_pred)
        tst_metrics = _metrics(y_tst_true, y_tst_pred)

        params = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
        row = {
            "tag": cfg.get("tag", ""),
            "layer_type": cfg.get("layer_type", "lstm"),
            "layer_sizes": json.dumps(cfg.get("layer_sizes", [50,50,50])),
            "bidirectional": bool(cfg.get("bidirectional", False)),
            "dropout": float(cfg.get("dropout", 0.2)),
            "optimizer": cfg.get("optimizer", "adam"),
            "loss": cfg.get("loss", "mean_squared_error"),
            "epochs": int(cfg.get("epochs", 25)),
            "batch_size": int(cfg.get("batch_size", 32)),
            "train_time_sec": round(train_time, 3),
            "params": params,
            "val_mse":  round(val_metrics["mse"], 6),
            "val_rmse": round(val_metrics["rmse"], 6),
            "val_mae":  round(val_metrics["mae"], 6),
            "val_mape": round(val_metrics["mape"], 4),
            "test_mse":  round(tst_metrics["mse"], 6),
            "test_rmse": round(tst_metrics["rmse"], 6),
            "test_mae":  round(tst_metrics["mae"], 6),
            "test_mape": round(tst_metrics["mape"], 4),
        }
        rows.append(row)

        del model
        K.clear_session()

    df_res = pd.DataFrame(rows).sort_values(["test_rmse", "val_rmse"])
    df_res.to_csv(results_csv, index=False)
    print(f"[experiments] Wrote {results_csv}")
    print(df_res.head(10).to_string(index=False))

    return df_res
# ======================== END EXPERIMENT HARNESS (v0.3) =======================
# =============================== v0.4 — Additive Multistep & Multivariate ===============================


def _inverse_multistep(y_scaled: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform a (N, H) block with your 1D target scaler."""
    if scaler is None:
        return y_scaled
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)

def train_univariate_multistep(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    lookback: int = 60,
    horizon: int = 5,
    start_offset: int = 0,
    epochs: int = 25,
    batch_size: int = 32,
    layer_type: str = "lstm",
    layer_sizes = (64,64),
    dropout: float = 0.2,
    bidirectional: bool = False,
):
    X_tr, Y_tr, X_te, Y_te, df, scalers = load_process_multifeature_v04(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_columns=("Close",),      # univariate
        target_column="Close",
        lookback=lookback,
        horizon=horizon,
        start_offset=start_offset,
        split_method="ratio",
        train_ratio=0.8,
        handle_nan="ffill",
        scaler_type="minmax",
        cache_dir="data_cache",
    )
    model = build_dl_model(
        layer_type=layer_type,
        layer_sizes=list(layer_sizes),
        input_timesteps=lookback,
        input_features=1,
        dropout=dropout,
        bidirectional=bidirectional,
        dense_units=horizon,             # multi-output head
        optimizer="adam",
        loss="mean_squared_error",
    )
    model.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    # Evaluate (report step-1; plot all k)
    Yp_tr = model.predict(X_tr, verbose=0); Yp_te = model.predict(X_te, verbose=0)
    ytr   = _inverse_multistep(Y_tr, scalers["target"]);  yte   = _inverse_multistep(Y_te, scalers["target"])
    yptr  = _inverse_multistep(Yp_tr, scalers["target"]); ypte  = _inverse_multistep(Yp_te, scalers["target"])
    met = _metrics(yte[:,0], ypte[:,0])   # uses your existing _metrics
    print(f"[Univariate k={horizon}] Test step-1 RMSE={met['rmse']:.4f} MAE={met['mae']:.4f} MAPE={met['mape']:.2f}%")
    _plot_multistep_example(yte, ypte, horizon, title=f"{ticker} Univariate {horizon}-step Forecast")
    return model, (yte, ypte)


# --- Task 2: Multivariate single-step (OHLCV → next-day Close) --------------------------------------
def train_multivariate_single_step(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    lookback: int = 60,
    epochs: int = 25,
    batch_size: int = 32,
    layer_type: str = "lstm",
    layer_sizes = (64,64),
    dropout: float = 0.2,
    bidirectional: bool = False,
    feature_columns=("Open","High","Low","Close","Adj Close","Volume"),
):
    X_tr, y_tr, X_te, y_te, df, scalers = load_process_multifeature(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_columns=list(feature_columns),
        target_column="Close",
        lookback=lookback,
        handle_nan="ffill",
        split_method="ratio",
        train_ratio=0.8,
        scale_features=True,
        scaler_type="minmax",
        cache_dir="data_cache",
        use_cache=True,
        save_cache=True,
    )
    n_features = X_tr.shape[2]
    model = build_dl_model(
        layer_type=layer_type,
        layer_sizes=list(layer_sizes),
        input_timesteps=lookback,
        input_features=n_features,
        dropout=dropout,
        bidirectional=bidirectional,
        dense_units=1,
        optimizer="adam",
        loss="mean_squared_error",
    )
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    y_pred = model.predict(X_te, verbose=0).reshape(-1, 1)

    if scalers["target"] is not None:
        y_true = scalers["target"].inverse_transform(y_te.reshape(-1, 1)).ravel()
        y_pred_inv = scalers["target"].inverse_transform(y_pred).ravel()
    else:
        y_true = y_te.ravel()
        y_pred_inv = y_pred.ravel()

    met = _metrics(y_true, y_pred_inv)
    print(f"[Multivariate 1-step] Test RMSE={met['rmse']:.4f} MAE={met['mae']:.4f} MAPE={met['mape']:.2f}%")

    return model, (y_true, y_pred_inv)


# --- Task 3: Multivariate + multistep (OHLCV → next k Close) ----------------------------------------
def train_multivariate_multistep(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    lookback: int = 60,
    horizon: int = 5,
    start_offset: int = 0,
    epochs: int = 25,
    batch_size: int = 32,
    layer_type: str = "lstm",
    layer_sizes = (128,128),
    dropout: float = 0.2,
    bidirectional: bool = False,
    feature_columns=("Open","High","Low","Close","Adj Close","Volume"),
):
    X_tr, Y_tr, X_te, Y_te, df, scalers = load_process_multifeature_v04(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_columns=list(feature_columns),
        target_column="Close",
        lookback=lookback,
        horizon=horizon,
        start_offset=start_offset,
        split_method="ratio",
        train_ratio=0.8,
        handle_nan="ffill",
        scaler_type="minmax",
        cache_dir="data_cache",
    )
    n_features = X_tr.shape[2]
    model = build_dl_model(
        layer_type=layer_type,
        layer_sizes=list(layer_sizes),
        input_timesteps=lookback,
        input_features=n_features,
        dropout=dropout,
        bidirectional=bidirectional,
        dense_units=horizon,
        optimizer="adam",
        loss="mean_squared_error",
    )
    model.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    Yp_te = model.predict(X_te, verbose=0)
    Y_true = _inverse_multistep(Y_te, scalers["target"])
    Y_pred = _inverse_multistep(Yp_te, scalers["target"])
    met_1 = _metrics(Y_true[:,0], Y_pred[:,0])
    met_k = _metrics(Y_true[:,-1], Y_pred[:,-1])
    print(f"[Multivariate {horizon}-step] step-1 RMSE={met_1['rmse']:.4f} | step-{horizon} RMSE={met_k['rmse']:.4f}")
    _plot_multistep_example(Y_true, Y_pred, horizon, title=f"{ticker} Multivariate {horizon}-step Forecast")
    return model, (Y_true, Y_pred)

# --- (Optional) example calls to drop at the bottom of __main__ (kept commented) ---------------------
# model_u, _  = train_univariate_multistep(COMPANY, TRAIN_START, TRAIN_END, lookback=60, horizon=5, epochs=20)
# model_m1    = train_multivariate_single_step(COMPANY, TRAIN_START, TRAIN_END, lookback=60, epochs=20)
# model_mm, _ = train_multivariate_multistep(COMPANY, TRAIN_START, TRAIN_END, lookback=60, horizon=5, epochs=20)
# =====================================================================================================


def save_step_series_plot(y_true_2d, y_pred_2d, *, step:int, ticker:str, tag:str, save_path:str):
    step_idx = step - 1
    y_true_1d = np.asarray(y_true_2d)[:, step_idx].ravel()
    y_pred_1d = np.asarray(y_pred_2d)[:, step_idx].ravel()

    plt.figure(figsize=(18, 5))
    plt.plot(y_true_1d, label=f"Actual (t+{step})")
    plt.plot(y_pred_1d, label=f"Predicted (t+{step})")
    plt.title(f"{ticker} {tag}: Step-{step} Actual vs Predicted")
    plt.xlabel("Test sample index"); plt.ylabel("Price"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def _plot_multistep_example(y_true: np.ndarray, y_pred: np.ndarray, k: int, title: str):
    """Quick look: first sample's k-step truth vs prediction."""
    plt.figure(figsize=(8,4))
    plt.plot(range(1,k+1), y_true[0], label="Actual")
    plt.plot(range(1,k+1), y_pred[0], label="Predicted")
    plt.xlabel("Steps ahead"); plt.ylabel("Price"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig("multistep_example.png", dpi=150); plt.show()


# =============================== v0.5 — ENSEMBLES ===============================
def _series_for_sequences_alignment(df: pd.DataFrame, target_col: str, lookback: int) -> np.ndarray:
    """
    For 1-step setups, our supervised targets start at index `lookback`.
    This returns the target series aligned to those targets: s = df[target][lookback:].
    """
    s = df[target_col].values.astype(float)
    if len(s) <= lookback:
        raise ValueError("Not enough rows to align series with lookback.")
    return s[lookback:]  # len = total_targets (y_all)

def _fit_predict_arima_like(
    y: np.ndarray,
    *,
    train_points: int,
    val_points: int,
    test_points: int,
    arima_order=(5,1,0),
    seasonal_order=None,
):
    """
    Fit ARIMA/SARIMA on:
      - validation model: first (train_points - val_points) points -> predict next val_points
      - test model: first train_points points -> predict next test_points
    Returns:
      y_val_pred, y_tst_pred  (both in original scale; y itself passed in original scale)
    """
    if ARIMA is None and SARIMAX is None:
        raise ImportError("statsmodels is required for ARIMA/SARIMA. Install: pip install statsmodels")

    core_points = train_points - val_points
    if core_points < 5:
        raise ValueError("Too few points for ARIMA validation fit; increase training size or lower val_ratio.")

    # Choose model class
    use_sarima = seasonal_order is not None
    Model = SARIMAX if use_sarima else ARIMA

    # VAL model
    if use_sarima:
        m_val = Model(endog=y[:core_points], order=arima_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    else:
        m_val = Model(endog=y[:core_points], order=arima_order)
    res_val = m_val.fit(method_kwargs={"warn_convergence": False})
    pred_val = res_val.get_forecast(steps=val_points).predicted_mean
    y_val_pred = np.asarray(pred_val, dtype=float)

    # TEST model
    if use_sarima:
        m_tst = Model(endog=y[:train_points], order=arima_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    else:
        m_tst = Model(endog=y[:train_points], order=arima_order)
    res_tst = m_tst.fit(method_kwargs={"warn_convergence": False})
    pred_tst = res_tst.get_forecast(steps=test_points).predicted_mean
    y_tst_pred = np.asarray(pred_tst, dtype=float)

    return y_val_pred, y_tst_pred

def _weighted_blend(p1: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    """w * p1 + (1 - w) * p2"""
    return (w * p1) + ((1.0 - w) * p2)

def _grid_search_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, n=21) -> float:
    """Brute-force weight in [0,1] to minimize RMSE on validation."""
    best_w, best_rmse = 0.5, 1e18
    for k in range(n):
        w = k / (n - 1)
        blend = _weighted_blend(p1, p2, w)
        rmse = _metrics(y_true, blend)["rmse"]
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    return best_w

def _rf_baseline_preds(X_tr, y_tr, X_val, X_tst, *, target_scaler):
    """
    Optional tree baseline for Task 2 experimentation.
    Flattens time windows and trains a RandomForest on tabular features.
    """
    if RandomForestRegressor is None:
        raise ImportError("scikit-learn is required for RandomForestRegressor. Install: pip install scikit-learn")

    n_tr, t, f = X_tr.shape
    n_val = X_val.shape[0]
    n_ts  = X_tst.shape[0]

    Xtr_flat = X_tr.reshape(n_tr, t * f)
    Xval_flat= X_val.reshape(n_val, t * f)
    Xts_flat = X_tst.reshape(n_ts, t * f)

    # y in original scale
    y_tr_orig = target_scaler.inverse_transform(y_tr.reshape(-1,1)).ravel() if target_scaler is not None else y_tr.ravel()

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(Xtr_flat, y_tr_orig)

    p_val = rf.predict(Xval_flat)
    p_tst = rf.predict(Xts_flat)
    return p_val, p_tst

def run_ensemble_lstm_arima_single_step(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    lookback: int = 60,
    feature_columns=("Open","High","Low","Close","Adj Close","Volume"),
    target_column="Close",
    split_method="ratio",
    train_ratio=0.8,
    val_ratio=0.1,
    handle_nan="ffill",
    scaler_type="minmax",
    cache_dir="data_cache",
    # DL params
    dl_layer_type="lstm",
    dl_layer_sizes=(128,128),
    dl_dropout=0.2,
    dl_bidirectional=False,
    dl_epochs=20,
    dl_batch_size=64,
    # ARIMA/SARIMA params
    arima_order=(5,1,0),
    seasonal_order=None,   # e.g., (1,1,1,12) for monthly seasonality
    # Ensembling
    try_rf: bool = True,
    results_prefix: str = "v05",
):
    """
    Train LSTM (or GRU/RNN) 1-step model, train ARIMA/SARIMA 1-step model on the same data,
    auto-tune a blend weight on validation, and evaluate on the test set.
    Also optionally trains a RandomForest baseline and supports 3-way blending.
    """
    os.makedirs("charts", exist_ok=True)

    # 1) Prepare sequences and splits
    X_train, y_train, X_test, y_test, df, scalers = load_process_multifeature(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_columns=list(feature_columns),
        target_column=target_column,
        lookback=lookback,
        handle_nan=handle_nan,
        split_method=split_method,
        train_ratio=train_ratio,
        scale_features=True,
        scaler_type=scaler_type,
        cache_dir=cache_dir,
        use_cache=True,
        save_cache=True,
        random_state=42,
    )
    X_tr, y_tr, X_val, y_val = _train_val_split_time_series(X_train, y_train, val_ratio=val_ratio)
    tgt_scaler = scalers["target"]

    # 2) DL model
    n_features = X_train.shape[2]
    dl_model = build_dl_model(
        layer_type=dl_layer_type,
        layer_sizes=list(dl_layer_sizes),
        input_timesteps=lookback,
        input_features=n_features,
        dropout=dl_dropout,
        bidirectional=dl_bidirectional,
        dense_units=1,
        optimizer="adam",
        loss="mean_squared_error",
    )
    dl_model.fit(X_tr, y_tr, epochs=dl_epochs, batch_size=dl_batch_size, validation_data=(X_val, y_val), verbose=1)

    # Predictions (inverse scale)
    y_val_pred_dl_s = dl_model.predict(X_val, verbose=0)
    y_tst_pred_dl_s = dl_model.predict(X_test, verbose=0)
    if tgt_scaler is not None:
        y_val_true = tgt_scaler.inverse_transform(y_val.reshape(-1,1)).ravel()
        y_val_pred_dl = tgt_scaler.inverse_transform(y_val_pred_dl_s).ravel()
        y_tst_true = tgt_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
        y_tst_pred_dl = tgt_scaler.inverse_transform(y_tst_pred_dl_s).ravel()
    else:
        y_val_true, y_val_pred_dl = y_val.ravel(), y_val_pred_dl_s.ravel()
        y_tst_true, y_tst_pred_dl = y_test.ravel(), y_tst_pred_dl_s.ravel()

    # 3) ARIMA/SARIMA aligned to the same targets
    s_aligned = _series_for_sequences_alignment(df, target_col=target_column, lookback=lookback)
    total_targets = len(s_aligned)
    train_points = int(total_targets * train_ratio)
    val_points   = max(1, int(train_points * val_ratio))
    core_points  = train_points - val_points
    test_points  = total_targets - train_points

    y_val_pred_arima, y_tst_pred_arima = _fit_predict_arima_like(
        s_aligned,
        train_points=train_points,
        val_points=val_points,
        test_points=test_points,
        arima_order=arima_order,
        seasonal_order=seasonal_order,
    )

    # Sanity: align lengths to our DL splits
    if len(y_val_pred_arima) != len(y_val_true):
        # Best-effort trim/pad (trim is safer)
        m = min(len(y_val_pred_arima), len(y_val_true))
        y_val_pred_arima = y_val_pred_arima[-m:]
        y_val_true = y_val_true[-m:]
        y_val_pred_dl = y_val_pred_dl[-m:]

    if len(y_tst_pred_arima) != len(y_tst_true):
        m = min(len(y_tst_pred_arima), len(y_tst_true))
        y_tst_pred_arima = y_tst_pred_arima[-m:]
        y_tst_true = y_tst_true[-m:]
        y_tst_pred_dl = y_tst_pred_dl[-m:]

    # 4) Optional RandomForest baseline (for Task 2 experimentation)
    have_rf = False
    if try_rf and RandomForestRegressor is not None:
        try:
            y_val_pred_rf, y_tst_pred_rf = _rf_baseline_preds(X_tr, y_tr, X_val, X_test, target_scaler=tgt_scaler)
            have_rf = (len(y_val_pred_rf) == len(y_val_true)) and (len(y_tst_pred_rf) == len(y_tst_true))
        except Exception as e:
            print("[RF] Skipping RF baseline due to error:", e)
            have_rf = False

    # 5) Weight search on validation for 2-way blend (DL vs ARIMA)
    w_best = _grid_search_weight(y_val_true, y_val_pred_dl, y_val_pred_arima, n=21)
    print(f"[Ensemble] Best DL weight on validation = {w_best:.3f} (ARIMA weight = {1.0 - w_best:.3f})")

    # Apply to test
    y_tst_pred_ens = _weighted_blend(y_tst_pred_dl, y_tst_pred_arima, w_best)

    m_dl   = _metrics(y_tst_true, y_tst_pred_dl)
    m_ar   = _metrics(y_tst_true, y_tst_pred_arima)
    m_ens2 = _metrics(y_tst_true, y_tst_pred_ens)

    print(f"[Test] DL       RMSE={m_dl['rmse']:.4f}  MAE={m_dl['mae']:.4f}  MAPE={m_dl['mape']:.2f}%")
    print(f"[Test] ARIMA    RMSE={m_ar['rmse']:.4f}  MAE={m_ar['mae']:.4f}  MAPE={m_ar['mape']:.2f}%")
    print(f"[Test] Ensemble RMSE={m_ens2['rmse']:.4f}  MAE={m_ens2['mae']:.4f}  MAPE={m_ens2['mape']:.2f}%")

    # 6) (Optional) 3-way blend (DL/ARIMA/RF) with simple uniform average or small grid
    y_tst_pred_ens3 = None
    if have_rf:
        # small ternary grid search over weights that sum to 1
        best_combo, best_rmse = (1/3, 1/3, 1/3), 1e18
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1 - a, 11):
                c = 1.0 - a - b
                blend = a*y_tst_pred_dl + b*y_tst_pred_arima + c*y_tst_pred_rf
                rmse = _metrics(y_tst_true, blend)["rmse"]
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_combo = (a, b, c)
                    y_tst_pred_ens3 = blend
        print(f"[Test] 3-way Ensemble (DL/ARIMA/RF) best weights={best_combo} RMSE={best_rmse:.4f}")

    # 7) Plots
    def _save_series_plot(y_true, preds: dict, title, path):
        plt.figure(figsize=(18,5))
        plt.plot(y_true, label="Actual")
        for name, arr in preds.items():
            plt.plot(arr, label=name)
        plt.title(title); plt.xlabel("Test sample index"); plt.ylabel("Price"); plt.legend()
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

    _save_series_plot(
        y_tst_true,
        {"DL": y_tst_pred_dl, "ARIMA": y_tst_pred_arima, "DL+ARIMA": y_tst_pred_ens},
        title=f"{ticker} v0.5 Ensemble (1-step)",
        path=os.path.join("charts", f"{results_prefix}_{ticker}_ens2_test.png"),
    )

    if have_rf and y_tst_pred_ens3 is not None:
        _save_series_plot(
            y_tst_true,
            {"DL": y_tst_pred_dl, "ARIMA": y_tst_pred_arima, "RF": y_tst_pred_rf, "DL+ARIMA+RF": y_tst_pred_ens3},
            title=f"{ticker} v0.5 Ensemble (3-way, 1-step)",
            path=os.path.join("charts", f"{results_prefix}_{ticker}_ens3_test.png"),
        )

    # 8) Return artifacts
    results = {
        "test": {
            "y_true": y_tst_true,
            "DL": y_tst_pred_dl,
            "ARIMA": y_tst_pred_arima,
            "DL+ARIMA": y_tst_pred_ens,
            "metrics": {"DL": m_dl, "ARIMA": m_ar, "DL+ARIMA": m_ens2},
        },
        "val_weight": w_best,
    }
    if have_rf and y_tst_pred_ens3 is not None:
        results["test"]["RF"] = y_tst_pred_rf
        results["test"]["DL+ARIMA+RF"] = y_tst_pred_ens3
    return results


# ============================================================================
# ENHANCED ENSEMBLE: ARIMA + LSTM with Side-by-Side Comparison Plots
# ============================================================================
def run_ensemble_with_detailed_plots(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    lookback: int = 60,
    feature_columns=("Open","High","Low","Close","Adj Close","Volume"),
    target_column="Close",
    split_method="ratio",
    train_ratio=0.8,
    val_ratio=0.1,
    # DL params
    dl_layer_type="lstm",
    dl_layer_sizes=(128,128),
    dl_dropout=0.2,
    dl_epochs=20,
    dl_batch_size=64,
    # ARIMA params
    arima_order=(5,1,0),
    seasonal_order=None,  # e.g., (1,1,1,12) for SARIMA
    # Output
    results_dir="ensemble_results",
):
    """
    Complete ensemble pipeline with detailed comparison plots.

    Steps:
    1. Load and prepare data
    2. Train LSTM model
    3. Train ARIMA/SARIMA model
    4. Find optimal ensemble weight on validation set
    5. Generate comparison plots:
       - Combined vs Actual
       - Combined vs LSTM only
       - Combined vs ARIMA only
       - All models together
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    os.makedirs(results_dir, exist_ok=True)

    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading and Preparing Data")
    print("="*70)

    # Reuse local helpers already in this file
    X_train, y_train, X_test, y_test, df, scalers = load_process_multifeature(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_columns=list(feature_columns),
        target_column=target_column,
        lookback=lookback,
        handle_nan="ffill",
        split_method=split_method,
        train_ratio=train_ratio,
        scale_features=True,
        scaler_type="minmax",
        cache_dir="data_cache",
        use_cache=True,
        save_cache=True,
        random_state=42,
    )

    # Split train into train/validation for weight optimization
    X_tr, y_tr, X_val, y_val = _train_val_split_time_series(
        X_train, y_train, val_ratio=val_ratio
    )

    tgt_scaler = scalers["target"]
    n_features = X_train.shape[2]

    print(f"✓ Data loaded: {len(X_tr)} train, {len(X_val)} val, {len(X_test)} test samples")
    print(f"✓ Features: {n_features} columns, Lookback: {lookback} days")

    # ========================================================================
    # STEP 2: TRAIN LSTM MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Training LSTM Model")
    print("="*70)

    # Build LSTM architecture via your model factory
    dl_model = build_dl_model(
        layer_type=dl_layer_type,
        layer_sizes=list(dl_layer_sizes),
        input_timesteps=lookback,
        input_features=n_features,
        dropout=dl_dropout,
        bidirectional=False,
        dense_units=1,
        optimizer="adam",
        loss="mean_squared_error",
    )

    print(f" Model architecture: {dl_layer_type.upper()} with {dl_layer_sizes}")
    print(f"Parameters: dropout={dl_dropout}, epochs={dl_epochs}, batch_size={dl_batch_size}")

    # Train LSTM
    history = dl_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=dl_epochs,
        batch_size=dl_batch_size,
        verbose=1
    )

    # Get predictions and inverse transform to original scale
    y_val_pred_dl_scaled = dl_model.predict(X_val, verbose=0)
    y_tst_pred_dl_scaled = dl_model.predict(X_test, verbose=0)

    if tgt_scaler is not None:
        y_val_true = tgt_scaler.inverse_transform(y_val.reshape(-1,1)).ravel()
        y_val_pred_dl = tgt_scaler.inverse_transform(y_val_pred_dl_scaled).ravel()
        y_tst_true = tgt_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
        y_tst_pred_dl = tgt_scaler.inverse_transform(y_tst_pred_dl_scaled).ravel()
    else:
        y_val_true = y_val.ravel()
        y_val_pred_dl = y_val_pred_dl_scaled.ravel()
        y_tst_true = y_test.ravel()
        y_tst_pred_dl = y_tst_pred_dl_scaled.ravel()

    # Calculate LSTM metrics
    lstm_val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred_dl))
    lstm_tst_rmse = np.sqrt(mean_squared_error(y_tst_true, y_tst_pred_dl))
    lstm_tst_mae = mean_absolute_error(y_tst_true, y_tst_pred_dl)

    print(f"✓ LSTM trained successfully")
    print(f"  Validation RMSE: ${lstm_val_rmse:.4f}")
    print(f"  Test RMSE: ${lstm_tst_rmse:.4f}")
    print(f"  Test MAE:  ${lstm_tst_mae:.4f}")

    # ========================================================================
    # STEP 3: TRAIN ARIMA/SARIMA MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Training ARIMA/SARIMA Model")
    print("="*70)

    # Align ARIMA to same targets as LSTM (important for fair comparison!)
    s_aligned = _series_for_sequences_alignment(df, target_col=target_column, lookback=lookback)
    total_targets = len(s_aligned)
    train_points = int(total_targets * train_ratio)
    val_points = max(1, int(train_points * val_ratio))
    test_points = total_targets - train_points

    model_type = "SARIMA" if seasonal_order else "ARIMA"
    print(f" Model type: {model_type}")
    print(f"Order: {arima_order}" + (f", Seasonal: {seasonal_order}" if seasonal_order else ""))
    print(f" Training on {train_points} points, predicting {test_points} test points")

    # Fit ARIMA and get predictions
    y_val_pred_arima, y_tst_pred_arima = _fit_predict_arima_like(
        s_aligned,
        train_points=train_points,
        val_points=val_points,
        test_points=test_points,
        arima_order=arima_order,
        seasonal_order=seasonal_order,
    )

    # Align lengths (safety check)
    min_val = min(len(y_val_pred_arima), len(y_val_true))
    y_val_pred_arima = y_val_pred_arima[-min_val:]
    y_val_true_aligned = y_val_true[-min_val:]
    y_val_pred_dl_aligned = y_val_pred_dl[-min_val:]

    min_tst = min(len(y_tst_pred_arima), len(y_tst_true))
    y_tst_pred_arima = y_tst_pred_arima[-min_tst:]
    y_tst_true_aligned = y_tst_true[-min_tst:]
    y_tst_pred_dl_aligned = y_tst_pred_dl[-min_tst:]

    # Calculate ARIMA metrics
    arima_val_rmse = np.sqrt(mean_squared_error(y_val_true_aligned, y_val_pred_arima))
    arima_tst_rmse = np.sqrt(mean_squared_error(y_tst_true_aligned, y_tst_pred_arima))
    arima_tst_mae = mean_absolute_error(y_tst_true_aligned, y_tst_pred_arima)

    print(f"{model_type} trained successfully")
    print(f"  Validation RMSE: ${arima_val_rmse:.4f}")
    print(f"  Test RMSE: ${arima_tst_rmse:.4f}")
    print(f"  Test MAE:  ${arima_tst_mae:.4f}")

    # ========================================================================
    # STEP 4: ENSEMBLE COMBINATION (Weight Optimization)
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Creating Ensemble (Weighted Combination)")
    print("="*70)
    print("Searching for optimal weight w in: Combined = w*LSTM + (1-w)*ARIMA")

    # Grid search over weights
    best_w = 0.5
    best_rmse = float('inf')
    weight_results = []

    for i in range(21):  # Test 21 weights from 0.0 to 1.0
        w = i / 20.0
        blend = w * y_val_pred_dl_aligned + (1 - w) * y_val_pred_arima
        rmse = np.sqrt(mean_squared_error(y_val_true_aligned, blend))
        weight_results.append((w, rmse))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w

    print(f"Optimal weight found: w = {best_w:.3f}")
    print(f"  This means: Combined = {best_w:.1%} LSTM + {(1-best_w):.1%} ARIMA")
    print(f"  Validation RMSE with ensemble: ${best_rmse:.4f}")

    # Apply optimal weight to test set
    y_tst_pred_ensemble = best_w * y_tst_pred_dl_aligned + (1 - best_w) * y_tst_pred_arima

    # Calculate ensemble metrics
    ens_tst_rmse = np.sqrt(mean_squared_error(y_tst_true_aligned, y_tst_pred_ensemble))
    ens_tst_mae = mean_absolute_error(y_tst_true_aligned, y_tst_pred_ensemble)

    print(f"Ensemble applied to test set")
    print(f"  Test RMSE: ${ens_tst_rmse:.4f}")
    print(f"  Test MAE:  ${ens_tst_mae:.4f}")

    # ========================================================================
    # STEP 5: GENERATE COMPARISON PLOTS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Generating Comparison Plots")
    print("="*70)

    # Plot 1: Weight Search Curve
    plt.figure(figsize=(10, 5))
    weights = [w for w, _ in weight_results]
    rmses = [r for _, r in weight_results]
    plt.plot(weights, rmses, 'b-', linewidth=2)
    plt.axvline(best_w, color='r', linestyle='--', label=f'Optimal w={best_w:.3f}')
    plt.xlabel('LSTM Weight (w)', fontsize=12)
    plt.ylabel('Validation RMSE ($)', fontsize=12)
    plt.title(f'{ticker}: Ensemble Weight Optimization\nCombined = w*LSTM + (1-w)*ARIMA', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{ticker}_weight_search.png", dpi=150)
    plt.close()
    print(f" Saved: {ticker}_weight_search.png")

    # Plot 2: Combined vs Actual (main result)
    plt.figure(figsize=(18, 6))
    plt.plot(y_tst_true_aligned, 'k-', linewidth=2, label='Actual Price', alpha=0.8)
    plt.plot(y_tst_pred_ensemble, 'g-', linewidth=2, label=f'Ensemble (RMSE=${ens_tst_rmse:.2f})', alpha=0.8)
    plt.fill_between(range(len(y_tst_true_aligned)),
                     y_tst_true_aligned, y_tst_pred_ensemble,
                     alpha=0.2, color='green', label='Prediction Error')
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title(f'{ticker}: Ensemble Prediction vs Actual\nLSTM ({best_w:.1%}) + ARIMA ({(1-best_w):.1%})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{ticker}_ensemble_vs_actual.png", dpi=150)
    plt.close()
    print(f" Saved: {ticker}_ensemble_vs_actual.png")

    # Plot 3: Combined vs LSTM Only
    plt.figure(figsize=(18, 6))
    plt.plot(y_tst_true_aligned, 'k-', linewidth=2, label='Actual Price', alpha=0.8)
    plt.plot(y_tst_pred_dl_aligned, 'b--', linewidth=1.5, label=f'LSTM Only (RMSE=${lstm_tst_rmse:.2f})', alpha=0.7)
    plt.plot(y_tst_pred_ensemble, 'g-', linewidth=2, label=f'Ensemble (RMSE=${ens_tst_rmse:.2f})', alpha=0.8)
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title(f'{ticker}: Ensemble vs LSTM-Only Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{ticker}_ensemble_vs_lstm.png", dpi=150)
    plt.close()
    print(f" Saved: {ticker}_ensemble_vs_lstm.png")

    # Plot 4: Combined vs ARIMA Only
    plt.figure(figsize=(18, 6))
    plt.plot(y_tst_true_aligned, 'k-', linewidth=2, label='Actual Price', alpha=0.8)
    plt.plot(y_tst_pred_arima, 'r--', linewidth=1.5, label=f'{model_type} Only (RMSE=${arima_tst_rmse:.2f})', alpha=0.7)
    plt.plot(y_tst_pred_ensemble, 'g-', linewidth=2, label=f'Ensemble (RMSE=${ens_tst_rmse:.2f})', alpha=0.8)
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title(f'{ticker}: Ensemble vs {model_type}-Only Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{ticker}_ensemble_vs_arima.png", dpi=150)
    plt.close()
    print(f" Saved: {ticker}_ensemble_vs_arima.png")

    # Plot 5: All Models Together (comprehensive view)
    plt.figure(figsize=(20, 8))
    plt.plot(y_tst_true_aligned, 'k-', linewidth=2.5, label='Actual Price', alpha=0.9)
    plt.plot(y_tst_pred_dl_aligned, 'b:', linewidth=1.5, label=f'LSTM (RMSE=${lstm_tst_rmse:.2f})', alpha=0.7)
    plt.plot(y_tst_pred_arima, 'r:', linewidth=1.5, label=f'{model_type} (RMSE=${arima_tst_rmse:.2f})', alpha=0.7)
    plt.plot(y_tst_pred_ensemble, 'g-', linewidth=2, label=f'Ensemble (RMSE=${ens_tst_rmse:.2f})', alpha=0.9)
    plt.xlabel('Test Sample Index', fontsize=13)
    plt.ylabel('Stock Price ($)', fontsize=13)
    plt.title(f'{ticker}: Complete Model Comparison\nEnsemble = {best_w:.1%} LSTM + {(1-best_w):.1%} {model_type}', fontsize=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{ticker}_all_models_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {ticker}_all_models_comparison.png")

    # Plot 6: Error Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    errors_lstm = y_tst_true_aligned - y_tst_pred_dl_aligned
    errors_arima = y_tst_true_aligned - y_tst_pred_arima
    errors_ensemble = y_tst_true_aligned - y_tst_pred_ensemble

    axes[0].hist(errors_lstm, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'LSTM Errors\nRMSE=${lstm_tst_rmse:.2f}')
    axes[0].set_xlabel('Error ($)')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(errors_arima, bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title(f'{model_type} Errors\nRMSE=${arima_tst_rmse:.2f}')
    axes[1].set_xlabel('Error ($)')

    axes[2].hist(errors_ensemble, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[2].set_title(f'Ensemble Errors\nRMSE=${ens_tst_rmse:.2f}')
    axes[2].set_xlabel('Error ($)')

    plt.suptitle(f'{ticker}: Prediction Error Distributions', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{ticker}_error_distributions.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {ticker}_error_distributions.png")

    # ========================================================================
    # STEP 6: SUMMARY TABLE
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Performance Summary")
    print("="*70)

    summary = pd.DataFrame({
        'Model': ['LSTM Only', f'{model_type} Only', 'Ensemble (LSTM+ARIMA)'],
        'Test RMSE ($)': [lstm_tst_rmse, arima_tst_rmse, ens_tst_rmse],
        'Test MAE ($)': [lstm_tst_mae, arima_tst_mae, ens_tst_mae],
        'Improvement vs LSTM': ['—',
                                f'{((lstm_tst_rmse - arima_tst_rmse)/lstm_tst_rmse*100):+.2f}%',
                                f'{((lstm_tst_rmse - ens_tst_rmse)/lstm_tst_rmse*100):+.2f}%'],
    })

    print(summary.to_string(index=False))
    summary.to_csv(f"{results_dir}/{ticker}_ensemble_summary.csv", index=False)
    print(f"\n✓ Saved: {ticker}_ensemble_summary.csv")

    # Return all results
    return {
        'lstm_predictions': y_tst_pred_dl_aligned,
        'arima_predictions': y_tst_pred_arima,
        'ensemble_predictions': y_tst_pred_ensemble,
        'actual': y_tst_true_aligned,
        'optimal_weight': best_w,
        'metrics': {
            'lstm_rmse': lstm_tst_rmse,
            'arima_rmse': arima_tst_rmse,
            'ensemble_rmse': ens_tst_rmse,
        }
    }




# ============================================================================
# IMPROVED THREE-WAY ENSEMBLE (LSTM + ARIMA + RandomForest)
# ============================================================================
def run_three_way_ensemble(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    lookback: int = 60,
    arima_order: tuple = (5, 1, 0),
    seasonal_order: tuple | None = None,
    results_dir="ensemble_results",
):
    """
    Three-model ensemble: LSTM + ARIMA + RandomForest
    Shows individual model performance before combining.
    """
    print("\n" + "="*70)
    print("THREE-WAY ENSEMBLE: LSTM + ARIMA + RandomForest")
    print("="*70)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.ensemble import RandomForestRegressor
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test, df, scalers = load_process_multifeature(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        feature_columns=["Open","High","Low","Close","Adj Close","Volume"],
        target_column="Close",
        lookback=lookback,
        handle_nan="ffill",
        split_method="ratio",
        train_ratio=0.8,
        scale_features=True,
        scaler_type="minmax",
        cache_dir="data_cache",
    )
    
    X_tr, y_tr, X_val, y_val = _train_val_split_time_series(X_train, y_train, val_ratio=0.1)
    tgt_scaler = scalers["target"]
    
    # =======================================================================
    # MODEL 1: LSTM
    # =======================================================================
    print("\n[1/3] Training LSTM...")
    lstm = build_dl_model("lstm", [128,128], lookback, X_train.shape[2], dropout=0.2)
    lstm.fit(X_tr, y_tr, epochs=20, batch_size=64, validation_data=(X_val, y_val), verbose=0)
    
    y_val_lstm_s = lstm.predict(X_val, verbose=0)
    y_tst_lstm_s = lstm.predict(X_test, verbose=0)
    
    if tgt_scaler:
        y_val_true = tgt_scaler.inverse_transform(y_val.reshape(-1,1)).ravel()
        y_val_lstm = tgt_scaler.inverse_transform(y_val_lstm_s).ravel()
        y_tst_true = tgt_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
        y_tst_lstm = tgt_scaler.inverse_transform(y_tst_lstm_s).ravel()
    else:
        y_val_true, y_val_lstm = y_val.ravel(), y_val_lstm_s.ravel()
        y_tst_true, y_tst_lstm = y_test.ravel(), y_tst_lstm_s.ravel()
    
    lstm_val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_lstm))
    lstm_tst_rmse = np.sqrt(mean_squared_error(y_tst_true, y_tst_lstm))
    print(f"  ✓ LSTM - Val RMSE: ${lstm_val_rmse:.4f} | Test RMSE: ${lstm_tst_rmse:.4f}")
    
    # =======================================================================
    # MODEL 2: ARIMA/SARIMA
    # =======================================================================
    print(f"\n[2/3] Training {'SARIMA' if seasonal_order else 'ARIMA'}...")
    print(f"  Order: {arima_order}" + (f", Seasonal: {seasonal_order}" if seasonal_order else ""))
    
    s_aligned = _series_for_sequences_alignment(df, "Close", lookback)
    train_points = int(len(s_aligned) * 0.8)
    val_points = max(1, int(train_points * 0.1))
    test_points = len(s_aligned) - train_points
    
    try:
        y_val_arima, y_tst_arima = _fit_predict_arima_like(
            s_aligned, 
            train_points=train_points, 
            val_points=val_points,
            test_points=test_points, 
            arima_order=arima_order, 
            seasonal_order=seasonal_order
        )
        
        # Align lengths
        min_val = min(len(y_val_arima), len(y_val_true))
        y_val_arima = y_val_arima[-min_val:]
        y_val_true_aligned = y_val_true[-min_val:]
        y_val_lstm_aligned = y_val_lstm[-min_val:]
        
        min_tst = min(len(y_tst_arima), len(y_tst_true))
        y_tst_arima = y_tst_arima[-min_tst:]
        y_tst_true_aligned = y_tst_true[-min_tst:]
        y_tst_lstm_aligned = y_tst_lstm[-min_tst:]
        
        arima_val_rmse = np.sqrt(mean_squared_error(y_val_true_aligned, y_val_arima))
        arima_tst_rmse = np.sqrt(mean_squared_error(y_tst_true_aligned, y_tst_arima))
        print(f"  ✓ ARIMA - Val RMSE: ${arima_val_rmse:.4f} | Test RMSE: ${arima_tst_rmse:.4f}")
        
        arima_failed = False
    except Exception as e:
        print(f"  ✗ ARIMA failed: {e}")
        print(f"  → Using zero predictions as fallback (will get 0% weight)")
        y_val_arima = np.zeros_like(y_val_true)
        y_tst_arima = np.zeros_like(y_tst_true)
        y_val_true_aligned = y_val_true
        y_val_lstm_aligned = y_val_lstm
        y_tst_true_aligned = y_tst_true
        y_tst_lstm_aligned = y_tst_lstm
        arima_val_rmse = float('inf')
        arima_tst_rmse = float('inf')
        arima_failed = True
    
    # =======================================================================
    # MODEL 3: RandomForest (on flattened windows)
    # =======================================================================
    print("\n[3/3] Training RandomForest...")
    n_tr, t, f = X_tr.shape
    n_val = X_val.shape[0]
    n_tst = X_test.shape[0]
    
    X_tr_flat = X_tr.reshape(n_tr, t*f)
    X_val_flat = X_val.reshape(n_val, t*f)
    X_tst_flat = X_test.reshape(n_tst, t*f)
    
    y_tr_orig = tgt_scaler.inverse_transform(y_tr.reshape(-1,1)).ravel() if tgt_scaler else y_tr.ravel()
    
    rf = RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_tr_flat, y_tr_orig)
    
    y_val_rf_full = rf.predict(X_val_flat)
    y_tst_rf_full = rf.predict(X_tst_flat)
    
    # Align with ARIMA lengths
    y_val_rf = y_val_rf_full[-len(y_val_true_aligned):]
    y_tst_rf = y_tst_rf_full[-len(y_tst_true_aligned):]
    
    rf_val_rmse = np.sqrt(mean_squared_error(y_val_true_aligned, y_val_rf))
    rf_tst_rmse = np.sqrt(mean_squared_error(y_tst_true_aligned, y_tst_rf))
    print(f"  ✓ RF - Val RMSE: ${rf_val_rmse:.4f} | Test RMSE: ${rf_tst_rmse:.4f}")
    
    # =======================================================================
    # COMPARISON TABLE
    # =======================================================================
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL PERFORMANCE (Validation Set)")
    print("="*70)
    print(f"{'Model':<15} {'Val RMSE':<15} {'Test RMSE':<15} {'Notes':<20}")
    print("-"*70)
    print(f"{'LSTM':<15} ${lstm_val_rmse:<14.4f} ${lstm_tst_rmse:<14.4f}")
    if arima_failed:
        print(f"{'ARIMA':<15} {'FAILED':<15} {'FAILED':<15} {'Will get 0% weight':<20}")
    else:
        print(f"{'ARIMA':<15} ${arima_val_rmse:<14.4f} ${arima_tst_rmse:<14.4f}")
    print(f"{'RandomForest':<15} ${rf_val_rmse:<14.4f} ${rf_tst_rmse:<14.4f}")
    print("="*70)
    
    # =======================================================================
    # WEIGHT OPTIMIZATION (Grid search for 3-way weights)
    # =======================================================================
    print("\n[Optimization] Searching for optimal 3-way weights...")
    print("Testing combinations: a*LSTM + b*ARIMA + c*RF where a+b+c=1")
    
    best_combo = (1/3, 1/3, 1/3)
    best_rmse = float('inf')
    best_blend = None
    
    # Store top 5 combinations for analysis
    top_combos = []
    
    for a in np.linspace(0, 1, 11):
        for b in np.linspace(0, 1-a, 11):
            c = 1.0 - a - b
            blend = a*y_val_lstm_aligned + b*y_val_arima + c*y_val_rf
            rmse = np.sqrt(mean_squared_error(y_val_true_aligned, blend))
            
            top_combos.append((rmse, a, b, c))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_combo = (a, b, c)
    
    # Show top 3 weight combinations
    top_combos.sort()
    print(f"\nTop 3 Weight Combinations (by validation RMSE):")
    print(f"{'Rank':<6} {'LSTM':<8} {'ARIMA':<8} {'RF':<8} {'Val RMSE':<12}")
    print("-"*50)
    for i, (rmse, a, b, c) in enumerate(top_combos[:3], 1):
        print(f"{i:<6} {a:>6.1%}  {b:>6.1%}  {c:>6.1%}  ${rmse:<11.4f}")
    
    a_opt, b_opt, c_opt = best_combo
    print(f"\n✓ Optimal weights: LSTM={a_opt:.2f}, ARIMA={b_opt:.2f}, RF={c_opt:.2f}")
    
    # Explain why ARIMA got low weight
    if b_opt < 0.05 and not arima_failed:
        print(f"\n⚠ ARIMA received low weight ({b_opt:.1%}) because:")
        print(f"  • ARIMA val RMSE (${arima_val_rmse:.4f}) >> RF val RMSE (${rf_val_rmse:.4f})")
        print(f"  • Grid search found other models more accurate")
        print(f"  • This is normal for volatile stocks like {ticker}")
        print(f"  Suggestions:")
        print(f"  → Try SARIMA: --seasonal-order 1 1 1 12")
        print(f"  → Adjust order: --arima-order 3 1 2")
    
    # =======================================================================
    # APPLY TO TEST SET
    # =======================================================================
    y_tst_3way = a_opt*y_tst_lstm_aligned + b_opt*y_tst_arima + c_opt*y_tst_rf
    ens_tst_rmse = np.sqrt(mean_squared_error(y_tst_true_aligned, y_tst_3way))
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'Test RMSE':<15} {'vs Best Single':<20}")
    print("-"*70)
    best_single_rmse = min(lstm_tst_rmse, rf_tst_rmse, arima_tst_rmse if not arima_failed else float('inf'))
    print(f"{'LSTM':<20} ${lstm_tst_rmse:<14.4f}")
    if not arima_failed:
        print(f"{'ARIMA':<20} ${arima_tst_rmse:<14.4f}")
    print(f"{'RandomForest':<20} ${rf_tst_rmse:<14.4f}")
    improvement = ((best_single_rmse - ens_tst_rmse) / best_single_rmse * 100)
    print(f"{'3-Way Ensemble':<20} ${ens_tst_rmse:<14.4f} {improvement:+.2f}%")
    print("="*70)
    
    # =======================================================================
    # PLOT
    # =======================================================================
    plt.figure(figsize=(20, 8))
    plt.plot(y_tst_true_aligned, 'k-', linewidth=2.5, label='Actual', alpha=0.9)
    plt.plot(y_tst_lstm_aligned, 'b:', linewidth=1.5, label=f'LSTM (RMSE=${lstm_tst_rmse:.2f})', alpha=0.7)
    if not arima_failed:
        plt.plot(y_tst_arima, 'r:', linewidth=1.5, label=f'ARIMA (RMSE=${arima_tst_rmse:.2f})', alpha=0.7)
    plt.plot(y_tst_rf, 'm:', linewidth=1.5, label=f'RF (RMSE=${rf_tst_rmse:.2f})', alpha=0.7)
    plt.plot(y_tst_3way, 'g-', linewidth=2.5, label=f'3-Way Ensemble (RMSE=${ens_tst_rmse:.2f})', alpha=0.9)
    plt.xlabel('Test Sample Index', fontsize=13)
    plt.ylabel('Stock Price ($)', fontsize=13)
    plt.title(f'{ticker}: Three-Way Ensemble\n{a_opt:.0%} LSTM + {b_opt:.0%} ARIMA + {c_opt:.0%} RandomForest', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{ticker}_3way_ensemble.png", dpi=150)
    plt.close()
    
    print(f"\n✓ 3-Way Ensemble RMSE: ${ens_tst_rmse:.4f}")
    print(f"✓ Plot saved: {ticker}_3way_ensemble.png")
    
    return {
        'predictions': y_tst_3way,
        'actual': y_tst_true_aligned,
        'weights': best_combo,
        'rmse': ens_tst_rmse,
        'individual_rmse': {
            'lstm': lstm_tst_rmse,
            'arima': arima_tst_rmse if not arima_failed else None,
            'rf': rf_tst_rmse
        }
    }

# ============================== SENTIMENT + CLASSIFICATION PIPELINE ==============================
# These functions are self-contained and do not change existing forecasting code.
# They implement:
#   (1) Data collection & preprocessing for text (news) aligned to price dates
#   (2) Sentiment analysis (VADER by default, optional FinBERT if transformers is installed)
#   (3) Feature engineering: technical indicators + sentiment features (daily)
#   (4) Classification model: predict if next day's close > today's close (Up/Down)
#   (5) Evaluation: accuracy, precision, recall, F1, confusion matrix, comparison to baseline w/o sentiment
#   (6) Independent research enhancement: recency-weighted daily sentiment + sentiment volatility features

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional models if available
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Optional NLP libs
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# ----------------------------- (1) Data Collection & Preprocessing -----------------------------
def collect_text_data_yfinance_or_csv(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    csv_path: str | None = None,
    cache_dir: str = "data_cache",
    max_articles: int = 5000,
) -> pd.DataFrame:
    """
    Collect textual news-like data, time-alignable to prices.

    Strategy:
    - If csv_path is provided, load it. Expected columns:
        ['published', 'title'] or ['published', 'title', 'summary'/'content']
      'published' must be parseable datetime in UTC or local (we normalize).
    - Else, attempt yfinance.Ticker(ticker).news (recent only; not fully historical).
      This is best-effort and may return None/limited data depending on the ticker.

    Returns
    -------
    DataFrame with columns: ['published', 'text', 'date', 'hours_since_midnight']
      - 'published' = timezone-aware UTC timestamp
      - 'text'      = concatenated headline + summary if available
      - 'date'      = calendar date (YYYY-MM-DD) in the 'published' timezone
      - 'hours_since_midnight' = numeric hours within the day (for recency weighting)
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_dt = pd.to_datetime(start_date).tz_localize(None)
    end_dt   = pd.to_datetime(end_date).tz_localize(None)

    rows = []
    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        # Normalize expected fields
        time_col = None
        for c in ["published", "time", "date", "datetime", "pubDate"]:
            if c in df.columns:
                time_col = c; break
        if time_col is None:
            raise ValueError("CSV must include a 'published' (or datetime-like) column.")

        text_cols = [c for c in ["title", "summary", "content", "description", "text"] if c in df.columns]
        if not text_cols:
            raise ValueError("CSV should include at least one text column (title/summary/content/description/text).")

        df = df[[time_col] + text_cols].copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col])
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        df["text"] = df[text_cols].agg(". ".join, axis=1).str.strip()

        rows = df[[time_col, "text"]].rename(columns={time_col: "published"}).to_dict("records")

    # --- replace the yfinance branch inside collect_text_data_yfinance_or_csv ---
    else:
        import yfinance as yf
        news = []
        try:
            news = (yf.Ticker(ticker).news or [])[:max_articles]
        except Exception:
            news = []
        for item in news:
            ts = item.get("providerPublishTime", None)
            # Coerce everything to text safely (titles/contents can be dicts)
            title   = _coerce_to_text(item.get("title", ""))
            content = _coerce_to_text(item.get("content", ""))
            text = (f"{title}. {content}").strip()
            if ts is None or not text:
                continue
            pub = pd.to_datetime(ts, unit="s", utc=True)
            rows.append({"published": pub, "text": text})


    df_news = pd.DataFrame(rows)
    if df_news.empty:
        # Return empty dataframe with expected schema; caller will handle gracefully
        return pd.DataFrame(columns=["published", "text", "date", "hours_since_midnight"])

    # Clip to desired date range (inclusive)
    df_news["published"] = pd.to_datetime(df_news["published"], utc=True)
    df_news = df_news[(df_news["published"] >= start_dt.tz_localize("UTC")) & (df_news["published"] <= end_dt.tz_localize("UTC"))]

    # Basic cleaning
    df_news["text"] = df_news["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # Daily bucketing helpers
    local = df_news["published"].dt.tz_convert(timezone.utc)  # keep UTC but compute hour-of-day
    df_news["date"] = local.dt.date.astype("datetime64[ns]")  # normalize to date column (UTC day)
    df_news["hours_since_midnight"] = local.dt.hour + local.dt.minute/60.0 + local.dt.second/3600.0

    return df_news.reset_index(drop=True)


# ---------- Text coercion for sentiment ----------
def _coerce_to_text(x):
    """
    Make ANY object safe for text models:
    - dict -> join key:value strings
    - list/tuple/set -> join items
    - None/NaN -> empty string
    - other -> str(x)
    """
    import pandas as _pd
    import numpy as _np
    if x is None:
        return ""
    # pandas NA
    try:
        if _pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            parts.append(f"{k}: {'' if v is None else str(v)}")
        return " | ".join(parts)
    if isinstance(x, (list, tuple, set)):
        return " | ".join("" if i is None else str(i) for i in x)
    return str(x)

# # assuming df_text has a column 'text' (title+description+body etc.)
# df_text["text"] = normalise_text_series(df_text["text"])

# # VADER
# try:
#     from nltk.sentiment import SentimentIntensityAnalyzer
#     sia = SentimentIntensityAnalyzer()
#     scores = df_text["text"].map(lambda t: sia.polarity_scores(t)["compound"] if t else 0.0)
#     df_text["vader_compound"] = scores.astype(float)
# except Exception as e:
#     print(f"[Sentiment] VADER failed: {e}")
#     df_text["vader_compound"] = 0.0

# # TF-IDF (must be strings!)
# from sklearn.feature_extraction.text import TfidfVectorizer
# vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
# X_text = vec.fit_transform(df_text["text"])

def normalise_text_series(s):
    """Return a clean string Series with newlines collapsed and trimmed."""
    import pandas as _pd
    s = _pd.Series(s).fillna("").map(_coerce_to_text)
    # Remove control characters & collapse whitespace
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

# def combine_fields(row, fields=("title","description","content")):
#     return " ".join(_coerce_to_text(row.get(f, "")) for f in fields)

# df_text["text"] = df_text.apply(combine_fields, axis=1)
# df_text["text"] = normalise_text_series(df_text["text"])





# ----------------------------- (2) Sentiment Analysis (VADER / FinBERT) -----------------------------
def _ensure_vader():
    """Download the VADER lexicon on first use (if nltk is available)."""
    global _HAS_VADER
    if not _HAS_VADER:
        raise ImportError("NLTK + VADER required. Install: pip install nltk; then nltk.download('vader_lexicon')")
    try:
        # Try creating analyzer (will fail if lexicon not present)
        SentimentIntensityAnalyzer()
    except Exception:
        import nltk
        nltk.download("vader_lexicon")
    _HAS_VADER = True




def sentiment_scores_vader(df_news: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Compute VADER compound scores in [-1,1] for each news row.
    Adds columns: ['sent_compound', 'sent_pos', 'sent_neg', 'sent_neu'].
    """
    _ensure_vader()
    sid = SentimentIntensityAnalyzer()
    s = df_news[text_col].fillna("").astype(str).tolist()
    scores = [sid.polarity_scores(x) for x in s]
    out = df_news.copy()
    out["sent_compound"] = [d["compound"] for d in scores]
    out["sent_pos"] = [d["pos"] for d in scores]
    out["sent_neg"] = [d["neg"] for d in scores]
    out["sent_neu"] = [d["neu"] for d in scores]
    return out

def _build_finbert_pipeline():
    """
    Load FinBERT for finance-specific sentiment (optional).
    Model: 'yiyanghkust/finbert-tone' with labels: positive/neutral/negative.
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers not installed. Install: pip install transformers torch")
    name = "yiyanghkust/finbert-tone"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    return TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, truncation=True, max_length=256, top_k=None)

def sentiment_scores_finbert(df_news: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Compute FinBERT scores per row; convert to a single signed score: (pos - neg).
    Adds: ['finbert_pos','finbert_neu','finbert_neg','finbert_signed'].
    """
    pipe = _build_finbert_pipeline()
    texts = df_news[text_col].fillna("").astype(str).tolist()
    preds = pipe(texts, batch_size=32)
    # Convert list of [{'label':'positive','score':...}, ..] to columns
    pos, neu, neg = [], [], []
    for triple in preds:
        d = {x["label"].lower(): float(x["score"]) for x in triple}
        pos.append(d.get("positive", 0.0))
        neu.append(d.get("neutral", 0.0))
        neg.append(d.get("negative", 0.0))
    out = df_news.copy()
    out["finbert_pos"] = pos
    out["finbert_neu"] = neu
    out["finbert_neg"] = neg
    out["finbert_signed"] = out["finbert_pos"] - out["finbert_neg"]
    return out


# ----------------------------- (3) Aggregate to Daily with Recency Weighting -----------------------------
def aggregate_sentiment_daily(
    df_sent: pd.DataFrame,
    *,
    engine: str = "vader",            # 'vader' or 'finbert'
    weight_by_recency: bool = True,   # independent research enhancement
    decay_per_hour: float = 0.10,     # exponential decay lambda; 0.10 ~ half-life ≈ 6.9 h
) -> pd.DataFrame:
    """
    Aggregate per-article sentiment to daily features that match stock frequency.

    If weight_by_recency=True, we apply exponential decay within the day so that
    articles published later in the day have more impact on that day's signal.
    This approximates “closer to market close has higher weight”.

    Returns columns (depending on engine):
      - base score: 'sent_compound' (vader) OR 'finbert_signed' (finbert)
      - daily_mean, daily_weighted_mean, daily_std, daily_sum, daily_count
      - optional: pos/neg share if available
    """
    df = df_sent.copy()
    if df.empty:
        # return empty daily frame with expected schema
        return pd.DataFrame(columns=["date", "daily_mean", "daily_weighted_mean", "daily_std", "daily_sum", "daily_count"]).set_index("date")

    if engine == "finbert":
        score_col = "finbert_signed"
        # ensure finbert columns exist
        if score_col not in df.columns:
            raise ValueError("FinBERT columns not found; run sentiment_scores_finbert first.")
    else:
        engine = "vader"
        score_col = "sent_compound"
        if score_col not in df.columns:
            raise ValueError("VADER columns not found; run sentiment_scores_vader first.")

    # Recency weights within the day: w = exp(+decay_per_hour * (h - max_h))
    # i.e., highest weight for the latest article that day.
    def _daily_weights(x: pd.DataFrame):
        if not weight_by_recency:
            return np.ones(len(x), dtype=float)
        h = x["hours_since_midnight"].astype(float).values
        if len(h) == 0:
            return np.ones(0, dtype=float)
        h0 = h.max()
        return np.exp(-decay_per_hour * (h0 - h))

    agg_rows = []
    for d, grp in df.groupby("date"):
        w = _daily_weights(grp)
        s = grp[score_col].astype(float).values
        if len(s) == 0:
            continue
        w = w / (w.sum() if w.sum() > 0 else 1.0)
        daily = {
            "date": pd.to_datetime(d),
            "daily_mean": float(np.mean(s)),
            "daily_weighted_mean": float(np.sum(w * s)),
            "daily_std": float(np.std(s)) if len(s) > 1 else 0.0,
            "daily_sum": float(np.sum(s)),
            "daily_count": int(len(s)),
        }
        # optional shares if present
        if engine == "vader":
            if {"sent_pos","sent_neg","sent_neu"}.issubset(grp.columns):
                daily["pos_mean"] = float(np.mean(grp["sent_pos"]))
                daily["neg_mean"] = float(np.mean(grp["sent_neg"]))
                daily["neu_mean"] = float(np.mean(grp["sent_neu"]))
        else:
            if {"finbert_pos","finbert_neg","finbert_neu"}.issubset(grp.columns):
                daily["pos_mean"] = float(np.mean(grp["finbert_pos"]))
                daily["neg_mean"] = float(np.mean(grp["finbert_neg"]))
                daily["neu_mean"] = float(np.mean(grp["finbert_neu"]))
        agg_rows.append(daily)

    daily_df = pd.DataFrame(agg_rows).sort_values("date").set_index("date")
    return daily_df


# ----------------------------- (4) Technical Indicators & Label -----------------------------
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def engineer_technical_indicators(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common indicators on Close/HL/Volume; keeps index as Date.
    Produces: returns, lagged returns, SMA, EMA, RSI, MACD, volume change.
    """
    p = df_prices.copy().sort_index()
    if "Close" not in p.columns:
        raise ValueError("engineer_technical_indicators expects a 'Close' column.")

    p["ret_1"] = p["Close"].pct_change()
    p["lag_ret_1"] = p["ret_1"].shift(1)
    p["lag_ret_2"] = p["ret_1"].shift(2)
    p["lag_ret_3"] = p["ret_1"].shift(3)

    for w in (5, 10, 20):
        p[f"sma_{w}"] = p["Close"].rolling(w).mean()
        p[f"ema_{w}"] = _ema(p["Close"], w)

    p["rsi_14"] = _rsi(p["Close"], 14)
    macd, sig, hist = _macd(p["Close"])
    p["macd"] = macd; p["macd_signal"] = sig; p["macd_hist"] = hist

    if "Volume" in p.columns:
        p["vol_chg"] = p["Volume"].pct_change()

    return p


def build_features_and_labels_for_direction(
    df_prices: pd.DataFrame,
    df_daily_sent: pd.DataFrame,
    *,
    use_weighted_sentiment: bool = True,
    add_sent_volatility: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Join technical indicators with daily sentiment features; generate a binary label:
       y = 1 if next_day_close > today_close, else 0.
    We forward-fill missing sentiment to trading days (no-leak because only same-day signal).
    """
    # Technicals
    tech = engineer_technical_indicators(df_prices)
    tech = tech.dropna()

    # Choose which sentiment column to use as main signal
    sent_col = "daily_weighted_mean" if (use_weighted_sentiment and "daily_weighted_mean" in df_daily_sent.columns) else "daily_mean"
    sent = df_daily_sent.copy()
    if sent.empty or (sent_col not in sent.columns):
        # no sentiment available; make empty frame to be merged (baseline will still work)
        sent = pd.DataFrame(index=tech.index)
    else:
        # Optional enhancement: rolling volatility of daily sentiment
        if add_sent_volatility and "daily_mean" in sent.columns:
            sent["sent_vol_5"] = sent["daily_mean"].rolling(5).std()
            sent["sent_vol_10"] = sent["daily_mean"].rolling(10).std()

    # Align by date and forward-fill sentiment to trading days (no look-ahead)
    feats = tech.join(sent[[c for c in sent.columns if c != "daily_sum"]], how="left")
    feats = feats.ffill().dropna()

    # Label: next day's direction (based on Close)
    close = df_prices["Close"].reindex(feats.index).ffill()
    y = (close.shift(-1) > close).astype(int).loc[feats.index]
    feats = feats.iloc[:-1, :]
    y = y.iloc[:-1]

    # Drop raw prices to avoid leakage (keep differences/ratios/indicators)
    drop_cols = ["Open","High","Low","Adj Close","Close"]
    keep_cols = [c for c in feats.columns if c not in drop_cols]
    X = feats[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()

    # Align y with X after dropping NA
    y = y.reindex(X.index)

    return X, y


# ----------------------------- (5) Modeling & Evaluation -----------------------------
def _make_classifier(model_type: str = "logreg", random_state: int = 42):
    """
    Factory for simple, well-regularized classifiers.
    """
    model_type = (model_type or "logreg").lower()
    if model_type == "logreg":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=200, class_weight=None, C=1.0, solver="lbfgs")),
        ])
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        )
    elif model_type == "xgb" and _HAS_XGB:
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=random_state, eval_metric="logloss", n_jobs=-1
        )
    else:
        # default fallback
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=200, C=1.0)),
        ])

def evaluate_classifier_with_baseline(
    X: pd.DataFrame, y: pd.Series, *,
    sentiment_cols_prefix=("daily_", "pos_", "neg_", "neu_", "finbert_", "sent_"),
    model_type: str = "logreg",
    test_size: float = 0.25,
    random_state: int = 42,
    results_dir: str = "charts",
    results_tag: str = "sentclf",
):
    """
    Train/test split, fit model with ALL features (tech + sentiment),
    and compare to a baseline that excludes sentiment features.

    Saves a confusion-matrix plot. Returns a metrics dict.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Chronological split to avoid leakage
    n = len(X)
    cut = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    # Separate baseline (drop sentiment-ish columns heuristically by prefix)
    sent_like = [c for c in X.columns if any(c.startswith(p) for p in sentiment_cols_prefix)]
    X_train_base = X_train.drop(columns=sent_like, errors="ignore")
    X_test_base  = X_test.drop(columns=sent_like, errors="ignore")

    # Models
    clf = _make_classifier(model_type=model_type, random_state=random_state)
    clf_base = _make_classifier(model_type=model_type, random_state=random_state)

    clf.fit(X_train, y_train)
    clf_base.fit(X_train_base, y_train)

    y_pred = clf.predict(X_test)
    y_pred_b = clf_base.predict(X_test_base)

    def _metrics_block(y_true, y_hat, label):
        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        cm = confusion_matrix(y_true, y_hat)
        print(f"[{label}] Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

    m_full = _metrics_block(y_test, y_pred, "With Sentiment")
    m_base = _metrics_block(y_test, y_pred_b, "Baseline (No Sentiment)")

    # Plot confusion matrices
    def _plot_cm(cm, title, path):
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap="Blues")
        plt.title(title)
        plt.xticks([0,1], ["Down","Up"])
        plt.yticks([0,1], ["Down","Up"])
        for (i,j), v in np.ndenumerate(cm):
            plt.text(j, i, int(v), ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

    _plot_cm(m_full["confusion_matrix"], f"Confusion Matrix — {results_tag} (With Sentiment)",
             os.path.join(results_dir, f"{results_tag}_cm_full.png"))
    _plot_cm(m_base["confusion_matrix"], f"Confusion Matrix — {results_tag} (Baseline)",
             os.path.join(results_dir, f"{results_tag}_cm_base.png"))

    return {"with_sentiment": m_full, "baseline": m_base}


# ----------------------------- (6) Orchestrator: Run the full pipeline -----------------------------
def run_sentiment_classification_pipeline(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    prices_df: pd.DataFrame | None = None,  # optional: pass preloaded price DF with Date index
    csv_news: str | None = None,            # optional: path to CSV with news
    sentiment_engine: str = "vader",        # "vader" or "finbert"
    model_type: str = "logreg",             # "logreg" | "rf" | "xgb"
    weight_by_recency: bool = True,         # enhancement
    decay_per_hour: float = 0.10,
    results_dir: str = "charts",
):
    """
    High-level helper that:
      (a) loads price OHLCV via yfinance if not supplied
      (b) collects/loads text data
      (c) computes sentiment per article (VADER or FinBERT)
      (d) aggregates to daily
      (e) builds features + labels (Up/Down)
      (f) trains classifier & evaluates vs baseline; saves confusion matrices

    Returns:
      X, y, daily_sentiment, metrics_dict
    """
    os.makedirs(results_dir, exist_ok=True)

    # Load prices if not provided (uses your existing normalization if available)
    if prices_df is None:
        import yfinance as yf
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, group_by="column", progress=False)
        # If you have _normalize_ohlcv in scope (from your earlier code), prefer it:
        try:
            df_prices = _normalize_ohlcv(raw, require_ohlc=False)
        except Exception:
            df_prices = raw.copy()
        df_prices.index.name = "Date"
    else:
        df_prices = prices_df.copy().sort_index()

    # Collect text
    df_news = collect_text_data_yfinance_or_csv(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        csv_path=csv_news,
        cache_dir="data_cache",
    )

    # If no news found, warn but continue to enable baseline
    if df_news.empty:
        print("[Sentiment] No news found in the range; proceeding with baseline-only features.")

    # Compute per-article sentiment
    if sentiment_engine.lower() == "finbert":
        try:
            df_sent = sentiment_scores_finbert(df_news, text_col="text")
            engine_used = "finbert"
        except Exception as e:
            print("[FinBERT] Failed, falling back to VADER. Error:", e)
            df_sent = sentiment_scores_vader(df_news, text_col="text")
            engine_used = "vader"
    else:
        df_sent = sentiment_scores_vader(df_news, text_col="text")
        engine_used = "vader"

    # Aggregate to daily with enhancement (recency weighting + volatility)
    daily_sent = aggregate_sentiment_daily(
        df_sent, engine=engine_used,
        weight_by_recency=weight_by_recency,
        decay_per_hour=decay_per_hour
    )

    # Build features & labels
    X, y = build_features_and_labels_for_direction(
        df_prices=df_prices,
        df_daily_sent=daily_sent,
        use_weighted_sentiment=True,
        add_sent_volatility=True,
    )

    # Evaluate vs baseline
    metrics = evaluate_classifier_with_baseline(
        X, y,
        model_type=model_type,
        results_dir=results_dir,
        results_tag=f"{ticker}_{engine_used}_{model_type}"
    )

    # Optional: quick feature importances if RF/XGB
    if model_type in ("rf", "xgb"):
        # Refit on full dataset for simple importance view (illustrative only)
        clf = _make_classifier(model_type=model_type, random_state=42)
        clf.fit(X, y)
        try:
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "named_steps") and "clf" in clf.named_steps and hasattr(clf.named_steps["clf"], "feature_importances_"):
                importances = clf.named_steps["clf"].feature_importances_
            else:
                importances = None
            if importances is not None:
                imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:20]
                plt.figure(figsize=(8,6))
                imp.iloc[:20].plot(kind="bar")
                plt.title(f"Top Feature Importances — {ticker} ({model_type})")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"{ticker}_{engine_used}_{model_type}_feat_importance.png"), dpi=150)
                plt.close()
        except Exception:
            pass

    return X, y, daily_sent, metrics
# ============================== END SENTIMENT + CLASSIFICATION PIPELINE ==============================


if __name__ == "__main__":
    # ====================== CLI-ENABLED TASK RUNNER ======================
    import os, sys, json, numpy as np, argparse
    import yfinance as yf
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    from datetime import datetime

    os.makedirs("charts", exist_ok=True)

    # ---------- helpers ----------
    def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def safe(task_name, fn, *, on_error="continue"):
        """Run a task safely. on_error: 'continue' | 'raise' """
        try:
            log(f"▶ {task_name} ...")
            out = fn()
            log(f"✓ {task_name} done.")
            return out
        except Exception as e:
            log(f"✗ {task_name} failed: {e}")
            if on_error == "raise":
                raise
            return None

    def save_step_series_plot(y_true_2d, y_pred_2d, *, step:int, ticker:str, tag:str, save_path:str):
        idx = step - 1
        y_t = np.asarray(y_true_2d)[:, idx].ravel()
        y_p = np.asarray(y_pred_2d)[:, idx].ravel()
        plt.figure(figsize=(18, 5))
        plt.plot(y_t, label=f"Actual (t+{step})")
        plt.plot(y_p, label=f"Predicted (t+{step})")
        plt.title(f"{ticker} {tag}: Step-{step} Actual vs Predicted")
        plt.xlabel("Test sample index"); plt.ylabel("Price"); plt.legend()
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

    # ---------- CLI argument parser ----------
    parser = argparse.ArgumentParser(
        description="Run stock prediction tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python final.py --all                      # Run all tasks
  python final.py --tasks 1 3 5              # Run tasks 1, 3, and 5
  python final.py --univariate --ensemble    # Run specific tasks
  python final.py --ensemble-3way --ticker AAPL  # 3-way ensemble
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all tasks (default if no tasks specified)')
    parser.add_argument('--tasks', '-t', nargs='+', type=int, choices=range(1, 9),
                        metavar='N', help='Task numbers to run (1-8)')
    
    # Named task flags
    parser.add_argument('--univariate', action='store_true', help='Run Task 1: Univariate multistep')
    parser.add_argument('--mv-single', action='store_true', help='Run Task 2: Multivariate single-step')
    parser.add_argument('--mv-multistep', action='store_true', help='Run Task 3: Multivariate multistep')
    parser.add_argument('--experiments', action='store_true', help='Run Task 4: Model factory experiments')
    parser.add_argument('--charts', action='store_true', help='Run Task 5: Charts (Candlestick + Boxplot)')
    parser.add_argument('--ensemble', action='store_true', help='Run Task 6: Ensemble (ARIMA + DL)')
    parser.add_argument('--sentiment', action='store_true', help='Run Task 7: Sentiment classification')
    parser.add_argument('--ensemble-3way', action='store_true', help='Run Task 8: 3-Way Ensemble (LSTM+ARIMA+RF)')
    
    # Config overrides
    parser.add_argument('--ticker', default=None, help='Override stock ticker (default: COMPANY variable)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window (default: 60)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default: 20)')
    parser.add_argument('--arima-order', type=int, nargs=3, default=[5,1,0], 
                       metavar=('p','d','q'), help='ARIMA order (default: 5 1 0)')
    parser.add_argument('--seasonal-order', type=int, nargs=4, default=None,
                       metavar=('P','D','Q','s'), help='SARIMA seasonal order (e.g., 1 1 1 12)')
    
    args = parser.parse_args()

    # ---------- determine which tasks to run ----------
    tasks_to_run = set()
    
    if args.tasks:
        tasks_to_run.update(args.tasks)
    
    if args.univariate: tasks_to_run.add(1)
    if args.mv_single: tasks_to_run.add(2)
    if args.mv_multistep: tasks_to_run.add(3)
    if args.experiments: tasks_to_run.add(4)
    if args.charts: tasks_to_run.add(5)
    if args.ensemble: tasks_to_run.add(6)
    if args.sentiment: tasks_to_run.add(7)
    if args.ensemble_3way: tasks_to_run.add(8)
    
    # If no tasks specified, run all (or just exit with help)
    if not tasks_to_run and not args.all:
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        args.all = True
    
    if args.all:
        tasks_to_run = {1, 2, 3, 4, 5, 6, 7, 8}

    # ---------- shared config ----------
    TICKER = args.ticker if args.ticker else COMPANY
    k_univariate = 5
    k_multi = 5
    LOOKBACK = args.lookback
    EPOCHS = args.epochs
    ARIMA_ORDER = tuple(args.arima_order)
    SEASONAL_ORDER = tuple(args.seasonal_order) if args.seasonal_order else None

    log(f"Running tasks: {sorted(tasks_to_run)}")
    log(f"Config: ticker={TICKER}, lookback={LOOKBACK}, epochs={EPOCHS}")
    if 8 in tasks_to_run or 6 in tasks_to_run:
        log(f"ARIMA: order={ARIMA_ORDER}, seasonal={SEASONAL_ORDER}")

    # ---------- task definitions ----------
    results = {}

    # ====================== Task 1 — Univariate multistep ======================
    def _task1():
        model_u, (u_true, u_pred) = train_univariate_multistep(
            ticker=TICKER, start_date=TRAIN_START, end_date=TRAIN_END,
            lookback=LOOKBACK, horizon=k_univariate, start_offset=0,
            epochs=EPOCHS, batch_size=64, layer_type="lstm", layer_sizes=(128, 128),
            dropout=0.2, bidirectional=False,
        )
        save_step_series_plot(u_true, u_pred,
            step=1, ticker=TICKER, tag="Univariate",
            save_path=f"charts/{TICKER}_uni_step1.png")
        save_step_series_plot(u_true, u_pred,
            step=k_univariate, ticker=TICKER, tag="Univariate",
            save_path=f"charts/{TICKER}_uni_step{k_univariate}.png")
        if os.path.exists("multistep_example.png"):
            os.replace("multistep_example.png", f"charts/{TICKER}_uni_{k_univariate}step.png")
        return {"model": model_u, "y_true": u_true, "y_pred": u_pred}

    # ====================== Task 2 — Multivariate single-step ======================
    def _task2():
        model_m1, (mv1_true, mv1_pred) = train_multivariate_single_step(
            ticker=TICKER, start_date=TRAIN_START, end_date=TRAIN_END,
            lookback=LOOKBACK, epochs=EPOCHS, batch_size=64,
            layer_type="lstm", layer_sizes=(128, 128),
            dropout=0.2, bidirectional=False,
            feature_columns=("Open", "High", "Low", "Close", "Adj Close", "Volume"),
        )
        save_step_series_plot(
            y_true_2d=np.expand_dims(mv1_true, 1),
            y_pred_2d=np.expand_dims(mv1_pred, 1),
            step=1, ticker=TICKER, tag="Multivariate 1-step",
            save_path=f"charts/{TICKER}_mv1_step1.png"
        )
        return {"model": model_m1, "y_true": mv1_true, "y_pred": mv1_pred}

    # ====================== Task 3 — Multivariate multistep ======================
    def _task3():
        model_mm, (mm_true, mm_pred) = train_multivariate_multistep(
            ticker=TICKER, start_date=TRAIN_START, end_date=TRAIN_END,
            lookback=LOOKBACK, horizon=k_multi, start_offset=0,
            epochs=EPOCHS, batch_size=64, layer_type="lstm", layer_sizes=(128, 128),
            dropout=0.2, bidirectional=False,
            feature_columns=("Open", "High", "Low", "Close", "Adj Close", "Volume"),
        )
        save_step_series_plot(mm_true, mm_pred,
            step=1, ticker=TICKER, tag="Multivariate",
            save_path=f"charts/{TICKER}_mvk_step1.png")
        save_step_series_plot(mm_true, mm_pred,
            step=k_multi, ticker=TICKER, tag="Multivariate",
            save_path=f"charts/{TICKER}_mvk_step{k_multi}.png")
        if os.path.exists("multistep_example.png"):
            os.replace("multistep_example.png", f"charts/{TICKER}_mv_{k_multi}step.png")
        return {"model": model_mm, "y_true": mm_true, "y_pred": mm_pred}

    # # ====================== Task 4 — Model factory experiments ======================
    # def _task4():
    #     MY_CONFIGS = [
    #         {"tag": "LSTM_Baseline", "layer_type": "lstm", "layer_sizes": [50, 50, 50], "dropout": 0.2, "bidirectional": False, "epochs": 25},
    #         {"tag": "GRU_64x2", "layer_type": "gru", "layer_sizes": [64, 64], "dropout": 0.2, "epochs": 30},
    #         {"tag": "BiLSTM_64x2", "layer_type": "lstm", "layer_sizes": [64, 64], "dropout": 0.2, "bidirectional": True, "epochs": 30},
    #         {"tag": "GRU_Deep", "layer_type": "gru", "layer_sizes": [128, 128, 64], "dropout": 0.3, "epochs": 40},
    #     ]
    #     return run_experiments_v03(
    #         ticker=TICKER,
    #         start_date="2020-01-01",
    #         end_date="2023-08-01",
    #         lookback=LOOKBACK,
    #         feature_columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    #         configs=MY_CONFIGS,
    #         verbose=1,
    #     )

    # ====================== Task 4 Model factory experiments ======================
    def _task4():
        MY_CONFIGS = [
            {"tag": "LSTM_Baseline", "layer_type": "lstm", "layer_sizes": [50, 50, 50], "dropout": 0.2, "bidirectional": False, "epochs": 25},
            {"tag": "GRU_64x2", "layer_type": "gru", "layer_sizes": [64, 64], "dropout": 0.2, "epochs": 30},
            {"tag": "BiLSTM_64x2", "layer_type": "lstm", "layer_sizes": [64, 64], "dropout": 0.2, "bidirectional": True, "epochs": 30},
            {"tag": "GRU_Deep", "layer_type": "gru", "layer_sizes": [128, 128, 64], "dropout": 0.3, "epochs": 40},
        ]
        return run_experiments_v03(
            ticker=TICKER,  # Fixed: use actual ticker variable
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            lookback=LOOKBACK,
            feature_columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
            configs=MY_CONFIGS,
            verbose=1,
        )

    # ====================== Task 5 — Charts ======================
    def _task5_charts():
        df_for_plots = yf.download(
            TICKER, start=TRAIN_START, end=TRAIN_END,
            auto_adjust=False, group_by="column", progress=False,
        )
        df_for_plots = _normalize_ohlcv(df_for_plots, require_ohlc=False)

        plot_candlestick(
            df=df_for_plots, n=1, title=f"{TICKER} Candlestick (1-day candles)",
            volume=True, mav=(20, 50), style="yahoo", figsize=(12, 6),
            save_path=f"charts/{TICKER}_candles_1d.png",
        )
        plot_candlestick(
            df=df_for_plots, n=3, title=f"{TICKER} Candlestick (3-day candles)",
            volume=True, mav=(20, 50), style="yahoo", figsize=(12, 6),
            save_path=f"charts/{TICKER}_candles_3d.png",
        )
        plot_boxplot_moving_window(
            df=df_for_plots, column="Close", window=20, step=5, showfliers=False,
            title=f"{TICKER} Close — 20-day Window Boxplots",
            figsize=(12, 6), save_path="boxplots.png",
        )
        return True

    # ====================== Task 6 — Ensemble (2-way: ARIMA + DL) ======================
    def _task6_ensemble():
        ens_results = run_ensemble_lstm_arima_single_step(
            ticker=TICKER,
            start_date=TRAIN_START, end_date=TRAIN_END,
            lookback=LOOKBACK,
            feature_columns=("Open","High","Low","Close","Adj Close","Volume"),
            target_column="Close",
            split_method="ratio", train_ratio=0.8, val_ratio=0.15,
            handle_nan="ffill",
            dl_layer_type="lstm", dl_layer_sizes=(128,128),
            dl_dropout=0.2, dl_bidirectional=False, dl_epochs=EPOCHS, dl_batch_size=64,
            arima_order=ARIMA_ORDER, seasonal_order=SEASONAL_ORDER,
            try_rf=True, results_prefix="v05",
        )
        try:
            y_true = ens_results["test"]["y_true"]
            y_dl = ens_results["test"]["DL"]
            y_arima = ens_results["test"]["ARIMA"]
            y_combined = ens_results["test"]["DL+ARIMA"]

            def _save_series_plot(y, lines: dict, title, path):
                plt.figure(figsize=(18,5))
                plt.plot(y, label="Actual")
                for k, v in lines.items():
                    plt.plot(v, label=k)
                plt.title(title); plt.xlabel("Test sample index"); plt.ylabel("Price"); plt.legend()
                plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

            _save_series_plot(y_true, {"DL+ARIMA": y_combined, "DL": y_dl},
                              f"{TICKER} Combined vs DL (1-step)",
                              os.path.join("charts", f"v05_{TICKER}_combined_vs_DL.png"))
            _save_series_plot(y_true, {"DL+ARIMA": y_combined, "ARIMA": y_arima},
                              f"{TICKER} Combined vs ARIMA (1-step)",
                              os.path.join("charts", f"v05_{TICKER}_combined_vs_ARIMA.png"))
        except Exception as _e:
            log(f"[Ensemble] Extra plots skipped: {_e}")
        
        try:
            detailed_results = run_ensemble_with_detailed_plots(
                ticker=TICKER,
                start_date=TRAIN_START,
                end_date=TRAIN_END,
                lookback=60,
                dl_epochs=EPOCHS,
                arima_order=ARIMA_ORDER,
                seasonal_order=SEASONAL_ORDER,
                results_dir="ensemble_results",
            )
            log(f"✓ Detailed ensemble complete! Optimal blend: {detailed_results['optimal_weight']:.1%} LSTM")
        except Exception as e:
            log(f"[Detailed Ensemble] Skipped: {e}")
        
        return ens_results

    # ====================== Task 7 — Sentiment ======================
    def _task7_sentiment():
        results_dict = {}
        for mtype in ("logreg", "rf"):
            try:
                log(f"[Sentiment] Running classifier={mtype} ...")
                X, y, daily_sent, metrics = run_sentiment_classification_pipeline(
                    ticker=TICKER,
                    start_date=TRAIN_START, end_date=TRAIN_END,
                    csv_news=None,
                    sentiment_engine="vader",
                    model_type=mtype,
                    weight_by_recency=True,
                    decay_per_hour=0.10,
                    results_dir="charts",
                )
                results_dict[mtype] = metrics
            except Exception as e:
                log(f"[Sentiment] {mtype} failed: {e}")
        
        try:
            X, y, daily_sent, metrics = run_sentiment_classification_pipeline(
                ticker=TICKER,
                start_date=TRAIN_START, end_date=TRAIN_END,
                csv_news=None,
                sentiment_engine="finbert",
                model_type="logreg",
                weight_by_recency=True,
                decay_per_hour=0.10,
                results_dir="charts",
            )
            results_dict["finbert_logreg"] = metrics
        except Exception as e:
            log(f"[Sentiment] FinBERT run skipped: {e}")
        
        try:
            with open(os.path.join("charts", f"{TICKER}_sentiment_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(results_dict, f, indent=2)
        except Exception as e:
            log(f"[Sentiment] Saving metrics JSON failed: {e}")
        return results_dict

    # ====================== Task 8 — 3-Way Ensemble (LSTM + ARIMA + RF) ======================
    def _task8_ensemble_3way():
        return run_three_way_ensemble(
            ticker=TICKER,
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            lookback=LOOKBACK,
            arima_order=ARIMA_ORDER,
            seasonal_order=SEASONAL_ORDER,
            results_dir="charts",
        )

    # ---------- execute selected tasks ----------
    if 1 in tasks_to_run:
        results[1] = safe("Task 1: Univariate multistep", _task1)
    
    if 2 in tasks_to_run:
        results[2] = safe("Task 2: Multivariate single-step", _task2)
    
    if 3 in tasks_to_run:
        results[3] = safe("Task 3: Multivariate multistep", _task3)
    
    if 4 in tasks_to_run:
        results[4] = safe("Task 4: DL factory experiments", _task4)
    
    if 5 in tasks_to_run:
        results[5] = safe("Task 5: Charts", _task5_charts)
    
    if 6 in tasks_to_run:
        results[6] = safe("Task 6: Ensemble (ARIMA + DL)", _task6_ensemble)
    
    if 7 in tasks_to_run:
        results[7] = safe("Task 7: Sentiment + Classification", _task7_sentiment)
    
    if 8 in tasks_to_run:
        results[8] = safe("Task 8: 3-Way Ensemble (LSTM+ARIMA+RF)", _task8_ensemble_3way)

    # ====================== Final summary ======================
    log("Selected tasks completed.")
    try:
        summary = {
            "tasks_run": sorted(tasks_to_run),
            "T1_univariate": bool(results.get(1)),
            "T2_mv_single": bool(results.get(2)),
            "T3_mv_multistep": bool(results.get(3)),
            "T4_experiments": bool(results.get(4) is not None),
            "T5_charts": bool(results.get(5)),
            "T6_ensemble": bool(results.get(6)),
            "T7_sentiment": bool(results.get(7)),
            "T8_ensemble_3way": bool(results.get(8)),
            "config": {
                "ticker": TICKER,
                "lookback": LOOKBACK,
                "epochs": EPOCHS,
                "arima_order": ARIMA_ORDER,
                "seasonal_order": SEASONAL_ORDER,
            }
        }
        with open(os.path.join("charts", "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary: {summary}")
    except Exception as e:
        log(f"Failed to write summary: {e}")