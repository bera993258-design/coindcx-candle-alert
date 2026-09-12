import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests


PAIR = os.getenv(
    "PAIR",
    "B-BTC_USDT"
)

INTERVAL = os.getenv(
    "INTERVAL",
    "15m"
)

ATR_PERIOD = 14
ATR_MULTIPLIER = 1.2
MIN_BODY_RATIO = 0.60
LIMIT = 100

API_URL = (
    "https://public.coindcx.com/"
    "market_data/candles/"
)

IST = timezone(
    timedelta(hours=5, minutes=30)
)

HISTORY_FILE = (
    "output/closed_candles_history.csv"
)


def get_ist_time():
    return datetime.now(IST).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def timestamp_to_ist(timestamp):
    value = float(timestamp)

    if value < 100000000000:
        value *= 1000

    dt = datetime.fromtimestamp(
        value / 1000,
        tz=IST
    )

    return dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def get_candles():
    params = {
        "pair": PAIR,
        "interval": INTERVAL,
        "limit": LIMIT
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=headers,
        timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"CoinDCX API error: "
            f"{response.status_code} | "
            f"{response.text[:300]}"
        )

    data = response.json()

    if not isinstance(data, list) or not data:
        raise RuntimeError(
            "CoinDCX candle data পাওয়া যায়নি"
        )

    return pd.DataFrame(data)


def calculate(df):
    required = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df = df.copy()

    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "time",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # শেষ row চলমান candle হতে পারে।
    # তাই সেটি বাদ দিচ্ছি।
    if len(df) <= ATR_PERIOD:
        raise RuntimeError(
            "ATR calculation-এর জন্য "
            "যথেষ্ট candle নেই"
        )

    df = df.iloc[:-1].copy()

    if len(df) < ATR_PERIOD:
        raise RuntimeError(
            "Closed candle বাদ দেওয়ার পরে "
            "যথেষ্ট data নেই"
        )

    df["candle_start_time_ist"] = (
        df["time"].apply(timestamp_to_ist)
    )

    df["candle_end_time_ist"] = (
        pd.to_datetime(
            df["time"],
            unit="ms",
            utc=True
        )
        .dt.tz_convert("Asia/Kolkata")
        + pd.Timedelta(minutes=15)
    ).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df["body"] = (
        df["close"] - df["open"]
    ).abs()

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, pd.NA)
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"] - previous_close
            ).abs(),
            (
                df["low"] - previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    df["true_range"] = true_range

    df["atr14"] = (
        df["true_range"]
        .rolling(
            ATR_PERIOD,
            min_periods=ATR_PERIOD
        )
        .mean()
    )

    df["required_body"] = (
        ATR_MULTIPLIER * df["atr14"]
    )

    df["condition_1"] = (
        df["body"] >= df["required_body"]
    )

    df["condition_2"] = (
        df["body_ratio"] >= MIN_BODY_RATIO
    )

    df["final_signal"] = (
        df["condition_1"] &
        df["condition_2"]
    )

    df["signal_text"] = df[
        "final_signal"
    ].map(
        {
            True: "VALID SIGNAL",
            False: "NO SIGNAL"
        }
    )

    df["checked_time_ist"] = get_ist_time()

    df["pair"] = PAIR
    df["timeframe"] = INTERVAL

    return df


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(
            HISTORY_FILE
        )
    except Exception:
        return pd.DataFrame()


def save_history(df):
    os.makedirs(
        "output",
        exist_ok=True
    )

    history_columns = [
        "candle_start_time_ist",
        "candle_end_time_ist",
        "checked_time_ist",
        "pair",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "body",
        "range",
        "body_ratio",
        "body_ratio_percent",
        "true_range",
        "atr14",
        "required_body",
        "condition_1",
        "condition_2",
        "final_signal",
        "signal_text"
    ]

    available = [
        column
        for column in history_columns
        if column in df.columns
    ]

    output = df[available].copy()

    output["body_ratio_percent"] = (
        output["body_ratio"] * 100
    )

    output.to_csv(
        HISTORY_FILE,
        index=False
    )

    # Latest result
    latest = output.tail(1)

    latest.to_csv(
        "output/live_result.csv",
        index=False
    )

    latest.to_json(
        "output/live_result.json",
        orient="records",
        indent=2
    )


def main():
    print(
        f"Fetching {PAIR} {INTERVAL} candles"
    )

    raw = get_candles()
    calculated = calculate(raw)

    old_history = load_history()

    new_rows = calculated.copy()

    if not old_history.empty:
        combined = pd.concat(
            [
                old_history,
                new_rows
            ],
            ignore_index=True
        )
    else:
        combined = new_rows

    combined = combined.drop_duplicates(
        subset=[
            "pair",
            "timeframe",
            "candle_start_time_ist"
        ],
        keep="last"
    )

    combined = combined.sort_values(
        "candle_start_time_ist"
    ).reset_index(drop=True)

    # শেষ 1000টি candle রাখা
    combined = combined.tail(1000)

    save_history(combined)

    latest = combined.iloc[-1]

    print("")
    print(
        "Latest closed candle:"
    )
    print(
        f"Start IST: "
        f"{latest['candle_start_time_ist']}"
    )
    print(
        f"End IST: "
        f"{latest['candle_end_time_ist']}"
    )
    print(
        f"Signal: "
        f"{latest['signal_text']}"
    )


if __name__ == "__main__":
    main()
