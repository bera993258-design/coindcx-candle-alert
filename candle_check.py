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
    "5m"
)

ATR_PERIOD = 14
ATR_MULTIPLIER = 1.2
MIN_BODY_RATIO = 0.60
LIMIT = 100

COINDCX_URL = (
    "https://public.coindcx.com/"
    "market_data/candles/"
)

IST = timezone(
    timedelta(hours=5, minutes=30)
)


def get_ist_time():
    return datetime.now(IST).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def timestamp_to_ist(timestamp):
    """
    CoinDCX timestamp milliseconds
    অথবা seconds হলে সেটিকে IST time-এ রূপান্তর করে।
    """

    if timestamp is None:
        return ""

    try:
        timestamp_number = float(timestamp)
    except (TypeError, ValueError):
        return str(timestamp)

    # Unix milliseconds সাধারণত 13 digit
    # Unix seconds সাধারণত 10 digit
    if timestamp_number < 100000000000:
        timestamp_number *= 1000

    date_time = datetime.fromtimestamp(
        timestamp_number / 1000,
        tz=IST
    )

    return date_time.strftime(
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

    try:
        response = requests.get(
            COINDCX_URL,
            params=params,
            headers=headers,
            timeout=30
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"CoinDCX connection error: {error}"
        ) from error

    if response.status_code != 200:
        error_text = response.text[:500]

        raise RuntimeError(
            "CoinDCX API error: "
            f"{response.status_code} | "
            f"{error_text}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "CoinDCX valid JSON পাঠায়নি: "
            f"{response.text[:500]}"
        ) from error

    if not isinstance(data, list):
        raise RuntimeError(
            "CoinDCX response list নয়"
        )

    if len(data) == 0:
        raise RuntimeError(
            "CoinDCX থেকে কোনো candle data পাওয়া যায়নি"
        )

    return pd.DataFrame(data)


def calculate_signal(df):
    required_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "CoinDCX response-এ column নেই: "
            f"{missing_columns}"
        )

    df = df.copy()

    # Time numeric করে sort
    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["time"]
    )

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    ).reset_index(drop=True)

    if len(df) < ATR_PERIOD:
        raise RuntimeError(
            f"ATR({ATR_PERIOD}) calculation-এর "
            "জন্য যথেষ্ট candle নেই"
        )

    # Human-readable IST candle time
    df["candle_time_ist"] = df["time"].apply(
        timestamp_to_ist
    )

    # Body = ABS(Close - Open)
    df["body"] = (
        df["close"] - df["open"]
    ).abs()

    # Range = High - Low
    df["range"] = (
        df["high"] - df["low"]
    )

    # Body / Range
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

    # Simple rolling ATR(14)
    df["atr14"] = (
        df["true_range"]
        .rolling(
            window=ATR_PERIOD,
            min_periods=ATR_PERIOD
        )
        .mean()
    )

    # Body >= 1.2 x ATR(14)
    df["required_body"] = (
        ATR_MULTIPLIER * df["atr14"]
    )

    df["condition_1"] = (
        df["body"] >= df["required_body"]
    )

    # Body / Range >= 60%
    df["condition_2"] = (
        df["body_ratio"] >= MIN_BODY_RATIO
    )

    # Both conditions must be true
    df["final_signal"] = (
        df["condition_1"] &
        df["condition_2"]
    )

    return df


def safe_value(value):
    """
    NaN হলে blank return করে।
    Numpy value হলে normal Python value return করে।
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        return value.item()

    return value


def create_result(df):
    latest = df.iloc[-1]

    result = {
        "checked_time_ist": get_ist_time(),
        "pair": PAIR,
        "timeframe": INTERVAL,
        "candle_time_ist": latest[
            "candle_time_ist"
        ],
        "open": safe_value(
            latest["open"]
        ),
        "high": safe_value(
            latest["high"]
        ),
        "low": safe_value(
            latest["low"]
        ),
        "close": safe_value(
            latest["close"]
        ),
        "body": safe_value(
            latest["body"]
        ),
        "range": safe_value(
            latest["range"]
        ),
        "body_ratio_percent": safe_value(
            latest["body_ratio"] * 100
        ),
        "atr14": safe_value(
            latest["atr14"]
        ),
        "required_body": safe_value(
            latest["required_body"]
        ),
        "condition_1": bool(
            latest["condition_1"]
        ),
        "condition_2": bool(
            latest["condition_2"]
        ),
        "final_signal": bool(
            latest["final_signal"]
        ),
        "signal_text": (
            "VALID SIGNAL"
            if bool(latest["final_signal"])
            else "NO SIGNAL"
        )
    }

    return result


def save_files(df, result):
    os.makedirs(
        "output",
        exist_ok=True
    )

    # Latest result JSON
    with open(
        "output/live_result.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Latest result CSV
    pd.DataFrame(
        [result]
    ).to_csv(
        "output/live_result.csv",
        index=False
    )

    # All candle CSV
    all_columns = [
        "candle_time_ist",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "body",
        "range",
        "body_ratio",
        "true_range",
        "atr14",
        "required_body",
        "condition_1",
        "condition_2",
        "final_signal"
    ]

    available_columns = [
        column
        for column in all_columns
        if column in df.columns
    ]

    output_df = df[available_columns].copy()

    output_df["body_ratio_percent"] = (
        output_df["body_ratio"] * 100
    )

    output_df.to_csv(
        "output/all_candles.csv",
        index=False
    )


def print_result(result):
    print("")
    print("=" * 45)
    print("CoinDCX Candle Analysis")
    print("=" * 45)
    print(
        f"Checked Time IST: "
        f"{result['checked_time_ist']}"
    )
    print(
        f"Candle Time IST: "
        f"{result['candle_time_ist']}"
    )
    print(
        f"Pair: {result['pair']}"
    )
    print(
        f"Timeframe: {result['timeframe']}"
    )
    print(
        f"Open: {result['open']}"
    )
    print(
        f"High: {result['high']}"
    )
    print(
        f"Low: {result['low']}"
    )
    print(
        f"Close: {result['close']}"
    )
    print(
        f"Body: {result['body']}"
    )
    print(
        f"Range: {result['range']}"
    )
    print(
        f"Body/Range: "
        f"{result['body_ratio_percent']}%"
    )
    print(
        f"ATR(14): {result['atr14']}"
    )
    print(
        f"Required Body: "
        f"{result['required_body']}"
    )
    print(
        f"Condition 1: "
        f"{result['condition_1']}"
    )
    print(
        f"Condition 2: "
        f"{result['condition_2']}"
    )
    print(
        f"Final Signal: "
        f"{result['signal_text']}"
    )
    print("=" * 45)


def main():
    print(
        f"Fetching {PAIR} {INTERVAL} candles..."
    )

    candles_df = get_candles()

    calculated_df = calculate_signal(
        candles_df
    )

    result = create_result(
        calculated_df
    )

    save_files(
        calculated_df,
        result
    )

    print_result(result)

    print(
        "CSV files successfully saved in output/"
    )


if __name__ == "__main__":
    main()
