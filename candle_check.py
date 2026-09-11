import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests


PAIR = os.getenv("PAIR", "B-BTC_USDT")
INTERVAL = os.getenv("INTERVAL", "5m")

ATR_PERIOD = 14
ATR_MULTIPLIER = 1.2
MIN_BODY_RATIO = 0.60
LIMIT = 100

COINDCX_URL = (
    "https://public.coindcx.com/"
    "market_data/candles/"
)

IST = timezone(timedelta(hours=5, minutes=30))


def get_ist_time():
    return datetime.now(IST).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def get_candles():
    params = {
        "pair": PAIR,
        "interval": INTERVAL,
        "limit": LIMIT
    }

    response = requests.get(
        COINDCX_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(
            "CoinDCX থেকে কোনো candle data পাওয়া যায়নি"
        )

    return pd.DataFrame(data)


def calculate_signal(df):
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
        .rolling(ATR_PERIOD)
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

    return df


def create_result(df):
    latest = df.iloc[-1]

    result = {
        "checked_time_ist": get_ist_time(),
        "pair": PAIR,
        "timeframe": INTERVAL,
        "candle_time": str(latest["time"]),
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "close": latest["close"],
        "body": latest["body"],
        "range": latest["range"],
        "body_ratio_percent": (
            latest["body_ratio"] * 100
        ),
        "atr14": latest["atr14"],
        "required_body": latest["required_body"],
        "condition_1": latest["condition_1"],
        "condition_2": latest["condition_2"],
        "final_signal": latest["final_signal"]
    }

    return result


def clean_result(result):
    cleaned = {}

    for key, value in result.items():
        if pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value

    return cleaned


def main():
    df = get_candles()
    df = calculate_signal(df)

    result = create_result(df)
    result = clean_result(result)

    os.makedirs("output", exist_ok=True)

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

    pd.DataFrame([result]).to_csv(
        "output/live_result.csv",
        index=False
    )

    df.to_csv(
        "output/all_candles.csv",
        index=False
    )

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
