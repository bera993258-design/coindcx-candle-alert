import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

PAIR = os.getenv("PAIR", "B-BTC_USDT")
INTERVAL = os.getenv("INTERVAL", "5m")
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.2
MIN_BODY_RATIO = 0.60
LIMIT = 100

URL = "https://public.coindcx.com/market_data/candles/"

params = {
    "pair": PAIR,
    "interval": INTERVAL,
    "limit": LIMIT
}

response = requests.get(
    URL,
    params=params,
    timeout=30
)

response.raise_for_status()

data = response.json()

if not data:
    raise ValueError("CoinDCX থেকে candle data পাওয়া যায়নি")

df = pd.DataFrame(data)

df = df.sort_values("time").reset_index(drop=True)

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

tr = pd.concat(
    [
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ],
    axis=1
).max(axis=1)

df["true_range"] = tr

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

latest = df.iloc[-1]

result = {
    "checked_at_utc": datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S"),
    "pair": PAIR,
    "timeframe": INTERVAL,
    "candle_time": str(latest["time"]),
    "open": float(latest["open"]),
    "high": float(latest["high"]),
    "low": float(latest["low"]),
    "close": float(latest["close"]),
    "body": float(latest["body"]),
    "range": float(latest["range"]),
    "body_ratio_percent": float(
        latest["body_ratio"] * 100
    ),
    "atr14": None,
    "required_body": None,
    "condition_1": bool(latest["condition_1"]),
    "condition_2": bool(latest["condition_2"]),
    "final_signal": bool(latest["final_signal"])
}

if pd.notna(latest["atr14"]):
    result["atr14"] = float(latest["atr14"])

if pd.notna(latest["required_body"]):
    result["required_body"] = float(
        latest["required_body"]
    )

print(json.dumps(result, indent=2))

os.makedirs("output", exist_ok=True)

result_df = pd.DataFrame([result])

result_df.to_csv(
    "output/live_result.csv",
    index=False
)

df.to_csv(
    "output/all_candles.csv",
    index=False
)
