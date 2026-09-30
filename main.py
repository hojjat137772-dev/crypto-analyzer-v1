import os
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

APP_VERSION = "1.4.0"
BINANCE_URLS = [
    "https://api.binance.com",
    "https://api.binance.us",
]
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"

app = FastAPI(title="تحلیل‌گر بازار کریپتو", version=APP_VERSION)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CryptoAnalyzer/1.0",
    "Accept": "application/json",
})

CACHE = {}
CACHE_TTL = 60
SYMBOLS_CACHE = {"time": 0, "rows": []}
SYMBOLS_CACHE_TTL = 600

WALLEX_BASE = "https://api.wallex.ir"
NOBITEX_BASE = "https://api.nobitex.ir"


def get_json(path, params=None, timeout=15):
    last_error = None
    for base in BINANCE_URLS:
        try:
            r = SESSION.get(base + path, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"دریافت اطلاعات بازار ناموفق بود: {last_error}")


def clean_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("/", "").replace("-", "").strip()


def interval_to_text(interval: str) -> str:
    allowed = {
        "15m": "15 دقیقه",
        "1h": "1 ساعت",
        "4h": "4 ساعت",
        "1d": "روزانه",
    }
    return allowed.get(interval, interval)


def _normalize_market(base, quote, symbol, source):
    base = str(base or "").upper()
    quote = str(quote or "").upper()
    symbol = clean_symbol(symbol)
    return {"symbol": symbol, "base": base, "quote": quote, "source": source}


def get_binance_symbols():
    data = get_json("/api/v3/exchangeInfo")
    rows = []
    for s in data.get("symbols", []):
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT" and s.get("isSpotTradingAllowed", True):
            rows.append(_normalize_market(s.get("baseAsset"), "USDT", s.get("symbol"), "Binance"))
    return rows


def get_nobitex_symbols():
    r = SESSION.get(NOBITEX_BASE + "/market/stats", timeout=15)
    r.raise_for_status()
    data = r.json()
    rows = []
    for key in data.get("stats", {}).keys():
        raw = str(key or "").upper().strip()
        # Nobitex commonly exposes pairs such as BTC-USDT / BTC_USDT.
        compact = raw.replace("/", "").replace("-", "").replace("_", "")
        if compact.endswith("USDT") and len(compact) > 4:
            base = compact[:-4]
            rows.append(_normalize_market(base, "USDT", compact, "Nobitex"))
            rows[-1]["market_symbol"] = raw
    return rows

def get_wallex_symbols():
    # Wallex public spot markets. Use the documented /v1/markets response
    # where result.symbols is a mapping keyed by the exact market symbol.
    try:
        r = SESSION.get(WALLEX_BASE + "/v1/markets", timeout=15)
        r.raise_for_status()
        data = r.json()
        symbols = ((data.get("result") or {}).get("symbols") or {}) if isinstance(data, dict) else {}
        rows = []
        if isinstance(symbols, dict):
            items = symbols.items()
        elif isinstance(symbols, list):
            items = [(None, x) for x in symbols]
        else:
            items = []
        for key, m in items:
            if not isinstance(m, dict):
                continue
            raw_symbol = str(m.get("symbol") or key or "").upper().strip()
            base = str(m.get("baseAsset") or "").upper().strip()
            quote = str(m.get("quoteAsset") or "").upper().strip()
            if not raw_symbol or not base or quote != "USDT":
                continue
            # Keep the exact Wallex market symbol for the UDF request.
            rows.append(_normalize_market(base, quote, raw_symbol, "Wallex"))
            rows[-1]["market_symbol"] = raw_symbol
        return rows
    except Exception:
        return []

def get_all_symbols(source="auto"):
    now = time.time()
    cache_key = f"symbols:{source}"
    cached = CACHE.get(cache_key)
    if cached and now - cached["time"] < SYMBOLS_CACHE_TTL:
        return cached["rows"]

    funcs = {"Binance": get_binance_symbols, "Wallex": get_wallex_symbols, "Nobitex": get_nobitex_symbols}
    names = [source] if source in funcs else list(funcs.keys())
    rows = []
    seen = set()
    for name in names:
        try:
            for row in funcs[name]():
                key = (name, row["symbol"], row.get("market_symbol", row["symbol"]))
                if key in seen:
                    continue
                seen.add(key)
                row = dict(row)
                row["source"] = name
                row["market_symbol"] = row.get("market_symbol", row["symbol"])
                rows.append(row)
        except Exception:
            continue
    rows.sort(key=lambda x: (x["symbol"], x["source"]))
    CACHE[cache_key] = {"time": now, "rows": rows}
    return rows

def resolution_for(interval):
    return {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}[interval]


def get_nobitex_history(symbol, interval, limit):
    symbol = clean_symbol(symbol)
    now = int(time.time())
    resolution = resolution_for(interval)
    seconds = {"15m": 15*60, "1h": 3600, "4h": 4*3600, "1d": 86400}[interval]
    frm = now - seconds * (limit + 10)
    r = SESSION.get(NOBITEX_BASE + "/market/udf/history", params={
        "symbol": symbol, "resolution": resolution, "from": frm, "to": now, "countback": limit
    }, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok" or not data.get("t"):
        raise ValueError("نوبیتکس برای این نماد/تایم‌فریم داده کافی ندارد.")
    df = pd.DataFrame({
        "time": pd.to_datetime(data["t"], unit="s", utc=True),
        "open": pd.to_numeric(data["o"], errors="coerce"),
        "high": pd.to_numeric(data["h"], errors="coerce"),
        "low": pd.to_numeric(data["l"], errors="coerce"),
        "close": pd.to_numeric(data["c"], errors="coerce"),
        "volume": pd.to_numeric(data.get("v", [0]*len(data["t"])), errors="coerce"),
    }).dropna().reset_index(drop=True)
    return df


def get_wallex_history(symbol, interval, limit):
    # Official Wallex UDF candle endpoint.
    # Wallex documents 1m, 60m and 1D directly; 15m and 4h are built
    # locally from smaller supported candles.
    symbol = clean_symbol(symbol)
    now = int(time.time())
    if interval == "15m":
        resolution = "1"
        source_minutes = 1
        fetch_limit = limit * 15 + 20
    elif interval == "1h":
        resolution = "60"
        source_minutes = 60
        fetch_limit = limit + 20
    elif interval == "4h":
        resolution = "60"
        source_minutes = 60
        fetch_limit = limit * 4 + 20
    else:
        resolution = "1D"
        source_minutes = 1440
        fetch_limit = limit + 10

    frm = now - source_minutes * 60 * (fetch_limit + 5)
    r = SESSION.get(WALLEX_BASE + "/v1/udf/history", params={
        "symbol": symbol,
        "resolution": resolution,
        "from": frm,
        "to": now,
        "countback": fetch_limit,
    }, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok" or not data.get("t"):
        raise ValueError("والکس برای این نماد/تایم‌فریم داده کافی ندارد.")

    df = pd.DataFrame({
        "time": pd.to_datetime(data["t"], unit="s", utc=True),
        "open": pd.to_numeric(data["o"], errors="coerce"),
        "high": pd.to_numeric(data["h"], errors="coerce"),
        "low": pd.to_numeric(data["l"], errors="coerce"),
        "close": pd.to_numeric(data["c"], errors="coerce"),
        "volume": pd.to_numeric(data.get("v", [0] * len(data["t"])), errors="coerce"),
    }).dropna().sort_values("time")

    if interval in {"15m", "4h"}:
        rule = "15min" if interval == "15m" else "4h"
        df = (df.set_index("time")
                .resample(rule)
                .agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"})
                .dropna()
                .reset_index())

    return df.tail(limit).reset_index(drop=True)

def get_binance_history(symbol: str, interval: str, limit: int):
    raw = get_json("/api/v3/klines", params={"symbol": clean_symbol(symbol), "interval": interval, "limit": limit}, timeout=20)
    if not raw: raise ValueError("Binance داده تاریخی برای این نماد ندارد.")
    cols=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]
    df=pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c], errors="coerce")
    df["time"]=pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["time","open","high","low","close","volume"]].dropna().reset_index(drop=True)


def get_history_with_source(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = 300, source: str = "auto"):
    symbol = clean_symbol(symbol)
    limit = max(100, min(int(limit), 1000))
    key = f"{source}:{symbol}:{interval}:{limit}"
    now = time.time()
    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        cached = CACHE[key]
        return cached["df"].copy(), cached.get("source", source)

    funcs = {
        "Binance": get_binance_history,
        "Wallex": get_wallex_history,
        "Nobitex": get_nobitex_history,
    }

    # In v1.4, the selected exchange is preferred, but if its candle endpoint
    # fails, the app automatically tries the other public sources as fallback.
    if source in funcs:
        order = [source] + [x for x in ["Binance", "Wallex", "Nobitex"] if x != source]
    else:
        order = ["Binance", "Wallex", "Nobitex"]

    errors = []
    for name in order:
        try:
            market_symbol = symbol
            matches = [x for x in get_all_symbols(name) if x["symbol"] == symbol]
            if matches:
                market_symbol = matches[0].get("market_symbol", symbol)

            df = funcs[name](market_symbol, interval, limit)
            if len(df) >= 50:
                CACHE[key] = {"time": now, "df": df.copy(), "source": name}
                return df, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise ValueError(
        "داده کندلی قابل‌استفاده پیدا نشد. "
        "هر سه منبع عمومی بررسی شدند. " + " | ".join(errors[-3:])
    )


def get_history(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = 300, source: str = "auto"):
    df, _ = get_history_with_source(symbol, interval, limit, source)
    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def support_resistance(df, window=40):
    recent = df.tail(window)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    return support, resistance


def score_trend(price, e20, e50, e200):
    score = 0
    if price > e20:
        score += 10
    if e20 > e50:
        score += 10
    if e50 > e200:
        score += 10
    if price > e200:
        score += 5
    return score


def build_signal(df):
    price = float(df["close"].iloc[-1])
    e20 = float(df["ema20"].iloc[-1])
    e50 = float(df["ema50"].iloc[-1])
    e200 = float(df["ema200"].iloc[-1])
    r = float(df["rsi"].iloc[-1])
    ml = float(df["macd"].iloc[-1])
    ms = float(df["macd_signal"].iloc[-1])
    mh = float(df["macd_hist"].iloc[-1])
    a = float(df["atr"].iloc[-1])

    support, resistance = support_resistance(df)

    score = score_trend(price, e20, e50, e200)

    # RSI: not treated as a standalone buy/sell trigger.
    if 45 <= r <= 68:
        score += 15
    elif r < 35:
        score += 8
    elif r > 75:
        score -= 8

    if ml > ms:
        score += 15
    if mh > 0:
        score += 5

    distance_to_res = max(0.0, resistance - price)
    if resistance > price and price > support:
        score += 4

    score = int(max(0, min(100, score)))

    if score >= 65 and price > e20 and ml >= ms:
        status = "صعودی"
        signal = "خرید مشروط / بررسی ورود"
    elif score <= 40 and price < e20 and ml < ms:
        status = "نزولی"
        signal = "فروش / عدم ورود"
    else:
        status = "خنثی"
        signal = "انتظار"

    # Risk levels are mechanical reference levels, not guaranteed targets.
    stop = max(0.0, min(support, price - 1.5 * a))
    if stop >= price:
        stop = max(0.0, price - 1.5 * a)

    risk = price - stop
    if risk <= 0:
        risk = max(a, price * 0.01)
        stop = max(0.0, price - risk)

    tp1 = price + 1.0 * risk
    tp2 = price + 2.0 * risk
    tp3 = price + 3.0 * risk

    return {
        "score": score,
        "signal": signal,
        "status": status,
        "price": price,
        "entry": price,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "support": support,
        "resistance": resistance,
        "rsi": r,
        "ema20": e20,
        "ema50": e50,
        "ema200": e200,
        "macd": ml,
        "macd_signal": ms,
        "atr": a,
        "risk_pct": abs(price - stop) / price * 100 if price else 0,
        "distance_to_resistance_pct": distance_to_res / price * 100 if price else 0,
    }


def analyze_symbol(symbol: str, interval: str, source: str = "auto"):
    df, actual_source = get_history_with_source(symbol, interval, 300, source)

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    ml, ms, mh = macd(df["close"])
    df["macd"] = ml
    df["macd_signal"] = ms
    df["macd_hist"] = mh
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        raise ValueError("داده کافی برای تحلیل وجود ندارد.")

    result = build_signal(df)
    result["symbol"] = clean_symbol(symbol)
    result["source"] = actual_source
    result["requested_source"] = source
    result["interval"] = interval
    result["interval_text"] = interval_to_text(interval)
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    return result

def run_backtest(symbol: str, interval: str, limit: int = 300, source: str = "auto"):
    df, actual_source = get_history_with_source(symbol, interval, limit, source)
    df["ema10"] = ema(df["close"], 10)
    df["ema20"] = ema(df["close"], 20)
    df["rsi"] = rsi(df["close"], 14)
    ml, ms, mh = macd(df["close"])
    df["macd"] = ml
    df["macd_signal"] = ms
    df["macd_hist"] = mh
    df = df.dropna().reset_index(drop=True)

    if len(df) < 60:
        raise ValueError("داده کافی برای بک‌تست وجود ندارد.")

    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        buy = (
            prev["ema10"] <= prev["ema20"]
            and row["ema10"] > row["ema20"]
            and row["rsi"] < 70
            and row["macd_hist"] > 0
        )

        sell = (
            prev["ema10"] >= prev["ema20"]
            and row["ema10"] < row["ema20"]
        )

        if position is None and buy:
            position = {
                "entry": float(row["close"]),
                "entry_time": row["time"],
            }
        elif position is not None and sell:
            exit_price = float(row["close"])
            ret = (exit_price / position["entry"] - 1) * 100
            trades.append({
                "entry": position["entry"],
                "exit": exit_price,
                "return_pct": ret,
                "entry_time": str(position["entry_time"]),
                "exit_time": str(row["time"]),
            })
            position = None

    if position is not None:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        ret = (exit_price / position["entry"] - 1) * 100
        trades.append({
            "entry": position["entry"],
            "exit": exit_price,
            "return_pct": ret,
            "entry_time": str(position["entry_time"]),
            "exit_time": str(last["time"]),
        })

    if not trades:
        return {
            "symbol": clean_symbol(symbol),
            "interval": interval,
            "source": actual_source,
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "compound_return": 0,
            "max_drawdown": 0,
            "trades_detail": [],
            "note": "در این بازه معامله‌ای با قوانین بک‌تست ایجاد نشد.",
        }

    returns = np.array([t["return_pct"] / 100 for t in trades], dtype=float)
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = (equity - peaks) / peaks * 100

    win_rate = float((returns > 0).mean() * 100)
    avg_return = float(returns.mean() * 100)
    compound_return = float((equity[-1] - 1) * 100)
    max_drawdown = float(drawdown.min())

    return {
        "symbol": clean_symbol(symbol),
        "interval": interval,
        "source": actual_source,
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "compound_return": compound_return,
        "max_drawdown": max_drawdown,
        "trades_detail": trades[-30:],
        "note": "بک‌تست با کارمزد، اسلیپیج و محدودیت نقدشوندگی محاسبه نشده است.",
    }


def _cluster_levels(values, current, side, max_levels=2):
    vals = sorted([float(v) for v in values if np.isfinite(v)])
    if side == "support":
        vals = [v for v in vals if v < current]
        vals = sorted(vals, reverse=True)
    else:
        vals = [v for v in vals if v > current]
        vals = sorted(vals)

    levels = []
    tolerance = max(abs(current) * 0.006, 1e-12)
    for value in vals:
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(value)
        if len(levels) >= max_levels:
            break
    return levels


def chart_payload(symbol: str, interval: str, source: str = "auto"):
    raw, actual_source = get_history_with_source(symbol, interval, 320, source)
    df = raw.copy().sort_values("time").reset_index(drop=True)

    if len(df) < 60:
        raise ValueError("داده کافی برای رسم نمودار وجود ندارد.")

    recent = df.tail(180).copy()
    current = float(recent["close"].iloc[-1])

    lows = []
    highs = []
    w = 3
    for i in range(w, len(recent) - w):
        lo = float(recent["low"].iloc[i])
        hi = float(recent["high"].iloc[i])
        if lo <= float(recent["low"].iloc[i-w:i+w+1].min()):
            lows.append(lo)
        if hi >= float(recent["high"].iloc[i-w:i+w+1].max()):
            highs.append(hi)

    support_levels = _cluster_levels(lows, current, "support", 2)
    resistance_levels = _cluster_levels(highs, current, "resistance", 2)

    if not support_levels:
        support_levels = [float(recent["low"].tail(40).min())]
    if not resistance_levels:
        resistance_levels = [float(recent["high"].tail(40).max())]

    # Historical entry/exit markers from the same backtest rules.
    bt = run_backtest(symbol, interval, limit=320, source=source)
    trades = bt.get("trades_detail", [])

    candles = []
    for _, row in recent.iterrows():
        candles.append({
            "t": int(pd.Timestamp(row["time"]).timestamp() * 1000),
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
        })

    return {
        "symbol": clean_symbol(symbol),
        "interval": interval,
        "interval_text": interval_to_text(interval),
        "source": actual_source,
        "candles": candles,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "current_price": current,
        "trades": trades,
        "trade_count": len(trades),
        "note": "نقاط سبز/قرمز مربوط به معاملات تاریخی همین منطق بک‌تست هستند؛ نقطه سبز آخر نیز ورود مرجع فعلی را نشان می‌دهد.",
    }

def html_page():
    return """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>تحلیل‌گر بازار کریپتو</title>
<style>
*{box-sizing:border-box}
body{font-family:Tahoma,Arial,sans-serif;background:#eef3f8;color:#18212b;margin:0}
.wrap{max-width:980px;margin:auto;padding:18px}
.card{background:#fff;border-radius:22px;padding:20px;margin:14px 0;box-shadow:0 8px 30px #00000010}
h1{margin:0 0 8px;font-size:26px}
h2{margin:0 0 15px}
.sub{color:#68717b}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.item{background:#f8fafc;border:1px solid #e6ebf0;border-radius:16px;padding:15px}
.label{font-size:13px;color:#68717b}
.value{font-size:22px;font-weight:bold;margin-top:5px}
select,input,button{width:100%;box-sizing:border-box;padding:13px;border-radius:12px;border:1px solid #d7dee8;font-size:16px}
button{background:#2447a8;color:white;border:0;font-weight:bold;cursor:pointer;margin-top:10px}
button.secondary{background:#344256}
.row{display:grid;grid-template-columns:2fr 1fr;gap:10px}
.good{color:#16815c}.bad{color:#c33}.neutral{color:#555}
ul{line-height:2}
.small{font-size:11px;color:#6b7280}\n.chartbox{background:#0f1722;border-radius:16px;padding:6px;overflow:hidden}\ncanvas{display:block;width:100%;height:380px;touch-action:none}\n.legend{display:flex;gap:12px;flex-wrap:wrap;padding:8px 2px 2px;font-size:11px;color:#cbd5e1}\n.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-left:4px}.green{background:#16c784}.red{background:#ef5350}.support{background:#38bdf8}.resist{background:#f59e0b}\n.levels{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.level{background:#f8fafc;border:1px solid #e7ebf0;border-radius:12px;padding:9px;font-size:11px}\n@media(max-width:600px){canvas{height:320px}}
pre{white-space:pre-wrap;direction:ltr;text-align:left}
@media(max-width:600px){.grid,.row{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<h1>تحلیل‌گر بازار کریپتو 🪙</h1>
<div class="sub">نسخه 1.4.0 — نمودار حرفه‌ای، ورود/خروج، حمایت و مقاومت</div>
</div>

<div class="card">
<div class="row">
<select id="symbol">
<option value="BTCUSDT" selected>Bitcoin — BTCUSDT</option>
</select>
<select id="source">
<option value="auto" selected>خودکار (هر سه منبع)</option>
<option value="Binance">Binance</option>
<option value="Wallex">والکس</option>
<option value="Nobitex">نوبیتکس</option>
</select>
<select id="interval">
<option value="15m">15 دقیقه</option>
<option value="1h" selected>1 ساعت</option>
<option value="4h">4 ساعت</option>
<option value="1d">روزانه</option>
</select>
</div>
<div class="small" style="margin-top:8px">در حالت خودکار، اگر یک صرافی داده کندلی نداشته باشد، برنامه به منبع بعدی می‌رود. انتخاب مستقیم صرافی فقط همان منبع را بررسی می‌کند.</div>
<button onclick="analyze()">🔎 تحلیل</button>\n<button onclick="loadChart()">📈 نمودار</button>
<button class="secondary" onclick="backtest()">🧪 بک‌تست</button>
<button onclick="scan()">🔍 اسکن بازار</button>
</div>

<div id="out"></div>
</div>

<script>
function n(x,d=2){
  if(x===null||x===undefined||Number.isNaN(Number(x))) return "-";
  return Number(x).toLocaleString("en-US",{maximumFractionDigits:d});
}
function card(label,value){return `<div class="item"><div class="label">${label}</div><div class="value">${value}</div></div>`}
function esc(s){
  return String(s ?? "").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
}
async function api(url){
  const r=await fetch(url);
  const j=await r.json();
  if(!r.ok) throw new Error(j.detail||"خطا");
  return j;
}

async function loadChart(silent=false){
  const market=selectedMarket();
  const i=document.getElementById("interval").value;
  if(!silent) out.innerHTML='<div class="card">در حال ساخت نمودار...</div>';
  try{
    const j=await api(`/chart?symbol=${encodeURIComponent(market.symbol)}&interval=${i}&source=${encodeURIComponent(market.source)}`);
    const chart=`<div class="card">
      <h2>📈 نمودار ${esc(j.symbol)} — ${esc(j.interval_text)}</h2>
      <div class="small">منبع واقعی: ${esc(j.source)} • قیمت فعلی: ${n(j.current_price)}</div>
      <div class="chartbox">
        <canvas id="priceChart"></canvas>
        <div class="legend">
          <span><i class="dot green"></i>ورود</span>
          <span><i class="dot red"></i>خروج</span>
          <span><i class="dot support"></i>حمایت</span>
          <span><i class="dot resist"></i>مقاومت</span>
        </div>
      </div>
      <div class="levels">
        <div class="level"><b>حمایت‌ها</b><br>${(j.support_levels||[]).map(x=>n(x)).join(" • ")}</div>
        <div class="level"><b>مقاومت‌ها</b><br>${(j.resistance_levels||[]).map(x=>n(x)).join(" • ")}</div>
      </div>
      <div class="small" style="margin-top:7px">نقاط سبز و قرمز معاملات تاریخی همین استراتژی بک‌تست هستند.</div>
    </div>`;
    if(silent) out.innerHTML += chart; else out.innerHTML=chart;
    requestAnimationFrame(()=>drawChart(j));
  }catch(e){
    if(!silent) out.innerHTML=`<div class="card"><b>خطا:</b> ${esc(e.message)}</div>`;
  }
}

function drawChart(j){
  const canvas=document.getElementById("priceChart");
  if(!canvas) return;
  const dpr=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect();
  canvas.width=Math.max(320,rect.width*dpr);
  canvas.height=Math.max(260,rect.height*dpr);
  const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
  const W=rect.width,H=rect.height, cs=j.candles||[];
  if(!cs.length) return;
  const pad={l:8,r:62,t:18,b:22}, cw=(W-pad.l-pad.r)/cs.length;
  let hi=Math.max(...cs.map(x=>x.h)), lo=Math.min(...cs.map(x=>x.l));
  [...(j.support_levels||[]),...(j.resistance_levels||[])].forEach(v=>{hi=Math.max(hi,v);lo=Math.min(lo,v)});
  const m=(hi-lo)*0.06||1; hi+=m; lo-=m;
  const X=i=>pad.l+(i+.5)*cw, Y=v=>pad.t+(hi-v)/(hi-lo)*(H-pad.t-pad.b);

  ctx.fillStyle="#0f1722"; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle="#263445"; ctx.lineWidth=1;
  for(let k=0;k<5;k++){
    const yy=pad.t+k*(H-pad.t-pad.b)/4;
    ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();
    ctx.fillStyle="#91a0b2";ctx.font="10px Arial";
    ctx.fillText(n(hi-k*(hi-lo)/4),W-pad.r+5,yy+3);
  }

  function line(v,label,dash){
    const yy=Y(v);ctx.save();ctx.setLineDash(dash?[5,5]:[]);
    ctx.strokeStyle=label==="حمایت"?"#38bdf8":"#f59e0b";ctx.lineWidth=1.2;
    ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();ctx.restore();
    ctx.fillStyle=label==="حمایت"?"#38bdf8":"#f59e0b";ctx.font="10px Arial";ctx.fillText(label,W-pad.r+5,yy-3);
  }
  (j.support_levels||[]).forEach(v=>line(v,"حمایت",true));
  (j.resistance_levels||[]).forEach(v=>line(v,"مقاومت",true));

  cs.forEach((c,i)=>{
    const xx=X(i),up=c.c>=c.o,bw=Math.max(1,Math.min(7,cw*.62));
    ctx.strokeStyle=up?"#16c784":"#ef5350";ctx.fillStyle=ctx.strokeStyle;
    ctx.beginPath();ctx.moveTo(xx,Y(c.h));ctx.lineTo(xx,Y(c.l));ctx.stroke();
    const top=Y(Math.max(c.o,c.c)),bot=Y(Math.min(c.o,c.c));
    ctx.fillRect(xx-bw/2,top,bw,Math.max(1,bot-top));
  });

  const idx=new Map(cs.map((c,i)=>[String(c.t),i]));
  function marker(ts,price,color,letter){
    let i=idx.get(String(ts));
    if(i===undefined){
      i=0;let best=Infinity;
      cs.forEach((c,k)=>{const d=Math.abs(c.t-ts);if(d<best){best=d;i=k;}});
    }
    const xx=X(i),yy=Y(price);
    ctx.fillStyle=color;ctx.beginPath();ctx.arc(xx,yy,5,0,Math.PI*2);ctx.fill();
    ctx.fillStyle="#fff";ctx.font="bold 8px Arial";ctx.textAlign="center";ctx.fillText(letter,xx,yy+3);ctx.textAlign="start";
  }
  (j.trades||[]).forEach(t=>{
    marker(new Date(t.entry_time).getTime(),t.entry,"#16c784","E");
    marker(new Date(t.exit_time).getTime(),t.exit,"#ef5350","X");
  });

  const py=Y(j.current_price);
  ctx.strokeStyle="#e5e7eb";ctx.setLineDash([2,3]);
  ctx.beginPath();ctx.moveTo(pad.l,py);ctx.lineTo(W-pad.r,py);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle="#e5e7eb";ctx.font="bold 10px Arial";ctx.fillText(n(j.current_price),W-pad.r+5,py+3);
}

async function loadSymbols(){
  const select=document.getElementById("symbol");
  const src=document.getElementById("source").value;
  select.innerHTML='<option>در حال دریافت فهرست ارزها...</option>';
  try{
    const r=await fetch(`/symbols?source=${encodeURIComponent(src)}`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||"خطا در دریافت فهرست ارزها");
    const rows=Array.isArray(j.symbols)?j.symbols:[];
    select.innerHTML="";
    rows.forEach(x=>{
      const o=document.createElement("option");
      o.value=`${x.source}::${x.symbol}`;
      o.textContent=`${x.base || x.symbol.replace(/USDT$/,"")} — ${x.symbol} — ${x.source}`;
      select.appendChild(o);
    });
    const btc=rows.find(x=>x.symbol==="BTCUSDT");
    if(btc) select.value=`${btc.source}::BTCUSDT`;
    else if(rows.length) select.selectedIndex=0;
  }catch(e){
    select.innerHTML='<option value="BTCUSDT">BTC — BTCUSDT</option>';
    console.error(e);
  }
}

document.getElementById("source").addEventListener("change", loadSymbols);

function selectedMarket(){
  const raw=document.getElementById("symbol").value || "";
  const parts=raw.split("::");
  const uiSource=document.getElementById("source").value;
  // Auto mode must stay auto so the backend can fail over between exchanges.
  if(uiSource==="auto") return {symbol:(parts.length===2?parts[1]:raw), source:"auto"};
  if(parts.length===2) return {symbol:parts[1], source:uiSource};
  return {symbol:raw, source:uiSource};
}

async function analyze(){
  const market=selectedMarket();
  const s=market.symbol;
  const i=document.getElementById("interval").value;
  const src=market.source;
  out.innerHTML='<div class="card">در حال دریافت اطلاعات بازار...</div>';
  try{
    const r=await fetch(`/analyze?symbol=${encodeURIComponent(s)}&interval=${i}&source=${encodeURIComponent(src)}`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||"خطا");
    const cls=j.status==="صعودی"?"good":j.status==="نزولی"?"bad":"neutral";
    out.innerHTML=`
    <div class="card">
      <h2>${j.symbol} — ${j.interval_text}</h2><div class="small">منبع داده: ${j.source||src}</div>
      <div class="grid">
        ${card("وضعیت",`<span class="${cls}">${j.status}</span>`)}
        ${card("امتیاز سیگنال (0 تا 100)",n(j.score,0))}
        ${card("قیمت آخر",n(j.price))}
        ${card("ورود مرجع",n(j.entry))}
        ${card("حد ضرر",n(j.stop))}
        ${card("حد سود 1",n(j.tp1))}
        ${card("حد سود 2",n(j.tp2))}
        ${card("حد سود 3",n(j.tp3))}
        ${card("RSI",n(j.rsi))}
        ${card("EMA20",n(j.ema20))}
        ${card("EMA50",n(j.ema50))}
        ${card("EMA200",n(j.ema200))}
        ${card("حمایت",n(j.support))}
        ${card("مقاومت",n(j.resistance))}
        ${card("ریسک تا حد ضرر",n(j.risk_pct)+"%")}
      </div>
      <h2 style="margin-top:20px">نتیجه تحلیل</h2>
      <p>${j.signal}</p>
      <ul>
        <li>قیمت نسبت به EMA20: ${j.price>j.ema20?"بالاتر":"پایین‌تر"}</li>
        <li>EMA20 نسبت به EMA50: ${j.ema20>j.ema50?"بالاتر":"پایین‌تر"}</li>
        <li>EMA50 نسبت به EMA200: ${j.ema50>j.ema200?"بالاتر":"پایین‌تر"}</li>
        <li>MACD: ${j.macd>=j.macd_signal?"مثبت/صعودی":"منفی/نزولی"}</li>
        <li>فاصله تا مقاومت: ${n(j.distance_to_resistance_pct)}%</li>
      </ul>
      <div class="small">این خروجی ابزار تحلیل است و تضمین نتیجه معامله یا سود نیست.</div>
    </div>`;
    await loadChart(true);
  }catch(e){out.innerHTML=`<div class="card"><b>خطا:</b> ${esc(e.message)}</div>`}
}
async function backtest(){
  const market=selectedMarket();
  const s=market.symbol;
  const i=document.getElementById("interval").value;
  const src=market.source;
  out.innerHTML='<div class="card">در حال اجرای بک‌تست...</div>';
  try{
    const r=await fetch(`/backtest?symbol=${encodeURIComponent(s)}&interval=${i}&source=${encodeURIComponent(src)}`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||"خطا");
    out.innerHTML=`
    <div class="card">
      <h2>نتیجه بک‌تست ${j.symbol}</h2>
      <div class="grid">
        ${card("تعداد معاملات",n(j.trades,0))}
        ${card("درصد برد",n(j.win_rate)+"%")}
        ${card("میانگین بازده هر معامله",n(j.avg_return)+"%")}
        ${card("بازده مرکب",n(j.compound_return)+"%")}
        ${card("حداکثر افت سرمایه",n(j.max_drawdown)+"%")}
      </div>
      <p class="small">${j.note}</p>
    </div>`;
    await loadChart(true);
  }catch(e){out.innerHTML=`<div class="card"><b>خطا:</b> ${esc(e.message)}</div>`}
}
async function scan(){
  out.innerHTML='<div class="card">در حال اسکن ارزهای USDT...</div>';
  try{
    const i=document.getElementById("interval").value;
    const src=document.getElementById("source").value;
    const r=await fetch(`/scan?interval=${i}&limit=20&source=${encodeURIComponent(src)}`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||"خطا");
    let rows=j.results.map(x=>`<div class="item"><b>${x.symbol}</b><br>امتیاز: ${x.score} — ${x.status}<br>قیمت: ${n(x.price)}</div>`).join("");
    out.innerHTML=`<div class="card"><h2>نتیجه اسکن بازار</h2><div class="grid">${rows}</div></div>`;
  }catch(e){out.innerHTML=`<div class="card"><b>خطا:</b> ${e.message}</div>`}
}
loadSymbols();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return html_page()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "data_sources": ["Binance", "Wallex", "Nobitex"],
        "trading_enabled": False,
    }


@app.get("/symbols")
def symbols(source: str = Query("auto")):
    rows = get_all_symbols(source)
    return {"count": len(rows), "symbols": rows}


@app.get("/analyze")
def analyze(
    symbol: str = Query(DEFAULT_SYMBOL),
    interval: str = Query(DEFAULT_INTERVAL),
    source: str = Query("auto"),
):
    allowed = {"15m", "1h", "4h", "1d"}
    if interval not in allowed:
        return JSONResponse({"detail": "تایم‌فریم نامعتبر است."}, status_code=400)

    try:
        return analyze_symbol(symbol, interval, source)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/chart")
def chart(
    symbol: str = Query(DEFAULT_SYMBOL),
    interval: str = Query(DEFAULT_INTERVAL),
    source: str = Query("auto"),
):
    allowed = {"15m", "1h", "4h", "1d"}
    if interval not in allowed:
        return JSONResponse({"detail": "تایم‌فریم نامعتبر است."}, status_code=400)
    try:
        return chart_payload(symbol, interval, source)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/backtest")
def backtest(
    symbol: str = Query(DEFAULT_SYMBOL),
    interval: str = Query(DEFAULT_INTERVAL),
    source: str = Query("auto"),
):
    allowed = {"15m", "1h", "4h", "1d"}
    if interval not in allowed:
        return JSONResponse({"detail": "تایم‌فریم نامعتبر است."}, status_code=400)

    try:
        return run_backtest(symbol, interval, source=source)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/scan")
def scan(
    interval: str = Query(DEFAULT_INTERVAL),
    limit: int = Query(20, ge=1, le=50),
    source: str = Query("auto"),
):
    try:
        symbols = get_all_symbols(source)

        # A deterministic first group is used to avoid thousands of API calls.
        # The app remains analysis-only; it never places orders.
        preferred = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
            "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT",
            "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT",
        ]
        available = {x["symbol"] for x in symbols}
        selected = [x for x in preferred if x in available][:limit]

        results = []
        for s in selected:
            try:
                r = analyze_symbol(s, interval, source)
                results.append({
                    "symbol": s,
                    "score": r["score"],
                    "status": r["status"],
                    "price": r["price"],
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "interval": interval,
            "count": len(results),
            "results": results,
            "note": "اسکن نمونه‌ای از ارزهای نقدشونده USDT است؛ برای اسکن گسترده‌تر می‌توان مرحله بعد رتبه‌بندی حجم بازار را اضافه کرد.",
        }
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
