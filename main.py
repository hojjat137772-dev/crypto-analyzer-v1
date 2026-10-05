
import os
import time
import math
import hashlib
import hmac
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CRYPTO SIGNAL ENGINE — FRESH BUILD
# Tabdeal-first / Spot + Margin/Futures analysis / No auto trading
# No trading fees are included in backtest calculations by design.
# ============================================================

st.set_page_config(
    page_title="Tabdeal Crypto Signal Engine",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Configuration
# -----------------------------
TABDEAL = os.getenv("TABDEAL_API_BASE", "https://api1.tabdeal.org").rstrip("/")
TABDEAL_API_KEY = os.getenv("TABDEAL_API_KEY", "").strip()
TABDEAL_API_SECRET = os.getenv("TABDEAL_API_SECRET", "").strip()
TABDEAL_ONLY = True
TRADE_LIMIT = 1000
PRIVATE_READ_ONLY = True


# ============================================================
# HTTP / API
# ============================================================
def _get(url, params=None, timeout=REQUEST_TIMEOUT, signed=False):
    params = dict(params or {})
    headers = {}
    if signed:
        if not TABDEAL_API_KEY or not TABDEAL_API_SECRET:
            raise RuntimeError("TABDEAL_API_KEY/TABDEAL_API_SECRET تنظیم نشده‌اند.")
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params, doseq=True)
        params["signature"] = hmac.new(TABDEAL_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        headers["X-MBX-APIKEY"] = TABDEAL_API_KEY
    r = SESSION.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def tabdeal_private_get(path, params=None):
    return _get(TABDEAL + path, params=params, signed=True)

@st.cache_data(ttl=30, show_spinner=False)
def get_account():
    return tabdeal_private_get("/r/api/v1/account")

@st.cache_data(ttl=30, show_spinner=False)
def get_open_orders():
    return tabdeal_private_get("/r/api/v1/openOrders")

@st.cache_data(ttl=30, show_spinner=False)
def get_funding_assets():
    return tabdeal_private_get("/r/api/v1/asset/get-funding-asset")


@st.cache_data(ttl=300, show_spinner=False)
def get_exchange_info():
    """Return all Tabdeal spot markets. The API docs state that omitting
    symbol filters returns all markets."""
    urls = [
        f"{TABDEAL}/r/api/v1/exchangeInfo",
        f"{TABDEAL}/api/v1/exchangeInfo",
    ]
    last_error = None
    for url in urls:
        try:
            data = _get(url)
            if isinstance(data, dict):
                for key in ("symbols", "data", "result", "markets"):
                    if isinstance(data.get(key), list):
                        return data[key]
            if isinstance(data, list):
                return data
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Tabdeal exchangeInfo failed: {last_error}")


@st.cache_data(ttl=60, show_spinner=False)
def get_depth(symbol, limit=20):
    for key in ("symbol", "tabdealSymbol"):
        params = {key: symbol, "limit": limit}
        for path in ("/r/api/v1/depth", "/api/v1/depth"):
            try:
                d = _get(TABDEAL + path, params)
                if isinstance(d, dict) and ("bids" in d or "asks" in d):
                    return d
            except Exception:
                pass
    return {"bids": [], "asks": []}


@st.cache_data(ttl=20, show_spinner=False)
def get_ticker(symbol):
    for path in ("/r/api/v1/ticker/24hr", "/api/v1/ticker/24hr"):
        try:
            d = _get(TABDEAL + path, {"symbol": symbol})
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


def _parse_klines(raw):
    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()
    # Binance-compatible candle shape:
    # [open_time, open, high, low, close, volume, close_time, ...]
    rows = []
    for x in raw:
        if isinstance(x, (list, tuple)) and len(x) >= 6:
            rows.append(x[:6])
        elif isinstance(x, dict):
            # tolerate object-style responses
            rows.append([
                x.get("openTime", x.get("timestamp", x.get("time"))),
                x.get("open"),
                x.get("high"),
                x.get("low"),
                x.get("close"),
                x.get("volume", x.get("vol")),
            ])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    if df.empty:
        return df
    unit = "ms" if df["timestamp"].max() > 10_000_000_000 else "s"
    df["time"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


@st.cache_data(ttl=60, show_spinner=False)
def get_tabdeal_trades(symbol, limit=TRADE_LIMIT):
    for key in ("symbol", "tabdealSymbol"):
        try:
            return _get(TABDEAL + "/r/api/v1/trades", {key: symbol, "limit": min(int(limit), 1000)})
        except Exception:
            pass
    return []

def _interval_seconds(interval):
    return {"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,"4h":14400,"6h":21600,"12h":43200,"1d":86400,"3d":259200,"1w":604800}[interval]

def trades_to_ohlcv(trades, interval):
    rows=[]
    for t in trades if isinstance(trades,list) else []:
        try:
            price=float(t["price"]); qty=float(t["qty"]); ts=int(t.get("time",0))
            if ts>10_000_000_000: ts//=1000
            if price>0 and ts>0: rows.append((pd.to_datetime(ts,unit="s",utc=True),price,qty))
        except Exception: pass
    if len(rows)<30: return pd.DataFrame()
    x=pd.DataFrame(rows,columns=["time","price","qty"]).sort_values("time")
    sec=_interval_seconds(interval); epoch=x["time"].astype("int64")//10**9
    x["bucket"]=pd.to_datetime((epoch//sec)*sec,unit="s",utc=True)
    g=x.groupby("bucket",sort=True)
    out=g["price"].agg(open="first",high="max",low="min",close="last").reset_index().rename(columns={"bucket":"time"})
    out["volume"]=g["qty"].sum().values
    return out[["time","open","high","low","close","volume"]].reset_index(drop=True)

@st.cache_data(ttl=60, show_spinner=False)
def get_klines(symbol, interval, limit=CANDLE_LIMIT):
    df=trades_to_ohlcv(get_tabdeal_trades(symbol,TRADE_LIMIT),interval)
    if len(df)>=50: return df.tail(min(limit,len(df))).reset_index(drop=True), "Tabdeal trades→OHLCV"
    return pd.DataFrame(), "Tabdeal: داده کندلی کافی نیست"


# ============================================================
# Market normalization
# ============================================================
def normalize_markets(raw):
    rows = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        symbol = str(x.get("symbol", "")).replace("_", "").upper()
        tabdeal_symbol = str(x.get("tabdealSymbol", x.get("symbol", ""))).upper()
        status = str(x.get("status", "")).upper()
        base = str(x.get("baseAsset", "")).upper()
        quote = str(x.get("quoteAsset", "")).upper()

        if not symbol or not base or not quote:
            # best-effort extraction for unusual payloads
            continue

        rows.append({
            "symbol": symbol,
            "tabdealSymbol": tabdeal_symbol,
            "base": base,
            "quote": quote,
            "status": status,
            "spot": bool(x.get("isSpotTradingAllowed", True)),
            "margin": bool(x.get("isMarginTradingAllowed", False)),
            "permissions": ",".join(map(str, x.get("permissions", []))),
            "raw": x,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("symbol").reset_index(drop=True)
    return df


def market_type(m):
    if m["margin"]:
        return "Spot + Margin"
    return "Spot"


def fmt_price(v):
    if v is None or not np.isfinite(v):
        return "-"
    av = abs(float(v))
    if av >= 1000:
        return f"{v:,.2f}"
    if av >= 1:
        return f"{v:,.4f}"
    if av >= 0.01:
        return f"{v:,.6f}"
    return f"{v:,.10f}".rstrip("0").rstrip(".")


# ============================================================
# Indicators — causal only
# ============================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atrv = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrv
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrv
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean(), pdi, mdi


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def ichimoku(df):
    high9 = df["high"].rolling(9, min_periods=9).max()
    low9 = df["low"].rolling(9, min_periods=9).min()
    tenkan = (high9 + low9) / 2

    high26 = df["high"].rolling(26, min_periods=26).max()
    low26 = df["low"].rolling(26, min_periods=26).min()
    kijun = (high26 + low26) / 2

    high52 = df["high"].rolling(52, min_periods=52).max()
    low52 = df["low"].rolling(52, min_periods=52).min()
    span_a = (tenkan + kijun) / 2
    span_b = (high52 + low52) / 2
    return tenkan, kijun, span_a, span_b


def add_indicators(df):
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)
    x["atr"] = atr(x, 14)
    x["adx"], x["plus_di"], x["minus_di"] = adx(x, 14)
    x["macd"], x["macd_signal"], x["macd_hist"] = macd(x["close"])
    x["tenkan"], x["kijun"], x["span_a"], x["span_b"] = ichimoku(x)
    x["vol_ma20"] = x["volume"].rolling(20, min_periods=20).mean()
    x["body"] = (x["close"] - x["open"]).abs()
    x["range"] = (x["high"] - x["low"]).replace(0, np.nan)
    x["body_ratio"] = x["body"] / x["range"]
    # Causal support/resistance: shifted rolling extrema; no centered windows.
    x["support"] = x["low"].shift(1).rolling(40, min_periods=20).min()
    x["resistance"] = x["high"].shift(1).rolling(40, min_periods=20).max()
    x["ret_5"] = x["close"].pct_change(5)
    x["ret_20"] = x["close"].pct_change(20)
    return x


# ============================================================
# Signal scoring
# ============================================================
def candle_bias(row):
    if row["close"] > row["open"] and row["body_ratio"] >= 0.55:
        return 1
    if row["close"] < row["open"] and row["body_ratio"] >= 0.55:
        return -1
    return 0


def score_row(row):
    if not np.isfinite(row.get("close", np.nan)):
        return 0, 0, 0, []

    long_score = 0
    short_score = 0
    reasons_long = []
    reasons_short = []

    # Trend
    if row["close"] > row["ema50"] > row["ema200"]:
        long_score += 2
        reasons_long.append("روند EMA صعودی")
    if row["close"] < row["ema50"] < row["ema200"]:
        short_score += 2
        reasons_short.append("روند EMA نزولی")

    # Momentum
    if row["rsi"] >= 52 and row["rsi"] <= 72:
        long_score += 1
        reasons_long.append("RSI متمایل به صعود")
    if row["rsi"] <= 48 and row["rsi"] >= 28:
        short_score += 1
        reasons_short.append("RSI متمایل به نزول")

    # MACD
    if row["macd"] > row["macd_signal"] and row["macd_hist"] > 0:
        long_score += 2
        reasons_long.append("MACD مثبت")
    if row["macd"] < row["macd_signal"] and row["macd_hist"] < 0:
        short_score += 2
        reasons_short.append("MACD منفی")

    # ADX / directional movement
    if row["adx"] >= 20 and row["plus_di"] > row["minus_di"]:
        long_score += 1
        reasons_long.append("قدرت روند صعودی")
    if row["adx"] >= 20 and row["minus_di"] > row["plus_di"]:
        short_score += 1
        reasons_short.append("قدرت روند نزولی")

    # Ichimoku
    cloud_top = max(row["span_a"], row["span_b"])
    cloud_bottom = min(row["span_a"], row["span_b"])
    if row["close"] > cloud_top and row["tenkan"] >= row["kijun"]:
        long_score += 2
        reasons_long.append("بالای ابر ایچیموکو")
    if row["close"] < cloud_bottom and row["tenkan"] <= row["kijun"]:
        short_score += 2
        reasons_short.append("پایین ابر ایچیموکو")

    # Volume
    if row["volume"] > row["vol_ma20"] * 1.15:
        if row["close"] > row["open"]:
            long_score += 1
            reasons_long.append("حجم تأییدکننده خرید")
        elif row["close"] < row["open"]:
            short_score += 1
            reasons_short.append("حجم تأییدکننده فروش")

    # Price action
    cb = candle_bias(row)
    if cb > 0:
        long_score += 1
        reasons_long.append("کندل صعودی")
    elif cb < 0:
        short_score += 1
        reasons_short.append("کندل نزولی")

    # Avoid chasing extreme RSI
    if row["rsi"] > 78:
        long_score -= 2
    if row["rsi"] < 22:
        short_score -= 2

    total = max(long_score, short_score)
    if long_score >= short_score + 2 and long_score >= 6:
        direction = 1
    elif short_score >= long_score + 2 and short_score >= 6:
        direction = -1
    else:
        direction = 0

    return direction, long_score, short_score, (
        reasons_long if direction == 1 else reasons_short if direction == -1 else []
    )


def levels_for(row, direction):
    price = float(row["close"])
    a = float(row["atr"]) if np.isfinite(row["atr"]) else price * 0.01
    risk = max(a * ATR_SL, price * 0.003)

    if direction == 1:
        sl = price - risk
        tp1 = price + risk * RR1
        tp2 = price + risk * RR2
        tp3 = price + risk * RR3
    elif direction == -1:
        sl = price + risk
        tp1 = price - risk * RR1
        tp2 = price - risk * RR2
        tp3 = price - risk * RR3
    else:
        return price, np.nan, np.nan, np.nan, np.nan

    rr = abs(tp2 - price) / max(abs(price - sl), 1e-12)
    return price, sl, tp1, tp2, tp3, rr


# ============================================================
# Backtest — signal on closed bar, entry on next bar open
# ============================================================
def simulate_trades(df, start_idx=0, end_idx=None):
    if end_idx is None:
        end_idx = len(df) - 1

    trades = []
    # Need next bar for execution.
    for i in range(max(start_idx, 200), min(end_idx, len(df) - 2)):
        row = df.iloc[i]
        direction, ls, ss, reasons = score_row(row)
        if direction == 0:
            continue

        # Entry occurs at next candle open, not current close.
        entry_bar = df.iloc[i + 1]
        entry = float(entry_bar["open"])

        a = float(row["atr"]) if np.isfinite(row["atr"]) else entry * 0.01
        risk = max(a * ATR_SL, entry * 0.003)

        if direction == 1:
            sl = entry - risk
            tp1 = entry + risk * RR1
            tp2 = entry + risk * RR2
        else:
            sl = entry + risk
            tp1 = entry - risk * RR1
            tp2 = entry - risk * RR2

        tp1_hit = False
        tp2_hit = False
        outcome = "TIMEOUT"
        exit_price = np.nan
        exit_idx = min(i + 1 + MAX_HOLD_BARS, len(df) - 1)

        for j in range(i + 1, exit_idx + 1):
            bar = df.iloc[j]
            hi, lo = float(bar["high"]), float(bar["low"])

            if direction == 1:
                # Conservative rule: if SL and TP are both touched in one bar,
                # count SL first because intrabar order is unknown.
                if lo <= sl:
                    outcome = "SL"
                    exit_price = sl
                    exit_idx = j
                    break
                if hi >= tp2:
                    tp2_hit = True
                    tp1_hit = True
                    outcome = "TP2"
                    exit_price = tp2
                    exit_idx = j
                    break
                if hi >= tp1:
                    tp1_hit = True
            else:
                if hi >= sl:
                    outcome = "SL"
                    exit_price = sl
                    exit_idx = j
                    break
                if lo <= tp2:
                    tp2_hit = True
                    tp1_hit = True
                    outcome = "TP2"
                    exit_price = tp2
                    exit_idx = j
                    break
                if lo <= tp1:
                    tp1_hit = True

        if np.isnan(exit_price):
            exit_price = float(df.iloc[exit_idx]["close"])
            if direction == 1:
                if exit_price > entry:
                    outcome = "TIMEOUT_WIN"
                else:
                    outcome = "TIMEOUT_LOSS"
            else:
                if exit_price < entry:
                    outcome = "TIMEOUT_WIN"
                else:
                    outcome = "TIMEOUT_LOSS"

        ret = (exit_price - entry) / entry if direction == 1 else (entry - exit_price) / entry

        trades.append({
            "signal_idx": i,
            "entry_idx": i + 1,
            "exit_idx": exit_idx,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp1_hit": bool(tp1_hit),
            "tp2_hit": bool(tp2_hit),
            "outcome": outcome,
            "return": ret,
        })

    return pd.DataFrame(trades)


def stats_from_trades(trades):
    if trades is None or trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "tp1_rate": 0.0, "tp2_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "max_dd": 0.0
        }

    r = trades["return"].astype(float)
    wins = int((r > 0).sum())
    losses = int((r <= 0).sum())
    gross_win = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades),
        "tp1_rate": float(trades["tp1_hit"].mean()),
        "tp2_rate": float(trades["tp2_hit"].mean()),
        "profit_factor": pf,
        "expectancy": float(r.mean()),
        "max_dd": max_dd,
    }


def walk_forward_backtest(df):
    if len(df) < MIN_BARS + 80:
        return None

    cut = int(len(df) * (1 - OOS_RATIO))
    train = simulate_trades(df, 0, cut - 1)
    oos = simulate_trades(df, cut, len(df) - 1)
    all_trades = simulate_trades(df, 0, len(df) - 1)

    st_train = stats_from_trades(train)
    st_oos = stats_from_trades(oos)
    st_all = stats_from_trades(all_trades)

    stats_ok = (
        st_oos["trades"] >= MIN_OOS_TRADES
        and st_oos["profit_factor"] > MIN_PF
        and st_oos["expectancy"] > MIN_EXPECTANCY
        and st_oos["tp1_rate"] >= MIN_TP1_RATE
    )

    return {
        "train": st_train,
        "oos": st_oos,
        "all": st_all,
        "stats_ok": stats_ok,
        "oos_trades": st_oos["trades"],
    }


# ============================================================
# Multi-timeframe confirmation
# ============================================================
@st.cache_data(ttl=120, show_spinner=False)
def analyze_timeframe(symbol, interval):
    df, source = get_klines(symbol, interval, CANDLE_LIMIT)
    if df.empty or len(df) < MIN_BARS:
        return None

    x = add_indicators(df).dropna().reset_index(drop=True)
    if len(x) < 100:
        return None

    row = x.iloc[-1]
    direction, ls, ss, reasons = score_row(row)
    price, sl, tp1, tp2, tp3, rr = levels_for(row, direction)

    bt = walk_forward_backtest(x)

    return {
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "df": x,
        "direction": direction,
        "long_score": ls,
        "short_score": ss,
        "reasons": reasons,
        "price": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr": rr,
        "rsi": float(row["rsi"]),
        "adx": float(row["adx"]),
        "atr_pct": float(row["atr"] / row["close"] * 100),
        "volume_ratio": float(row["volume"] / row["vol_ma20"]) if row["vol_ma20"] else np.nan,
        "bt": bt,
    }


def mtf_confirmation(results):
    valid = [v for v in results.values() if v is not None]
    dirs = [v["direction"] for v in valid if v["direction"] != 0]
    if len(valid) < 3 or len(dirs) < 3:
        return 0, False, "داده کافی نیست"

    long_n = sum(d == 1 for d in dirs)
    short_n = sum(d == -1 for d in dirs)

    if long_n >= 3 and long_n / len(dirs) >= 0.60:
        return 1, True, f"تأیید لانگ {long_n}/{len(dirs)}"
    if short_n >= 3 and short_n / len(dirs) >= 0.60:
        return -1, True, f"تأیید شورت {short_n}/{len(dirs)}"
    return 0, False, "تأیید چندتایم‌فریمی ناقص"


def final_decision(mtf_dir, mtf_ok, primary, bt):
    if bt is None:
        return "عدم معامله", False
    oos = bt["oos"]
    stats_ok = (
        oos["trades"] >= MIN_OOS_TRADES
        and oos["profit_factor"] > MIN_PF
        and oos["expectancy"] > MIN_EXPECTANCY
        and oos["tp1_rate"] >= MIN_TP1_RATE
    )
    if not stats_ok:
        return "عدم معامله", False
    if not mtf_ok:
        return "صبر", False
    if primary is None or primary["direction"] != mtf_dir:
        return "صبر", False
    return "معامله", True


# ============================================================
# Scan
# ============================================================
@st.cache_data(ttl=120, show_spinner=False)
def scan_market(symbols, interval):
    out = []
    for symbol in symbols:
        try:
            a = analyze_timeframe(symbol, interval)
            if a is None:
                continue
            bt = a["bt"]
            oos = bt["oos"] if bt else None
            out.append({
                "ارز": symbol,
                "جهت": "لانگ" if a["direction"] == 1 else "شورت" if a["direction"] == -1 else "خنثی",
                "قیمت": a["price"],
                "امتیاز": max(a["long_score"], a["short_score"]),
                "RSI": a["rsi"],
                "ADX": a["adx"],
                "TP1": oos["tp1_rate"] if oos else np.nan,
                "PF": oos["profit_factor"] if oos else np.nan,
                "Expectancy": oos["expectancy"] if oos else np.nan,
                "OOS": oos["trades"] if oos else 0,
                "وضعیت آماری": "قبول" if bt and bt["stats_ok"] else "رد",
                "_analysis": a,
            })
        except Exception:
            continue

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    df["_rank"] = (
        df["وضعیت آماری"].eq("قبول").astype(int) * 100
        + df["PF"].fillna(0).clip(0, 5) * 10
        + df["TP1"].fillna(0) * 20
        + df["امتیاز"].fillna(0)
    )
    return df.sort_values("_rank", ascending=False).reset_index(drop=True)


# ============================================================
# UI
# ============================================================
st.title("Tabdeal Crypto Signal Engine")
st.caption(
    "نسخه اختصاصی API تبدیل — فقط Tabdeal، تحلیل Spot + Margin/Futures، "
    "بدون Auto Trading و بدون لحاظ کارمزد در بک‌تست."
)

try:
    raw_markets = get_exchange_info()
    markets = normalize_markets(raw_markets)
except Exception as e:
    st.error(f"دریافت بازارهای تبدیل ناموفق بود: {e}")
    st.stop()

if markets.empty:
    st.error("هیچ بازار فعالی از تبدیل دریافت نشد.")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("اتصال API تبدیل")
    if TABDEAL_API_KEY and TABDEAL_API_SECRET:
        try:
            acct = get_account()
            st.success("API حساب: متصل")
            st.caption(f"Account: {acct.get('accountType','-')}")
            st.caption(f"Can Trade: {acct.get('canTrade','-')}")
        except Exception as e:
            st.error(f"API حساب: {e}")
    else:
        st.warning("TABDEAL_API_KEY و TABDEAL_API_SECRET تنظیم نشده‌اند؛ فقط API عمومی فعال است.")

active = markets[
    markets["status"].isin(["TRADING", ""])
].copy()

# All markets are available; default focus is USDT for the main scanner,
# while the user can switch to every quote asset.
quote_options = ["همه"] + sorted(active["quote"].dropna().unique().tolist())
base_options = sorted(active["base"].dropna().unique().tolist())

with st.sidebar:
    st.header("تنظیمات")
    quote_filter = st.selectbox("Quote", quote_options, index=quote_options.index("USDT") if "USDT" in quote_options else 0)
    scan_tf = st.selectbox("تایم‌فریم اسکن", list(TIMEFRAMES.keys()), index=list(TIMEFRAMES.keys()).index(DEFAULT_SCAN_TF))
    detail_tfs = st.multiselect(
        "تایم‌فریم‌های تحلیل دقیق",
        list(TIMEFRAMES.keys()),
        default=["15m", "1H", "4H", "1D"]
    )
    market_mode = st.selectbox("نوع بازار", ["همه", "Spot", "Margin/Futures"])

    st.divider()
    st.subheader("قوانین اعتبارسنجی")
    st.write(f"OOS حداقل: {MIN_OOS_TRADES}")
    st.write(f"Profit Factor > {MIN_PF}")
    st.write("Expectancy > 0")
    st.write(f"TP1 Rate ≥ {MIN_TP1_RATE:.0%}")
    st.write("کارمزد: لحاظ نمی‌شود")
    st.write("سیگنال فقط روی کندل بسته‌شده")

if quote_filter == "همه":
    filtered = active.copy()
else:
    filtered = active[active["quote"] == quote_filter].copy()

if market_mode == "Spot":
    filtered = filtered[filtered["spot"]]
elif market_mode == "Margin/Futures":
    filtered = filtered[filtered["margin"]]

symbols = filtered["symbol"].tolist()

tab1, tab2, tab3 = st.tabs(["تحلیل ارز", "اسکن کل بازار", "اطلاعات سیستم"])

with tab1:
    st.subheader("تحلیل دقیق")
    if not symbols:
        st.warning("بازاری با این فیلتر وجود ندارد.")
    else:
        default_symbol = symbols[0]
        selected = st.selectbox("ارز", symbols, index=0)

        if st.button("اجرای تحلیل", type="primary", use_container_width=True):
            if not detail_tfs:
                st.warning("حداقل یک تایم‌فریم انتخاب کنید.")
            else:
                details = {}
                progress = st.progress(0)
                for i, tf in enumerate(detail_tfs):
                    details[tf] = analyze_timeframe(selected, TIMEFRAMES[tf])
                    progress.progress((i + 1) / len(detail_tfs))

                mtf_dir, mtf_ok, mtf_msg = mtf_confirmation(details)
                primary_tf = "1H" if "1H" in details else detail_tfs[0]
                primary = details.get(primary_tf)
                bt = primary["bt"] if primary else None
                decision, trade_ok = final_decision(mtf_dir, mtf_ok, primary, bt)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("تصمیم نهایی", decision)
                c2.metric("تأیید MTF", "قبول" if mtf_ok else "رد")
                if bt:
                    c3.metric("OOS معاملات", bt["oos"]["trades"])
                    c4.metric("Profit Factor", f'{bt["oos"]["profit_factor"]:.2f}')

                if primary:
                    direction = "لانگ" if primary["direction"] == 1 else "شورت" if primary["direction"] == -1 else "خنثی"
                    st.markdown(f"### {selected} — {direction}")

                    if primary["direction"] != 0:
                        l1, l2, l3, l4, l5 = st.columns(5)
                        l1.metric("ورود", fmt_price(primary["price"]))
                        l2.metric("حد ضرر", fmt_price(primary["sl"]))
                        l3.metric("TP1", fmt_price(primary["tp1"]))
                        l4.metric("TP2", fmt_price(primary["tp2"]))
                        l5.metric("TP3", fmt_price(primary["tp3"]))

                    if bt:
                        o = bt["oos"]
                        t = bt["train"]
                        st.write(
                            f"**OOS:** {o['trades']} معامله | "
                            f"Win Rate: {o['win_rate']:.1%} | "
                            f"TP1: {o['tp1_rate']:.1%} | "
                            f"TP2: {o['tp2_rate']:.1%} | "
                            f"PF: {o['profit_factor']:.2f} | "
                            f"Expectancy: {o['expectancy']:.2%} | "
                            f"Max DD: {o['max_dd']:.2%}"
                        )
                        st.caption(
                            f"Train: {t['trades']} معامله — "
                            f"داده OOS برابر {OOS_RATIO:.0%} بخش پایانی تاریخچه است."
                        )

                    st.info(
                        f"وضعیت چندتایم‌فریمی: {mtf_msg}. "
                        f"منبع کندل تایم‌فریم اصلی: {primary['source']}."
                    )
                    st.write("دلایل سیگنال:", "، ".join(primary["reasons"]) if primary["reasons"] else "تأیید کافی وجود ندارد.")

                rows = []
                for tf, a in details.items():
                    if a is None:
                        rows.append({"تایم‌فریم": tf, "جهت": "داده ناکافی"})
                        continue
                    rows.append({
                        "تایم‌فریم": tf,
                        "جهت": "لانگ" if a["direction"] == 1 else "شورت" if a["direction"] == -1 else "خنثی",
                        "RSI": round(a["rsi"], 1),
                        "ADX": round(a["adx"], 1),
                        "امتیاز لانگ": a["long_score"],
                        "امتیاز شورت": a["short_score"],
                        "منبع": a["source"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("اسکن کل بازار")
    st.write(f"تعداد بازارهای قابل بررسی: **{len(symbols)}**")
    if st.button("اسکن همه بازارها", type="primary", use_container_width=True):
        with st.spinner("در حال دریافت داده و محاسبه سیگنال‌ها..."):
            scan = scan_market(tuple(symbols), TIMEFRAMES[scan_tf])

        if scan.empty:
            st.warning("نتیجه‌ای دریافت نشد. اتصال API یا محدودیت داده را بررسی کنید.")
        else:
            # Final status for the scanner uses statistical gate + primary signal.
            display = scan.copy()
            display["سیگنال"] = display.apply(
                lambda r: (
                    "معامله" if r["وضعیت آماری"] == "قبول" and r["جهت"] in ("لانگ", "شورت")
                    else "عدم معامله"
                ),
                axis=1
            )
            display["TP1"] = display["TP1"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
            display["Expectancy"] = display["Expectancy"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
            display["PF"] = display["PF"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            display["قیمت"] = display["قیمت"].map(fmt_price)

            cols = ["ارز", "قیمت", "جهت", "امتیاز", "RSI", "ADX", "TP1", "PF", "Expectancy", "OOS", "وضعیت آماری", "سیگنال"]
            st.dataframe(display[cols], use_container_width=True, hide_index=True)

            st.download_button(
                "دانلود CSV نتایج",
                data=display[cols].to_csv(index=False).encode("utf-8-sig"),
                file_name="tabdeal_market_scan.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab3:
    st.subheader("منطق سیستم")
    st.markdown("""
- داده بازار ابتدا از **Tabdeal** دریافت می‌شود.
- هیچ صرافی خارجی استفاده نمی‌شود؛ داده بازار فقط از Tabdeal دریافت می‌شود.
- به‌دلیل نبود endpoint کندل تاریخی در مستندات عمومی فعلی، کندل‌ها از معاملات عمومی اخیر Tabdeal تجمیع می‌شوند و عمق OOS محدود است.
- اندیکاتورها کاملاً **causal** هستند؛ از rolling centered یا اطلاعات آینده استفاده نمی‌شود.
- سیگنال از **کندل بسته‌شده** ساخته می‌شود.
- اجرای فرضی بک‌تست روی **Open کندل بعدی** انجام می‌شود.
- اگر در یک کندل هم SL و هم TP لمس شوند، برای حالت محافظه‌کارانه **SL اول** در نظر گرفته می‌شود.
- بخش پایانی تاریخچه به‌صورت **Out-of-Sample** جدا می‌شود.
- «درصد موفقیت» از عملکرد تاریخی OOS می‌آید، نه از تبدیل امتیاز تکنیکال به درصد ساختگی.
- برای «معامله» حداقل‌های آماری سخت‌گیرانه اعمال می‌شود.
- کارمزد در بک‌تست **عمداً لحاظ نشده**؛ slippage کوچک به‌عنوان پارامتر مدل اجرای بک‌تست باقی مانده است.
- این برنامه **هیچ سفارش واقعی ارسال نمی‌کند**.
""")
    if TABDEAL_API_KEY and TABDEAL_API_SECRET:
        try:
            acct=get_account()
            balances=pd.DataFrame(acct.get("balances",[]))
            if not balances.empty:
                balances["free"]=pd.to_numeric(balances["free"],errors="coerce")
                balances["freeze"]=pd.to_numeric(balances["freeze"],errors="coerce")
                balances=balances[(balances["free"]>0)|(balances["freeze"]>0)]
                st.markdown("### موجودی حساب تبدیل")
                st.dataframe(balances,use_container_width=True,hide_index=True)
        except Exception as e:
            st.warning(f"خواندن موجودی حساب ممکن نشد: {e}")
    st.write(f"تعداد کل بازارهای دریافتی: {len(markets)}")
    st.write(f"بازارهای فعال پس از فیلتر: {len(active)}")
    st.dataframe(
        active[["symbol", "base", "quote", "status", "spot", "margin", "permissions"]]
        .sort_values(["quote", "base"]),
        use_container_width=True,
        hide_index=True,
    )

st.caption("این نسخه فقط API تبدیل را می‌خواند و هیچ سفارش واقعی ارسال نمی‌کند.")
st.caption("این نرم‌افزار ابزار تحلیل و سیگنال است و سود یا موفقیت معامله را تضمین نمی‌کند.")
