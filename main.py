import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import hmac
import hashlib
import json
import urllib.parse
import uuid
from decimal import Decimal, ROUND_DOWN
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Crypto Analyzer Pro — RENDER AUTO-TRADER V2
# ============================================================
# Architecture:
# 1) Fast scan of the complete Tabdeal USDT market.
# 2) Rank candidates by trend + price action + momentum + volume.
# 3) Deep multi-timeframe analysis of only the best candidates.
# 4) Historical/OOS probability gate.
# 5) Optional automatic spot BUY on Tabdeal.
# 6) Automatic OCO SELL = TP1 + SL.
#
# IMPORTANT:
# - Live trading is controlled by TABDEAL_LIVE_TRADING in Render.
# - Set TABDEAL_API_KEY / TABDEAL_API_SECRET in Streamlit secrets
#   or environment variables.
# - Spot mode is long-only.
# - The percentage shown is an estimated probability score based on
#   historical/OOS outcomes + current confluence. It is NOT a guarantee.
# - Only CLOSED candles are used for signals.
# ============================================================

st.set_page_config(
    page_title="Crypto Analyzer Pro — Autonomous",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- CONFIG -------------------------

TABDEAL = "https://api1.tabdeal.org"
BINANCE_DATA = "https://data-api.binance.vision"

TIMEFRAMES = {
    "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h",
    "6H": "6h", "12H": "12h", "1D": "1d",
    "3D": "3d", "1W": "1w",
}

SCAN_TFS = ["1H", "4H", "1D"]
DEEP_TFS = ["15m", "1H", "4H", "1D"]

DEFAULT_SCAN_INTERVAL = 60
DEFAULT_SCAN_WORKERS = 6
DEFAULT_TRADE_USDT = 10.0
DEFAULT_MAX_POSITIONS = 1

MIN_HISTORY = 100
BACKTEST_BARS = 180
OOS_MIN_TRADES = 25
MIN_PROBABILITY = 58.0
MIN_SCORE = 68.0
MIN_RISK_REWARD = 1.20
MAX_ATR_PCT = 15.0

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoAnalyzerPro-Autonomous/1.0"})

TABDEAL_MARKET_ENDPOINTS = [
    "/r/api/v1/exchangeInfo",
    "/api/v1/exchangeInfo",
]

# ----------------------------- HTTP ----------------------------

@st.cache_data(ttl=20, show_spinner=False)
def public_json(url, params=None, timeout=12):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _tabdeal_signature_params(params):
    """Build the exact query/body string used for Tabdeal HMAC signing."""
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    return urllib.parse.urlencode(clean, doseq=True)


def tabdeal_server_time():
    data = public_json(TABDEAL + "/r/api/v1/time", timeout=8)
    if isinstance(data, dict):
        try:
            return int(data["serverTime"])
        except Exception:
            pass
    return int(time.time() * 1000)


def signed_request(method, path, api_key, api_secret, params=None, timeout=15):
    """
    Real Tabdeal signed request.
    TRADE endpoints require X-MBX-APIKEY + HMAC-SHA256 signature.
    POST parameters are sent in the request body and the signature is
    calculated over that same encoded parameter string.
    """
    if not api_key or not api_secret:
        return {"_error": "TABDEAL_API_KEY / TABDEAL_API_SECRET تنظیم نشده است."}

    params = dict(params or {})
    # Use exchange time to reduce timestamp rejection after Render sleeps/restarts.
    params["timestamp"] = tabdeal_server_time()

    payload = _tabdeal_signature_params(params)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "X-MBX-APIKEY": api_key,
        "User-Agent": "CryptoAnalyzerPro-RenderAutoTrader/2.0",
    }

    try:
        method = method.upper()
        if method == "GET":
            r = SESSION.get(
                TABDEAL + path,
                params={**params, "signature": signature},
                headers=headers,
                timeout=timeout,
            )
        elif method == "DELETE":
            r = SESSION.delete(
                TABDEAL + path,
                data={**params, "signature": signature},
                headers=headers,
                timeout=timeout,
            )
        else:
            r = SESSION.post(
                TABDEAL + path,
                data={**params, "signature": signature},
                headers=headers,
                timeout=timeout,
            )

        if not r.ok:
            return {"_error": f"HTTP {r.status_code}: {r.text[:800]}"}

        try:
            return r.json()
        except Exception:
            return {"_error": f"پاسخ JSON معتبر نبود: {r.text[:500]}"}
    except Exception as e:
        return {"_error": str(e)}


# ----------------------------- HELPERS -------------------------

def normalize_symbol(s):
    return str(s).upper().replace("-", "").replace("_", "").replace("/", "")


def base_asset(symbol):
    s = normalize_symbol(symbol)
    return s[:-4] if s.endswith("USDT") else s


def fmt_num(v, decimals=8):
    try:
        x = float(v)
        if not np.isfinite(x):
            return "-"
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        if abs(x) >= 1:
            return f"{x:,.4f}"
        return f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return "-"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def flatten_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from flatten_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten_dicts(v)


# ----------------------------- MARKET --------------------------

@st.cache_data(ttl=180, show_spinner=False)
def get_tabdeal_exchange_info():
    for ep in TABDEAL_MARKET_ENDPOINTS:
        data = public_json(TABDEAL + ep, timeout=15)
        if data:
            return data
    return {}


@st.cache_data(ttl=180, show_spinner=False)
def get_universe():
    data = get_tabdeal_exchange_info()
    symbols = set()

    for d in flatten_dicts(data):
        if not isinstance(d, dict):
            continue

        raw = d.get("symbol") or d.get("tabdealSymbol")
        if not isinstance(raw, str):
            continue

        s = normalize_symbol(raw)
        quote = str(d.get("quoteAsset", "")).upper()
        status = str(d.get("status", "TRADING")).upper()

        if s.endswith("USDT") and status in ("TRADING", "ACTIVE", "ENABLED", ""):
            if quote in ("", "USDT"):
                symbols.add(s)

    if symbols:
        return sorted(symbols)

    # Fallback only for analysis data if Tabdeal's market list is unavailable.
    data = public_json(
        BINANCE_DATA + "/api/v3/exchangeInfo",
        timeout=15,
    )
    if isinstance(data, dict):
        for d in data.get("symbols", []):
            if (
                str(d.get("status", "")).upper() == "TRADING"
                and str(d.get("quoteAsset", "")).upper() == "USDT"
            ):
                symbols.add(normalize_symbol(d.get("symbol", "")))

    return sorted(s for s in symbols if s.endswith("USDT"))


def tabdeal_symbol(symbol):
    s = normalize_symbol(symbol)
    return s[:-4] + "_USDT" if s.endswith("USDT") else s


# ----------------------------- DATA ----------------------------

def rows_to_df(rows):
    if not isinstance(rows, list) or len(rows) < 40:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]

        t = pd.to_numeric(df["time"], errors="coerce")
        unit = "us" if t.dropna().median() > 1e14 else "ms"
        df["time"] = pd.to_datetime(t, unit=unit, utc=True)

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return (
            df.dropna()
            .drop_duplicates("time")
            .sort_values("time")
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=25, show_spinner=False)
def get_klines(symbol, interval, limit=500):
    # Analysis data is pulled from Binance public market data because it
    # provides stable historical OHLCV. Orders are always sent to Tabdeal.
    data = public_json(
        BINANCE_DATA + "/api/v3/klines",
        {
            "symbol": normalize_symbol(symbol),
            "interval": interval,
            "limit": min(int(limit), 1000),
        },
        timeout=15,
    )
    return rows_to_df(data)


# ----------------------------- INDICATORS ----------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()

    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atrv = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atrv
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atrv

    dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    return dx.ewm(alpha=1/n, adjust=False).mean()


def add_indicators(df):
    x = df.copy()

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)

    x["macd"], x["macd_signal"], x["macd_hist"] = macd(x["close"])
    x["atr"] = atr(x, 14)
    x["adx"] = adx(x, 14)

    # Ichimoku
    high9 = x["high"].rolling(9).max()
    low9 = x["low"].rolling(9).min()
    high26 = x["high"].rolling(26).max()
    low26 = x["low"].rolling(26).min()
    high52 = x["high"].rolling(52).max()
    low52 = x["low"].rolling(52).min()

    x["tenkan"] = (high9 + low9) / 2
    x["kijun"] = (high26 + low26) / 2
    x["senkou_a"] = (x["tenkan"] + x["kijun"]) / 2
    x["senkou_b"] = (high52 + low52) / 2

    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["swing_high"] = x["high"].rolling(20).max()
    x["swing_low"] = x["low"].rolling(20).min()

    return x.dropna().reset_index(drop=True)


# ----------------------------- PRICE ACTION -------------------

def candle_features(x):
    o, h, l, c = x["open"], x["high"], x["low"], x["close"]
    rng = (h - l).replace(0, np.nan)

    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    return {
        "body_pct": float((body.iloc[-1] / rng.iloc[-1]) * 100),
        "upper_wick_pct": float((upper.iloc[-1] / rng.iloc[-1]) * 100),
        "lower_wick_pct": float((lower.iloc[-1] / rng.iloc[-1]) * 100),
        "bullish_close": bool(c.iloc[-1] > o.iloc[-1]),
    }


def price_action_score(x):
    score = 50.0
    reasons = []

    last = x.iloc[-1]
    prev = x.iloc[-2]

    f = candle_features(x)

    # Trend structure
    if last["close"] > last["ema20"] > last["ema50"]:
        score += 10
        reasons.append("ساختار صعودی EMA")
    elif last["close"] < last["ema20"] < last["ema50"]:
        score -= 10
        reasons.append("ساختار نزولی EMA")

    # Breakout / continuation
    prior_high = x["high"].iloc[-21:-1].max()
    prior_low = x["low"].iloc[-21:-1].min()

    if last["close"] > prior_high:
        score += 14
        reasons.append("شکست سقف ۲۰ کندل")
    elif last["close"] < prior_low:
        score -= 14
        reasons.append("شکست کف ۲۰ کندل")

    # Bullish engulfing
    if (
        last["close"] > last["open"]
        and prev["close"] < prev["open"]
        and last["close"] >= prev["open"]
        and last["open"] <= prev["close"]
    ):
        score += 8
        reasons.append("Bullish Engulfing")

    # Rejection from lower area
    if f["lower_wick_pct"] > 45 and last["close"] > last["open"]:
        score += 7
        reasons.append("رد قیمت از پایین")

    # Volume confirmation
    if last["volume"] > 1.35 * last["vol_ma20"]:
        score += 8 if last["close"] > last["open"] else -8
        reasons.append("افزایش حجم")

    return clamp(score, 0, 100), reasons


# ----------------------------- CURRENT TREND ------------------

def tf_snapshot(df):
    if len(df) < MIN_HISTORY:
        return None

    x = add_indicators(df)
    if len(x) < 60:
        return None

    # Always use the last fully closed candle.
    x = x.iloc[:-1].copy()
    if len(x) < 50:
        return None

    last = x.iloc[-1]
    prev = x.iloc[-2]

    score = 50.0
    reasons = []

    if last["close"] > last["ema20"]:
        score += 6
        reasons.append("قیمت بالای EMA20")
    else:
        score -= 6

    if last["ema20"] > last["ema50"]:
        score += 7
        reasons.append("EMA20 بالای EMA50")
    else:
        score -= 7

    if last["ema50"] > last["ema200"]:
        score += 8
        reasons.append("EMA50 بالای EMA200")
    else:
        score -= 8

    if last["macd"] > last["macd_signal"]:
        score += 6
        reasons.append("MACD مثبت")
    else:
        score -= 6

    if last["macd_hist"] > prev["macd_hist"]:
        score += 4
        reasons.append("شتاب MACD رو به افزایش")
    else:
        score -= 3

    if 52 <= last["rsi"] <= 72:
        score += 7
        reasons.append("RSI مناسب روند صعودی")
    elif last["rsi"] > 78:
        score -= 6
        reasons.append("RSI بیش‌خرید")
    elif last["rsi"] < 35:
        score -= 6
        reasons.append("RSI ضعیف")

    cloud_top = max(last["senkou_a"], last["senkou_b"])
    cloud_bottom = min(last["senkou_a"], last["senkou_b"])

    if last["close"] > cloud_top:
        score += 8
        reasons.append("قیمت بالای ابر ایچیموکو")
    elif last["close"] < cloud_bottom:
        score -= 8
        reasons.append("قیمت زیر ابر ایچیموکو")

    if last["adx"] >= 20:
        score += 4
        reasons.append("روند دارای قدرت")

    pa, pa_reasons = price_action_score(x)
    score = 0.65 * score + 0.35 * pa
    reasons.extend(pa_reasons)

    atr_pct = (last["atr"] / last["close"]) * 100

    # Short-term return and momentum.
    ret5 = (last["close"] / x["close"].iloc[-6] - 1) * 100
    ret20 = (last["close"] / x["close"].iloc[-21] - 1) * 100

    return {
        "score": float(clamp(score, 0, 100)),
        "close": float(last["close"]),
        "atr": float(last["atr"]),
        "atr_pct": float(atr_pct),
        "rsi": float(last["rsi"]),
        "adx": float(last["adx"]),
        "ret5": float(ret5),
        "ret20": float(ret20),
        "reasons": reasons[-8:],
        "data": x,
    }


# ----------------------------- FAST SCAN ----------------------

def fast_scan_one(symbol):
    try:
        snaps = {}
        for tf in SCAN_TFS:
            df = get_klines(symbol, TIMEFRAMES[tf], 240)
            snap = tf_snapshot(df)
            if snap is not None:
                snaps[tf] = snap

        if len(snaps) < 2:
            return None

        weights = {"1H": 0.30, "4H": 0.35, "1D": 0.35}
        total_w = sum(weights[k] for k in snaps)
        score = sum(snaps[k]["score"] * weights[k] for k in snaps) / total_w

        trend_agreement = sum(
            1 for v in snaps.values() if v["score"] >= 62
        ) / len(snaps) * 100

        # Penalize excessive volatility.
        atr_pct = np.mean([v["atr_pct"] for v in snaps])
        if atr_pct > MAX_ATR_PCT:
            score -= 8

        return {
            "symbol": symbol,
            "fast_score": float(clamp(score, 0, 100)),
            "trend_agreement": float(trend_agreement),
            "atr_pct": float(atr_pct),
            "1H": snaps.get("1H", {}).get("score", np.nan),
            "4H": snaps.get("4H", {}).get("score", np.nan),
            "1D": snaps.get("1D", {}).get("score", np.nan),
        }
    except Exception:
        return None


def run_fast_market_scan(symbols, workers, progress_cb=None):
    rows = []
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fast_scan_one, s): s for s in symbols}

        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if row:
                rows.append(row)

            if progress_cb:
                progress_cb(i / max(total, 1))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(
        ["fast_score", "trend_agreement"],
        ascending=False,
    ).reset_index(drop=True)


# ----------------------------- BACKTEST -----------------------

def historical_probability(df):
    """
    Rolling out-of-sample style test.
    Entry = next candle open after a bullish setup.
    TP = +1.5 ATR, SL = -1 ATR.
    Conservative rule: if both touched in one candle, SL wins.
    """

    if len(df) < BACKTEST_BARS + 80:
        return np.nan, 0, np.nan, np.nan

    x = add_indicators(df).iloc[:-1].copy()
    if len(x) < BACKTEST_BARS + 60:
        return np.nan, 0, np.nan, np.nan

    start = max(50, len(x) - BACKTEST_BARS)
    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0

    for i in range(start, len(x) - 2):
        cur = x.iloc[i]

        # Setup must be bullish at the close.
        bullish = (
            cur["close"] > cur["ema20"] > cur["ema50"]
            and cur["macd"] > cur["macd_signal"]
            and 48 <= cur["rsi"] <= 76
        )

        if not bullish:
            continue

        entry = float(x.iloc[i + 1]["open"])
        risk = max(float(cur["atr"]), entry * 0.002)
        tp = entry + 1.5 * risk
        sl = entry - 1.0 * risk

        outcome = None

        # Test next 24 candles.
        for j in range(i + 1, min(i + 1 + 24, len(x))):
            hi = float(x.iloc[j]["high"])
            lo = float(x.iloc[j]["low"])

            hit_sl = lo <= sl
            hit_tp = hi >= tp

            if hit_sl and hit_tp:
                outcome = "loss"
                break
            if hit_sl:
                outcome = "loss"
                break
            if hit_tp:
                outcome = "win"
                break

        if outcome == "win":
            wins += 1
            gross_win += 1.5
        elif outcome == "loss":
            losses += 1
            gross_loss += 1.0

    trades = wins + losses
    if trades == 0:
        return np.nan, 0, np.nan, np.nan

    raw_prob = wins / trades * 100
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf

    # Bayesian smoothing prevents tiny samples from showing extreme numbers.
    calibrated = (wins + 10) / (trades + 20) * 100

    return float(calibrated), int(trades), float(pf), float(raw_prob)


# ----------------------------- LEVELS --------------------------

def build_trade_plan(df, probability, trend_score):
    x = add_indicators(df).iloc[:-1].copy()
    if len(x) < 80:
        return None

    last = x.iloc[-1]
    entry = float(last["close"])
    atrv = float(last["atr"])

    # Support / resistance from recent closed candles.
    support = float(x["low"].iloc[-31:-1].min())
    resistance = float(x["high"].iloc[-31:-1].max())

    sl = min(entry - 1.15 * atrv, support * 0.997)
    if sl >= entry:
        sl = entry - 1.15 * atrv

    risk = entry - sl
    tp1 = entry + 1.35 * risk
    tp2 = entry + 2.10 * risk
    tp3 = entry + 3.00 * risk

    rr1 = (tp1 - entry) / risk if risk > 0 else 0

    # Current trend + historical probability + momentum.
    rsi_v = float(last["rsi"])
    macd_up = float(last["macd_hist"]) > 0

    momentum_bonus = 5 if macd_up and 50 <= rsi_v <= 72 else 0
    probability = float(probability) if np.isfinite(probability) else 50.0

    final_score = (
        0.45 * trend_score
        + 0.40 * probability
        + 0.15 * (50 + momentum_bonus)
    )

    # Outlook is a scenario, not a promise.
    outlook_pct = (tp1 / entry - 1) * 100

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "support": support,
        "resistance": resistance,
        "rr1": rr1,
        "score": float(clamp(final_score, 0, 100)),
        "outlook_pct": float(outlook_pct),
        "rsi": rsi_v,
    }


# ----------------------------- DEEP ANALYSIS ------------------

def deep_analyze(symbol):
    tf_rows = {}

    for tf in DEEP_TFS:
        df = get_klines(symbol, TIMEFRAMES[tf], 700)
        snap = tf_snapshot(df)
        if snap:
            tf_rows[tf] = snap

    if len(tf_rows) < 3:
        return None

    weights = {
        "15m": 0.10,
        "1H": 0.25,
        "4H": 0.35,
        "1D": 0.30,
    }

    wsum = sum(weights[k] for k in tf_rows)
    trend_score = sum(
        tf_rows[k]["score"] * weights[k]
        for k in tf_rows
    ) / wsum

    bullish_tfs = sum(v["score"] >= 60 for v in tf_rows.values())
    agreement = bullish_tfs / len(tf_rows) * 100

    # Use 4H first for risk construction, then 1H.
    base_snap = tf_rows.get("4H") or tf_rows.get("1H")
    base_df = base_snap["data"]

    probability, trades, pf, raw_prob = historical_probability(base_df)

    plan = build_trade_plan(
        base_df,
        probability,
        trend_score,
    )
    if not plan:
        return None

    # Hard confirmation.
    confirmed = (
        plan["score"] >= MIN_SCORE
        and agreement >= 65
        and trades >= OOS_MIN_TRADES
        and np.isfinite(probability)
        and probability >= MIN_PROBABILITY
        and pf > 1.0
        and plan["rr1"] >= MIN_RISK_REWARD
        and base_snap["atr_pct"] <= MAX_ATR_PCT
    )

    decision = "معامله" if confirmed else "صبر"

    if plan["score"] < 55 or agreement < 50:
        decision = "عدم معامله"

    # Detect direction of current trend.
    if trend_score >= 67 and agreement >= 75:
        trend = "صعودی قوی"
    elif trend_score >= 58:
        trend = "صعودی"
    elif trend_score <= 38:
        trend = "نزولی قوی"
    elif trend_score <= 46:
        trend = "نزولی"
    else:
        trend = "خنثی"

    return {
        "symbol": symbol,
        "trend": trend,
        "decision": decision,
        "confirmed": confirmed,
        "score": float(plan["score"]),
        "probability": float(probability) if np.isfinite(probability) else np.nan,
        "raw_probability": float(raw_prob) if np.isfinite(raw_prob) else np.nan,
        "oos_trades": int(trades),
        "profit_factor": float(pf) if np.isfinite(pf) else np.inf,
        "agreement": float(agreement),
        "outlook_pct": float(plan["outlook_pct"]),
        "entry": plan["entry"],
        "sl": plan["sl"],
        "tp1": plan["tp1"],
        "tp2": plan["tp2"],
        "tp3": plan["tp3"],
        "support": plan["support"],
        "resistance": plan["resistance"],
        "rr1": plan["rr1"],
        "rsi": plan["rsi"],
        "timeframes": {
            tf: round(v["score"], 1) for tf, v in tf_rows.items()
        },
    }


def deep_scan_candidates(candidates, workers, progress_cb=None):
    rows = []
    total = len(candidates)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(deep_analyze, s): s for s in candidates}

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
                if r:
                    rows.append(r)
            except Exception:
                pass

            if progress_cb:
                progress_cb(i / max(total, 1))

    return rows


# ----------------------------- TRADING API --------------------

@st.cache_data(ttl=120, show_spinner=False)
def get_trade_filters():
    data = get_tabdeal_exchange_info()
    result = {}

    for d in flatten_dicts(data):
        if not isinstance(d, dict):
            continue

        raw = d.get("symbol") or d.get("tabdealSymbol")
        if not isinstance(raw, str):
            continue

        s = normalize_symbol(raw)
        if not s.endswith("USDT"):
            continue

        item = {
            "min_qty": 0.0,
            "step_size": 0.00000001,
            "min_notional": 0.0,
        }

        filters = d.get("filters")
        if isinstance(filters, list):
            for f in filters:
                ft = str(f.get("filterType", "")).upper()
                if ft == "LOT_SIZE":
                    item["min_qty"] = float(f.get("minQty", 0) or 0)
                    item["step_size"] = float(f.get("stepSize", 0) or 0.00000001)
                elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                    item["min_notional"] = float(
                        f.get("minNotional", f.get("notional", 0)) or 0
                    )

        result[s] = item

    return result


def floor_step(value, step):
    if step <= 0:
        return float(value)
    d = Decimal(str(value))
    s = Decimal(str(step))
    return float((d / s).to_integral_value(rounding=ROUND_DOWN) * s)


def get_credentials():
    # Render Environment Variables take priority.
    key = os.getenv("TABDEAL_API_KEY", "")
    secret = os.getenv("TABDEAL_API_SECRET", "")

    if not key or not secret:
        try:
            if not key and "TABDEAL_API_KEY" in st.secrets:
                key = st.secrets["TABDEAL_API_KEY"]
            if not secret and "TABDEAL_API_SECRET" in st.secrets:
                secret = st.secrets["TABDEAL_API_SECRET"]
        except Exception:
            pass

    return str(key).strip(), str(secret).strip()


def env_bool(name, default=False):
    value = os.getenv(name, "")
    if value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


def get_account(api_key, api_secret):
    return signed_request(
        "GET",
        "/r/api/v1/account",
        api_key,
        api_secret,
        {},
    )


def get_spot_balance(api_key, api_secret, asset):
    account = get_account(api_key, api_secret)
    if not isinstance(account, dict) or "_error" in account:
        return 0.0

    for b in account.get("balances", []):
        if str(b.get("asset", "")).upper() == asset.upper():
            try:
                return float(b.get("free", 0) or 0)
            except Exception:
                return 0.0
    return 0.0


def get_open_orders(api_key, api_secret, symbol=None):
    params = {}
    if symbol:
        params["symbol"] = normalize_symbol(symbol)
    return signed_request(
        "GET",
        "/r/api/v1/openOrders",
        api_key,
        api_secret,
        params,
    )


def get_open_oco_lists(api_key, api_secret):
    return signed_request(
        "GET",
        "/r/api/v1/openOrderList",
        api_key,
        api_secret,
        {},
    )


def get_order(api_key, api_secret, symbol, order_id):
    return signed_request(
        "GET",
        "/r/api/v1/order",
        api_key,
        api_secret,
        {
            "symbol": normalize_symbol(symbol),
            "orderId": int(order_id),
        },
    )


def bot_oco_ids(oco_data):
    """Return open OCO IDs belonging to this bot's CAP2 client-id prefix."""
    ids = []
    if not isinstance(oco_data, list):
        return ids

    for item in oco_data:
        cid = str(item.get("listClientOrderId") or "")
        if cid.startswith("CAP2_"):
            try:
                ids.append(int(item.get("orderListId")))
            except Exception:
                pass
    return ids


def count_bot_open_positions(api_key, api_secret):
    """
    Render can restart, so Streamlit session state is not used as the source
    of truth. Open OCO lists on Tabdeal are the persistent source of truth.
    """
    oco = get_open_oco_lists(api_key, api_secret)
    return len(bot_oco_ids(oco))


def verify_account_can_trade(account):
    if not isinstance(account, dict) or "_error" in account:
        return False, "اطلاعات حساب از تبدیل دریافت نشد."

    if account.get("canTrade") is False:
        return False, "API حساب اجازه معامله ندارد."

    permissions = [str(x).upper() for x in account.get("permissions", [])]
    if permissions and "SPOT" not in permissions:
        return False, "مجوز SPOT برای API فعال نیست."

    return True, "حساب آماده معامله است."


def tabdeal_last_price(symbol):
    data = public_json(
        TABDEAL + "/r/api/v1/ticker/24hr",
        {"symbol": normalize_symbol(symbol)},
        timeout=10,
    )
    if isinstance(data, dict):
        for key in ("lastPrice", "last", "price"):
            try:
                value = float(data.get(key))
                if value > 0:
                    return value
            except Exception:
                pass
    return None


def place_market_buy(symbol, usdt_amount, api_key, api_secret):
    """
    Real spot MARKET BUY on Tabdeal.
    The amount is converted to base-asset quantity using Tabdeal's current
    market price, then rounded to the market LOT_SIZE.
    """
    price = tabdeal_last_price(symbol)
    if not price:
        return {"_error": "قیمت لحظه‌ای بازار تبدیل دریافت نشد."}

    filters = get_trade_filters().get(normalize_symbol(symbol), {})
    step = float(filters.get("step_size", 0.00000001))
    min_qty = float(filters.get("min_qty", 0))
    min_notional = float(filters.get("min_notional", 0))

    qty = floor_step(float(usdt_amount) / price, step)

    if qty <= 0 or qty < min_qty:
        return {
            "_error": f"حجم معامله کمتر از حداقل مقدار بازار است. qty={qty}, min={min_qty}"
        }

    if min_notional > 0 and qty * price < min_notional:
        return {
            "_error": (
                f"ارزش سفارش کمتر از حداقل مجاز بازار است. "
                f"minNotional={min_notional}"
            )
        }

    # Unique bot id makes it possible to distinguish bot OCOs after Render restarts.
    client_id = "CAP2_" + uuid.uuid4().hex[:20]

    return signed_request(
        "POST",
        "/api/v1/order",
        api_key,
        api_secret,
        {
            "symbol": normalize_symbol(symbol),
            "side": "BUY",
            "type": "MARKET",
            "quantity": f"{qty:.12f}".rstrip("0").rstrip("."),
            "newClientOrderId": client_id,
        },
    )


def place_oco_sell(symbol, quantity, tp, sl, api_key, api_secret):
    """
    Real Tabdeal OCO SELL:
      - limit leg = TP1
      - stopPrice = SL trigger
      - stopLimitPrice = slightly below SL for a long position
    """
    filters = get_trade_filters().get(normalize_symbol(symbol), {})
    step = float(filters.get("step_size", 0.00000001))
    qty = floor_step(quantity, step)

    if qty <= 0:
        return {"_error": "حجم قابل فروش صفر است."}

    tp = float(tp)
    sl = float(sl)

    if not (sl < tp):
        return {"_error": "مقادیر TP/SL برای OCO معتبر نیستند."}

    # Stop-limit slightly below stop trigger for a long position.
    stop_limit = sl * 0.9985

    token = uuid.uuid4().hex[:12]
    list_client_id = f"CAP2_{token}"
    limit_client_id = f"CAP2L_{token}"
    stop_client_id = f"CAP2S_{token}"

    return signed_request(
        "POST",
        "/api/v1/order/oco",
        api_key,
        api_secret,
        {
            "symbol": normalize_symbol(symbol),
            "side": "SELL",
            "quantity": f"{qty:.12f}".rstrip("0").rstrip("."),
            "price": f"{tp:.12f}".rstrip("0").rstrip("."),
            "stopPrice": f"{sl:.12f}".rstrip("0").rstrip("."),
            "stopLimitPrice": f"{stop_limit:.12f}".rstrip("0").rstrip("."),
            "listClientOrderId": list_client_id,
            "limitClientOrderId": limit_client_id,
            "stopClientOrderId": stop_client_id,
        },
    )


def execute_trade(plan, trade_usdt, max_positions):
    """
    Full real-trading sequence:
      1. Verify credentials.
      2. Verify account can trade.
      3. Check persistent bot OCO positions (survives Render restart).
      4. Verify enough USDT is available.
      5. Submit MARKET BUY.
      6. Query the order until FILLED/PARTIALLY_FILLED timeout.
      7. Submit OCO using the actual executed quantity.
      8. Refuse to silently continue if protection could not be placed.
    """
    api_key, api_secret = get_credentials()

    if not api_key or not api_secret:
        return {"ok": False, "message": "API Key/Secret تبدیل در Render پیدا نشد."}

    account = get_account(api_key, api_secret)
    can_trade, account_msg = verify_account_can_trade(account)
    if not can_trade:
        return {"ok": False, "message": account_msg, "account": account}

    try:
        open_positions = count_bot_open_positions(api_key, api_secret)
    except Exception as e:
        return {
            "ok": False,
            "message": f"نتوانستم معاملات باز ربات را از تبدیل بررسی کنم: {e}",
        }

    if open_positions >= int(max_positions):
        return {
            "ok": False,
            "message": (
                f"حداکثر معاملات همزمان پر است: "
                f"{open_positions}/{int(max_positions)}"
            ),
        }

    usdt_free = get_spot_balance(api_key, api_secret, "USDT")
    if usdt_free < float(trade_usdt):
        return {
            "ok": False,
            "message": (
                f"موجودی USDT کافی نیست: "
                f"{usdt_free:.4f} USDT < {float(trade_usdt):.4f} USDT"
            ),
        }

    symbol = normalize_symbol(plan["symbol"])

    # Conservative duplicate check: if this symbol already has any open
    # order/OCO, do not add another position.
    open_orders = get_open_orders(api_key, api_secret, symbol)
    if isinstance(open_orders, list) and open_orders:
        return {
            "ok": False,
            "message": f"برای {symbol} سفارش باز وجود دارد؛ خرید جدید لغو شد.",
            "open_orders": open_orders,
        }

    buy = place_market_buy(symbol, float(trade_usdt), api_key, api_secret)

    if not isinstance(buy, dict) or "_error" in buy:
        return {
            "ok": False,
            "message": str(buy.get("_error", "BUY ناموفق")),
            "buy": buy,
        }

    order_id = buy.get("orderId")
    executed_qty = float(buy.get("executedQty") or 0)

    # MARKET order normally returns FILLED, but query again so the bot does
    # not create an OCO from an unconfirmed quantity.
    if order_id and executed_qty <= 0:
        deadline = time.time() + 12
        while time.time() < deadline:
            checked = get_order(api_key, api_secret, symbol, order_id)
            if isinstance(checked, dict) and "_error" not in checked:
                executed_qty = float(checked.get("executedQty") or 0)
                status = str(checked.get("status", "")).upper()
                buy = checked
                if status == "FILLED":
                    break
                if status in ("CANCELED", "REJECTED", "EXPIRED"):
                    return {
                        "ok": False,
                        "message": f"سفارش BUY با وضعیت {status} پایان یافت.",
                        "buy": checked,
                    }
            time.sleep(1)

    if executed_qty <= 0:
        return {
            "ok": False,
            "message": (
                "BUY ارسال شد اما مقدار واقعی اجراشده تأیید نشد؛ "
                "برای جلوگیری از سفارش محافظتی اشتباه، OCO ارسال نشد."
            ),
            "buy": buy,
        }

    # Refresh the real filled quantity from account/order before protection.
    base = base_asset(symbol)
    balance_after_buy = get_spot_balance(api_key, api_secret, base)
    protected_qty = min(executed_qty, balance_after_buy) if balance_after_buy > 0 else executed_qty

    if protected_qty <= 0:
        return {
            "ok": False,
            "message": "خرید انجام شد اما موجودی دارایی برای ثبت OCO تأیید نشد.",
            "buy": buy,
        }

    oco = place_oco_sell(
        symbol,
        protected_qty,
        plan["tp1"],
        plan["sl"],
        api_key,
        api_secret,
    )

    if isinstance(oco, dict) and "_error" in oco:
        return {
            "ok": False,
            "message": (
                "هشدار جدی: BUY انجام شد ولی OCO ثبت نشد. "
                "موقعیت بدون حدضرر خودکار باقی مانده است."
            ),
            "buy": buy,
            "oco": oco,
        }

    return {
        "ok": True,
        "message": f"معامله واقعی {symbol} انجام شد و OCO ثبت شد.",
        "buy": buy,
        "oco": oco,
        "executed_qty": protected_qty,
    }


# ----------------------------- UI HELPERS ---------------------

def scan_table(rows):
    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append({
            "ارز": r["symbol"],
            "درصد": round(r["probability"], 1) if np.isfinite(r["probability"]) else np.nan,
            "امتیاز": round(r["score"], 1),
            "روند": r["trend"],
            "آینده/سناریو": f"+{r['outlook_pct']:.2f}%",
            "ورود": fmt_num(r["entry"]),
            "حدضرر": fmt_num(r["sl"]),
            "TP1": fmt_num(r["tp1"]),
            "TP2": fmt_num(r["tp2"]),
            "RR": round(r["rr1"], 2),
            "تأیید MTF": f"{r['agreement']:.0f}%",
            "OOS": r["oos_trades"],
            "PF": round(r["profit_factor"], 2) if np.isfinite(r["profit_factor"]) else 99.0,
            "تصمیم": r["decision"],
        })

    return (
        pd.DataFrame(data)
        .sort_values(["درصد", "امتیاز"], ascending=False)
        .reset_index(drop=True)
    )


# ----------------------------- MAIN ---------------------------

st.title("Crypto Analyzer Pro — Render Auto-Trader V2")
st.caption(
    "نسخه V2: سفارش واقعی فقط وقتی فعال است که TABDEAL_LIVE_TRADING=true باشد "
    "و موتور معامله در رابط کاربری نیز روشن باشد."
)

with st.sidebar:
    st.header("تنظیمات")

    auto_scan = st.checkbox(
        "اسکن خودکار بازار",
        value=env_bool("AUTO_SCAN_ENABLED", True),
    )

    env_live = env_bool("TABDEAL_LIVE_TRADING", env_bool("AUTO_TRADE", False))
    live_trading = st.checkbox(
        "موتور معامله واقعی",
        value=env_live,
        help="باید در Render با TABDEAL_LIVE_TRADING=true فعال شده باشد.",
    )

    trade_usdt_default = env_float("AUTO_TRADE_USDT", DEFAULT_TRADE_USDT)
    trade_usdt = st.number_input(
        "مبلغ هر معامله (USDT)",
        min_value=1.0,
        max_value=100000.0,
        value=max(1.0, trade_usdt_default),
        step=1.0,
    )

    max_positions = st.number_input(
        "حداکثر معاملات همزمان",
        min_value=1,
        max_value=20,
        value=max(1, env_int("AUTO_MAX_POSITIONS", DEFAULT_MAX_POSITIONS)),
        step=1,
    )

    scan_interval = st.number_input(
        "فاصله اسکن خودکار (ثانیه)",
        min_value=30,
        max_value=3600,
        value=max(30, env_int("AUTO_SCAN_INTERVAL", DEFAULT_SCAN_INTERVAL)),
        step=10,
    )

    workers = st.slider(
        "تعداد پردازش همزمان",
        min_value=1,
        max_value=12,
        value=DEFAULT_SCAN_WORKERS,
    )

    deep_candidates = st.slider(
        "تعداد کاندیدا برای تحلیل عمیق",
        min_value=3,
        max_value=30,
        value=12,
    )

    st.caption(
        "درصد = احتمال تخمینی موفقیت TP1 بر پایه داده تاریخی/OOS و هم‌جهتی فعلی؛ تضمین سود نیست."
    )

    api_key, api_secret = get_credentials()
    if api_key and api_secret:
        st.success("API تبدیل: دریافت شد")
    else:
        st.error("API تبدیل: پیدا نشد")

    if env_live:
        st.warning("AUTO-TRADING: فعال")
    else:
        st.info("AUTO-TRADING: خاموش")

# Session state
if "last_scan" not in st.session_state:
    st.session_state.last_scan = 0.0

if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

# ----------------------------- SCAN ---------------------------

if auto_scan:
    symbols = get_universe()

    if live_trading and env_live and not symbols:
        st.error(
            "بازارهای تبدیل دریافت نشدند؛ برای جلوگیری از معامله اشتباه، "
            "AUTO-TRADING متوقف شد."
        )
        st.stop()

    st.subheader(f"اسکن بازار — {len(symbols)} بازار USDT")

    progress = st.progress(0)
    status = st.empty()

    def fast_progress(v):
        pct = int(v * 70)
        progress.progress(pct)
        status.caption(f"مرحله ۱: اسکن سریع بازار — {pct}%")

    fast = run_fast_market_scan(
        symbols,
        workers,
        progress_cb=fast_progress,
    )

    if fast.empty:
        progress.progress(100)
        status.error("داده کافی برای اسکن بازار دریافت نشد.")
        st.stop()

    candidates = (
        fast.head(deep_candidates)["symbol"]
        .astype(str)
        .tolist()
    )

    def deep_progress(v):
        pct = 70 + int(v * 30)
        progress.progress(min(pct, 100))
        status.caption(f"مرحله ۲: تحلیل عمیق کاندیداها — {pct}%")

    deep_rows = deep_scan_candidates(
        candidates,
        workers,
        progress_cb=deep_progress,
    )

    progress.progress(100)
    status.success("اسکن کامل شد — 100%")

    table = scan_table(deep_rows)

    if not table.empty:
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        # ---------------- AUTO DECISION ----------------
        ranked = sorted(
            deep_rows,
            key=lambda r: (
                r["probability"] if np.isfinite(r["probability"]) else -1,
                r["score"],
            ),
            reverse=True,
        )

        executable = [
            r for r in ranked
            if r["confirmed"]
            and r["decision"] == "معامله"
        ]

        if executable:
            best = executable[0]

            st.success(
                f"بهترین کاندیدای فعلی: {best['symbol']} | "
                f"درصد {best['probability']:.1f}% | "
                f"روند {best['trend']} | "
                f"امتیاز {best['score']:.1f}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ورود", fmt_num(best["entry"]))
            c2.metric("حدضرر", fmt_num(best["sl"]))
            c3.metric("TP1", fmt_num(best["tp1"]))
            c4.metric("TP2", fmt_num(best["tp2"]))
            c5.metric("RR", f"{best['rr1']:.2f}")

            st.write(
                f"چشم‌انداز سناریویی تا TP1: +{best['outlook_pct']:.2f}% | "
                f"تأیید چندتایم‌فریمی: {best['agreement']:.0f}% | "
                f"OOS: {best['oos_trades']} | PF: "
                f"{best['profit_factor']:.2f}"
            )

            # ---------------- LIVE TRADE ----------------
            if live_trading and env_live:
                # Session state is only a secondary guard. The actual position
                # limit is checked against Tabdeal's persistent open OCO orders.
                already = any(
                    x.get("symbol") == best["symbol"]
                    and x.get("status") == "success"
                    for x in st.session_state.trade_log
                )

                if already:
                    st.info("این ارز در همین نشست قبلاً توسط ربات معامله شده است.")
                else:
                    result = execute_trade(
                        best,
                        float(trade_usdt),
                        int(max_positions),
                    )

                    if result.get("ok"):
                        st.session_state.trade_log.append({
                            "symbol": best["symbol"],
                            "status": "success",
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "message": result["message"],
                            "executed_qty": result.get("executed_qty", ""),
                        })
                        st.success(result["message"])
                    else:
                        st.session_state.trade_log.append({
                            "symbol": best["symbol"],
                            "status": "error",
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "message": result["message"],
                        })
                        st.error(result["message"])
            else:
                if not env_live:
                    st.warning(
                        "TABDEAL_LIVE_TRADING در Render فعال نیست؛ "
                        "اپ فقط تحلیل می‌کند و هیچ سفارش واقعی نمی‌فرستد."
                    )
                else:
                    st.info("موتور معامله واقعی توسط کلید داخل رابط کاربری خاموش شده است.")

        else:
            st.warning("در این اسکن هیچ ارز شرایط کامل ورود را نداشت.")

    else:
        st.warning("هیچ کاندیدای قابل تحلیل عمیق پیدا نشد.")

    st.session_state.last_scan = time.time()

    # Auto-refresh after the configured interval.
    if auto_scan:
        time.sleep(int(scan_interval))
        st.rerun()

else:
    st.info("اسکن خودکار خاموش است.")

# ----------------------------- LOG ----------------------------

if st.session_state.trade_log:
    st.subheader("گزارش معاملات خودکار")
    st.dataframe(
        pd.DataFrame(st.session_state.trade_log),
        use_container_width=True,
        hide_index=True,
    )
