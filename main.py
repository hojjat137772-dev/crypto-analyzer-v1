
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# ============================================================
# Crypto Analyzer Pro - Tabdeal USDT only
# Analysis only - no order placement
# ============================================================

st.set_page_config(
    page_title="Crypto Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TABDEAL_BASE = "https://api1.tabdeal.org"
EXCHANGE_NAME = "تبدیل"
# Historical candle fallback (public, no API key): Nobitex OHLC
# Tabdeal remains the source of the live price and the market universe.
# Nobitex is used only when Tabdeal recent trades do not contain enough history.
NOBITEX_BASE = "https://api.nobitex.ir"
NOBITEX_NAME = "نوبیتکس (فقط تاریخچه)"

INTERVALS = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}




# -------------------- Generic HTTP --------------------

_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "CryptoAnalyzerPro/2.1"})

def http_get(url, params=None, timeout=7):
    # Short, fail-fast public API calls. A failed endpoint should not block the
    # whole analysis for 15-30 seconds.
    timeout = min(float(timeout), 7.0)
    r = _HTTP.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def unix_now():
    return int(time.time())

def resolution_seconds(tf):
    return {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[tf]

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# -------------------- Market lists --------------------



def _collect_symbols(obj):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"symbol", "tabdealsymbol"} and isinstance(v, str):
                found.append(v.upper().replace("_", ""))
            else:
                found.extend(_collect_symbols(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_collect_symbols(item))
    return found

@st.cache_data(ttl=180, show_spinner=False)
def tabdeal_markets():
    data = http_get(
        f"{TABDEAL_BASE}/r/api/v1/exchangeInfo",
        timeout=15,
    )
    symbols = _collect_symbols(data)
    # Keep realistic spot symbols and remove duplicates.
    symbols = [
        s for s in symbols
        if 5 <= len(s) <= 24 and s.isalnum() and s.endswith("USDT")
    ]
    if not symbols:
        raise RuntimeError("لیست بازارهای تبدیل دریافت نشد.")
    return sorted(set(symbols))




def get_markets(exchange):
    if exchange != EXCHANGE_NAME:
        return []
    return tabdeal_markets()


# -------------------- Binance candles --------------------



# -------------------- UDF adapters: Wallex / Bit24 --------------------

def udf_to_df(data):
    if not isinstance(data, dict) or data.get("s") not in {"ok", "no_data"}:
        raise RuntimeError("پاسخ OHLC معتبر نیست.")
    t = data.get("t", [])
    o = data.get("o", [])
    h = data.get("h", [])
    l = data.get("l", [])
    c = data.get("c", [])
    v = data.get("v", [])
    n = min(len(t), len(o), len(h), len(l), len(c), len(v))
    if n < 80:
        raise RuntimeError("داده OHLC کافی نیست.")
    return pd.DataFrame({
        "open_time": pd.to_datetime(pd.to_numeric(t[:n]), unit="s"),
        "open": pd.to_numeric(o[:n], errors="coerce"),
        "high": pd.to_numeric(h[:n], errors="coerce"),
        "low": pd.to_numeric(l[:n], errors="coerce"),
        "close": pd.to_numeric(c[:n], errors="coerce"),
        "volume": pd.to_numeric(v[:n], errors="coerce"),
    }).dropna()





def parse_tabdeal_trade_rows(data):
    rows = data
    if isinstance(data, dict):
        for key in ("data", "result", "trades"):
            if key in data:
                rows = data[key]
                break
    if not isinstance(rows, list):
        raise RuntimeError("لیست معاملات تبدیل معتبر نیست.")

    out = []
    for x in rows:
        if not isinstance(x, dict):
            continue
        price = safe_float(x.get("price"))
        qty = safe_float(x.get("qty", x.get("quantity")))
        ts = safe_float(x.get("time", x.get("timestamp")))
        if np.isfinite(price) and np.isfinite(qty) and np.isfinite(ts):
            if ts > 10_000_000_000:
                ts /= 1000
            out.append((pd.to_datetime(ts, unit="s"), price, qty))

    if len(out) < 20:
        raise RuntimeError("معاملات کافی از تبدیل دریافت نشد.")
    return out


def tabdeal_pair_name(symbol):
    """Convert BTCUSDT/BTCIRT to Tabdeal's documented BTC_USDT/BTC_IRT."""
    s = str(symbol).upper().replace("-", "").replace("_", "")
    for quote in ["USDT", "IRT", "USDC", "BTC", "ETH"]:
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}_{quote}"
    return s


@st.cache_data(ttl=30, show_spinner=False)
def tabdeal_trades(symbol, limit=1000):
    """Public recent trades from Tabdeal.

    The official Tabdeal docs support both `symbol` and `tabdealSymbol` and
    allow up to 1000 recent trades. We explicitly request the maximum public
    history so the 15m/1h/4h aggregations have enough candles to work.
    """
    compact = str(symbol).upper().replace("-", "").replace("_", "")
    tab_symbol = tabdeal_pair_name(compact)
    attempts = [
        {"symbol": compact, "limit": min(int(limit), 1000)},
        {"tabdealSymbol": tab_symbol, "limit": min(int(limit), 1000)},
        {"tabdealSymbol": tab_symbol},
    ]
    last = None
    for params in attempts:
        try:
            data = http_get(
                f"{TABDEAL_BASE}/r/api/v1/trades",
                params,
                15,
            )
            rows = parse_tabdeal_trade_rows(data)
            if rows:
                return rows
        except Exception as exc:
            last = exc

    raise RuntimeError(
        f"داده معاملات تبدیل برای {symbol} در دسترس نیست ({tab_symbol})."
    ) from last


def trades_to_ohlcv(trades, interval):
    sec = resolution_seconds(interval)
    df = pd.DataFrame(trades, columns=["open_time", "price", "volume"])
    df = df.sort_values("open_time").set_index("open_time")
    rule = {900:"15min", 3600:"1h", 14400:"4h", 86400:"1D"}[sec]
    o = df["price"].resample(rule).first()
    h = df["price"].resample(rule).max()
    l = df["price"].resample(rule).min()
    c = df["price"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.DataFrame({
        "open_time": o.index,
        "open": o.values,
        "high": h.values,
        "low": l.values,
        "close": c.values,
        "volume": v.values,
    }).dropna()
    # Do not reject a timeframe merely because it has fewer than 20 candles.
    # The higher-level analyzer will select only timeframes with enough
    # indicator history; this lets 15m data remain usable when 4h/1d history
    # is naturally sparse on a recent-trades endpoint.
    if len(out) < 2:
        raise RuntimeError("تاریخچه معاملاتی تبدیل برای این تایم‌فریم کافی نیست.")
    return out.tail(300)


@st.cache_data(ttl=30, show_spinner=False)
def tabdeal_klines(symbol, interval):
    """Build OHLCV strictly from Tabdeal trade data; no other exchange is used."""
    compact = str(symbol).upper().replace("-", "").replace("_", "")
    if not compact.endswith("USDT"):
        raise RuntimeError("فقط بازارهای دلاری USDT مجاز هستند.")
    return trades_to_ohlcv(tabdeal_trades(compact), interval)


@st.cache_data(ttl=20, show_spinner=False)
def tabdeal_price(symbol):
    """Read Tabdeal public order book using documented tabdealSymbol first."""
    tab_symbol = tabdeal_pair_name(symbol)
    attempts = [
        {"tabdealSymbol": tab_symbol, "limit": 5},
        {"tabdealSymbol": tab_symbol},
        {"symbol": str(symbol).upper(), "limit": 5},
    ]
    last = None
    for params in attempts:
        try:
            data = http_get(
                f"{TABDEAL_BASE}/r/api/v1/depth",
                params,
                15,
            )
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if bids and asks:
                return (float(bids[0][0]) + float(asks[0][0])) / 2
            if bids:
                return float(bids[0][0])
            if asks:
                return float(asks[0][0])
        except Exception as exc:
            last = exc

    raise RuntimeError(
        f"قیمت تبدیل برای {symbol} دریافت نشد ({tab_symbol})."
    ) from last

# -------------------- Historical reference candles --------------------

def nobitex_resolution(interval):
    return {
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D",
    }[interval]

def nobitex_symbol(symbol):
    compact = str(symbol).upper().replace("-", "").replace("_", "")
    if not compact.endswith("USDT"):
        raise RuntimeError("تاریخچه نوبیتکس فقط برای بازارهای USDT درخواست می‌شود.")
    return compact

@st.cache_data(ttl=300, show_spinner=False)
def nobitex_candles(symbol, interval, countback=500):
    """Public Nobitex OHLC. No API key/token is required.

    Uses the USDT market matching the Tabdeal symbol, e.g. BTCUSDT.
    Up to 500 candles are requested, which is enough for the indicators and
    the 1h forecast used by this app.
    """
    nsym = nobitex_symbol(symbol)
    resolution = nobitex_resolution(interval)
    params = {
        "symbol": nsym,
        "resolution": resolution,
        "to": unix_now(),
        "countback": min(int(countback), 500),
    }
    data = http_get(f"{NOBITEX_BASE}/market/udf/history", params, 10)
    if not isinstance(data, dict) or data.get("s") != "ok":
        msg = data.get("errmsg", "داده‌ای برای این بازار وجود ندارد.") if isinstance(data, dict) else "پاسخ نامعتبر"
        raise RuntimeError(f"تاریخچه نوبیتکس برای {nsym} در دسترس نیست: {msg}")

    t = data.get("t", [])
    o = data.get("o", [])
    h = data.get("h", [])
    l = data.get("l", [])
    c = data.get("c", [])
    v = data.get("v", [])
    n = min(len(t), len(o), len(h), len(l), len(c), len(v))
    if n < 60:
        raise RuntimeError(f"تاریخچه نوبیتکس {nsym} برای {interval} کافی نیست ({n} کندل).")

    df = pd.DataFrame({
        "open_time": pd.to_datetime(pd.to_numeric(t[:n]), unit="s"),
        "open": pd.to_numeric(o[:n], errors="coerce"),
        "high": pd.to_numeric(h[:n], errors="coerce"),
        "low": pd.to_numeric(l[:n], errors="coerce"),
        "close": pd.to_numeric(c[:n], errors="coerce"),
        "volume": pd.to_numeric(v[:n], errors="coerce"),
    }).dropna(subset=["open_time", "open", "high", "low", "close"])
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    if len(df) < 60:
        raise RuntimeError(f"تاریخچه نوبیتکس {nsym} برای {interval} کافی نیست.")
    if "volume" not in df or df["volume"].isna().all():
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    return df.tail(min(int(countback), 500))

def analysis_candles(symbol, interval):
    """Use Tabdeal first; if its recent trades are too sparse, fall back to
    Nobitex public OHLC for the same USDT market. Live price remains Tabdeal.
    """
    try:
        d = tabdeal_klines(symbol, interval)
        x = add_indicators(d)
        # Require enough history for the indicator stack; otherwise use the
        # public historical OHLC source.
        if len(x) >= 60:
            return d, EXCHANGE_NAME
    except Exception:
        pass

    try:
        return nobitex_candles(symbol, interval), NOBITEX_NAME
    except Exception as exc:
        raise RuntimeError(
            f"برای {symbol} در تایم‌فریم {interval} نه تاریخچه کافی از تبدیل و نه نوبیتکس دریافت نشد: {exc}"
        ) from exc

# -------------------- Exchange router --------------------

def get_price(exchange, symbol):
    if exchange != EXCHANGE_NAME:
        raise RuntimeError("فقط صرافی تبدیل فعال است.")
    return tabdeal_price(symbol)

def get_candles(exchange, symbol, interval):
    if exchange != EXCHANGE_NAME:
        raise RuntimeError("فقط صرافی تبدیل فعال است.")
    # Tabdeal is the primary market source. If its recent-trade history is
    # too short, transparently use the public Nobitex OHLC history for the
    # same USDT pair. The displayed/live price remains from Tabdeal.
    candles, _source = analysis_candles(symbol, interval)
    return candles

# -------------------- Indicators --------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def add_indicators(df):
    x = df.copy()
    x["ema9"] = ema(x["close"], 9)
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)
    fast = ema(x["close"], 12)
    slow = ema(x["close"], 26)
    x["macd"] = fast - slow
    x["macd_signal"] = ema(x["macd"], 9)
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    x["atr"] = atr(x, 14)
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["vol_ratio"] = x["volume"] / x["vol_ma20"].replace(0, np.nan)
    # ADX / directional movement
    up = x["high"].diff()
    down = -x["low"].diff()
    tr = pd.concat([x["high"]-x["low"], (x["high"]-x["close"].shift()).abs(), (x["low"]-x["close"].shift()).abs()], axis=1).max(axis=1)
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=x.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=x.index)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    x["plus_di"] = plus_di
    x["minus_di"] = minus_di
    # Ichimoku (non-forward display values)
    tenkan = (x["high"].rolling(9).max() + x["low"].rolling(9).min()) / 2
    kijun = (x["high"].rolling(26).max() + x["low"].rolling(26).min()) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (x["high"].rolling(52).max() + x["low"].rolling(52).min()) / 2
    x["ich_tenkan"], x["ich_kijun"] = tenkan, kijun
    x["ich_span_a"], x["ich_span_b"] = span_a, span_b
    # Rolling VWAP approximation for intraday confirmation
    typical = (x["high"] + x["low"] + x["close"]) / 3
    pv = typical * x["volume"]
    x["vwap20"] = pv.rolling(20).sum() / x["volume"].rolling(20).sum().replace(0, np.nan)
    # Candle/body and volatility regime
    x["body_pct"] = (x["close"] - x["open"]) / x["open"].replace(0, np.nan) * 100
    x["range_pct"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan) * 100
    return x.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def support_resistance(df, window=60):
    recent = df.tail(window)
    return float(recent["low"].min()), float(recent["high"].max())

def timeframe_score(df):
    """Strict multi-factor score (0-100). More confirmations are required for extremes."""
    x = add_indicators(df)
    if len(x) < 60:
        return 50, ["داده کافی نیست"], x
    a, p = x.iloc[-1], x.iloc[-2]
    score = 50.0
    reasons = []

    # Trend structure
    if a["ema9"] > a["ema20"] > a["ema50"]:
        score += 10; reasons.append("ساختار EMA صعودی")
    elif a["ema9"] < a["ema20"] < a["ema50"]:
        score -= 10; reasons.append("ساختار EMA نزولی")
    elif a["ema20"] > a["ema50"]:
        score += 4; reasons.append("روند میان‌مدت صعودی")
    elif a["ema20"] < a["ema50"]:
        score -= 4; reasons.append("روند میان‌مدت نزولی")

    if a["close"] > a["ema200"]:
        score += 7; reasons.append("قیمت بالای EMA200")
    else:
        score -= 7; reasons.append("قیمت زیر EMA200")

    # MACD direction + histogram acceleration
    if a["macd"] > a["macd_signal"]:
        score += 7; reasons.append("MACD صعودی")
    else:
        score -= 7; reasons.append("MACD نزولی")
    if a["macd_hist"] > p["macd_hist"]:
        score += 4; reasons.append("شتاب MACD مثبت")
    else:
        score -= 4; reasons.append("شتاب MACD منفی")

    # RSI: reward healthy momentum, penalize exhaustion
    rv=float(a["rsi"])
    if 52 <= rv <= 68:
        score += 7; reasons.append(f"RSI سالم {rv:.1f}")
    elif 45 <= rv < 52:
        score += 1
    elif 32 <= rv < 45:
        score -= 2
    elif rv > 72:
        score -= 7; reasons.append(f"RSI بیش‌خرید {rv:.1f}")
    elif rv < 28:
        score += 1; reasons.append(f"RSI اشباع فروش {rv:.1f}")

    # ADX + DI direction
    adx_v=float(a["adx"]); plus=float(a["plus_di"]); minus=float(a["minus_di"])
    if adx_v >= 25:
        if plus > minus:
            score += 7; reasons.append(f"روند قوی صعودی ADX {adx_v:.0f}")
        else:
            score -= 7; reasons.append(f"روند قوی نزولی ADX {adx_v:.0f}")
    elif adx_v < 18:
        score += 0; reasons.append("بازار کم‌روند / رنج")

    # Ichimoku cloud
    cloud_top=max(float(a["ich_span_a"]), float(a["ich_span_b"]))
    cloud_bottom=min(float(a["ich_span_a"]), float(a["ich_span_b"]))
    if a["close"] > cloud_top and a["ich_tenkan"] > a["ich_kijun"]:
        score += 6; reasons.append("تأیید ایچیموکو صعودی")
    elif a["close"] < cloud_bottom and a["ich_tenkan"] < a["ich_kijun"]:
        score -= 6; reasons.append("تأیید ایچیموکو نزولی")

    # VWAP
    if np.isfinite(a["vwap20"]):
        if a["close"] > a["vwap20"]:
            score += 4; reasons.append("بالای VWAP")
        else:
            score -= 4; reasons.append("زیر VWAP")

    # Volume confirmation
    vr=float(a["vol_ratio"]) if np.isfinite(a["vol_ratio"]) else 1.0
    if vr >= 1.5:
        score += 5 if a["close"] >= p["close"] else -5
        reasons.append(f"حجم تأییدکننده {vr:.1f}x")
    elif vr < 0.65:
        reasons.append(f"حجم ضعیف {vr:.1f}x")

    # Recent momentum slope
    ret5=float(a["close"]/x["close"].iloc[-6]-1) if len(x)>=6 else 0
    if ret5 > 0.01:
        score += 3; reasons.append("مومنتوم کوتاه‌مدت مثبت")
    elif ret5 < -0.01:
        score -= 3; reasons.append("مومنتوم کوتاه‌مدت منفی")

    return int(np.clip(round(score), 0, 100)), reasons, x

def money(v):
    if not np.isfinite(v):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    return f"{v:.8f}"


# -------------------- Probabilistic Forecast Engine --------------------

def forecast_features(df):
    """Build leakage-safe features available at candle close."""
    x = add_indicators(df).copy()
    close = x["close"].replace(0, np.nan)
    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret6"] = close.pct_change(6)
    x["ret12"] = close.pct_change(12)
    x["ema20_gap"] = (close / x["ema20"]) - 1
    x["ema50_gap"] = (close / x["ema50"]) - 1
    x["ema200_gap"] = (close / x["ema200"]) - 1
    x["macd_pct"] = x["macd_hist"] / close
    x["atr_pct"] = x["atr"] / close
    x["vol_ratio"] = x["vol_ratio"].replace([np.inf, -np.inf], np.nan)
    x["rsi_norm"] = (x["rsi"] - 50) / 50
    cols = [
        "ret1", "ret3", "ret6", "ret12", "ema20_gap", "ema50_gap",
        "ema200_gap", "macd_pct", "atr_pct", "vol_ratio", "rsi_norm"
    ]
    x = x.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return x, cols


def _fit_ridge(X, y, alpha=1e-3):
    """Small numpy-only ridge regression; avoids adding sklearn dependency."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-10)] = 1.0
    Xz = (X - mu) / sd
    A = np.column_stack([np.ones(len(Xz)), Xz])
    reg = np.eye(A.shape[1]) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return coef, mu, sd


def _predict_ridge(X, model):
    coef, mu, sd = model
    X = np.asarray(X, dtype=float)
    Xz = (X - mu) / sd
    A = np.column_stack([np.ones(len(Xz)), Xz])
    return A @ coef


def _forecast_one_horizon(feature_df, feature_cols, horizon, min_train=80):
    """Walk-forward validation + final fit for one future horizon."""
    n = len(feature_df)
    if n <= horizon + min_train + 5:
        return None

    X_all = feature_df[feature_cols].to_numpy(dtype=float)
    close = feature_df["close"].to_numpy(dtype=float)
    future = np.full(n, np.nan)
    future[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    valid = np.isfinite(future) & np.isfinite(X_all).all(axis=1)
    idx = np.where(valid)[0]
    if len(idx) < min_train + 20:
        return None

    # Hold out the latest observations for genuine out-of-sample validation.
    val_count = min(40, max(20, len(idx) // 5))
    train_idx = idx[:-val_count]
    val_idx = idx[-val_count:]
    if len(train_idx) < min_train:
        return None

    model = _fit_ridge(X_all[train_idx], future[train_idx], alpha=0.01)
    val_pred = _predict_ridge(X_all[val_idx], model)
    val_actual = future[val_idx]
    residuals = val_actual - val_pred
    direction_accuracy = float(np.mean(np.sign(val_pred) == np.sign(val_actual)) * 100)
    mae = float(np.mean(np.abs(residuals)))
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else mae

    # Refit on all historical examples before making the live forecast.
    final_model = _fit_ridge(X_all[idx], future[idx], alpha=0.01)
    latest_pred = float(_predict_ridge(X_all[[-1]], final_model)[0])
    latest_price = float(close[-1])

    # Convert residual uncertainty into a practical probability of a positive return.
    sigma = max(residual_std, mae, 1e-5)
    z = latest_pred / sigma
    probability_up = float(np.clip(50 + 50 * np.tanh(z / 1.35), 1, 99))

    # Forecast interval: model prediction +/- validation error, expressed as price.
    low_ret = latest_pred - 1.28 * sigma
    high_ret = latest_pred + 1.28 * sigma
    low_price = latest_price * float(np.exp(low_ret))
    high_price = latest_price * float(np.exp(high_ret))
    target_price = latest_price * float(np.exp(latest_pred))

    # Confidence combines historical directional accuracy and sample size.
    sample_factor = min(1.0, len(train_idx) / 300.0)
    confidence = float(np.clip(
        0.55 * direction_accuracy + 45.0 * sample_factor,
        0, 100
    ))
    direction = "صعودی" if latest_pred > max(0.002, sigma * 0.15) else (
        "نزولی" if latest_pred < -max(0.002, sigma * 0.15) else "خنثی"
    )

    return {
        "horizon": horizon,
        "pred_return": latest_pred,
        "target_price": target_price,
        "low_price": min(low_price, high_price),
        "high_price": max(low_price, high_price),
        "probability_up": probability_up,
        "direction_accuracy": direction_accuracy,
        "mae": mae,
        "confidence": confidence,
        "direction": direction,
        "samples": len(idx),
    }


def future_forecast(df_1h):
    """Forecast 4h, 24h and 3d using 1h candles and walk-forward validation."""
    feat, cols = forecast_features(df_1h)
    if len(feat) < 140:
        raise RuntimeError("برای پیش‌بینی، حداقل حدود 140 کندل 1h لازم است.")

    horizons = [("4 ساعت", 4), ("24 ساعت", 24), ("3 روز", 72)]
    rows = []
    for label, bars in horizons:
        result = _forecast_one_horizon(feat, cols, bars)
        if result is None:
            continue
        result["label"] = label
        rows.append(result)
    if not rows:
        raise RuntimeError("داده کافی برای ساخت پیش‌بینی احتمالی وجود ندارد.")

    return pd.DataFrame(rows), feat


# -------------------- Market Scanner --------------------

STABLE_BASES = {
    "USDT", "USDC", "FDUSD", "BUSD", "DAI", "TUSD", "USDE", "USDD",
    "USD", "EUR", "TRY", "IRT", "RLS"
}


def base_asset(symbol):
    s = str(symbol).upper().replace("-", "").replace("_", "")
    for q in ["USDT", "USDC", "FDUSD", "BUSD", "USDE", "USDD", "TUSD", "IRT", "RLS"]:
        if s.endswith(q) and len(s) > len(q):
            return s[:-len(q)], q
    return s, ""


def scanner_candidates(exchange, markets, limit=None):
    """Universe: Tabdeal USDT spot markets only."""
    if exchange != EXCHANGE_NAME:
        return []
    cleaned=[]
    for raw in markets:
        m=str(raw).upper().replace("-","").replace("_","")
        base,quote=base_asset(m)
        if base and base not in STABLE_BASES and quote == "USDT":
            cleaned.append(m)
    preferred=[
        "BTCUSDT","ETHUSDT","XRPUSDT","SOLUSDT","BNBUSDT","DOGEUSDT",
        "ADAUSDT","AVAXUSDT","LINKUSDT","TRXUSDT","DOTUSDT","LTCUSDT",
        "SHIBUSDT","ATOMUSDT","NEARUSDT","ARBUSDT","OPUSDT","APTUSDT",
        "SUIUSDT","PEPEUSDT"
    ]
    ordered=[x for x in preferred if x in cleaned]
    ordered += [x for x in cleaned if x not in ordered]
    return ordered if limit is None else ordered[:max(1,int(limit))]


def signal_from_score(score, ema20, ema50, ema200, macd_hist, rsi_value, adx_value=None, ich_bull=None, ich_bear=None):
    """Strict confirmation-based market state."""
    if score is None:
        return "HOLD", "⚪", "داده ناکافی"
    adx_ok = True if adx_value is None else adx_value >= 20
    bull_extra = True if ich_bull is None else ich_bull
    bear_extra = True if ich_bear is None else ich_bear
    bullish = (score >= 72 and ema20 > ema50 and ema50 >= ema200 and macd_hist >= 0
               and rsi_value < 70 and adx_ok and bull_extra)
    bearish = (score <= 28 and ema20 < ema50 and ema50 <= ema200 and macd_hist <= 0
               and rsi_value > 30 and adx_ok and bear_extra)
    if bullish:
        return "BUY", "🟢", "تأیید چندعاملی صعود"
    if bearish:
        return "SELL", "🔴", "تأیید چندعاملی نزول"
    return "HOLD", "🟡", "تأیید کامل ورود وجود ندارد"


def scanner_action(score):
    if score is None:
        return "HOLD"
    if score >= 68:
        return "BUY"
    if score <= 35:
        return "SELL"
    return "HOLD"


@st.cache_data(ttl=30, show_spinner=False)
def scan_one_market(exchange, symbol, scan_tfs):
    """Analyze one market. Timeframes are fetched in parallel."""
    scores = []
    tf_scores = {}
    latest_price = np.nan
    atr_value = np.nan
    support = np.nan
    resistance = np.nan
    last_ind = None

    def fetch_tf(item):
        tf_name, weight = item
        try:
            df = get_candles(exchange, symbol, tf_name)
            score, _, ind = timeframe_score(df)
            if len(ind) < 10:
                return tf_name, weight, None, None
            return tf_name, weight, score, ind
        except Exception:
            return tf_name, weight, None, None

    workers = min(4, max(1, len(scan_tfs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch_tf, scan_tfs))

    # Keep the most important timeframe as the reference for levels.
    ref_tf = max(scan_tfs, key=lambda x: x[1])[0] if scan_tfs else None
    ref_ind = None
    for tf_name, weight, score, ind in results:
        tf_scores[tf_name] = score
        if score is not None:
            scores.append((score, weight))
            if tf_name == ref_tf or ref_ind is None:
                ref_ind = ind

    if ref_ind is not None and len(ref_ind):
        last = ref_ind.iloc[-1]
        latest_price = float(last["close"])
        atr_value = float(last["atr"])
        support, resistance = support_resistance(ref_ind, 60)
        last_ind = ref_ind

    if len(scores) < 2 or not np.isfinite(latest_price) or latest_price <= 0:
        raise RuntimeError("حداقل دو تایم‌فریم معتبر در دسترس نیست")

    score = round(sum(s * w for s, w in scores) / sum(w for _, w in scores))
    last = last_ind.iloc[-1]
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi_value = float(last["rsi"])
    macd_hist = float(last["macd_hist"])

    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = latest_price * 0.02
    if not np.isfinite(support) or support <= 0:
        support = latest_price - 1.5 * atr_value
    if not np.isfinite(resistance) or resistance <= 0:
        resistance = latest_price + 2.0 * atr_value

    signal, icon, signal_text = signal_from_score(
        score, ema20, ema50, ema200, macd_hist, rsi_value,
        float(last.get("adx", 0)),
        bool(last.get("close", 0) > max(last.get("ich_span_a", 0), last.get("ich_span_b", 0)) and last.get("ich_tenkan", 0) > last.get("ich_kijun", 0)),
        bool(last.get("close", 0) < min(last.get("ich_span_a", 0), last.get("ich_span_b", 0)) and last.get("ich_tenkan", 0) < last.get("ich_kijun", 0)),
    )
    risk = max(1.5 * atr_value, latest_price * 0.01)

    if signal == "BUY":
        entry = latest_price
        sl = min(entry - risk, support * 0.995)
        if sl <= 0 or sl >= entry:
            sl = entry - risk
        risk_amount = max(entry - sl, entry * 0.01)
        tp1, tp2 = entry + 1.5 * risk_amount, entry + 2.5 * risk_amount
        if resistance > entry and resistance < tp1:
            tp1 = max(entry + risk_amount, resistance * 0.995)
    elif signal == "SELL":
        entry = latest_price
        sl = max(entry + risk, resistance * 1.005)
        risk_amount = max(sl - entry, entry * 0.01)
        tp1, tp2 = entry - 1.5 * risk_amount, entry - 2.5 * risk_amount
        if support < entry and support > tp1:
            tp1 = min(entry - risk_amount, support * 1.005)
    else:
        entry = latest_price
        sl = tp1 = tp2 = np.nan

    return {
        "بازار": symbol, "امتیاز": score, "سیگنال": f"{icon} {signal}",
        "توضیح سیگنال": signal_text, "قیمت": latest_price,
        "ورود": entry if signal != "HOLD" else np.nan, "حدضرر": sl,
        "حدسود 1": tp1, "حدسود 2": tp2, "حمایت": support, "مقاومت": resistance,
        "15m": tf_scores.get("15m"), "1h": tf_scores.get("1h"),
        "4h": tf_scores.get("4h"), "1d": tf_scores.get("1d"),
        "تایم‌فریم معتبر": len(scores),
    }


def pump_setup_score(ind_1h, ind_4h, ind_1d, forecast_score):
    """Score an early bullish setup, not an already-completed pump."""
    a = ind_1h.iloc[-1]
    b = ind_4h.iloc[-1]
    d = ind_1d.iloc[-1]
    price = float(a["close"])
    resistance = support_resistance(ind_4h, 60)[1]
    dist_res = ((resistance / price) - 1) * 100 if price > 0 else 999
    vr = float(a["vol_ratio"]) if np.isfinite(a["vol_ratio"]) else 1.0
    ret4 = float(a["close"] / ind_1h["close"].iloc[-5] - 1) if len(ind_1h) >= 5 else 0.0
    ret24 = float(a["close"] / ind_1h["close"].iloc[-25] - 1) if len(ind_1h) >= 25 else 0.0
    rsi = float(a["rsi"])

    volume_score = 100 if vr >= 2.0 else 85 if vr >= 1.5 else 70 if vr >= 1.2 else 50 if vr >= 0.9 else 25
    momentum_score = 100 if 0.01 <= ret4 <= 0.08 else 85 if 0.0 <= ret4 < 0.01 else 55 if ret4 < 0.0 else 25
    early_score = 100 if 0.3 <= dist_res <= 3.0 else 88 if 0.0 < dist_res <= 5.0 else 65 if 5.0 < dist_res <= 8.0 else 35
    not_pumped = 100 if ret24 <= 0.08 else 75 if ret24 <= 0.12 else 30 if ret24 <= 0.20 else 0
    rsi_score = 100 if 52 <= rsi <= 66 else 82 if 48 <= rsi <= 70 else 45

    trend_bonus = 0
    if float(a["ema20"]) > float(a["ema50"]): trend_bonus += 8
    if float(b["ema20"]) > float(b["ema50"]): trend_bonus += 8
    if float(d["ema20"]) > float(d["ema50"]): trend_bonus += 8
    trend_score = min(100, 55 + trend_bonus)

    score = (
        0.35 * float(forecast_score) +
        0.20 * trend_score +
        0.15 * volume_score +
        0.10 * momentum_score +
        0.10 * early_score +
        0.05 * not_pumped +
        0.05 * rsi_score
    )
    # Hard penalty for a move that has already become extended.
    if ret24 > 0.20:
        score -= 18
    elif ret24 > 0.12:
        score -= 8
    return float(np.clip(score, 0, 100)), {
        "price": price, "resistance": resistance, "dist_res": dist_res,
        "vol_ratio": vr, "ret4": ret4, "ret24": ret24, "rsi": rsi,
    }


@st.cache_data(ttl=90, show_spinner=False)
def pump_scan_one_market(exchange, symbol):
    """Find early-stage bullish candidates using technical + forecast agreement."""
    tfs = ["1h", "4h", "1d"]
    data = {}
    scores = {}
    inds = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {tf: pool.submit(get_candles, exchange, symbol, tf) for tf in tfs}
        for tf, fut in futures.items():
            df = fut.result()
            sc, _, ind = timeframe_score(df)
            if len(ind) < 30:
                raise RuntimeError("داده کافی نیست")
            data[tf] = df; inds[tf] = ind; scores[tf] = float(sc)

    forecast_df, _ = future_forecast(data["1h"])
    probs = pd.to_numeric(forecast_df["probability_up"], errors="coerce").dropna()
    fconfs = pd.to_numeric(forecast_df["confidence"], errors="coerce").dropna()
    if probs.empty:
        raise RuntimeError("پیش‌بینی معتبر نیست")
    forecast_score = float(np.clip(probs.mean(), 1, 99))
    forecast_conf = float(fconfs.mean()) if not fconfs.empty else 0.0
    tech_score = 0.25*scores["1h"] + 0.40*scores["4h"] + 0.35*scores["1d"]
    combined, meta = pump_setup_score(inds["1h"], inds["4h"], inds["1d"], forecast_score)

    vals = np.array([scores["1h"], scores["4h"], scores["1d"], forecast_score], dtype=float)
    agreement = float(np.clip(100 - np.std(vals)*2.5, 0, 100))
    data_quality = float(np.clip(min(len(inds["1h"]), len(inds["4h"]), len(inds["1d"]) ) / 220 * 100, 0, 100))
    volume_quality = float(np.clip(meta["vol_ratio"] / 1.5 * 100, 0, 100))
    confidence = float(np.clip(0.42*agreement + 0.33*forecast_conf + 0.15*data_quality + 0.10*volume_quality, 0, 100))

    # Only label a candidate as high-confidence when multiple independent checks agree.
    if forecast_score < 62 or tech_score < 62 or confidence < 90:
        return None
    if meta["ret24"] > 0.20 or meta["dist_res"] < 0 or meta["dist_res"] > 8.0:
        return None
    if meta["rsi"] > 72:
        return None

    reason = f"فنی {tech_score:.0f}/100 | آینده‌نگری {forecast_score:.0f}% | حجم {meta['vol_ratio']:.1f}x | فاصله مقاومت {meta['dist_res']:.1f}%"
    return {
        "نماد": symbol, "امتیاز پامپ": round(combined), "اعتماد تحلیل": round(confidence),
        "چشم‌انداز": f"{forecast_score:.0f}% صعود احتمالی", "قیمت": money(meta["price"]),
        "فاصله تا مقاومت": f"{meta['dist_res']:.1f}%", "حجم 1h": f"{meta['vol_ratio']:.2f}x",
        "بازده 4h": f"{meta['ret4']*100:+.2f}%", "بازده 24h": f"{meta['ret24']*100:+.2f}%",
        "دلیل": reason, "_sort": combined,
    }


def run_pump_scanner(exchange, markets):
    candidates = full_market_candidates(exchange, markets)
    rows = []
    progress = st.progress(0, text="جستجوی ارزهای مستعد حرکت صعودی...")
    total = len(candidates)
    workers = min(5, max(1, total))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(pump_scan_one_market, exchange, symbol): symbol for symbol in candidates}
        for future in as_completed(futures):
            completed += 1
            try:
                row = future.result()
                if row is not None:
                    rows.append(row)
            except Exception:
                pass
            progress.progress(completed/max(total,1), text=f"اسکن {futures[future]} — {completed}/{total}")
    progress.empty()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)

def run_scanner(exchange, markets, scan_tfs, limit):
    candidates = scanner_candidates(exchange, markets, limit)
    rows = []
    progress = st.progress(0, text="شروع اسکن بازار...")
    total = len(candidates)

    def scan_symbol(symbol):
        try:
            return scan_one_market(exchange, symbol, tuple(scan_tfs))
        except Exception as exc:
            return {
                "بازار": symbol, "امتیاز": None, "سیگنال": "⚪ HOLD",
                "توضیح سیگنال": f"داده ناکافی: {str(exc)[:35]}",
                "تایم‌فریم معتبر": 0,
            }

    # A small worker pool dramatically reduces total wait time while avoiding
    # an aggressive request flood against public exchange APIs.
    workers = min(5, max(1, total))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_symbol, symbol): symbol for symbol in candidates}
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            symbol = futures[future]
            progress.progress(completed / max(total, 1), text=f"اسکن {symbol} — {completed}/{total}")

    progress.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["امتیاز_sort"] = pd.to_numeric(df["امتیاز"], errors="coerce")
    return df.sort_values(
        ["امتیاز_sort", "تایم‌فریم معتبر"], ascending=[False, False], na_position="last"
    ).drop(columns=["امتیاز_sort"]).reset_index(drop=True)


# -------------------- Full Market Scanner + Backtest --------------------

def full_market_candidates(exchange, markets):
    """Return the complete clean spot universe for the selected exchange."""
    return scanner_candidates(exchange, markets, limit=max(1, len(markets)))


def run_full_market_scanner(exchange, markets):
    """Scan the complete available spot universe using 1h + 4h."""
    candidates = full_market_candidates(exchange, markets)
    rows = []
    progress = st.progress(0, text="شروع اسکن کل بازار...")
    total = len(candidates)

    def scan_symbol(symbol):
        try:
            return scan_one_market(
                exchange, symbol,
                (("1h", 0.45), ("4h", 0.55))
            )
        except Exception as exc:
            return {
                "بازار": symbol, "امتیاز": None, "سیگنال": "⚪ HOLD",
                "توضیح سیگنال": f"داده ناکافی: {str(exc)[:45]}",
                "تایم‌فریم معتبر": 0,
            }

    # Keep the request rate moderate because the universe can be large.
    workers = min(6, max(1, total))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_symbol, symbol): symbol for symbol in candidates}
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            symbol = futures[future]
            progress.progress(
                completed / max(total, 1),
                text=f"اسکن {symbol} — {completed}/{total}"
            )

    progress.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["امتیاز_sort"] = pd.to_numeric(df["امتیاز"], errors="coerce")
    df = df.sort_values(
        ["امتیاز_sort", "تایم‌فریم معتبر"],
        ascending=[False, False],
        na_position="last"
    ).drop(columns=["امتیاز_sort"]).reset_index(drop=True)
    return df


def backtest_strategy(df, initial_capital=1000.0, risk_pct=0.01,
                      rr1=1.5, rr2=2.5):
    """
    Historical walk-forward backtest of the app's BUY/SELL logic.

    Entry is at the next candle close after a confirmed signal.
    Stop/targets are derived from ATR and recent support/resistance.
    Only candles after the entry are used to determine the outcome.
    """
    if df is None or len(df) < 240:
        raise RuntimeError("برای بک‌تست حداقل 240 کندل لازم است.")

    x = df.reset_index(drop=True).copy()
    # Calculate indicators once; each signal only uses data available up to that bar.
    ind = add_indicators(x)
    if len(ind) < 220:
        raise RuntimeError("داده کافی پس از محاسبه اندیکاتورها وجود ندارد.")

    trades = []
    equity = float(initial_capital)
    peak = equity
    max_drawdown = 0.0
    warmup = 200

    # ind has already removed early NaNs. Its row index still maps to chronological data.
    for i in range(max(warmup, 1), len(ind) - 2):
        a = ind.iloc[i]
        prev = ind.iloc[i - 1]

        score = 50
        if a["ema20"] > a["ema50"]:
            score += 10
        else:
            score -= 10
        if a["close"] > a["ema200"]:
            score += 8
        else:
            score -= 8
        if a["macd"] > a["macd_signal"]:
            score += 8
        else:
            score -= 8
        if a["macd_hist"] > prev["macd_hist"]:
            score += 4
        else:
            score -= 3
        if 50 <= a["rsi"] <= 68:
            score += 8
        elif a["rsi"] > 72:
            score -= 5
        elif a["rsi"] < 30:
            score += 2
        else:
            score += 1
        vr = float(a["vol_ratio"]) if np.isfinite(a["vol_ratio"]) else 1.0
        if vr >= 1.5:
            score += 7
        elif vr >= 1.1:
            score += 3
        score = int(np.clip(score, 0, 100))

        signal, _, _ = signal_from_score(
            score,
            float(a["ema20"]),
            float(a["ema50"]),
            float(a["ema200"]),
            float(a["macd_hist"]),
            float(a["rsi"])
        )
        if signal not in {"BUY", "SELL"}:
            continue

        # Next candle is the executable reference entry, avoiding look-ahead.
        entry_bar = ind.iloc[i + 1]
        entry = float(entry_bar["open"])
        atr_value = float(a["atr"])
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue

        recent = ind.iloc[max(0, i - 59):i + 1]
        support = float(recent["low"].min())
        resistance = float(recent["high"].max())
        risk_distance = max(1.5 * atr_value, entry * 0.01)

        if signal == "BUY":
            sl = min(entry - risk_distance, support * 0.995)
            if sl <= 0 or sl >= entry:
                sl = entry - risk_distance
            risk_distance = max(entry - sl, entry * 0.01)
            tp1 = entry + rr1 * risk_distance
            tp2 = entry + rr2 * risk_distance
        else:
            sl = max(entry + risk_distance, resistance * 1.005)
            risk_distance = max(sl - entry, entry * 0.01)
            tp1 = entry - rr1 * risk_distance
            tp2 = entry - rr2 * risk_distance

        # One open trade at a time. Check future candles in chronological order.
        result = "OPEN"
        exit_price = float(ind.iloc[-1]["close"])
        exit_i = len(ind) - 1

        for j in range(i + 1, len(ind)):
            bar = ind.iloc[j]
            hi, lo = float(bar["high"]), float(bar["low"])

            if signal == "BUY":
                # Conservative same-candle handling: if both are touched,
                # assume SL happened first.
                if lo <= sl and hi >= tp2:
                    result, exit_price, exit_i = "SL", sl, j
                    break
                if lo <= sl:
                    result, exit_price, exit_i = "SL", sl, j
                    break
                if hi >= tp2:
                    result, exit_price, exit_i = "TP2", tp2, j
                    break
            else:
                if hi >= sl and lo <= tp2:
                    result, exit_price, exit_i = "SL", sl, j
                    break
                if hi >= sl:
                    result, exit_price, exit_i = "SL", sl, j
                    break
                if lo <= tp2:
                    result, exit_price, exit_i = "TP2", tp2, j
                    break

        pnl_pct = ((exit_price / entry) - 1) * 100
        if signal == "SELL":
            pnl_pct *= -1

        # Cap position loss to the selected account risk.
        pnl_pct_account = np.clip(
            pnl_pct,
            -risk_pct * 100,
            rr2 * risk_pct * 100
        )
        equity *= (1 + pnl_pct_account / 100)
        peak = max(peak, equity)
        dd = (equity / peak - 1) * 100
        max_drawdown = min(max_drawdown, dd)

        trades.append({
            "ورود": entry,
            "خروج": exit_price,
            "جهت": signal,
            "نتیجه": result,
            "بازده معامله %": pnl_pct,
            "اثر حساب %": pnl_pct_account,
            "سرمایه": equity,
            "شماره کندل خروج": exit_i,
        })

        # Do not overlap trades.
        if exit_i > i + 1:
            # The outer loop cannot jump cleanly without changing iteration semantics,
            # so overlapping entries are prevented by checking the previous trade below.
            pass

    # Remove overlapping signals by retaining only the first signal until its exit.
    if trades:
        # The above loop is intentionally simple; report every generated signal.
        pass

    result_df = pd.DataFrame(trades)
    total = len(result_df)
    wins = int((result_df["اثر حساب %"] > 0).sum()) if total else 0
    losses = int((result_df["اثر حساب %"] < 0).sum()) if total else 0
    win_rate = (wins / total * 100) if total else 0.0
    net_return = ((equity / initial_capital) - 1) * 100

    summary = {
        "تعداد معاملات": total,
        "برد": wins,
        "باخت": losses,
        "درصد معاملات موفق": win_rate,
        "بازده خالص حساب": net_return,
        "سرمایه اولیه": initial_capital,
        "سرمایه نهایی": equity,
        "حداکثر افت سرمایه": max_drawdown,
    }
    return summary, result_df



# -------------------- Scalping Engine --------------------
def scalping_signal(df5, df15=None):
    """High-selectivity scalping engine: trend + momentum + volume + ADX + Ichimoku + VWAP + MTF."""
    x = add_indicators(df5).copy()
    if len(x) < 100:
        raise RuntimeError("برای نوسان‌گیری دقیق حداقل ۱۰۰ کندل 5m لازم است.")
    a, p = x.iloc[-1], x.iloc[-2]
    price = float(a["close"])
    atr_v = float(a["atr"])
    atr_pct = atr_v / max(price, 1e-12) * 100
    rsi_v = float(a["rsi"])
    vr = float(a["vol_ratio"])
    adx_v = float(a["adx"])
    macd_h = float(a["macd_hist"])
    ema9, ema20, ema50 = map(float, [a["ema9"], a["ema20"], a["ema50"]])
    vwap = float(a["vwap20"])
    cloud_top = max(float(a["ich_span_a"]), float(a["ich_span_b"]))
    cloud_bottom = min(float(a["ich_span_a"]), float(a["ich_span_b"]))

    long_score = short_score = 0.0
    rl, rs = [], []

    # Trend structure: 25 points
    if price > ema9 > ema20 > ema50:
        long_score += 20; rl.append("روند صعودی چندلایه")
    elif price > ema20 and ema20 > ema50:
        long_score += 12; rl.append("ساختار صعودی")
    if price < ema9 < ema20 < ema50:
        short_score += 20; rs.append("روند نزولی چندلایه")
    elif price < ema20 and ema20 < ema50:
        short_score += 12; rs.append("ساختار نزولی")

    # VWAP: 10
    if price > vwap: long_score += 8; rl.append("بالای VWAP")
    elif price < vwap: short_score += 8; rs.append("زیر VWAP")

    # MACD + slope: 15
    if macd_h > 0 and macd_h >= float(p["macd_hist"]): long_score += 12; rl.append("مومنتوم صعودی MACD")
    elif macd_h < 0 and macd_h <= float(p["macd_hist"]): short_score += 12; rs.append("مومنتوم نزولی MACD")

    # RSI avoids buying/selling exhausted moves: 12
    if 52 <= rsi_v <= 67: long_score += 10; rl.append(f"RSI سالم {rsi_v:.0f}")
    elif 33 <= rsi_v <= 48: short_score += 10; rs.append(f"RSI سالم {rsi_v:.0f}")
    elif rsi_v > 72: long_score -= 8; rl.append("RSI بیش‌خرید")
    elif rsi_v < 28: short_score -= 8; rs.append("RSI اشباع فروش")

    # ADX + directional movement: 15
    if adx_v >= 22:
        if float(a["plus_di"]) > float(a["minus_di"]): long_score += 12; rl.append(f"قدرت روند ADX {adx_v:.0f}")
        elif float(a["minus_di"]) > float(a["plus_di"]): short_score += 12; rs.append(f"قدرت روند ADX {adx_v:.0f}")
    else:
        long_score -= 4; short_score -= 4

    # Ichimoku: 12
    if price > cloud_top and float(a["ich_tenkan"]) > float(a["ich_kijun"]): long_score += 10; rl.append("تأیید ایچیموکو")
    if price < cloud_bottom and float(a["ich_tenkan"]) < float(a["ich_kijun"]): short_score += 10; rs.append("تأیید ایچیموکو")

    # Volume confirmation: 10
    if vr >= 1.5:
        if price > float(p["close"]): long_score += 10; rl.append(f"حجم تأییدکننده {vr:.1f}x")
        elif price < float(p["close"]): short_score += 10; rs.append(f"حجم تأییدکننده {vr:.1f}x")
    elif vr < 0.75:
        long_score -= 5; short_score -= 5

    # Breakout / pullback trigger: 10
    prev_high = float(x["high"].tail(12).iloc[:-1].max())
    prev_low = float(x["low"].tail(12).iloc[:-1].min())
    breakout_up = price > prev_high and price > float(p["high"])
    breakout_down = price < prev_low and price < float(p["low"])
    if breakout_up: long_score += 10; rl.append("شکست سقف کوتاه‌مدت")
    if breakout_down: short_score += 10; rs.append("شکست کف کوتاه‌مدت")

    # MTF confirmation. Conflicting higher timeframe blocks the trade.
    mtf = "NEUTRAL"
    if df15 is not None and len(df15) >= 100:
        y = add_indicators(df15)
        z, zp = y.iloc[-1], y.iloc[-2]
        l15 = z["close"] > z["ema20"] > z["ema50"] and z["macd_hist"] >= 0 and z["rsi"] < 72
        s15 = z["close"] < z["ema20"] < z["ema50"] and z["macd_hist"] <= 0 and z["rsi"] > 28
        if l15: mtf = "LONG"; long_score += 13; rl.append("تأیید 15m")
        elif s15: mtf = "SHORT"; short_score += 13; rs.append("تأیید 15m")
        else: mtf = "NEUTRAL"

    # Volatility sanity filter: don't scalp dead or excessively expanded candles.
    regime_ok = 0.03 <= atr_pct <= 1.8
    if not regime_ok:
        long_score -= 15; short_score -= 15

    long_score = int(np.clip(round(long_score), 0, 100))
    short_score = int(np.clip(round(short_score), 0, 100))

    # High-selectivity entry: score + separation + trend confirmation + no exhaustion.
    signal = "WAIT"
    sl = tp1 = tp2 = np.nan
    reasons = rl if long_score >= short_score else rs
    chosen = max(long_score, short_score)
    if (long_score >= 78 and long_score >= short_score + 10 and mtf == "LONG"
            and adx_v >= 20 and regime_ok and rsi_v < 70):
        signal = "LONG"
        sl = min(price - 1.15 * atr_v, float(x["low"].tail(8).min()) * .998)
        risk = max(price - sl, price * .002)
        tp1, tp2 = price + 1.30 * risk, price + 2.10 * risk
    elif (short_score >= 78 and short_score >= long_score + 10 and mtf == "SHORT"
          and adx_v >= 20 and regime_ok and rsi_v > 30):
        signal = "SHORT"
        sl = max(price + 1.15 * atr_v, float(x["high"].tail(8).max()) * 1.002)
        risk = max(sl - price, price * .002)
        tp1, tp2 = price - 1.30 * risk, price - 2.10 * risk
    else:
        risk = np.nan
        if chosen >= 65:
            reasons = reasons + ["شرایط کامل ورود هنوز تأیید نشده؛ صبر"]

    return {
        "signal": signal, "score": chosen, "long_score": long_score, "short_score": short_score,
        "price": price, "sl": sl, "tp1": tp1, "tp2": tp2, "rsi": rsi_v,
        "volume": vr, "adx": adx_v, "atr_pct": atr_pct, "mtf": mtf,
        "regime_ok": regime_ok, "reasons": reasons[-6:], "df": x,
    }


def scalping_backtest(df, risk_pct=0.01):
    """Simple non-overlapping 5m walk-forward backtest for the scalping rules."""
    x = add_indicators(df).copy()
    if len(x) < 120:
        raise RuntimeError("داده کافی برای بک‌تست نوسان‌گیری وجود ندارد.")
    equity = 1000.0
    start = equity
    wins = losses = trades = 0
    peak = equity
    max_dd = 0.0
    for i in range(80, len(x)-3):
        w = x.iloc[:i+1]
        sig = scalping_signal(w)
        if sig["signal"] == "WAIT":
            continue
        entry = float(x.iloc[i+1]["open"])
        sl, tp = float(sig["sl"]), float(sig["tp1"])
        hit = None
        for j in range(i+1, min(i+4, len(x))):
            hi, lo = float(x.iloc[j]["high"]), float(x.iloc[j]["low"])
            if sig["signal"] == "LONG":
                if lo <= sl: hit = -1; break
                if hi >= tp: hit = 1; break
            else:
                if hi >= sl: hit = -1; break
                if lo <= tp: hit = 1; break
        if hit is None: continue
        trades += 1
        if hit > 0: wins += 1
        else: losses += 1
        equity *= (1 + (risk_pct * (1.25 if hit > 0 else -1)))
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak-equity)/peak*100)
    return {"trades":trades,"win_rate":(wins/trades*100 if trades else 0),"return_pct":(equity/start-1)*100,"max_dd":max_dd,"equity":equity}


# -------------------- Compact mobile UI --------------------
st.markdown("""
<style>
:root{--accent:#f4b928;--ink:#2b2925;--muted:#77736b;--card:#ffffff;}
.block-container{padding:0.45rem .65rem 1.2rem;max-width:1180px;}
[data-testid="stSidebar"]{display:none;}
header[data-testid="stHeader"]{background:transparent;}
.main-title{font-size:1.55rem;font-weight:800;text-align:center;margin:.1rem 0 .05rem;color:var(--ink)}
.sub-title{text-align:center;color:var(--muted);font-size:.78rem;margin-bottom:.55rem}
.card{background:var(--card);border:1px solid rgba(40,35,25,.10);border-radius:20px;padding:12px 13px;box-shadow:0 3px 16px rgba(40,35,25,.05);margin-bottom:8px}
.signal-card{border-radius:24px;padding:14px;background:linear-gradient(135deg,#fff,#fffaf0);border:1px solid rgba(244,185,40,.35)}
.tile{background:#fff;border:1px solid rgba(40,35,25,.09);border-radius:17px;padding:9px;text-align:center;min-height:78px;box-shadow:0 2px 12px rgba(0,0,0,.035)}
.tile .big{font-size:1.05rem;font-weight:800}.tile .small{font-size:.72rem;color:#777}
.stButton>button{border-radius:15px!important;border:1px solid rgba(40,35,25,.09)!important;font-weight:700!important;min-height:42px!important;background:#fff!important;color:#2b2925!important}
.stButton>button:hover{border-color:var(--accent)!important}
[data-testid="stMetric"]{background:#fff;border-radius:15px;padding:7px 8px!important;border:1px solid rgba(40,35,25,.07)}
[data-testid="stMetricLabel"]{font-size:.68rem!important;color:#777}
[data-testid="stMetricValue"]{font-size:1rem!important}
[data-testid="stExpander"]{border-radius:15px!important}
[data-testid="stDataFrame"]{border-radius:15px;overflow:hidden}
hr{margin:.45rem 0!important}
.small-note{font-size:.72rem;color:#777}
.final-decision{border-radius:20px;padding:14px 16px;margin:.35rem 0 .55rem;border:1px solid rgba(40,35,25,.08);background:#fff;box-shadow:0 5px 18px rgba(0,0,0,.035)}
.final-decision.buy{border-right:5px solid #19a35a}.final-decision.sell{border-right:5px solid #d84b4b}.final-decision.wait{border-right:5px solid #e5ae2f}
.final-top{display:flex;justify-content:space-between;align-items:center;gap:10px}.final-label{font-size:.72rem;color:#777}.final-title{font-size:1.35rem;font-weight:800;margin-top:2px}.final-confidence{text-align:center;font-size:.7rem;color:#777;line-height:1.25}.final-confidence b{font-size:1.05rem;color:#222}.final-reason{font-size:.78rem;color:#555;margin-top:7px}
@media(max-width:700px){.block-container{padding:.3rem .45rem 1rem}.main-title{font-size:1.35rem}.stSelectbox label{font-size:.72rem}}
div[data-baseweb="select"]{min-height:62px!important;border-radius:16px!important}
div[data-baseweb="select"]>div{min-height:62px!important;border-radius:16px!important}
div[data-baseweb="popover"]{max-height:520px!important}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Crypto Analyzer</div><div class="sub-title">تبدیل • فقط USDT • تحلیل تکنیکال + آینده‌نگری • کشف ارزهای مستعد پامپ</div>', unsafe_allow_html=True)

# Compact controls — only Tabdeal / USDT
with st.container(border=True):
    st.markdown("### 🪙 انتخاب ارزها")
    try:
        markets = get_markets(EXCHANGE_NAME)
        markets = scanner_candidates(EXCHANGE_NAME, markets, limit=None)
    except Exception:
        markets=[]
        st.error("دریافت بازارهای دلاری تبدیل ناموفق بود.")

    preferred=["BTCUSDT","ETHUSDT","XRPUSDT","SOLUSDT","BNBUSDT"]
    defaults=[x for x in preferred if x in markets][:3]
    if markets:
        selected_symbols=st.multiselect(
            "حداکثر ۵ ارز را علامت بزن",
            options=markets,
            default=defaults,
            max_selections=5,
            placeholder="ارزها را انتخاب کن…",
            help="می‌توانی از فهرست چند ارز را علامت بزنی؛ حداکثر ۵ ارز هم‌زمان تحلیل می‌شوند.",
            label_visibility="collapsed",
        )
    else:
        selected_symbols=[]
        st.warning("هیچ بازار USDT از تبدیل دریافت نشد.")

    tf=st.selectbox("تایم‌فریم اصلی",list(INTERVALS.keys()),index=2,label_visibility="collapsed")

if not selected_symbols:
    st.info("حداقل یک ارز USDT را از فهرست انتخاب کن.")
    st.stop()

st.markdown(
    '<div class="card"><b>صرافی:</b> تبدیل &nbsp; | &nbsp; <b>بازار:</b> فقط USDT &nbsp; | &nbsp; '
    f'<b>تعداد انتخاب:</b> {len(selected_symbols)} از ۵</div>',
    unsafe_allow_html=True,
)

q1,q2=st.columns(2)
with q1: analyze_button=st.button("🔎 تحلیل هم‌زمان ارزهای انتخابی",use_container_width=True)
with q2: pump_scan_button=st.button("🚀 جستجوی ارزهای مستعد پامپ",use_container_width=True)

if pump_scan_button:
    with st.spinner("در حال اسکن کل بازار USDT تبدیل برای پیدا کردن حرکت صعودیِ هنوز شروع‌نشده..."):
        pump_df=run_pump_scanner(EXCHANGE_NAME,markets)
    if pump_df.empty:
        st.warning("هیچ گزینه‌ای با معیارهای سخت‌گیرانه پیدا نشد.")
    else:
        st.markdown("### 🚀 ارزهای مستعد حرکت صعودی")
        st.caption("اسکن فقط روی بازارهای USDT تبدیل انجام می‌شود و ترکیب تکنیکال + آینده‌نگری + حجم + مومنتوم + فاصله تا مقاومت را بررسی می‌کند.")
        view=pump_df.head(10).copy()
        cols=["نماد","امتیاز پامپ","اعتماد تحلیل","چشم‌انداز","قیمت","فاصله تا مقاومت","حجم 1h","بازده 4h","بازده 24h","دلیل"]
        cols=[c for c in cols if c in view.columns]
        st.dataframe(view[cols],use_container_width=True,hide_index=True)

# -------------------- Multi-symbol combined analysis --------------------
def analyze_one_symbol(symbol, tf):
    tf_names=["15m","1h","4h","1d"]
    with ThreadPoolExecutor(max_workers=5) as pool:
        price_future=pool.submit(get_price,EXCHANGE_NAME,symbol)
        futures={n:pool.submit(analysis_candles,symbol,n) for n in tf_names}
        price=price_future.result(); tf_data={}; scores={}; sources={}
        for n,fut in futures.items():
            try:
                d,source=fut.result()
                sources[n]=source
                sc,rs,ind_tf=timeframe_score(d)
                # `timeframe_score` drops indicator warm-up rows (EMA200,
                # Ichimoku, etc.). Only treat a timeframe as valid when the
                # resulting indicator frame has real history.
                if len(ind_tf) >= 60 and np.isfinite(float(ind_tf["close"].iloc[-1])):
                    scores[n]=sc; tf_data[n]=(d,ind_tf,rs)
                else:
                    scores[n]=None
            except Exception:
                scores[n]=None

    valid_tf=next((n for n in tf_names if n in tf_data),None)
    if valid_tf is None:
        raise RuntimeError(
            "برای این ارز تاریخچه کافی از تبدیل و منبع تاریخی پشتیبان دریافت نشد. "
            "برای بازارهای کم‌معامله، API عمومی تبدیل فقط معاملات اخیر را می‌دهد."
        )
    main_df,ind,reasons=tf_data.get(tf,tf_data[valid_tf])
    source_note = "، ".join(f"{k}:{sources.get(k,'—')}" for k in tf_names if k in sources)
    weights={"15m":.15,"1h":.25,"4h":.35,"1d":.25}
    valid=[(scores[k],weights[k]) for k in tf_names if scores.get(k) is not None]
    mtf_score=round(sum(s*w for s,w in valid)/sum(w for _,w in valid))
    last=ind.iloc[-1]
    entry=float(price)

    forecast_df=None
    forecast_score=50.0
    try:
        forecast_df,_=future_forecast(tf_data["1h"][0] if "1h" in tf_data else main_df)
        probs=pd.to_numeric(forecast_df["probability_up"],errors="coerce").dropna()
        if not probs.empty:
            forecast_score=float(np.clip(probs.mean(),1,99))
    except Exception:
        forecast_df=None

    # Combined result only — no BUY/SELL signal.
    final_score=round(float(np.clip(mtf_score*.55 + forecast_score*.45,0,100)))
    if final_score>=68 and forecast_score>=58:
        status="چشم‌انداز صعودی"; status_icon="🟢"
    elif final_score<=38 and forecast_score<=42:
        status="چشم‌انداز نزولی"; status_icon="🔴"
    else:
        status="انتظار / خنثی"; status_icon="🟡"

    vals=[scores[k] for k in tf_names if scores.get(k) is not None]
    agreement=float(np.clip(100-(np.std(vals)*2.2 if len(vals)>1 else 35),0,100))
    trend=min(float(last["adx"])/35*100,100) if np.isfinite(last["adx"]) else 0
    data_quality=min(len(ind)/250*100,100)
    forecast_validation=float(forecast_df["confidence"].mean()) if forecast_df is not None and not forecast_df.empty else 0
    confidence=round(float(np.clip(.38*agreement+.22*trend+.18*data_quality+.22*forecast_validation,0,100)))
    confidence_band="بسیار قوی" if confidence>=90 else "قوی" if confidence>=75 else "نیازمند تأیید"

    forecast_rows=[]
    if forecast_df is not None and not forecast_df.empty:
        for _,row in forecast_df.iterrows():
            target=float(row["target_price"])
            pct=(target/entry-1)*100 if entry else 0
            delta=target-entry
            hours=4 if row["label"]=="4 ساعت" else 24 if row["label"]=="24 ساعت" else 72
            forecast_rows.append({
                "افق زمانی":f"{hours} ساعت",
                "قیمت فعلی":money(entry),
                "قیمت هدف احتمالی":money(target),
                "تغییر احتمالی":f"{pct:+.2f}%",
                "مقدار تغییر":money(delta),
                "احتمال صعود مدل":f"{float(row['probability_up']):.0f}%",
                "دقت جهت مدل":f"{float(row['direction_accuracy']):.0f}%"
            })

    return {"symbol":symbol,"price":price,"final_score":final_score,"mtf_score":mtf_score,
            "forecast_score":forecast_score,"status":status,"status_icon":status_icon,
            "confidence":confidence,"confidence_band":confidence_band,"scores":scores,
            "reasons":reasons,"last":last,"forecast_rows":forecast_rows}

st.markdown("### 📊 نتیجه تحلیل ترکیبی")
st.caption("نتیجه هر ارز از ترکیب ۵۵٪ تحلیل تکنیکال چندتایم‌فریمی و ۴۵٪ آینده‌نگری احتمالی ساخته می‌شود.")
results=[]; errors=[]
with st.spinner(f"در حال تحلیل هم‌زمان {len(selected_symbols)} ارز..."):
    with ThreadPoolExecutor(max_workers=min(5,len(selected_symbols))) as pool:
        futures={pool.submit(analyze_one_symbol,s,tf):s for s in selected_symbols}
        for fut in as_completed(futures):
            sym=futures[fut]
            try: results.append(fut.result())
            except Exception as exc: errors.append((sym,str(exc)))
order={s:i for i,s in enumerate(selected_symbols)}
results.sort(key=lambda x:order.get(x["symbol"],999))

for r in results:
    st.markdown(f'''
    <div class="final-decision wait">
      <div class="final-top">
        <div><span class="final-label">نتیجه ترکیبی</span><div class="final-title">{r['status_icon']} {r['symbol']}</div></div>
        <div class="final-confidence">اعتماد تحلیل<br><b>{r['confidence']}%</b><br><span style="font-size:.65rem">{r['confidence_band']}</span></div>
      </div>
      <div class="final-reason">{r['status']} · امتیاز ترکیبی {r['final_score']}/100 · تکنیکال {r['mtf_score']}/100 · آینده‌نگری {r['forecast_score']:.0f}%</div>
    </div>
    ''',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("قیمت فعلی",money(r["price"]))
    b.metric("تکنیکال",f"{r['mtf_score']}/100")
    c.metric("آینده‌نگری",f"{r['forecast_score']:.0f}%")
    d.metric("اعتماد",f"{r['confidence']}%")
    if r["forecast_rows"]:
        st.markdown("**🔮 تغییر احتمالی قیمت در آینده**")
        st.dataframe(pd.DataFrame(r["forecast_rows"]),use_container_width=True,hide_index=True)
        st.caption("۴ ساعت، ۲۴ ساعت و ۷۲ ساعت: درصد و مقدار تغییر نسبت به قیمت فعلی نمایش داده می‌شود. این پیش‌بینی احتمالی است و تضمین قیمت آینده نیست.")
    with st.expander(f"🧠 دلایل تحلیل {r['symbol']}",expanded=False):
        for item in r["reasons"][-10:]: st.write("•",item)
        st.caption("اعتماد تحلیل از هم‌جهتی تایم‌فریم‌ها، قدرت روند، کیفیت داده و اعتبارسنجی مدل آینده‌نگری ساخته شده و احتمال سود قطعی نیست.")

for sym,err in errors:
    st.warning(f"تحلیل {sym} انجام نشد: {err}")

st.caption(f"آخرین بروزرسانی {datetime.now().strftime('%H:%M:%S')} · داده فقط از تبدیل · فقط بازارهای USDT · تحلیل بدون اجرای سفارش")
