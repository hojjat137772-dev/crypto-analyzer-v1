
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# ============================================================
# Crypto Analyzer Pro - Multi Exchange / Single File
# Binance + Wallex + Nobitex + Tabdeal
# Analysis only - no order placement
# ============================================================

st.set_page_config(
    page_title="Crypto Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
]
WALLEX_BASE = "https://api.wallex.ir"
NOBITEX_BASE = "https://api.nobitex.ir"
TABDEAL_BASE = "https://api1.tabdeal.org"

EXCHANGES = ["Binance", "والکس", "نوبیتکس", "تبدیل"]

INTERVALS = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

st.markdown("""
<style>
.block-container {padding-top: 1rem; max-width: 1450px;}
h1,h2,h3 {text-align:right;}
div[data-testid="stMetric"] {direction:rtl;}
.market-card {
    padding: 12px 16px; border-radius: 14px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 ربات تحلیل حرفه‌ای کریپتو")
st.caption(
    "اتصال داده بازار: Binance، والکس، نوبیتکس و تبدیل | "
    "تحلیل چندتایم‌فریمی | بدون ثبت سفارش"
)

# -------------------- Generic HTTP --------------------

def http_get(url, params=None, timeout=15):
    r = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "CryptoAnalyzerPro/2.0"},
    )
    r.raise_for_status()
    return r.json()

def unix_now():
    return int(time.time())

def resolution_seconds(tf):
    return {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[tf]

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# -------------------- Market lists --------------------

@st.cache_data(ttl=180, show_spinner=False)
def binance_markets():
    last = None
    for base in BINANCE_ENDPOINTS:
        try:
            data = http_get(f"{base}/api/v3/exchangeInfo", timeout=15)
            out = []
            for s in data.get("symbols", []):
                if s.get("status") != "TRADING":
                    continue
                sym = str(s.get("symbol", "")).upper()
                if sym.endswith("USDT"):
                    out.append(sym)
            if out:
                return sorted(set(out))
        except Exception as exc:
            last = exc
    raise RuntimeError("لیست بازارهای Binance دریافت نشد.") from last

@st.cache_data(ttl=180, show_spinner=False)
def wallex_markets():
    urls = [
        f"{WALLEX_BASE}/v1/markets",
        f"{WALLEX_BASE}/hector/web/v1/markets",
    ]
    last = None
    for url in urls:
        try:
            data = http_get(url, timeout=15)
            result = data.get("result", {})
            symbols = result.get("symbols", {})
            if isinstance(symbols, dict):
                out = [
                    k.upper() for k, v in symbols.items()
                    if isinstance(v, dict) and v.get("symbol")
                ]
                if out:
                    return sorted(set(out))
            markets = result.get("markets", [])
            if isinstance(markets, list):
                out = [
                    str(x.get("symbol", "")).upper()
                    for x in markets
                    if isinstance(x, dict) and x.get("symbol")
                ]
                if out:
                    return sorted(set(out))
        except Exception as exc:
            last = exc
    raise RuntimeError("لیست بازارهای والکس دریافت نشد.") from last

@st.cache_data(ttl=180, show_spinner=False)
def nobitex_markets():
    data = http_get(f"{NOBITEX_BASE}/market/stats", timeout=15)
    stats = data.get("stats", {})
    out = []
    for key in stats.keys():
        k = str(key).upper().replace("-", "")
        if k and k not in {"GLOBAL"}:
            out.append(k)
    return sorted(set(out))

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
        if 5 <= len(s) <= 24 and s.isalnum()
    ]
    if not symbols:
        raise RuntimeError("لیست بازارهای تبدیل دریافت نشد.")
    return sorted(set(symbols))

@st.cache_data(ttl=180, show_spinner=False)
def get_markets(exchange):
    if exchange == "Binance":
        return binance_markets()
    if exchange == "والکس":
        return wallex_markets()
    if exchange == "نوبیتکس":
        return nobitex_markets()
    if exchange == "تبدیل":
        return tabdeal_markets()
    return []

# -------------------- Binance candles --------------------

@st.cache_data(ttl=45, show_spinner=False)
def binance_klines(symbol, interval, limit=300):
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    last = None
    data = None
    for base in BINANCE_ENDPOINTS:
        try:
            candidate = http_get(f"{base}/api/v3/klines", params, 15)
            if isinstance(candidate, list) and len(candidate) >= 80:
                data = candidate
                break
        except Exception as exc:
            last = exc
    if data is None:
        raise RuntimeError("داده کندلی Binance در دسترس نیست.") from last

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time","open","high","low","close","volume","quote_volume"]].dropna()

@st.cache_data(ttl=30, show_spinner=False)
def binance_price(symbol):
    last = None
    for base in BINANCE_ENDPOINTS:
        try:
            d = http_get(
                f"{base}/api/v3/ticker/price",
                {"symbol": symbol},
                10,
            )
            return float(d["price"])
        except Exception as exc:
            last = exc
    raise RuntimeError("قیمت Binance دریافت نشد.") from last

# -------------------- UDF adapters: Wallex / Nobitex --------------------

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

@st.cache_data(ttl=45, show_spinner=False)
def wallex_klines(symbol, interval, limit=300):
    seconds = resolution_seconds(interval)
    end = unix_now()
    start = end - seconds * (limit + 50)
    resolution = {"15m": 15, "1h": 60, "4h": 240, "1d": "D"}[interval]
    data = http_get(
        f"{WALLEX_BASE}/v1/udf/history",
        {
            "symbol": symbol,
            "resolution": resolution,
            "from": start,
            "to": end,
        },
        15,
    )
    return udf_to_df(data).tail(limit)

@st.cache_data(ttl=30, show_spinner=False)
def wallex_price(symbol):
    data = http_get(f"{WALLEX_BASE}/v1/markets", timeout=15)
    symbols = data.get("result", {}).get("symbols", {})
    item = symbols.get(symbol)
    if not item:
        raise RuntimeError("بازار والکس پیدا نشد.")
    p = item.get("stats", {}).get("lastPrice")
    if p is None:
        raise RuntimeError("قیمت والکس دریافت نشد.")
    return float(p)

@st.cache_data(ttl=45, show_spinner=False)
def nobitex_klines(symbol, interval, limit=300):
    seconds = resolution_seconds(interval)
    end = unix_now()
    start = end - seconds * (limit + 50)
    resolution = {"15m": 15, "1h": 60, "4h": 240, "1d": "D"}[interval]
    data = http_get(
        f"{NOBITEX_BASE}/market/udf/history",
        {
            "symbol": symbol,
            "resolution": resolution,
            "from": start,
            "to": end,
        },
        15,
    )
    return udf_to_df(data).tail(limit)

@st.cache_data(ttl=30, show_spinner=False)
def nobitex_price(symbol):
    compact = symbol.upper().replace("-", "")
    # Try common destination pairs first.
    if compact.endswith("IRT"):
        src, dst = compact[:-3], "rls"
    elif compact.endswith("USDT"):
        src, dst = compact[:-4], "usdt"
    else:
        src, dst = compact, "usdt"
    data = http_get(
        f"{NOBITEX_BASE}/market/stats",
        {"srcCurrency": src.lower(), "dstCurrency": dst},
        15,
    )
    stats = data.get("stats", {})
    key = f"{src.lower()}-{dst}"
    item = stats.get(key)
    if not item:
        raise RuntimeError("بازار نوبیتکس پیدا نشد.")
    p = item.get("latest")
    if p is None:
        raise RuntimeError("قیمت نوبیتکس دریافت نشد.")
    return float(p)

# -------------------- Tabdeal --------------------

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
        ts = x.get("time", x.get("timestamp"))
        ts = safe_float(ts)
        if np.isfinite(price) and np.isfinite(qty) and np.isfinite(ts):
            if ts > 10_000_000_000:
                ts /= 1000
            out.append((pd.to_datetime(ts, unit="s"), price, qty))
    if len(out) < 80:
        raise RuntimeError("معاملات کافی از تبدیل دریافت نشد.")
    return out

def tabdeal_pair_name(symbol):
    """Convert compact symbol (BTCUSDT/BTCIRT) to Tabdeal's underscore form."""
    s = str(symbol).upper().replace("-", "").replace("_", "")
    for quote in ["USDT", "IRT", "RLS", "USDC", "BTC", "ETH"]:
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}_{quote}"
    return s


@st.cache_data(ttl=30, show_spinner=False)
def tabdeal_trades(symbol, limit=2000):
    # Tabdeal supports both compact `symbol` and underscore `tabdealSymbol`.
    # Some markets/endpoints reject the compact form, so try the documented
    # underscore form first and then fall back to compact form.
    tab_symbol = tabdeal_pair_name(symbol)
    attempts = [
        {"tabdealSymbol": tab_symbol, "limit": min(limit, 2000)},
        {"tabdealSymbol": tab_symbol},
        {"symbol": symbol, "limit": min(limit, 2000)},
        {"symbol": symbol},
    ]
    last = None
    for params in attempts:
        try:
            data = http_get(
                f"{TABDEAL_BASE}/r/api/v1/trades",
                params,
                15,
            )
            return parse_tabdeal_trade_rows(data)
        except Exception as exc:
            last = exc
    raise RuntimeError(
        f"داده معاملات تبدیل برای {symbol} دریافت نشد. "
        f"نماد Tabdeal: {tab_symbol}"
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
    if len(out) < 80:
        raise RuntimeError("تاریخچه معاملاتی تبدیل برای این تایم‌فریم کافی نیست.")
    return out.tail(300)

@st.cache_data(ttl=45, show_spinner=False)
def tabdeal_klines(symbol, interval):
    try:
        return trades_to_ohlcv(tabdeal_trades(symbol), interval)
    except Exception:
        # Tabdeal's public API does not expose a full historical kline feed
        # in the current public documentation. Use Binance OHLC only as a
        # technical-history fallback so the analyzer/scanner remains usable.
        return binance_klines(symbol, interval)

@st.cache_data(ttl=30, show_spinner=False)
def tabdeal_price(symbol):
    data = http_get(
        f"{TABDEAL_BASE}/r/api/v1/depth",
        {"symbol": symbol, "limit": 5},
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
    raise RuntimeError("قیمت تبدیل دریافت نشد.")

# -------------------- Exchange router --------------------

def get_price(exchange, symbol):
    if exchange == "Binance":
        return binance_price(symbol)
    if exchange == "والکس":
        return wallex_price(symbol)
    if exchange == "نوبیتکس":
        return nobitex_price(symbol)
    if exchange == "تبدیل":
        return tabdeal_price(symbol)
    raise RuntimeError("صرافی نامعتبر است.")

def get_candles(exchange, symbol, interval):
    if exchange == "Binance":
        return binance_klines(symbol, interval)
    if exchange == "والکس":
        return wallex_klines(symbol, interval)
    if exchange == "نوبیتکس":
        return nobitex_klines(symbol, interval)
    if exchange == "تبدیل":
        return tabdeal_klines(symbol, interval)
    raise RuntimeError("صرافی نامعتبر است.")

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
    return x.dropna().reset_index(drop=True)

def support_resistance(df, window=60):
    recent = df.tail(window)
    return float(recent["low"].min()), float(recent["high"].max())

def timeframe_score(df):
    x = add_indicators(df)
    if len(x) < 10:
        return 50, ["داده کافی نیست"], x
    a, p = x.iloc[-1], x.iloc[-2]
    score = 50
    reasons = []

    if a["ema20"] > a["ema50"]:
        score += 10; reasons.append("EMA20 بالاتر از EMA50 ✅")
    else:
        score -= 10; reasons.append("EMA20 پایین‌تر از EMA50 ⚠️")

    if a["close"] > a["ema200"]:
        score += 8; reasons.append("قیمت بالای EMA200 ✅")
    else:
        score -= 8; reasons.append("قیمت زیر EMA200 ⚠️")

    if a["macd"] > a["macd_signal"]:
        score += 8; reasons.append("MACD صعودی ✅")
    else:
        score -= 8; reasons.append("MACD نزولی ⚠️")

    if a["macd_hist"] > p["macd_hist"]:
        score += 4; reasons.append("مومنتوم MACD در حال تقویت ✅")
    else:
        score -= 3; reasons.append("مومنتوم MACD در حال تضعیف ⚠️")

    if 50 <= a["rsi"] <= 68:
        score += 8; reasons.append(f"RSI مناسب ({a['rsi']:.1f}) ✅")
    elif a["rsi"] > 72:
        score -= 5; reasons.append(f"RSI بیش‌خرید ({a['rsi']:.1f}) ⚠️")
    elif a["rsi"] < 30:
        score += 2; reasons.append(f"RSI اشباع فروش ({a['rsi']:.1f})")
    else:
        score += 1; reasons.append(f"RSI خنثی ({a['rsi']:.1f})")

    vr = float(a["vol_ratio"]) if np.isfinite(a["vol_ratio"]) else 1.0
    if vr >= 1.5:
        score += 7; reasons.append(f"حجم قوی ({vr:.1f}x) ✅")
    elif vr >= 1.1:
        score += 3; reasons.append(f"حجم مناسب ({vr:.1f}x)")
    else:
        reasons.append(f"حجم معمولی ({vr:.1f}x)")

    return int(np.clip(score, 0, 100)), reasons, x

def money(v):
    if not np.isfinite(v):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    return f"{v:.8f}"


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


def scanner_candidates(exchange, markets, limit=20):
    """Return a stable, reasonably sized universe for the scanner."""
    cleaned = []
    for m in markets:
        m = str(m).upper().replace("-", "").replace("_", "")
        base, quote = base_asset(m)
        if not base or base in STABLE_BASES:
            continue
        # For local exchanges prefer USDT/IRT markets; for Binance prefer USDT.
        if exchange == "Binance" and quote != "USDT":
            continue
        if exchange in {"والکس", "نوبیتکس", "تبدیل"} and quote not in {"USDT", "IRT", "RLS"}:
            continue
        cleaned.append(m)

    # Keep common liquid symbols first, then fill from the exchange list.
    preferred = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT",
        "DOTUSDT", "MATICUSDT", "LTCUSDT", "SHIBUSDT", "ATOMUSDT",
    ]
    ordered = [x for x in preferred if x in cleaned]
    ordered += [x for x in cleaned if x not in ordered]
    return ordered[:limit]


def scanner_action(score):
    if score is None:
        return "داده ناکافی"
    if score >= 80:
        return "🟢 شرایط صعودی قوی"
    if score >= 68:
        return "🟡 شرایط صعودی مشروط"
    if score <= 35:
        return "🔴 شرایط نزولی"
    return "⚪ خنثی"


def scan_one_market(exchange, symbol, scan_tfs):
    """Analyze one market without changing the main analyzer state."""
    scores = []
    tf_scores = {}
    latest_price = np.nan
    atr_value = np.nan
    support = np.nan
    resistance = np.nan

    for tf_name, weight in scan_tfs:
        try:
            df = get_candles(exchange, symbol, tf_name)
            score, _, ind = timeframe_score(df)
            if len(ind) < 10:
                continue
            scores.append((score, weight))
            tf_scores[tf_name] = score
            last = ind.iloc[-1]
            latest_price = float(last["close"])
            atr_value = float(last["atr"])
            support, resistance = support_resistance(ind, 60)
        except Exception:
            tf_scores[tf_name] = None

    if not scores:
        raise RuntimeError("داده تکنیکال کافی نیست")

    score = round(sum(s * w for s, w in scores) / sum(w for _, w in scores))
    try:
        latest_price = float(get_price(exchange, symbol))
    except Exception:
        pass

    if not np.isfinite(latest_price):
        raise RuntimeError("قیمت در دسترس نیست")

    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = latest_price * 0.02
    if not np.isfinite(support) or support <= 0:
        support = latest_price - 1.5 * atr_value
    if not np.isfinite(resistance) or resistance <= 0:
        resistance = latest_price + 2.0 * atr_value

    sl = max(support * 0.995, latest_price - 1.5 * atr_value)
    if sl >= latest_price:
        sl = latest_price * 0.97
    risk = max(latest_price - sl, latest_price * 0.01)
    tp1 = latest_price + 1.5 * risk
    tp2 = latest_price + 2.5 * risk
    if resistance > latest_price and resistance < tp1:
        tp1 = max(latest_price + risk, resistance * 0.995)

    return {
        "بازار": symbol,
        "امتیاز": score,
        "وضعیت": scanner_action(score),
        "قیمت": latest_price,
        "ورود مرجع": latest_price,
        "حدضرر": sl,
        "حدسود 1": tp1,
        "حدسود 2": tp2,
        "حمایت": support,
        "مقاومت": resistance,
        "15m": tf_scores.get("15m"),
        "1h": tf_scores.get("1h"),
        "4h": tf_scores.get("4h"),
        "1d": tf_scores.get("1d"),
    }


def run_scanner(exchange, markets, scan_tfs, limit):
    candidates = scanner_candidates(exchange, markets, limit)
    rows = []
    progress = st.progress(0, text="شروع اسکن بازار...")
    total = len(candidates)
    for i, symbol in enumerate(candidates, 1):
        try:
            rows.append(scan_one_market(exchange, symbol, scan_tfs))
        except Exception as exc:
            rows.append({
                "بازار": symbol,
                "امتیاز": None,
                "وضعیت": f"خطا: {str(exc)[:45]}",
            })
        progress.progress(i / max(total, 1), text=f"اسکن {symbol} — {i}/{total}")
    progress.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["امتیاز_sort"] = pd.to_numeric(df["امتیاز"], errors="coerce")
    return df.sort_values("امتیاز_sort", ascending=False, na_position="last").drop(columns=["امتیاز_sort"])

# -------------------- UI / Dropdowns --------------------

with st.sidebar:
    st.header("🏦 صرافی")
    exchange = st.selectbox(
        "منبع بازار",
        EXCHANGES,
        index=0,
    )

    try:
        markets = get_markets(exchange)
    except Exception as exc:
        markets = []
        st.error(f"دریافت بازارهای {exchange} ناموفق بود.")
        st.caption(str(exc))

    if markets:
        default_index = 0
        preferred = [
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT",
            "BTCIRT", "ETHIRT", "XRPIRT", "SOLIRT"
        ]
        for p in preferred:
            if p in markets:
                default_index = markets.index(p)
                break

        symbol = st.selectbox(
            "🪙 انتخاب ارز / بازار",
            markets,
            index=default_index,
            help="لیست به‌صورت زنده از صرافی انتخاب‌شده دریافت می‌شود."
        )
    else:
        symbol = st.text_input(
            "نماد بازار",
            value="BTCUSDT",
            placeholder="مثلاً BTCUSDT",
        ).strip().upper()

    tf = st.selectbox(
        "⏱ تایم‌فریم اصلی",
        list(INTERVALS.keys()),
        index=2,
    )

    analyze = st.button("🚀 تحلیل حرفه‌ای", use_container_width=True)

    st.divider()
    st.header("🔎 اسکن بازار")
    scan_count = st.slider("تعداد بازارها", min_value=5, max_value=30, value=15, step=5)
    scan_mode = st.selectbox(
        "نوع اسکن",
        ["سریع (1h + 4h)", "چندتایم‌فریمی کامل (15m + 1h + 4h + 1d)"],
        index=0,
    )
    scan_button = st.button("🔎 شروع اسکن", use_container_width=True)

    st.divider()
    st.caption(
        "🔒 فقط داده عمومی بازار استفاده می‌شود. "
        "این نسخه سفارش خرید/فروش ثبت نمی‌کند."
    )

if scan_button:
    if not markets:
        st.error("لیست بازارهای این صرافی در دسترس نیست؛ اسکن انجام نشد.")
    else:
        scan_tfs = [("1h", 0.45), ("4h", 0.55)]
        if scan_mode.startswith("چندتایم‌فریمی"):
            scan_tfs = [("15m", 0.15), ("1h", 0.25), ("4h", 0.35), ("1d", 0.25)]

        st.markdown("## 🔎 اسکن بازار")
        st.caption(
            f"صرافی: {exchange} | تعداد بازار: {scan_count} | "
            f"تایم‌فریم: {', '.join(tf for tf, _ in scan_tfs)}"
        )
        with st.spinner("در حال دریافت داده و محاسبه امتیاز بازارها..."):
            scan_df = run_scanner(exchange, markets, scan_tfs, scan_count)

        if scan_df.empty:
            st.warning("نتیجه‌ای برای اسکن به دست نیامد.")
        else:
            valid_scan = scan_df[pd.to_numeric(scan_df["امتیاز"], errors="coerce").notna()].copy()
            if not valid_scan.empty:
                st.success(f"اسکن تمام شد؛ {len(valid_scan)} بازار با داده معتبر بررسی شد.")

                # Highlight the three highest-scoring markets without claiming a trading recommendation.
                top = valid_scan.head(3)
                st.markdown("### 📌 ۳ بازار با بالاترین امتیاز محاسباتی")
                cols = st.columns(min(3, len(top)))
                for col, (_, row) in zip(cols, top.iterrows()):
                    with col:
                        st.markdown(f"**{row['بازار']}**")
                        st.metric("امتیاز", f"{int(row['امتیاز'])}/100")
                        st.write(row["وضعیت"])
                        if pd.notna(row.get("قیمت")):
                            st.caption(f"قیمت: {money(float(row['قیمت']))}")

            display_cols = [
                "بازار", "امتیاز", "وضعیت", "قیمت", "ورود مرجع",
                "حدضرر", "حدسود 1", "حدسود 2", "حمایت", "مقاومت",
            ]
            display_cols = [c for c in display_cols if c in scan_df.columns]
            view = scan_df[display_cols].copy()
            for c in ["قیمت", "ورود مرجع", "حدضرر", "حدسود 1", "حدسود 2", "حمایت", "مقاومت"]:
                if c in view.columns:
                    view[c] = view[c].apply(lambda x: money(float(x)) if pd.notna(x) else "-")
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.caption(
                "امتیاز اسکن بر پایه اندیکاتورهای تکنیکال و تایم‌فریم‌های انتخاب‌شده است؛ "
                "نتیجه اسکن به‌تنهایی تضمین‌کننده نتیجه معامله نیست."
            )

if not symbol:
    st.warning("یک بازار انتخاب کن.")
    st.stop()

# -------------------- Analysis --------------------

try:
    price = get_price(exchange, symbol)

    main_df = get_candles(exchange, symbol, tf)

    scores = {}
    tf_data = {}
    for name in ["15m", "1h", "4h", "1d"]:
        try:
            d = get_candles(exchange, symbol, name)
            sc, rs, ind = timeframe_score(d)
            scores[name] = sc
            tf_data[name] = (d, ind, rs)
        except Exception:
            scores[name] = None

    main_score, reasons, ind = timeframe_score(main_df)
    if len(ind) < 10:
        raise RuntimeError("داده کافی برای تحلیل تکنیکال وجود ندارد.")

    last = ind.iloc[-1]

    weights = {"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}
    valid = [(scores[k], weights[k]) for k in weights if scores.get(k) is not None]
    mtf_score = round(sum(s*w for s, w in valid) / sum(w for s, w in valid)) if valid else main_score

    support, resistance = support_resistance(ind, 60)
    atr_value = float(last["atr"])

    entry = float(price)
    sl = max(support * 0.995, entry - 1.5 * atr_value)
    if sl >= entry:
        sl = entry * 0.97

    risk = max(entry - sl, entry * 0.01)
    tp1 = entry + 1.5 * risk
    tp2 = entry + 2.5 * risk
    tp3 = entry + 4.0 * risk

    if resistance > entry and resistance < tp1:
        tp1 = resistance * 0.995
        if tp1 <= entry:
            tp1 = entry + risk

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi_v = float(last["rsi"])

    if mtf_score >= 75 and ema20 > ema50 and price > ema200:
        status, status_icon = "صعودی قوی", "🟢"
    elif mtf_score >= 60 and ema20 > ema50:
        status, status_icon = "صعودی", "🟢"
    elif mtf_score <= 35 and ema20 < ema50 and price < ema200:
        status, status_icon = "نزولی قوی", "🔴"
    elif mtf_score <= 45 and ema20 < ema50:
        status, status_icon = "نزولی", "🔴"
    else:
        status, status_icon = "رنج / نیازمند تأیید", "🟡"

    if mtf_score >= 80:
        action, action_icon = "ورود مشروط", "🟢"
    elif mtf_score >= 68:
        action, action_icon = "ورود با تأیید", "🟡"
    else:
        action, action_icon = "فعلاً صبر", "⚪"

    st.subheader(f"{exchange} | {symbol} — تحلیل چندتایم‌فریمی")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("قیمت فعلی", money(price))
    c2.metric("امتیاز ربات", f"{mtf_score}/100")
    c3.metric("وضعیت", f"{status_icon} {status}")
    c4.metric("تصمیم", f"{action_icon} {action}")

    st.markdown("---")

    a1, a2, a3 = st.columns(3)
    a1.metric("ورود مرجع", money(entry))
    a2.metric("حد ضرر", money(sl), f"-{(1-sl/entry)*100:.2f}%")
    a3.metric("حمایت", money(support))

    b1, b2, b3 = st.columns(3)
    b1.metric("حد سود 1", money(tp1), f"+{(tp1/entry-1)*100:.2f}%")
    b2.metric("حد سود 2", money(tp2), f"+{(tp2/entry-1)*100:.2f}%")
    b3.metric("حد سود 3", money(tp3), f"+{(tp3/entry-1)*100:.2f}%")

    st.metric("مقاومت", money(resistance))

    rr1 = (tp1-entry) / max(entry-sl, 1e-12)
    rr2 = (tp2-entry) / max(entry-sl, 1e-12)
    rr3 = (tp3-entry) / max(entry-sl, 1e-12)
    st.caption(
        f"ریسک/بازده تقریبی: TP1 = 1:{rr1:.1f} | "
        f"TP2 = 1:{rr2:.1f} | TP3 = 1:{rr3:.1f}"
    )

    st.markdown("### 🧭 هم‌جهتی تایم‌فریم‌ها")
    rows = []
    for k in ["15m", "1h", "4h", "1d"]:
        sc = scores.get(k)
        label = "داده ندارد" if sc is None else (
            "صعودی" if sc >= 60 else "نزولی" if sc <= 45 else "خنثی"
        )
        rows.append({
            "تایم‌فریم": k,
            "امتیاز": sc if sc is not None else "-",
            "وضعیت": label,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 📊 نمودار")
    chart_df = ind.tail(150).set_index("open_time")[["close", "ema20", "ema50"]]
    st.line_chart(chart_df, use_container_width=True)

    with st.expander("🧠 دلایل امتیاز ربات", expanded=True):
        for item in reasons:
            st.write("•", item)

    with st.expander("📐 اندیکاتورها"):
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("RSI", f"{rsi_v:.1f}")
        i2.metric("EMA20", money(ema20))
        i3.metric("EMA50", money(ema50))
        i4.metric("EMA200", money(ema200))

        j1, j2, j3 = st.columns(3)
        j1.metric("MACD", f"{last['macd']:.6f}")
        j2.metric("ATR", money(atr_value))
        j3.metric("نسبت حجم", f"{last['vol_ratio']:.2f}x")

    with st.expander("🏦 وضعیت اتصال صرافی‌ها"):
        st.write(f"منبع فعال تحلیل: **{exchange}**")
        st.write("Binance: متصل به داده عمومی")
        st.write("والکس: متصل به داده عمومی")
        st.write("نوبیتکس: متصل به داده عمومی")
        st.write("تبدیل: متصل به داده عمومی")
        st.caption(
            "در تبدیل، کندل‌های تکنیکال از معاملات عمومی اخیر ساخته می‌شوند؛ "
            "اگر تاریخچه کافی نباشد، همان بازار باید منبع داده جایگزین داشته باشد."
        )

    if action == "ورود مشروط":
        st.success(
            f"شرایط چندتایم‌فریمی مثبت است. ورود مرجع {money(entry)} و "
            f"حد ضرر محاسباتی {money(sl)}."
        )
    elif action == "ورود با تأیید":
        st.warning(
            f"سیگنال اولیه مثبت است؛ تثبیت بالای مقاومت {money(resistance)} "
            "یا تأیید کندلی را بررسی کن."
        )
    else:
        st.info("شرایط ایده‌آل ورود تأیید نشده؛ صبر برای تغییر ساختار یا افزایش امتیاز.")

    st.caption(
        f"آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "داده‌ها عمومی و صرفاً برای تحلیل هستند."
    )

except requests.exceptions.RequestException as exc:
    st.error("ارتباط با API صرافی برقرار نشد.")
    st.caption(str(exc))
except Exception as exc:
    st.error(f"تحلیل {symbol} در {exchange} انجام نشد.")
    st.caption(f"جزئیات فنی: {exc}")
