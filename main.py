
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "تحلیل چندتایم‌فریمی | BUY / SELL / HOLD | بدون ثبت سفارش"
)

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

@st.cache_data(ttl=30, show_spinner=False)
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

@st.cache_data(ttl=20, show_spinner=False)
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

@st.cache_data(ttl=30, show_spinner=False)
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

@st.cache_data(ttl=20, show_spinner=False)
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

@st.cache_data(ttl=30, show_spinner=False)
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

@st.cache_data(ttl=20, show_spinner=False)
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
def tabdeal_trades(symbol, limit=2000):
    """Public recent trades.

    Tabdeal's official public collection documents `tabdealSymbol` for this
    endpoint and shows the optional limit disabled. Do not send `limit=2000`
    because that combination can return HTTP 400 on the live API.
    """
    tab_symbol = tabdeal_pair_name(symbol)
    try:
        data = http_get(
            f"{TABDEAL_BASE}/r/api/v1/trades",
            {"tabdealSymbol": tab_symbol},
            15,
        )
        return parse_tabdeal_trade_rows(data)
    except Exception as exc:
        raise RuntimeError(
            f"داده معاملات تبدیل برای {symbol} در دسترس نیست ({tab_symbol})."
        ) from exc


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
    if len(out) < 20:
        raise RuntimeError("تاریخچه معاملاتی تبدیل برای این تایم‌فریم کافی نیست.")
    return out.tail(300)


@st.cache_data(ttl=30, show_spinner=False)
def tabdeal_klines(symbol, interval):
    """Get technical candles without letting a Tabdeal 400 break the app.

    For USDT markets, Binance is used as the technical-history source because
    it provides native OHLC candles. The live/reference price still comes
    from Tabdeal. For non-USDT markets, recent Tabdeal trades are converted
    to candles when available.
    """
    compact = str(symbol).upper().replace("-", "").replace("_", "")
    if compact.endswith("USDT"):
        return binance_klines(compact, interval)

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
    """Build a clean scanner universe from the selected exchange."""
    cleaned = []
    for raw in markets:
        m = str(raw).upper().replace("-", "").replace("_", "")
        base, quote = base_asset(m)
        if not base or base in STABLE_BASES or not quote:
            continue
        if exchange == "Binance" and quote != "USDT":
            continue
        if exchange in {"والکس", "نوبیتکس", "تبدیل"} and quote not in {"USDT", "IRT", "RLS"}:
            continue
        cleaned.append(m)

    preferred = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT",
        "DOTUSDT", "LTCUSDT", "SHIBUSDT", "ATOMUSDT", "NEARUSDT",
        "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT", "PEPEUSDT",
    ]
    ordered = [x for x in preferred if x in cleaned]
    ordered += [x for x in cleaned if x not in ordered]
    return ordered[:max(1, int(limit))]


def signal_from_score(score, ema20, ema50, ema200, macd_hist, rsi_value):
    """Return a neutral BUY/SELL/HOLD trading state.

    The signal is deliberately confirmation-based:
    - BUY requires bullish multi-timeframe score and trend structure.
    - SELL requires bearish multi-timeframe score and trend structure.
    - Everything else is HOLD.
    """
    if score is None:
        return "HOLD", "⚪", "داده ناکافی"

    bullish = (
        score >= 68
        and ema20 > ema50
        and ema50 >= ema200
        and macd_hist >= 0
        and rsi_value < 72
    )
    bearish = (
        score <= 35
        and ema20 < ema50
        and ema50 <= ema200
        and macd_hist <= 0
        and rsi_value > 28
    )

    if bullish:
        return "BUY", "🟢", "شرایط خرید تأییدشده"
    if bearish:
        return "SELL", "🔴", "شرایط فروش تأییدشده"
    return "HOLD", "🟡", "فعلاً دست نگه دار"


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
        score, ema20, ema50, ema200, macd_hist, rsi_value
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
                        st.write(row.get("سیگنال", "⚪ HOLD"))
                        st.caption(row.get("توضیح سیگنال", ""))
                        if pd.notna(row.get("قیمت")):
                            st.caption(f"قیمت: {money(float(row['قیمت']))}")

            display_cols = [
                "بازار", "امتیاز", "سیگنال", "توضیح سیگنال", "قیمت",
                "ورود", "حدضرر", "حدسود 1", "حدسود 2", "حمایت", "مقاومت",
            ]
            display_cols = [c for c in display_cols if c in scan_df.columns]
            view = scan_df[display_cols].copy()
            for c in ["قیمت", "ورود", "حدضرر", "حدسود 1", "حدسود 2", "حمایت", "مقاومت"]:
                if c in view.columns:
                    view[c] = view[c].apply(lambda x: money(float(x)) if pd.notna(x) else "-")
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.caption(
                "سیگنال اسکن بر پایه روند، EMA، MACD، RSI، حجم و تایم‌فریم‌های انتخاب‌شده است؛ "
                "نتیجه اسکن به‌تنهایی تضمین‌کننده نتیجه معامله نیست."
            )

if not symbol:
    st.warning("یک بازار انتخاب کن.")
    st.stop()

# -------------------- Analysis --------------------

try:
    # Fetch the live price and all four timeframes concurrently.
    tf_names = ["15m", "1h", "4h", "1d"]
    with ThreadPoolExecutor(max_workers=5) as pool:
        price_future = pool.submit(get_price, exchange, symbol)
        candle_futures = {name: pool.submit(get_candles, exchange, symbol, name) for name in tf_names}
        price = price_future.result()
        tf_data = {}
        scores = {}
        for name, future in candle_futures.items():
            try:
                d = future.result()
                sc, rs, ind_tf = timeframe_score(d)
                scores[name] = sc
                tf_data[name] = (d, ind_tf, rs)
            except Exception:
                scores[name] = None

    # Reuse the already-fetched selected timeframe; never request it twice.
    if tf in tf_data:
        main_df, ind, reasons = tf_data[tf]
    else:
        valid_tf = next((name for name in tf_names if name in tf_data), None)
        if valid_tf is None:
            raise RuntimeError("هیچ تایم‌فریم معتبری برای تحلیل دریافت نشد.")
        main_df, ind, reasons = tf_data[valid_tf]

    main_score = scores.get(tf) if scores.get(tf) is not None else timeframe_score(main_df)[0]
    if len(ind) < 10:
        raise RuntimeError("داده کافی برای تحلیل تکنیکال وجود ندارد.")

    last = ind.iloc[-1]
    weights = {"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}
    valid = [(scores[k], weights[k]) for k in weights if scores.get(k) is not None]
    mtf_score = round(sum(s * w for s, w in valid) / sum(w for _, w in valid)) if valid else main_score

    support, resistance = support_resistance(ind, 60)
    atr_value = float(last["atr"])

    entry = float(price)

    weights = {"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}
    valid = [(scores[k], weights[k]) for k in weights if scores.get(k) is not None]
    mtf_score = (
        round(sum(s * w for s, w in valid) / sum(w for s, w in valid))
        if valid else main_score
    )

    support, resistance = support_resistance(ind, 60)
    atr_value = float(last["atr"])
    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = entry * 0.02

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi_v = float(last["rsi"])
    macd_hist = float(last["macd_hist"])

    signal, signal_icon, signal_text = signal_from_score(
        mtf_score, ema20, ema50, ema200, macd_hist, rsi_v
    )

    risk = max(1.5 * atr_value, entry * 0.01)

    if signal == "BUY":
        sl = min(entry - risk, support * 0.995)
        if sl <= 0 or sl >= entry:
            sl = entry - risk
        risk_amount = max(entry - sl, entry * 0.01)
        tp1 = entry + 1.5 * risk_amount
        tp2 = entry + 2.5 * risk_amount
        tp3 = entry + 4.0 * risk_amount

        if resistance > entry and resistance < tp1:
            tp1 = max(entry + risk_amount, resistance * 0.995)

    elif signal == "SELL":
        sl = max(entry + risk, resistance * 1.005)
        risk_amount = max(sl - entry, entry * 0.01)
        tp1 = entry - 1.5 * risk_amount
        tp2 = entry - 2.5 * risk_amount
        tp3 = entry - 4.0 * risk_amount

        if support < entry and support > tp1:
            tp1 = min(entry - risk_amount, support * 1.005)

    else:
        sl = np.nan
        tp1 = np.nan
        tp2 = np.nan
        tp3 = np.nan

    status = (
        "صعودی" if mtf_score >= 60
        else "نزولی" if mtf_score <= 45
        else "رنج / خنثی"
    )
    status_icon = "🟢" if mtf_score >= 60 else "🔴" if mtf_score <= 45 else "🟡"
    action = signal
    action_icon = signal_icon

    st.subheader(f"{exchange} | {symbol} — تحلیل چندتایم‌فریمی")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("قیمت فعلی", money(price))
    c2.metric("امتیاز ربات", f"{mtf_score}/100")
    c3.metric("وضعیت", f"{status_icon} {status}")
    c4.metric("تصمیم", f"{action_icon} {action}")

    st.markdown("---")

    a1, a2, a3 = st.columns(3)
    a1.metric("قیمت / ورود", money(entry))
    a2.metric(
        "حد ضرر",
        money(sl),
        (
            f"{((sl/entry)-1)*100:.2f}%"
            if signal == "SELL" and np.isfinite(sl)
            else f"-{(1-sl/entry)*100:.2f}%"
            if signal == "BUY" and np.isfinite(sl)
            else "-"
        ),
    )
    a3.metric("حمایت", money(support))

    b1, b2, b3 = st.columns(3)
    if signal == "BUY":
        b1.metric("حد سود 1", money(tp1), f"+{(tp1/entry-1)*100:.2f}%")
        b2.metric("حد سود 2", money(tp2), f"+{(tp2/entry-1)*100:.2f}%")
        b3.metric("حد سود 3", money(tp3), f"+{(tp3/entry-1)*100:.2f}%")
    elif signal == "SELL":
        b1.metric("حد سود 1", money(tp1), f"{(tp1/entry-1)*100:.2f}%")
        b2.metric("حد سود 2", money(tp2), f"{(tp2/entry-1)*100:.2f}%")
        b3.metric("حد سود 3", money(tp3), f"{(tp3/entry-1)*100:.2f}%")
    else:
        b1.metric("حد سود 1", "-")
        b2.metric("حد سود 2", "-")
        b3.metric("حد سود 3", "-")

    st.metric("مقاومت", money(resistance))

    if signal in {"BUY", "SELL"}:
        rr1 = abs(tp1 - entry) / max(abs(entry - sl), 1e-12)
        rr2 = abs(tp2 - entry) / max(abs(entry - sl), 1e-12)
        rr3 = abs(tp3 - entry) / max(abs(entry - sl), 1e-12)
        st.caption(
            f"ریسک/بازده تقریبی: TP1 = 1:{rr1:.1f} | "
            f"TP2 = 1:{rr2:.1f} | TP3 = 1:{rr3:.1f}"
        )
    else:
        st.caption("برای HOLD حدضرر و حدسود فعال نمایش داده نمی‌شود؛ منتظر تأیید روند بمان.")


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

    if signal in {"BUY", "SELL"}:
        level_df = pd.DataFrame(index=chart_df.index)
        level_df["قیمت"] = chart_df["close"]
        level_df["ورود"] = entry
        level_df["حدضرر"] = sl
        level_df["حدسود1"] = tp1
        level_df["حدسود2"] = tp2
        st.markdown("### 🎯 سطوح معامله")
        st.line_chart(level_df, use_container_width=True)

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

    if action == "BUY":
        st.success(
            f"🟢 BUY — ورود مرجع {money(entry)} | "
            f"حدضرر {money(sl)} | TP1 {money(tp1)}"
        )
    elif action == "SELL":
        st.error(
            f"🔴 SELL — ورود مرجع {money(entry)} | "
            f"حدضرر {money(sl)} | TP1 {money(tp1)}"
        )
    else:
        st.info(
            "🟡 HOLD — فعلاً دست نگه دار؛ ساختار چندتایم‌فریمی برای ورود "
            "خرید یا فروش به اندازه کافی تأیید نشده است."
        )


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
