import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="Crypto Analyzer Pro",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TIMEFRAMES = {
    "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h",
    "6H": "6h", "12H": "12h", "1D": "1d",
    "3D": "3d", "1W": "1w",
}

BINANCE = "https://api.binance.com"
TABDEAL = "https://api.tabdeal.org"
OKX = "https://www.okx.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoAnalyzerPro/4.0"})

REQUEST_TIMEOUT = 8
KLINE_LIMIT = 220

@st.cache_data(ttl=900, show_spinner=False)
def get_tabdeal_universe():
    urls = [
        f"{TABDEAL}/r/api/v1/exchangeInfo",
        f"{TABDEAL}/api/v1/exchangeInfo",
        f"{TABDEAL}/v1/market/symbols",
        f"{TABDEAL}/v1/markets",
        f"{TABDEAL}/api/v1/markets",
    ]
    for url in urls:
        try:
            r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if r.ok:
                data = r.json()
                items = data.get("symbols") if isinstance(data, dict) else data
                if isinstance(data, dict) and items is None:
                    items = data.get("data")
                if not isinstance(items, list):
                    continue
                out = []
                for x in items:
                    if not isinstance(x, dict):
                        continue
                    raw = str(
                        x.get("symbol") or x.get("name") or x.get("code") or ""
                    ).upper().replace("-", "").replace("_", "").replace("/", "")
                    if raw.endswith("USDT"):
                        out.append(raw)
                out = sorted(set(out))
                if out:
                    return out, "Tabdeal"
        except Exception:
            pass

    # Reliable public fallback so the app remains usable if Tabdeal is unavailable.
    try:
        r = SESSION.get(f"{BINANCE}/api/v3/exchangeInfo", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        out = []
        for x in data.get("symbols", []):
            if x.get("quoteAsset") == "USDT" and x.get("status") == "TRADING":
                out.append(x["symbol"].upper())
        return sorted(set(out)), "Binance fallback"
    except Exception:
        return [], "Unavailable"

def normalize_symbol(symbol):
    s = str(symbol).upper().replace("/", "").replace("-", "").replace("_", "")
    return s

def fetch_binance_klines(symbol, interval, limit=KLINE_LIMIT):
    r = SESSION.get(
        f"{BINANCE}/api/v3/klines",
        params={"symbol": normalize_symbol(symbol), "interval": interval, "limit": limit},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError("empty")
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time","open","high","low","close","volume"]].dropna()

def fetch_okx_klines(symbol, interval, limit=KLINE_LIMIT):
    mapping = {
        "5m":"5m","15m":"15m","30m":"30m","1h":"1H","2h":"2H",
        "4h":"4H","6h":"6H","12h":"12H","1d":"1D","3d":"3D","1w":"1W"
    }
    inst = normalize_symbol(symbol)[:-4] + "-USDT"
    bar = mapping.get(interval, interval)
    r = SESSION.get(
        f"{OKX}/api/v5/market/candles",
        params={"instId": inst, "bar": bar, "limit": str(min(limit, 300))},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise ValueError("empty")
    data = list(reversed(data))
    rows = []
    for x in data:
        rows.append([pd.to_datetime(int(x[0]), unit="ms"),
                      float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])])
    return pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume"])

def get_klines(symbol, interval, limit=KLINE_LIMIT):
    # Tabdeal is the requested primary market universe. Public OHLCV fallbacks
    # keep analysis alive when Tabdeal's public candle endpoint is unavailable.
    for fn in (fetch_binance_klines, fetch_okx_klines):
        try:
            df = fn(symbol, interval, limit)
            if len(df) >= 80:
                return df, fn.__name__
        except Exception:
            continue
    return None, None

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(s):
    m = ema(s, 12) - ema(s, 26)
    sig = ema(m, 9)
    return m, sig, m - sig

def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atrv = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False).mean() / atrv.replace(0,np.nan)
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False).mean() / atrv.replace(0,np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)

def analyze_df(df):
    c = df["close"]
    e20, e50, e200 = ema(c,20), ema(c,50), ema(c,200)
    rv = rsi(c)
    m, ms, mh = macd(c)
    av = atr(df)
    ax = adx(df)

    last = float(c.iloc[-1])
    prev = float(c.iloc[-2])
    momentum = ((last / float(c.iloc[-6])) - 1) * 100
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]
    vol_ratio = float(df["volume"].iloc[-1] / vol_avg) if vol_avg else 1.0

    score = 50.0
    score += 10 if last > e20.iloc[-1] else -10
    score += 8 if e20.iloc[-1] > e50.iloc[-1] else -8
    score += 7 if e50.iloc[-1] > e200.iloc[-1] else -7
    score += 7 if m.iloc[-1] > ms.iloc[-1] else -7
    score += 5 if 52 <= rv.iloc[-1] <= 68 else (-4 if rv.iloc[-1] < 35 or rv.iloc[-1] > 75 else 0)
    score += 5 if momentum > 0 else -5
    score += 4 if ax.iloc[-1] >= 20 else 0
    score += 4 if vol_ratio >= 1.15 else 0
    score = float(np.clip(score, 0, 100))

    # Price-action confirmation.
    recent_high = float(df["high"].tail(20).max())
    recent_low = float(df["low"].tail(20).min())
    breakout = last >= recent_high * 0.997
    bullish_candle = last > float(df["open"].iloc[-1]) and last > prev

    if score >= 72 and (breakout or bullish_candle):
        trade = "خرید"
        decision = "معامله"
    elif score <= 34:
        trade = "فروش"
        decision = "اجتناب"
    else:
        trade = "صبر"
        decision = "عدم ورود"

    support = float(df["low"].tail(40).min())
    resistance = float(df["high"].tail(40).max())

    risk = max(float(av.iloc[-1]), last * 0.008)
    if trade == "خرید":
        entry = last
        sl = max(support * 0.995, entry - 1.5 * risk)
        tp1 = entry + 1.5 * (entry - sl)
        tp2 = entry + 2.5 * (entry - sl)
    elif trade == "فروش":
        entry = last
        sl = min(resistance * 1.005, entry + 1.5 * risk)
        tp1 = entry - 1.5 * (sl - entry)
        tp2 = entry - 2.5 * (sl - entry)
    else:
        entry, sl, tp1, tp2 = last, support, resistance, resistance

    return {
        "price": last, "score": round(score,1), "trade": trade,
        "decision": decision, "rsi": round(float(rv.iloc[-1]),1),
        "macd": "مثبت" if m.iloc[-1] > ms.iloc[-1] else "منفی",
        "adx": round(float(ax.iloc[-1]),1),
        "momentum": round(float(momentum),2),
        "volume_ratio": round(vol_ratio,2),
        "support": support, "resistance": resistance,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
    }

def analyze_symbol(symbol, timeframe="1H"):
    df, source = get_klines(symbol, TIMEFRAMES[timeframe])
    if df is None or len(df) < 80:
        return None
    try:
        a = analyze_df(df)
        a["symbol"] = symbol
        a["timeframe"] = timeframe
        a["source"] = source
        return a
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def scan_market(symbols, timeframe):
    results = []
    max_workers = min(12, max(2, len(symbols)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_symbol, s, timeframe): s for s in symbols}
        for f in as_completed(futures):
            try:
                x = f.result()
                if x:
                    results.append(x)
            except Exception:
                pass
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

def fmt_price(x):
    try:
        x = float(x)
        if x >= 1000: return f"{x:,.2f}"
        if x >= 1: return f"{x:,.4f}"
        if x >= .01: return f"{x:.6f}"
        return f"{x:.10f}"
    except Exception:
        return "-"

st.title("₿ Crypto Analyzer Pro")

symbols, source = get_tabdeal_universe()
if not symbols:
    st.error("فهرست ارزها دریافت نشد. اتصال اینترنت یا API صرافی را بررسی کنید.")
    st.stop()

with st.sidebar:
    st.caption(f"منبع فهرست: {source}")
    tf = st.selectbox("تایم‌فریم", list(TIMEFRAMES.keys()), index=3)
    search = st.text_input("جستجوی ارز", "")
    limit = st.slider("تعداد ارز برای اسکن", 10, min(250, len(symbols)), min(100, len(symbols)))

filtered = [s for s in symbols if search.upper() in s] if search else symbols
selected = st.multiselect(
    "انتخاب ارز برای تحلیل عمیق",
    filtered[:500],
    max_selections=5,
    placeholder="حداکثر ۵ ارز",
)

c1, c2 = st.columns([1,1])
with c1:
    run_selected = st.button("تحلیل ارزهای انتخاب‌شده", use_container_width=True)
with c2:
    run_scan = st.button("اسکن همه ارزها", use_container_width=True)

if run_selected and selected:
    rows = []
    with st.spinner("در حال تحلیل..."):
        for s in selected:
            x = analyze_symbol(s, tf)
            if x:
                rows.append(x)
    if rows:
        for x in rows:
            st.subheader(f"{x['symbol']} — {x['timeframe']}")
            cols = st.columns(7)
            cols[0].metric("قیمت", fmt_price(x["price"]))
            cols[1].metric("نوع معامله", x["trade"])
            cols[2].metric("درصد موفقیت", f"{x['score']:.1f}%")
            cols[3].metric("ورود", fmt_price(x["entry"]))
            cols[4].metric("حد ضرر", fmt_price(x["sl"]))
            cols[5].metric("هدف ۱", fmt_price(x["tp1"]))
            cols[6].metric("هدف ۲", fmt_price(x["tp2"]))
            st.write(
                f"تصمیم: **{x['decision']}** | RSI: {x['rsi']} | MACD: {x['macd']} | "
                f"ADX: {x['adx']} | مومنتوم: {x['momentum']}% | "
                f"حمایت: {fmt_price(x['support'])} | مقاومت: {fmt_price(x['resistance'])}"
            )
    else:
        st.warning("برای ارزهای انتخاب‌شده داده کافی دریافت نشد.")

if run_scan:
    scan_symbols = filtered[:limit]
    with st.spinner(f"اسکن {len(scan_symbols)} ارز..."):
        df = scan_market(tuple(scan_symbols), tf)
    if df.empty:
        st.warning("داده کافی برای اسکن دریافت نشد.")
    else:
        display = df[[
            "symbol","score","trade","decision","price",
            "entry","sl","tp1","tp2","rsi","momentum","source"
        ]].copy()
        display.columns = [
            "ارز","درصد موفقیت","نوع معامله","تصمیم","قیمت",
            "ورود","حد ضرر","هدف ۱","هدف ۲","RSI","مومنتوم %","منبع"
        ]
        for col in ["قیمت","ورود","حد ضرر","هدف ۱","هدف ۲"]:
            display[col] = display[col].map(fmt_price)
        display["درصد موفقیت"] = display["درصد موفقیت"].map(lambda x: f"{x:.1f}%")
        display["مومنتوم %"] = display["مومنتوم %"].map(lambda x: f"{x:.2f}%")
        st.subheader(f"جدول کامل بازار — {tf}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("پامپ کاندیدها")
        pump = df[(df["score"] >= 72) & (df["trade"] == "خرید")].head(20)
        if pump.empty:
            st.info("در این اسکن موردی با شرایط ورود تأییدشده پیدا نشد.")
        else:
            p = pump[["symbol","score","trade","price","entry","sl","tp1","tp2"]].copy()
            p.columns = ["ارز","درصد موفقیت","نوع معامله","قیمت","ورود","حد ضرر","هدف ۱","هدف ۲"]
            for col in ["قیمت","ورود","حد ضرر","هدف ۱","هدف ۲"]:
                p[col] = p[col].map(fmt_price)
            p["درصد موفقیت"] = p["درصد موفقیت"].map(lambda x: f"{x:.1f}%")
            st.dataframe(p, use_container_width=True, hide_index=True)

st.caption("این ابزار سیگنال قطعی یا تضمین سود نیست؛ درصد موفقیت، امتیاز مدل تحلیل است.")
