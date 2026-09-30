import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="ربات تحلیل حرفه‌ای کریپتو", page_icon="📈", layout="wide")

BINANCE = "https://api.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoAnalyzer/1.0"})

TIMEFRAMES = {
    "15 دقیقه": "15m",
    "1 ساعت": "1h",
    "4 ساعت": "4h",
    "روزانه": "1d",
}

COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "TRXUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "ETCUSDT"
]

def api_get(url, params=None, timeout=12):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=20)
def get_price(symbol):
    data = api_get(f"{BINANCE}/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])

@st.cache_data(ttl=30)
def get_klines(symbol, interval, limit=250):
    data = api_get(
        f"{BINANCE}/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not data:
        raise ValueError("داده کندلی دریافت نشد.")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if len(df) < 60:
        raise ValueError("تعداد کندل کافی نیست.")

    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist

def analyze_frame(df):
    close = df["close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)
    rv = rsi(close)
    ml, ms, mh = macd(close)
    av = atr(df)

    i = len(df) - 1
    price = float(close.iloc[i])

    score = 50.0
    reasons = []

    if price > e20.iloc[i]:
        score += 8
        reasons.append("قیمت بالای EMA20")
    else:
        score -= 8
        reasons.append("قیمت زیر EMA20")

    if e20.iloc[i] > e50.iloc[i]:
        score += 10
        reasons.append("EMA20 بالای EMA50")
    else:
        score -= 10
        reasons.append("EMA20 زیر EMA50")

    if price > e200.iloc[i]:
        score += 8
        reasons.append("قیمت بالای EMA200")
    else:
        score -= 8
        reasons.append("قیمت زیر EMA200")

    r = float(rv.iloc[i])
    if 50 <= r <= 68:
        score += 8
        reasons.append("RSI صعودی و متعادل")
    elif r >= 75:
        score -= 5
        reasons.append("RSI بیش‌خرید")
    elif r < 35:
        score -= 4
        reasons.append("RSI ضعیف")

    if ml.iloc[i] > ms.iloc[i]:
        score += 8
        reasons.append("MACD مثبت")
    else:
        score -= 8
        reasons.append("MACD منفی")

    if mh.iloc[i] > mh.iloc[i - 1]:
        score += 5
    else:
        score -= 3

    score = float(np.clip(score, 0, 100))

    recent = df.tail(60)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    atr_value = float(av.iloc[i])

    return {
        "price": price,
        "score": score,
        "rsi": r,
        "macd_hist": float(mh.iloc[i]),
        "atr": atr_value,
        "ema20": float(e20.iloc[i]),
        "ema50": float(e50.iloc[i]),
        "ema200": float(e200.iloc[i]),
        "support": support,
        "resistance": resistance,
        "reasons": reasons,
    }

def combine_signals(frames):
    scores = [x["score"] for x in frames]
    avg_score = float(np.mean(scores))
    bullish = sum(x["score"] >= 55 for x in frames)
    bearish = sum(x["score"] <= 45 for x in frames)

    latest = frames[0]
    price = latest["price"]
    atr_value = max(latest["atr"], price * 0.003)

    # فقط در صورت هم‌جهتی حداقل دو تایم‌فریم سیگنال صادر می‌شود.
    if bullish >= 2 and avg_score >= 62:
        action = "BUY"
        entry = price
        sl = entry - 1.5 * atr_value
        risk = entry - sl
        tp1 = entry + 1.0 * risk
        tp2 = entry + 2.0 * risk
        tp3 = entry + 3.0 * risk
    elif bearish >= 2 and avg_score <= 38:
        action = "SELL"
        entry = price
        sl = entry + 1.5 * atr_value
        risk = sl - entry
        tp1 = entry - 1.0 * risk
        tp2 = entry - 2.0 * risk
        tp3 = entry - 3.0 * risk
    else:
        action = "WAIT"
        entry = price
        sl = np.nan
        tp1 = np.nan
        tp2 = np.nan
        tp3 = np.nan

    return {
        "action": action,
        "score": round(avg_score, 1),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "bullish_frames": bullish,
        "bearish_frames": bearish,
    }

def full_analysis(symbol):
    # 1D برای ساختن سیگنال بلندمدت، 4H و 1H برای تأیید.
    intervals = ["1h", "4h", "1d"]
    frames = []
    for interval in intervals:
        df = get_klines(symbol, interval, 250)
        frames.append(analyze_frame(df))
    result = combine_signals(frames)
    result["frames"] = frames
    return result

@st.cache_data(ttl=60)
def scan_market(symbols):
    rows = []
    for symbol in symbols:
        try:
            result = full_analysis(symbol)
            rows.append({
                "ارز": symbol,
                "سیگنال": result["action"],
                "امتیاز": result["score"],
                "قیمت": result["entry"],
                "ورود": result["entry"],
                "حدضرر": result["sl"],
                "TP1": result["tp1"],
                "TP2": result["tp2"],
                "تایم‌فریم صعودی": result["bullish_frames"],
                "تایم‌فریم نزولی": result["bearish_frames"],
            })
        except Exception as e:
            rows.append({
                "ارز": symbol,
                "سیگنال": "ERROR",
                "امتیاز": 0,
                "قیمت": np.nan,
                "ورود": np.nan,
                "حدضرر": np.nan,
                "TP1": np.nan,
                "TP2": np.nan,
                "تایم‌فریم صعودی": 0,
                "تایم‌فریم نزولی": 0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("امتیاز", ascending=False).reset_index(drop=True)

st.title("📈 ربات تحلیل حرفه‌ای کریپتو")
st.caption("تحلیل تکنیکال | BUY / SELL / WAIT | بدون ثبت سفارش")

with st.sidebar:
    st.header("⚙️ تنظیمات")
    symbol = st.selectbox("ارز برای تحلیل", COINS, index=0)
    scan_count = st.slider("تعداد ارز برای اسکن", 5, len(COINS), 10)
    st.info("داده تکنیکال این نسخه از بازار عمومی Binance دریافت می‌شود.")

tab1, tab2 = st.tabs(["📊 تحلیل ارز", "🔎 اسکن بازار"])

with tab1:
    if st.button("تحلیل الآن", type="primary", use_container_width=True):
        try:
            result = full_analysis(symbol)
            action = result["action"]

            if action == "BUY":
                st.success("🟢 BUY — شرایط محاسباتی خرید")
            elif action == "SELL":
                st.error("🔴 SELL — شرایط محاسباتی فروش/خروج")
            else:
                st.warning("🟡 WAIT — فعلاً دست نگه دار")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("امتیاز", result["score"])
            c2.metric("ورود", f'{result["entry"]:,.8g}')
            c3.metric("TP1", "-" if np.isnan(result["tp1"]) else f'{result["tp1"]:,.8g}')
            c4.metric("TP2", "-" if np.isnan(result["tp2"]) else f'{result["tp2"]:,.8g}')

            c5, c6, c7 = st.columns(3)
            c5.metric("حدضرر", "-" if np.isnan(result["sl"]) else f'{result["sl"]:,.8g}')
            c6.metric("TP3", "-" if np.isnan(result["tp3"]) else f'{result["tp3"]:,.8g}')
            c7.metric("هم‌جهتی", f'{result["bullish_frames"]} صعودی / {result["bearish_frames"]} نزولی')

            st.subheader("تایم‌فریم‌ها")
            tf_names = ["1H", "4H", "1D"]
            tf_rows = []
            for name, frame in zip(tf_names, result["frames"]):
                tf_rows.append({
                    "تایم‌فریم": name,
                    "امتیاز": round(frame["score"], 1),
                    "RSI": round(frame["rsi"], 1),
                    "حمایت": frame["support"],
                    "مقاومت": frame["resistance"],
                })
            st.dataframe(pd.DataFrame(tf_rows), use_container_width=True, hide_index=True)

            st.subheader("دلایل تحلیل")
            for reason in result["frames"][0]["reasons"]:
                st.write("•", reason)

        except Exception as e:
            st.error(f"خطا در تحلیل: {e}")

with tab2:
    st.write("اسکنر چند ارز را بررسی می‌کند و سه امتیاز بالاتر را جداگانه نشان می‌دهد.")
    if st.button("🔎 شروع اسکن", type="primary", use_container_width=True):
        with st.spinner("در حال اسکن بازار..."):
            result_df = scan_market(COINS[:scan_count])

        valid = result_df[result_df["سیگنال"] != "ERROR"].copy()

        if valid.empty:
            st.error("هیچ داده‌ای برای اسکن دریافت نشد.")
        else:
            top3 = valid.head(3)

            st.subheader("🏆 سه ارز با بالاترین امتیاز محاسباتی")
            cols = st.columns(3)
            for idx, (_, row) in enumerate(top3.iterrows()):
                with cols[idx]:
                    signal = row["سیگنال"]
                    if signal == "BUY":
                        st.success(f'🥇 {row["ارز"]} — BUY')
                    elif signal == "SELL":
                        st.error(f'🔴 {row["ارز"]} — SELL')
                    else:
                        st.warning(f'🟡 {row["ارز"]} — WAIT')
                    st.metric("امتیاز", row["امتیاز"])
                    st.write(f'ورود: {row["ورود"]:,.8g}')
                    if pd.notna(row["حدضرر"]):
                        st.write(f'SL: {row["حدضرر"]:,.8g}')
                        st.write(f'TP1: {row["TP1"]:,.8g}')
                        st.write(f'TP2: {row["TP2"]:,.8g}')

            st.subheader("جدول کامل اسکن")
            display = valid.copy()
            for col in ["قیمت", "ورود", "حدضرر", "TP1", "TP2"]:
                display[col] = display[col].map(
                    lambda x: "-" if pd.isna(x) else f"{x:,.8g}"
                )
            st.dataframe(display, use_container_width=True, hide_index=True)

st.caption(f"آخرین اجرای برنامه: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
