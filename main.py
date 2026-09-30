import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ============================================================
# Crypto Analyzer Pro - Single File / No pandas-ta required
# ============================================================

st.set_page_config(
    page_title="Crypto Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BINANCE = "https://api.binance.com"

# ---------- UI ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
h1, h2, h3 {text-align: right;}
div[data-testid="stMetric"] {direction: rtl;}
.small-note {font-size: 0.85rem; color: #777;}
.signal-box {
    padding: 18px; border-radius: 14px; margin: 8px 0;
    border: 1px solid rgba(128,128,128,.25);
}
</style>
""", unsafe_allow_html=True)

st.title("📈 ربات تحلیل حرفه‌ای کریپتو")
st.caption("تحلیل تکنیکال چندتایم‌فریمی — داده عمومی Binance — بدون امکان ثبت سفارش")

# ---------- Helpers ----------
@st.cache_data(ttl=30, show_spinner=False)
def get_klines(symbol: str, interval: str, limit: int = 300):
    url = f"{BINANCE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 80:
        raise ValueError("داده کافی از صرافی دریافت نشد.")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]].dropna()

@st.cache_data(ttl=60, show_spinner=False)
def get_price(symbol: str):
    r = requests.get(f"{BINANCE}/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

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

def support_resistance(df, window=40):
    recent = df.tail(window)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    return support, resistance

def timeframe_score(df):
    """Returns score 0..100 and descriptive reasons."""
    x = add_indicators(df)
    if len(x) < 10:
        return 50, ["داده کافی نیست"], x

    a = x.iloc[-1]
    p = x.iloc[-2]
    score = 50
    reasons = []

    # Trend
    if a["ema20"] > a["ema50"]:
        score += 10
        reasons.append("EMA20 بالاتر از EMA50 ✅")
    else:
        score -= 10
        reasons.append("EMA20 پایین‌تر از EMA50 ⚠️")

    if a["close"] > a["ema200"]:
        score += 8
        reasons.append("قیمت بالای EMA200 ✅")
    else:
        score -= 8
        reasons.append("قیمت زیر EMA200 ⚠️")

    # MACD
    if a["macd"] > a["macd_signal"]:
        score += 8
        reasons.append("MACD صعودی ✅")
    else:
        score -= 8
        reasons.append("MACD نزولی ⚠️")

    if a["macd_hist"] > p["macd_hist"]:
        score += 4
        reasons.append("مومنتوم MACD در حال تقویت ✅")
    else:
        score -= 3
        reasons.append("مومنتوم MACD در حال تضعیف ⚠️")

    # RSI
    if 50 <= a["rsi"] <= 68:
        score += 8
        reasons.append(f"RSI مناسب ({a['rsi']:.1f}) ✅")
    elif a["rsi"] > 72:
        score -= 5
        reasons.append(f"RSI بیش‌خرید ({a['rsi']:.1f}) ⚠️")
    elif a["rsi"] < 30:
        score += 2
        reasons.append(f"RSI اشباع فروش ({a['rsi']:.1f})")
    else:
        score += 1
        reasons.append(f"RSI خنثی ({a['rsi']:.1f})")

    # Volume
    vr = float(a["vol_ratio"]) if np.isfinite(a["vol_ratio"]) else 1.0
    if vr >= 1.5:
        score += 7
        reasons.append(f"حجم قوی ({vr:.1f}x) ✅")
    elif vr >= 1.1:
        score += 3
        reasons.append(f"حجم مناسب ({vr:.1f}x)")
    else:
        reasons.append(f"حجم معمولی ({vr:.1f}x)")

    return int(np.clip(score, 0, 100)), reasons, x

def normalize_symbol(raw):
    s = raw.strip().upper().replace("/", "").replace("-", "").replace(" ", "")
    if not s:
        return ""
    if s.endswith("USDT"):
        return s
    return s + "USDT"

def money(v):
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.8f}"

# ---------- Sidebar / Coin Input ----------
with st.sidebar:
    st.header("🔎 انتخاب ارز")
    raw_symbol = st.text_input(
        "نام ارز را وارد کن",
        value="XRP",
        placeholder="مثلاً BTC ، ETH ، XRP ، SOL",
        help="فقط نماد ارز را بنویس؛ USDT به‌صورت خودکار اضافه می‌شود."
    )
    symbol = normalize_symbol(raw_symbol)

    tf = st.selectbox(
        "تایم‌فریم اصلی",
        ["15m", "1h", "4h", "1d"],
        index=2
    )

    analyze = st.button("🚀 شروع تحلیل", use_container_width=True)

    st.divider()
    st.info(
        "این برنامه ابزار تحلیل و تحقیق است و سفارش خرید/فروش ثبت نمی‌کند. "
        "سیگنال تضمینی نیست."
    )

if not symbol:
    st.warning("لطفاً نماد یک ارز را وارد کن.")
    st.stop()

# Run automatically on first load or button press
try:
    # Validate symbol and fetch main timeframe
    price = get_price(symbol)
    main_df = get_klines(symbol, tf, 300)

    # Multi-timeframe data
    tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    scores = {}
    tf_data = {}
    for name, interval in tf_map.items():
        try:
            d = get_klines(symbol, interval, 250)
            sc, rs, ind = timeframe_score(d)
            scores[name] = sc
            tf_data[name] = (d, ind, rs)
        except Exception:
            scores[name] = None

    main_score, reasons, ind = timeframe_score(main_df)
    last = ind.iloc[-1]

    # Weighted multi-timeframe score
    weights = {"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}
    valid = [(scores[k], weights[k]) for k in weights if scores.get(k) is not None]
    mtf_score = round(sum(s*w for s, w in valid) / sum(w for s, w in valid)) if valid else main_score

    support, resistance = support_resistance(ind, 60)
    atr_value = float(last["atr"])

    # Dynamic trade levels
    # Long scenario
    entry = float(price)
    sl = max(support * 0.995, entry - 1.5 * atr_value)
    if sl >= entry:
        sl = entry * 0.97

    risk = max(entry - sl, entry * 0.01)
    tp1 = entry + 1.5 * risk
    tp2 = entry + 2.5 * risk
    tp3 = entry + 4.0 * risk

    # Respect nearby resistance where sensible
    if resistance > entry and resistance < tp1:
        tp1 = resistance * 0.995
        if tp1 <= entry:
            tp1 = entry + risk

    # Market state
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi_v = float(last["rsi"])

    if mtf_score >= 75 and ema20 > ema50 and price > ema200:
        status = "صعودی قوی"
        status_icon = "🟢"
    elif mtf_score >= 60 and ema20 > ema50:
        status = "صعودی"
        status_icon = "🟢"
    elif mtf_score <= 35 and ema20 < ema50 and price < ema200:
        status = "نزولی قوی"
        status_icon = "🔴"
    elif mtf_score <= 45 and ema20 < ema50:
        status = "نزولی"
        status_icon = "🔴"
    else:
        status = "رنج / نیازمند تأیید"
        status_icon = "🟡"

    # Entry decision
    if mtf_score >= 80:
        action = "ورود مشروط"
        action_icon = "🟢"
    elif mtf_score >= 68:
        action = "ورود با تأیید"
        action_icon = "🟡"
    else:
        action = "فعلاً صبر"
        action_icon = "⚪"

    # ---------- Header ----------
    st.subheader(f"{symbol} — تحلیل چندتایم‌فریمی")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("قیمت فعلی", money(price), symbol.replace("USDT", " / USDT"))
    c2.metric("امتیاز ربات", f"{mtf_score}/100")
    c3.metric("وضعیت", f"{status_icon} {status}")
    c4.metric("تصمیم", f"{action_icon} {action}")

    # ---------- Main signal ----------
    st.markdown("---")
    a1, a2, a3 = st.columns(3)
    a1.metric("ورود مرجع", money(entry))
    a2.metric("حد ضرر", money(sl), f"-{(1-sl/entry)*100:.2f}%")
    a3.metric("مقاومت نزدیک", money(resistance))

    b1, b2, b3 = st.columns(3)
    b1.metric("حد سود 1", money(tp1), f"+{(tp1/entry-1)*100:.2f}%")
    b2.metric("حد سود 2", money(tp2), f"+{(tp2/entry-1)*100:.2f}%")
    b3.metric("حد سود 3", money(tp3), f"+{(tp3/entry-1)*100:.2f}%")

    rr1 = (tp1-entry) / max(entry-sl, 1e-12)
    rr2 = (tp2-entry) / max(entry-sl, 1e-12)
    rr3 = (tp3-entry) / max(entry-sl, 1e-12)

    st.caption(
        f"نسبت ریسک به بازده تقریبی: TP1 = 1:{rr1:.1f} | "
        f"TP2 = 1:{rr2:.1f} | TP3 = 1:{rr3:.1f}"
    )

    # ---------- MTF table ----------
    st.markdown("### 🧭 هم‌جهتی تایم‌فریم‌ها")
    rows = []
    for k in ["15m", "1h", "4h", "1d"]:
        sc = scores.get(k)
        label = "داده ندارد" if sc is None else (
            "صعودی" if sc >= 60 else "نزولی" if sc <= 45 else "خنثی"
        )
        rows.append({"تایم‌فریم": k, "امتیاز": sc if sc is not None else "-", "وضعیت": label})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---------- Chart ----------
    st.markdown("### 📊 نمودار قیمت")
    chart_df = ind.tail(120).set_index("open_time")[["close", "ema20", "ema50"]]
    st.line_chart(chart_df, use_container_width=True)

    # ---------- Support / Resistance ----------
    s1, s2, s3 = st.columns(3)
    s1.metric("حمایت", money(support))
    s2.metric("قیمت", money(price))
    s3.metric("مقاومت", money(resistance))

    # ---------- Reasons ----------
    with st.expander("🧠 چرا ربات این امتیاز را داده؟", expanded=True):
        for item in reasons:
            st.write("•", item)

    # ---------- Indicators ----------
    with st.expander("📐 جزئیات اندیکاتورها"):
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("RSI", f"{rsi_v:.1f}")
        i2.metric("EMA20", money(ema20))
        i3.metric("EMA50", money(ema50))
        i4.metric("EMA200", money(ema200))

        j1, j2, j3 = st.columns(3)
        j1.metric("MACD", f"{last['macd']:.6f}")
        j2.metric("ATR", money(atr_value))
        j3.metric("نسبت حجم", f"{last['vol_ratio']:.2f}x")

    # ---------- Rule-based confirmation ----------
    st.markdown("### 🎯 پلن معاملاتی ربات")
    if action == "ورود مشروط":
        st.success(
            f"سیگنال صعودی قوی است؛ با این حال ورود باید با مدیریت ریسک انجام شود. "
            f"ورود مرجع {money(entry)}، حد ضرر {money(sl)}."
        )
    elif action == "ورود با تأیید":
        st.warning(
            f"سیگنال اولیه مثبت است، اما بهتر است شکست/تثبیت مقاومت {money(resistance)} "
            f"یا تأیید کندلی را بررسی کنی."
        )
    else:
        st.info("فعلاً شرایط ایده‌آل ورود تأیید نشده؛ صبر برای تغییر ساختار یا افزایش امتیاز.")

    st.caption(
        f"آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "منبع داده: Binance Public API"
    )

except requests.exceptions.RequestException:
    st.error(
        "ارتباط با Binance برقرار نشد. اتصال اینترنت یا محدودیت منطقه‌ای API را بررسی کن و دوباره تلاش کن."
    )
except Exception as e:
    st.error(f"تحلیل برای {symbol} انجام نشد.")
    st.caption(f"جزئیات فنی: {str(e)}")
