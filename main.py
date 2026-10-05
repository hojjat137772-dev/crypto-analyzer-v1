import os
import time
import hmac
import hashlib
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# TABDEAL CRYPTO SIGNAL ENGINE — FINAL / TABDEAL ONLY
# ============================================================
# Read-only market analysis. No order endpoint is called.
# No Binance/OKX/Bybit/KuCoin fallback.
#
# Important:
# Tabdeal's documented public "trades" endpoint is recent-trades data,
# not a documented historical-kline API. Therefore this app NEVER invents
# history. Backtest statistics are shown only when enough real candles exist.
# ============================================================

st.set_page_config(
    page_title="Tabdeal Crypto Signal Engine",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = os.getenv("TABDEAL_BASE_URL", "https://api1.tabdeal.org")
API_KEY = os.getenv("TABDEAL_API_KEY", "")
API_SECRET = os.getenv("TABDEAL_API_SECRET", "")
TIMEOUT = int(os.getenv("TABDEAL_TIMEOUT", "15"))

TIMEFRAMES = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1H": 60,
    "2H": 120,
    "4H": 240,
    "6H": 360,
    "12H": 720,
    "1D": 1440,
    "3D": 4320,
    "1W": 10080,
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "TabdealCryptoSignalEngine/Final-1.0"})


# ============================================================
# HELPERS
# ============================================================

def norm_symbol(s):
    return str(s).upper().replace("-", "").replace("_", "").replace("/", "")


def money(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "-"
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        if abs(x) >= 1:
            return f"{x:,.5f}".rstrip("0").rstrip(".")
        return f"{x:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return "-"


def pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "-"


def api_get(path, params=None, signed=False):
    params = dict(params or {})
    if signed:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("TABDEAL_API_KEY / TABDEAL_API_SECRET تنظیم نشده است.")
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        params["signature"] = hmac.new(
            API_SECRET.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()

    headers = {}
    if API_KEY:
        headers["X-MBX-APIKEY"] = API_KEY

    r = SESSION.get(
        BASE.rstrip("/") + path,
        params=params,
        headers=headers,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ============================================================
# TABDEAL MARKET DATA
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def exchange_info():
    data = api_get("/r/api/v1/exchangeInfo")
    return data


def extract_markets(data):
    rows = []

    def walk(x):
        if isinstance(x, dict):
            # A market object
            if any(k in x for k in ("symbol", "tabdealSymbol")):
                rows.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)

    out = {}
    for d in rows:
        raw = d.get("symbol") or d.get("tabdealSymbol")
        if not raw:
            continue
        s = norm_symbol(raw)
        quote = str(d.get("quoteAsset", "")).upper()
        status = str(d.get("status", "TRADING")).upper()
        if s.endswith("USDT") and (not quote or quote == "USDT") and status in {
            "TRADING", "ACTIVE", "ENABLED", ""
        }:
            out[s] = d
    return out


def to_tabdeal_symbol(symbol):
    """
    Tabdeal's public API uses tabdealSymbol format, e.g. BTC_USDT,
    while the UI/internal symbol is normalized as BTCUSDT.
    """
    s = norm_symbol(symbol)
    if s.endswith("USDT"):
        return s[:-4] + "_USDT"
    if s.endswith("IRT"):
        return s[:-3] + "_IRT"
    return s


@st.cache_data(ttl=15, show_spinner=False)
def recent_trades(symbol, limit=1000):
    # Official public recent-trades endpoint.
    # Important: tabdealSymbol uses the exchange's underscore format.
    # Example: BTCUSDT -> BTC_USDT.
    tabdeal_symbol = to_tabdeal_symbol(symbol)
    data = api_get(
        "/r/api/v1/trades",
        {"tabdealSymbol": tabdeal_symbol, "limit": min(int(limit), 1000)},
    )

    if isinstance(data, dict):
        for key in ("trades", "data", "result"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []
    for t in data:
        if not isinstance(t, dict):
            continue

        price = t.get("price", t.get("p"))
        qty = t.get("qty", t.get("quantity", t.get("q")))
        ts = t.get("time", t.get("timestamp", t.get("T")))
        if price is None or qty is None or ts is None:
            continue

        try:
            rows.append({
                "time": pd.to_datetime(int(float(ts)), unit="ms", utc=True),
                "price": float(price),
                "qty": float(qty),
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["time", "price", "qty"])
    return df.sort_values("time").reset_index(drop=True)


def trades_to_candles(trades, minutes):
    if trades.empty:
        return pd.DataFrame()

    x = trades.copy()
    x = x.set_index("time")
    rule = f"{minutes}min"

    ohlc = x["price"].resample(rule).ohlc()
    vol = x["qty"].resample(rule).sum().rename("volume")
    count = x["price"].resample(rule).count().rename("trade_count")

    df = pd.concat([ohlc, vol, count], axis=1).dropna(subset=["open", "high", "low", "close"])
    return df


# ============================================================
# TECHNICAL INDICATORS — CAUSAL
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


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


def macd(close):
    line = ema(close, 12) - ema(close, 26)
    sig = ema(line, 9)
    return line, sig, line - sig


def adx(df, n=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(20)


def ichimoku(df):
    h9, l9 = df.high.rolling(9).max(), df.low.rolling(9).min()
    h26, l26 = df.high.rolling(26).max(), df.low.rolling(26).min()
    h52, l52 = df.high.rolling(52).max(), df.low.rolling(52).min()
    ten = (h9 + l9) / 2
    kij = (h26 + l26) / 2
    a = (ten + kij) / 2
    b = (h52 + l52) / 2
    return ten, kij, a, b


def causal_levels(df, lookback=40):
    x = df.tail(lookback)
    if len(x) < 10:
        return [], []

    support = float(x["low"].rolling(5, center=False).min().iloc[-2])
    resistance = float(x["high"].rolling(5, center=False).max().iloc[-2])
    price = float(x["close"].iloc[-1])

    supports = [support] if support < price else []
    resistances = [resistance] if resistance > price else []
    return supports, resistances


# ============================================================
# LIVE TECHNICAL ANALYSIS
# ============================================================

def technical_state(df):
    if len(df) < 80:
        return None

    c = df["close"]
    e20 = ema(c, 20)
    e50 = ema(c, 50)
    r = rsi(c)
    ml, ms, mh = macd(c)
    a = atr(df)
    ad = adx(df)
    ten, kij, sa, sb = ichimoku(df)

    i = -1
    price = float(c.iloc[i])
    cloud_hi = max(float(sa.iloc[i]), float(sb.iloc[i]))
    cloud_lo = min(float(sa.iloc[i]), float(sb.iloc[i]))

    score = 0
    reasons = []

    if price > cloud_hi:
        score += 20
        reasons.append("بالای ایچیموکو")
    elif price < cloud_lo:
        score -= 20
        reasons.append("زیر ایچیموکو")

    if float(ten.iloc[i]) > float(kij.iloc[i]):
        score += 10
        reasons.append("تنکن > کیجون")
    else:
        score -= 10

    if float(ml.iloc[i]) > float(ms.iloc[i]):
        score += 15
        reasons.append("MACD مثبت")
    else:
        score -= 15

    if 50 <= float(r.iloc[i]) <= 70:
        score += 12
        reasons.append("RSI مناسب روند")
    elif float(r.iloc[i]) > 75:
        score -= 8
        reasons.append("RSI اشباع خرید")
    elif float(r.iloc[i]) < 40:
        score -= 8
        reasons.append("RSI ضعیف")

    if price > float(e20.iloc[i]) > float(e50.iloc[i]):
        score += 18
        reasons.append("EMA20/50 صعودی")
    elif price < float(e20.iloc[i]) < float(e50.iloc[i]):
        score -= 18
        reasons.append("EMA20/50 نزولی")

    if float(ad.iloc[i]) >= 20:
        score += 8 if score > 0 else -8
        reasons.append("ADX رونددار")

    supports, resistances = causal_levels(df)
    return {
        "price": price,
        "score": float(np.clip(score, -100, 100)),
        "rsi": float(r.iloc[i]),
        "atr": float(a.iloc[i]),
        "adx": float(ad.iloc[i]),
        "reasons": reasons,
        "supports": supports,
        "resistances": resistances,
    }


def entry_plan(state):
    p = state["price"]
    a = max(state["atr"], p * 0.002)

    entry = p
    sl = p - 1.20 * a
    risk = max(entry - sl, p * 0.001)

    return {
        "entry": entry,
        "sl": sl,
        "tp1": entry + 1.00 * risk,
        "tp2": entry + 1.80 * risk,
        "tp3": entry + 2.60 * risk,
    }


# ============================================================
# BACKTEST — ONLY REAL CANDLES CURRENTLY AVAILABLE
# ============================================================

def signal_at(df, i):
    if i < 80:
        return False

    x = df.iloc[: i + 1]
    s = technical_state(x)
    if not s:
        return False

    # Causal entry condition. Signal is evaluated on bar i,
    # execution is assumed at bar i+1 open.
    return (
        s["score"] >= 55
        and 50 <= s["rsi"] <= 72
        and s["adx"] >= 18
    )


def backtest(df, max_hold=24, tp_r=1.0, sl_r=1.0):
    if len(df) < 120:
        return {
            "available": False,
            "trades": 0,
            "pf": np.nan,
            "expectancy": np.nan,
            "tp1": np.nan,
            "dd": np.nan,
            "reason": f"فقط {len(df)} کندل؛ حداقل 120 کندل لازم است.",
        }

    results = []
    equity = [0.0]

    i = 79
    while i < len(df) - 2:
        if not signal_at(df, i):
            i += 1
            continue

        entry_i = i + 1
        entry = float(df["open"].iloc[entry_i])
        a = float(atr(df.iloc[:i+1]).iloc[-1])

        risk = max(a, entry * 0.001)
        sl = entry - sl_r * risk
        tp1 = entry + tp_r * risk

        end = min(len(df) - 1, entry_i + max_hold)
        outcome = None
        r_mult = None

        for j in range(entry_i, end + 1):
            hi = float(df["high"].iloc[j])
            lo = float(df["low"].iloc[j])

            # Conservative same-bar rule: SL first.
            if lo <= sl:
                outcome = "SL"
                r_mult = -1.0
                break
            if hi >= tp1:
                outcome = "TP1"
                r_mult = 1.0
                break

        if outcome is None:
            exit_p = float(df["close"].iloc[end])
            r_mult = (exit_p - entry) / risk
            outcome = "WIN" if r_mult > 0 else "LOSS"

        results.append({
            "outcome": outcome,
            "r": float(r_mult),
        })

        equity.append(equity[-1] + float(r_mult))
        i = max(i + 1, end + 1)

    if not results:
        return {
            "available": True,
            "trades": 0,
            "pf": np.nan,
            "expectancy": np.nan,
            "tp1": np.nan,
            "dd": 0.0,
            "reason": "در تاریخچه موجود سیگنال کافی پیدا نشد.",
        }

    r = np.array([x["r"] for x in results], dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    exp = float(r.mean())
    tp1 = 100.0 * sum(x["outcome"] == "TP1" for x in results) / len(results)

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = float(np.min(eq - peak))

    return {
        "available": True,
        "trades": len(results),
        "pf": float(pf),
        "expectancy": exp,
        "tp1": tp1,
        "dd": dd,
        "reason": "محاسبه شد.",
    }


def oos_backtest(df):
    n = len(df)
    if n < 180:
        return {
            "available": False,
            "trades": 0,
            "pf": np.nan,
            "expectancy": np.nan,
            "tp1": np.nan,
            "dd": np.nan,
            "reason": f"{n} کندل موجود است؛ برای OOS حداقل 180 کندل لازم است.",
        }

    split = int(n * 0.60)
    # Train portion is deliberately not used to fabricate a score.
    # The fixed strategy is evaluated only on the OOS 40%.
    oos = df.iloc[split:].copy()
    result = backtest(oos)

    if result["available"]:
        result["reason"] = f"OOS: {len(oos)} کندل از {n} کندل."
    return result


# ============================================================
# MULTI-TIMEFRAME
# ============================================================

def analyze_symbol(symbol, trades):
    frames = {}
    for tf, minutes in TIMEFRAMES.items():
        frames[tf] = trades_to_candles(trades, minutes)

    live = {}
    for tf, df in frames.items():
        if len(df) >= 80:
            live[tf] = technical_state(df)
        else:
            live[tf] = None

    oos_1h = oos_backtest(frames["1H"])
    oos_4h = oos_backtest(frames["4H"])

    confirmations = []
    for tf in ("15m", "1H", "4H"):
        s = live.get(tf)
        if s is not None and s["score"] >= 55:
            confirmations.append(tf)

    available_mtf = [tf for tf in ("15m", "1H", "4H") if live.get(tf) is not None]
    mtf_ok = len(available_mtf) == 3 and len(confirmations) >= 2

    stats_ok = (
        oos_1h["available"]
        and oos_1h["trades"] >= 30
        and np.isfinite(oos_1h["pf"])
        and oos_1h["pf"] > 1
        and oos_1h["expectancy"] > 0
        and oos_1h["tp1"] >= 50
    )

    current = live.get("15m") or live.get("1H")

    if current is None:
        decision = "داده ناکافی"
    elif not oos_1h["available"] or oos_1h["trades"] < 30:
        decision = "عدم معامله"
    elif not stats_ok:
        decision = "عدم معامله"
    elif not mtf_ok:
        decision = "صبر"
    elif current["score"] < 55:
        decision = "صبر"
    else:
        decision = "معامله"

    plan = entry_plan(current) if current else None

    return {
        "frames": frames,
        "live": live,
        "oos_1h": oos_1h,
        "oos_4h": oos_4h,
        "stats_ok": stats_ok,
        "mtf_ok": mtf_ok,
        "decision": decision,
        "plan": plan,
    }


# ============================================================
# ACCOUNT — READ ONLY
# ============================================================

def account_status():
    if not API_KEY or not API_SECRET:
        return "کلید API تنظیم نشده"

    try:
        data = api_get("/r/api/v1/account", signed=True)
        return data
    except Exception as e:
        return f"خطا: {e}"


# ============================================================
# UI
# ============================================================

st.title("Tabdeal Crypto Signal Engine")
st.caption("Tabdeal-only • Read-only • بدون سفارش واقعی • بدون داده ساختگی")

try:
    markets = extract_markets(exchange_info())
except Exception as e:
    st.error(f"دریافت بازارهای تبدیل ناموفق بود: {e}")
    st.stop()

symbols = sorted(markets.keys())

if not symbols:
    st.error("هیچ بازار USDT فعالی از Tabdeal دریافت نشد.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["تحلیل ارز", "اسکن بازار", "وضعیت داده/API"])

with tab1:
    selected = st.selectbox("ارز", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)

    try:
        trades = recent_trades(selected, 1000)
    except Exception as e:
        st.error(f"دریافت معاملات {selected} ناموفق بود: {e}")
        trades = pd.DataFrame()

    if trades.empty:
        st.warning(
            f"برای {selected} داده معامله‌ای از API تبدیل دریافت نشد. "
            f"شناسه ارسالی به Tabdeal: {to_tabdeal_symbol(selected)}"
        )
    else:
        result = analyze_symbol(selected, trades)
        latest = trades["time"].max()
        coverage = trades["time"].max() - trades["time"].min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("تصمیم نهایی", result["decision"])
        c2.metric("قیمت", money(trades["price"].iloc[-1]))
        c3.metric("تعداد معاملات خام", f"{len(trades):,}")
        c4.metric("پوشش داده", str(coverage).split(".")[0])

        if result["plan"]:
            p = result["plan"]
            st.subheader("پلن معامله")
            cols = st.columns(5)
            cols[0].metric("Entry", money(p["entry"]))
            cols[1].metric("SL", money(p["sl"]))
            cols[2].metric("TP1", money(p["tp1"]))
            cols[3].metric("TP2", money(p["tp2"]))
            cols[4].metric("TP3", money(p["tp3"]))

        st.subheader("آمار OOS")
        o = result["oos_1h"]
        cols = st.columns(5)
        cols[0].metric("OOS معاملات", str(o["trades"]))
        cols[1].metric("Profit Factor", "-" if not np.isfinite(o["pf"]) else f"{o['pf']:.2f}")
        cols[2].metric("Expectancy", "-" if not np.isfinite(o["expectancy"]) else f"{o['expectancy']:.3f}R")
        cols[3].metric("TP1", "-" if not np.isfinite(o["tp1"]) else pct(o["tp1"]))
        cols[4].metric("Max DD", "-" if not np.isfinite(o["dd"]) else f"{o['dd']:.1f}R")

        st.write("گیت آماری:", "قبول" if result["stats_ok"] else "رد")
        st.write("تأیید چندتایم‌فریمی:", "قبول" if result["mtf_ok"] else "رد")

        rows = []
        for tf in TIMEFRAMES:
            df = result["frames"][tf]
            s = result["live"][tf]
            rows.append({
                "تایم‌فریم": tf,
                "کندل": len(df),
                "وضعیت": "آماده" if s else "داده ناکافی",
                "امتیاز": "-" if s is None else round(s["score"], 1),
                "RSI": "-" if s is None else round(s["rsi"], 1),
                "ADX": "-" if s is None else round(s["adx"], 1),
                "روند": "-" if s is None else ("صعودی" if s["score"] >= 55 else "نزولی" if s["score"] <= -35 else "خنثی"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if result["plan"] and result["decision"] == "معامله":
            st.success("شرایط آماری + تکنیکال + چندتایم‌فریمی تأیید شده است.")
        elif result["decision"] == "صبر":
            st.info("لبه آماری/داده موجود است، اما ورود چندتایم‌فریمی هنوز تأیید نشده.")
        else:
            st.warning("در شرایط فعلی معامله تأیید نمی‌شود.")

with tab2:
    st.write(f"تعداد بازارهای USDT: **{len(symbols)}**")
    limit = st.slider("تعداد ارز برای اسکن", 5, min(100, len(symbols)), min(20, len(symbols)))

    if st.button("شروع اسکن"):
        rows = []
        progress = st.progress(0)

        for n, symbol in enumerate(symbols[:limit], 1):
            try:
                tr = recent_trades(symbol, 1000)
                if tr.empty:
                    rows.append({
                        "ارز": symbol,
                        "تصمیم": "داده ناکافی",
                        "امتیاز": "-",
                        "OOS": 0,
                        "PF": "-",
                        "TP1": "-",
                    })
                else:
                    r = analyze_symbol(symbol, tr)
                    s = r["live"].get("15m") or r["live"].get("1H")
                    o = r["oos_1h"]
                    rows.append({
                        "ارز": symbol,
                        "تصمیم": r["decision"],
                        "امتیاز": "-" if s is None else round(s["score"], 1),
                        "OOS": o["trades"],
                        "PF": "-" if not np.isfinite(o["pf"]) else round(o["pf"], 2),
                        "TP1": "-" if not np.isfinite(o["tp1"]) else f"{o['tp1']:.1f}%",
                    })
            except Exception:
                rows.append({
                    "ارز": symbol,
                    "تصمیم": "خطا",
                    "امتیاز": "-",
                    "OOS": 0,
                    "PF": "-",
                    "TP1": "-",
                })
            progress.progress(n / limit)

        scan = pd.DataFrame(rows)
        if "امتیاز" in scan:
            scan["_sort"] = pd.to_numeric(scan["امتیاز"], errors="coerce")
            scan = scan.sort_values("_sort", ascending=False, na_position="last").drop(columns="_sort")
        st.dataframe(scan, use_container_width=True, hide_index=True)

with tab3:
    st.write("**منبع بازار:** فقط Tabdeal")
    st.write("**Fallback صرافی دیگر:** ندارد")
    st.write("**ارسال سفارش:** غیرفعال")
    st.write("**کارمزد در بک‌تست:** لحاظ نمی‌شود")

    if API_KEY and API_SECRET:
        st.success("TABDEAL_API_KEY و TABDEAL_API_SECRET در Environment Variables موجود هستند.")
        if st.button("بررسی اتصال حساب (فقط خواندنی)"):
            st.json(account_status())
    else:
        st.warning("TABDEAL_API_KEY و TABDEAL_API_SECRET تنظیم نشده‌اند.")

    st.info(
        "این نسخه تاریخچه‌ای که Tabdeal ارائه نمی‌کند تولید نمی‌کند. "
        "بنابراین ممکن است OOS برای بعضی تایم‌فریم‌ها «داده ناکافی» باشد؛ "
        "در این حالت درصد موفقیت ساختگی نمایش داده نمی‌شود."
    )
