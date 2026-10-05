import os
import time
import hmac
import hashlib
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_DOWN

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
RECV_WINDOW = int(os.getenv("TABDEAL_RECV_WINDOW", "5000"))
AUTO_LIVE_ENV = os.getenv("TABDEAL_LIVE_TRADING", "false").lower() == "true"
AUTO_TRADE_USDT = float(os.getenv("AUTO_TRADE_USDT", "0"))
AUTO_MAX_POSITIONS = int(os.getenv("AUTO_MAX_POSITIONS", "1"))
AUTO_SCAN_INTERVAL = int(os.getenv("AUTO_SCAN_INTERVAL", "60"))
AUTO_SCAN_WORKERS = int(os.getenv("AUTO_SCAN_WORKERS", "6"))

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


def _signed_params(params):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("TABDEAL_API_KEY / TABDEAL_API_SECRET تنظیم نشده است.")
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = RECV_WINDOW
    query = urlencode(p)
    p["signature"] = hmac.new(
        API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    return p


def api_post(path, data=None, signed=False):
    params = _signed_params(data) if signed else dict(data or {})
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    r = SESSION.post(
        BASE.rstrip("/") + path,
        data=params,
        headers=headers,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def api_delete(path, params=None, signed=False):
    p = _signed_params(params) if signed else dict(params or {})
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    r = SESSION.delete(
        BASE.rstrip("/") + path,
        params=p,
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

    # The Tabdeal public trades feed is recent-only, so higher timeframes
    # often do not have enough real candles. Instead of requiring 1H/4H
    # specifically, use the best available lower timeframe with >=180 real
    # candles for OOS validation. No synthetic history is created.
    stat_candidates = [
        tf for tf in ("5m", "15m", "30m", "1H", "2H", "4H")
        if len(frames[tf]) >= 180
    ]
    stat_tf = stat_candidates[0] if stat_candidates else None
    oos_stats = oos_backtest(frames[stat_tf]) if stat_tf else {
        "available": False,
        "trades": 0,
        "pf": np.nan,
        "expectancy": np.nan,
        "tp1": np.nan,
        "dd": np.nan,
        "reason": "هیچ تایم‌فریمی حداقل 180 کندل واقعی ندارد.",
    }

    # Adaptive MTF: use available real timeframes among 5m/15m/30m/1H/2H.
    # At least 3 must be available and at least 2 must confirm the setup.
    mtf_pool = ("5m", "15m", "30m", "1H", "2H")
    available_mtf = [tf for tf in mtf_pool if live.get(tf) is not None]
    confirmations = [tf for tf in available_mtf if live[tf]["score"] >= 55]
    mtf_ok = len(available_mtf) >= 3 and len(confirmations) >= 2

    stats_ok = (
        oos_stats["available"]
        and oos_stats["trades"] >= 30
        and np.isfinite(oos_stats["pf"])
        and oos_stats["pf"] > 1
        and oos_stats["expectancy"] > 0
        and oos_stats["tp1"] >= 50
    )

    # Prefer 15m for the live trigger; fall back to 5m/30m/1H if unavailable.
    current = next((live.get(tf) for tf in ("15m", "5m", "30m", "1H") if live.get(tf) is not None), None)

    if current is None:
        decision = "داده ناکافی"
    elif not oos_stats["available"] or oos_stats["trades"] < 30:
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
        "oos_1h": oos_stats,  # backward-compatible UI key
        "stat_tf": stat_tf,
        "oos_4h": oos_backtest(frames["4H"]) if len(frames["4H"]) >= 180 else {
            "available": False, "trades": 0, "pf": np.nan,
            "expectancy": np.nan, "tp1": np.nan, "dd": np.nan,
            "reason": f"{len(frames["4H"])} کندل واقعی موجود است.",
        },
        "stats_ok": stats_ok,
        "mtf_ok": mtf_ok,
        "decision": decision,
        "plan": plan,
    }


# ============================================================
# LIVE AUTO TRADING — SPOT LONG ONLY
# ============================================================

def _decimal(value):
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _floor_step(value, step):
    v = _decimal(value)
    st = _decimal(step)
    if st <= 0:
        return v
    return (v / st).to_integral_value(rounding=ROUND_DOWN) * st


def _market_rules(market):
    filters = market.get("filters", []) if isinstance(market, dict) else []
    if isinstance(filters, dict):
        filters = list(filters.values())
    out = {
        "min_qty": Decimal("0"),
        "step_size": Decimal("0.00000001"),
        "min_notional": Decimal("0"),
        "tick_size": Decimal("0.00000001"),
    }
    for f in filters:
        if not isinstance(f, dict):
            continue
        typ = str(f.get("filterType", "")).upper()
        if typ in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
            if _decimal(f.get("minQty", 0)) > out["min_qty"]:
                out["min_qty"] = _decimal(f.get("minQty", 0))
            if _decimal(f.get("stepSize", 0)) > 0:
                out["step_size"] = _decimal(f.get("stepSize"))
        elif typ == "MIN_NOTIONAL":
            out["min_notional"] = max(out["min_notional"], _decimal(f.get("minNotional", 0)))
        elif typ == "PRICE_FILTER":
            if _decimal(f.get("tickSize", 0)) > 0:
                out["tick_size"] = _decimal(f.get("tickSize"))
    return out


def account_info():
    return api_get("/r/api/v1/account", signed=True)


def usdt_balance():
    data = account_info()
    assets = data.get("balances", data.get("assets", [])) if isinstance(data, dict) else []
    if isinstance(assets, dict):
        assets = [assets]
    for a in assets:
        if str(a.get("asset", "")).upper() == "USDT":
            return float(a.get("free", a.get("available", a.get("availableBalance", 0))) or 0)
    return 0.0


def bot_open_orders():
    try:
        data = api_get("/r/api/v1/openOrderList", signed=True)
        if isinstance(data, dict):
            data = data.get("orders", data.get("data", data.get("result", [])))
        if not isinstance(data, list):
            return []
        result = []
        for item in data:
            cid = str(item.get("listClientOrderId") or item.get("clientOrderId") or "")
            if cid.startswith("AUTO_"):
                result.append(item)
        return result
    except Exception:
        return []


def place_market_buy(symbol, quote_usdt, market):
    rules = _market_rules(market)
    price = float(recent_trades(symbol, 100)["price"].iloc[-1])
    qty = _floor_step(Decimal(str(quote_usdt)) / Decimal(str(price)), rules["step_size"])
    if qty < rules["min_qty"]:
        raise RuntimeError(f"حجم سفارش {qty} کمتر از حداقل {rules['min_qty']} است.")
    if qty * Decimal(str(price)) < rules["min_notional"]:
        raise RuntimeError(f"ارزش سفارش کمتر از حداقل {rules['min_notional']} USDT است.")
    cid = f"AUTO_BUY_{norm_symbol(symbol)}_{int(time.time())}"
    return api_post(
        "/api/v1/order",
        {
            "tabdealSymbol": to_tabdeal_symbol(symbol),
            "newClientOrderId": cid,
            "side": "BUY",
            "type": "MARKET",
            "quantity": format(qty, "f"),
        },
        signed=True,
    )


def place_oco_sell(symbol, quantity, plan, market):
    rules = _market_rules(market)
    qty = _floor_step(Decimal(str(quantity)), rules["step_size"])
    tp = _floor_step(Decimal(str(plan["tp1"])), rules["tick_size"])
    sl = _floor_step(Decimal(str(plan["sl"])), rules["tick_size"])
    # Stop-limit must be slightly below stop trigger for a SELL OCO.
    stop_limit = _floor_step(sl * Decimal("0.998"), rules["tick_size"])
    if qty < rules["min_qty"]:
        raise RuntimeError("مقدار پرشده برای OCO کمتر از حداقل سفارش است.")
    cid = f"AUTO_OCO_{norm_symbol(symbol)}_{int(time.time())}"
    return api_post(
        "/api/v1/order/oco",
        {
            "tabdealSymbol": to_tabdeal_symbol(symbol),
            "listClientOrderId": cid,
            "limitClientOrderId": cid + "_TP",
            "stopClientOrderId": cid + "_SL",
            "side": "SELL",
            "quantity": format(qty, "f"),
            "price": format(tp, "f"),
            "stopPrice": format(sl, "f"),
            "stopLimitPrice": format(stop_limit, "f"),
        },
        signed=True,
    )


def emergency_market_sell(symbol, quantity):
    return api_post(
        "/api/v1/order",
        {
            "tabdealSymbol": to_tabdeal_symbol(symbol),
            "newClientOrderId": f"AUTO_EMERGENCY_{norm_symbol(symbol)}_{int(time.time())}",
            "side": "SELL",
            "type": "MARKET",
            "quantity": str(quantity),
        },
        signed=True,
    )


def execute_auto_trade(symbol, analysis, market, amount_usdt):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API Key/Secret تنظیم نشده است.")
    if not AUTO_LIVE_ENV:
        raise RuntimeError("TABDEAL_LIVE_TRADING=true فعال نشده است.")
    if analysis.get("decision") != "معامله":
        raise RuntimeError("این ارز توسط گیت استراتژی تأیید نشده است.")

    balance = usdt_balance()
    spend = min(float(amount_usdt), balance * 0.95)
    if spend <= 0:
        raise RuntimeError("موجودی آزاد USDT کافی نیست.")

    order = place_market_buy(symbol, spend, market)
    executed = float(order.get("executedQty", order.get("origQty", 0)) or 0)
    if executed <= 0:
        raise RuntimeError(f"خرید انجام نشد یا executedQty صفر است: {order}")

    try:
        oco = place_oco_sell(symbol, executed, analysis["plan"], market)
    except Exception:
        # Never leave a live position without protection if the OCO placement fails.
        try:
            emergency_market_sell(symbol, executed)
        except Exception as sell_error:
            raise RuntimeError(f"OCO ناموفق بود و فروش اضطراری هم ناموفق شد: {sell_error}")
        raise RuntimeError("OCO ناموفق بود؛ برای جلوگیری از پوزیشن بدون حدضرر، فروش اضطراری انجام شد.")

    return {"buy": order, "oco": oco, "spent_usdt": spend, "qty": executed}


def scan_one(symbol):
    try:
        tr = recent_trades(symbol, 1000)
        if tr.empty:
            return None
        r = analyze_symbol(symbol, tr)
        s = r["live"].get("15m") or r["live"].get("1H")
        if r["decision"] != "معامله" or not s:
            return None
        return {
            "symbol": symbol,
            "analysis": r,
            "score": float(s["score"]),
            "price": float(s["price"]),
        }
    except Exception:
        return None


def auto_scan_all(symbols):
    candidates = []
    workers = max(1, min(AUTO_SCAN_WORKERS, 10))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(scan_one, s): s for s in symbols}
        for f in as_completed(futures):
            item = f.result()
            if item:
                candidates.append(item)
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


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

st.title("Tabdeal Crypto Auto Trader")
st.caption("Tabdeal-only • اسکن خودکار • معاملات Spot فقط Long • گیت آماری و تکنیکال بدون داده ساختگی")

try:
    markets = extract_markets(exchange_info())
except Exception as e:
    st.error(f"دریافت بازارهای تبدیل ناموفق بود: {e}")
    st.stop()

symbols = sorted(markets.keys())

if not symbols:
    st.error("هیچ بازار USDT فعالی از Tabdeal دریافت نشد.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["تحلیل ارز", "اسکن بازار", "معاملات خودکار", "وضعیت داده/API"])

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
        st.caption(f"تایم‌فریم آماری انتخاب‌شده: {result.get("stat_tf") or "ندارد"}")
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
    st.subheader("معاملات خودکار")
    st.warning("فعال‌سازی این بخش می‌تواند سفارش واقعی در حساب Tabdeal ارسال کند.")

    live_confirm = st.checkbox("تأیید می‌کنم که معاملات واقعی را می‌خواهم")
    live_ready = live_confirm and AUTO_LIVE_ENV and bool(API_KEY and API_SECRET)

    c1, c2, c3 = st.columns(3)
    c1.metric("Live Trading ENV", "فعال" if AUTO_LIVE_ENV else "خاموش")
    c3.metric("حداکثر پوزیشن", str(AUTO_MAX_POSITIONS))

    default_amount = max(1.0, AUTO_TRADE_USDT)
    trade_amount = st.number_input(
        "مبلغ هر معامله (USDT)",
        min_value=1.0,
        max_value=100000.0,
        value=float(default_amount),
        step=1.0,
        key="auto_trade_amount",
        help="مبلغ هر خرید خودکار؛ قابل تغییر از داخل برنامه.",
    )
    c2.metric("مبلغ فعلی هر معامله", f"{trade_amount:.2f} USDT")

    if trade_amount <= 0:
        st.error("مبلغ معامله باید بیشتر از صفر باشد.")
    if not API_KEY or not API_SECRET:
        st.error("TABDEAL_API_KEY و TABDEAL_API_SECRET تنظیم نشده‌اند.")
    if not AUTO_LIVE_ENV:
        st.info("برای فعال‌سازی واقعی، TABDEAL_LIVE_TRADING=true را در Environment Variables قرار بده.")

    st.write("منطق اجرا: اسکن تمام بازارهای USDT → انتخاب فقط «معامله» → خرید Market → ثبت OCO برای TP1 و SL. مبلغ معامله از همین صفحه قابل تغییر است.")
    st.write("اگر OOS/گیت آماری/تأیید چندتایم‌فریمی رد شود، خرید انجام نمی‌شود.")

    if live_ready and trade_amount > 0:
        @st.fragment(run_every=max(30, AUTO_SCAN_INTERVAL))
        def auto_engine():
            if not live_ready:
                return
            active = bot_open_orders()
            if len(active) >= AUTO_MAX_POSITIONS:
                st.info(f"پوزیشن/سفارش خودکار فعال: {len(active)} — اسکن بدون سفارش جدید.")
                return
            with st.spinner("در حال اسکن بازارهای USDT..."):
                candidates = auto_scan_all(symbols)
            if not candidates:
                st.info("در این چرخه هیچ ارز با تمام شروط ورود تأیید نشد.")
                return
            best = candidates[0]
            st.success(f"کاندید تأییدشده: {best['symbol']} | امتیاز {best['score']:.1f} | قیمت {money(best['price'])}")
            try:
                result = execute_auto_trade(best["symbol"], best["analysis"], markets[best["symbol"]], trade_amount)
                st.success(f"خرید خودکار و OCO ارسال شد: {best['symbol']}")
                st.json(result)
            except Exception as e:
                st.error(f"سفارش خودکار ارسال نشد: {e}")
            st.dataframe(pd.DataFrame([{
                "ارز": x["symbol"], "امتیاز": round(x["score"], 1), "قیمت": money(x["price"])
            } for x in candidates[:10]]), use_container_width=True, hide_index=True)
        auto_engine()
    else:
        st.info("معاملات خودکار هنوز مسلح نشده است. پس از تنظیم ENV و تأیید بالا، موتور اسکن فعال می‌شود.")

with tab4:
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
