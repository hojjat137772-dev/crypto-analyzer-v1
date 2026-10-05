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
# AUTO TRADING — SPOT ONLY / EXPLICITLY ENABLED
# ============================================================
# Real orders are OFF by default. The user must explicitly enable
# Auto Trading in the UI. The engine uses the same final decision
# gate shown above and never trades when the decision is not "معامله".

AUTO_DEFAULT_USDT = float(os.getenv("AUTO_TRADE_USDT", "10"))
AUTO_MIN_USDT = float(os.getenv("AUTO_TRADE_MIN_USDT", "5"))
AUTO_COOLDOWN = int(os.getenv("AUTO_TRADE_COOLDOWN_SEC", "900"))
RECV_WINDOW = int(os.getenv("TABDEAL_RECV_WINDOW", "5000"))


def signed_request(method, path, params=None):
    params = dict(params or {})
    if not API_KEY or not API_SECRET:
        raise RuntimeError("TABDEAL_API_KEY / TABDEAL_API_SECRET تنظیم نشده است.")
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW
    query = urlencode(params)
    params["signature"] = hmac.new(
        API_SECRET.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    headers = {"X-MBX-APIKEY": API_KEY}
    r = SESSION.request(
        method.upper(),
        BASE.rstrip("/") + path,
        params=params if method.upper() in {"GET", "DELETE"} else None,
        data=params if method.upper() in {"POST", "PUT"} else None,
        headers=headers,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def spot_order(params):
    return signed_request("POST", "/api/v1/order", params)


def spot_oco(params):
    return signed_request("POST", "/api/v1/order/oco", params)


def spot_open_orders(tabdeal_symbol):
    return signed_request(
        "GET", "/r/api/v1/openOrders", {"tabdealSymbol": tabdeal_symbol}
    )


def account_info():
    return api_get("/r/api/v1/account", signed=True)


def extract_balances(data):
    if isinstance(data, dict):
        for key in ("balances", "balance", "assets"):
            v = data.get(key)
            if isinstance(v, list):
                return v
        for v in data.values():
            if isinstance(v, dict):
                b = extract_balances(v)
                if b:
                    return b
            elif isinstance(v, list):
                if any(isinstance(x, dict) and ("asset" in x or "currency" in x) for x in v):
                    return v
    elif isinstance(data, list):
        return data
    return []


def free_balance(data, asset):
    asset = asset.upper()
    for b in extract_balances(data):
        if not isinstance(b, dict):
            continue
        a = str(b.get("asset", b.get("currency", ""))).upper()
        if a != asset:
            continue
        for key in ("free", "available", "availableBalance", "freeBalance", "balance"):
            if b.get(key) is not None:
                try:
                    return float(b[key])
                except Exception:
                    pass
    return 0.0


def _find_filter_obj(market, filter_type):
    if not isinstance(market, dict):
        return None
    filters = market.get("filters")
    if isinstance(filters, list):
        for f in filters:
            if isinstance(f, dict) and str(f.get("filterType", "")).upper() == filter_type:
                return f
    return None


def _decimal_places(step):
    s = f"{float(step):.16f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def floor_step(value, step):
    step = float(step or 0)
    if step <= 0:
        return float(value)
    return float(np.floor(float(value) / step + 1e-12) * step)


def format_step(value, step):
    places = _decimal_places(step)
    return f"{float(value):.{places}f}"


def market_rules(market):
    lot = _find_filter_obj(market, "LOT_SIZE") or {}
    price = _find_filter_obj(market, "PRICE_FILTER") or {}
    min_notional = _find_filter_obj(market, "MIN_NOTIONAL") or _find_filter_obj(market, "NOTIONAL") or {}
    return {
        "step": float(lot.get("stepSize", market.get("stepSize", 0)) or 0),
        "min_qty": float(lot.get("minQty", market.get("minQty", 0)) or 0),
        "tick": float(price.get("tickSize", market.get("tickSize", 0)) or 0),
        "min_notional": float(min_notional.get("minNotional", min_notional.get("notional", 0)) or 0),
    }


def build_order_qty(symbol, market, price, usdt_amount, balance):
    quote = str(market.get("quoteAsset", "USDT")).upper() or "USDT"
    if quote != "USDT":
        raise RuntimeError(f"بازار {symbol} جفت USDT نیست.")
    spend = min(float(usdt_amount), float(balance))
    rules = market_rules(market)
    qty = spend / float(price)
    qty = floor_step(qty, rules["step"])
    if qty <= 0 or (rules["min_qty"] > 0 and qty < rules["min_qty"]):
        raise RuntimeError("مقدار سفارش از حداقل quantity بازار کمتر است.")
    if rules["min_notional"] > 0 and qty * price < rules["min_notional"]:
        raise RuntimeError("ارزش سفارش از حداقل Notional بازار کمتر است.")
    return qty, rules


def place_protective_oco(symbol, market, qty, plan):
    tabdeal_symbol = market.get("tabdealSymbol") or to_tabdeal_symbol(symbol)
    rules = market_rules(market)
    step = rules["step"]
    tick = rules["tick"]
    if tick <= 0:
        tick = max(plan["entry"] * 1e-8, 1e-12)

    # Split the filled position into three equal exit legs. Each leg has
    # its own OCO so the remaining position stays protected after a TP fills.
    q1 = floor_step(qty / 3.0, step)
    q2 = floor_step(qty / 3.0, step)
    q3 = floor_step(qty - q1 - q2, step)
    legs = [(q1, plan["tp1"]), (q2, plan["tp2"]), (q3, plan["tp3"])]
    placed = []

    for idx, (leg_qty, tp) in enumerate(legs, 1):
        if leg_qty <= 0:
            continue
        limit_price = floor_step(tp, tick)
        stop_price = floor_step(plan["sl"], tick)
        stop_limit = floor_step(stop_price * 0.998, tick)
        if not (limit_price > stop_price > stop_limit > 0):
            raise RuntimeError("قیمت‌های TP/SL برای OCO معتبر نیستند.")
        params = {
            "tabdealSymbol": tabdeal_symbol,
            "symbol": symbol,
            "listClientOrderId": f"auto_oco_{int(time.time())}_{idx}",
            "limitClientOrderId": f"auto_tp_{int(time.time())}_{idx}",
            "stopClientOrderId": f"auto_sl_{int(time.time())}_{idx}",
            "side": "SELL",
            "quantity": format_step(leg_qty, step),
            "price": format_step(limit_price, tick),
            "stopPrice": format_step(stop_price, tick),
            "stopLimitPrice": format_step(stop_limit, tick),
        }
        placed.append(spot_oco(params))
    return placed


def execute_auto_trade(symbol, market, result, usdt_amount):
    if result.get("decision") != "معامله":
        return {"ok": False, "message": "سیگنال نهایی «معامله» نیست."}
    if not result.get("plan"):
        return {"ok": False, "message": "پلن ورود/خروج موجود نیست."}
    if not API_KEY or not API_SECRET:
        return {"ok": False, "message": "کلید API تنظیم نشده است."}

    tabdeal_symbol = market.get("tabdealSymbol") or to_tabdeal_symbol(symbol)
    try:
        open_orders = spot_open_orders(tabdeal_symbol)
        if isinstance(open_orders, list) and open_orders:
            return {"ok": False, "message": "برای این ارز سفارش باز وجود دارد؛ معامله جدید ارسال نشد."}

        account = account_info()
        balance = free_balance(account, "USDT")
        if balance < max(AUTO_MIN_USDT, 0.01):
            return {"ok": False, "message": f"موجودی آزاد USDT کافی نیست: {balance:.4f}"}

        entry = float(result["plan"]["entry"])
        qty, rules = build_order_qty(symbol, market, entry, usdt_amount, balance)

        order = spot_order({
            "tabdealSymbol": tabdeal_symbol,
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quantity": format_step(qty, rules["step"]),
            "newClientOrderId": f"auto_buy_{int(time.time())}",
        })

        executed_qty = float(order.get("executedQty", 0) or 0)
        if executed_qty <= 0:
            return {"ok": False, "message": "سفارش خرید ارسال شد اما executedQty صفر است؛ سفارش‌های خروج ایجاد نشدند.", "order": order}

        quote_qty = float(order.get("cummulativeQuoteQty", order.get("cumulativeQuoteQty", 0)) or 0)
        avg_entry = quote_qty / executed_qty if quote_qty > 0 else entry
        plan = dict(result["plan"])
        if avg_entry > 0 and abs(avg_entry - entry) / entry > 0.0001:
            risk = max(avg_entry - plan["sl"], avg_entry * 0.001)
            plan = {
                "entry": avg_entry,
                "sl": avg_entry - risk,
                "tp1": avg_entry + risk,
                "tp2": avg_entry + 1.8 * risk,
                "tp3": avg_entry + 2.6 * risk,
            }

        try:
            oco = place_protective_oco(symbol, market, executed_qty, plan)
        except Exception as protection_error:
            return {
                "ok": False,
                "message": f"خرید انجام شد ولی ثبت سفارش‌های حفاظتی OCO ناموفق بود: {protection_error}",
                "order": order,
            }

        return {
            "ok": True,
            "message": "خرید بازار انجام شد و سه OCO برای TP1/TP2/TP3 + SL ثبت شد.",
            "order": order,
            "oco": oco,
            "plan": plan,
            "qty": executed_qty,
        }
    except Exception as e:
        return {"ok": False, "message": str(e)}


def auto_trade_cycle(symbol, market, result, usdt_amount):
    now = time.time()
    key = f"auto_last_{symbol}"
    last = float(st.session_state.get(key, 0))
    if now - last < AUTO_COOLDOWN:
        return {"ok": False, "message": f"Cooldown فعال است؛ {int(AUTO_COOLDOWN - (now-last))} ثانیه باقی مانده."}
    out = execute_auto_trade(symbol, market, result, usdt_amount)
    if out.get("ok"):
        st.session_state[key] = now
    return out


# ============================================================
# UI
# ============================================================

st.title("Tabdeal Crypto Signal Engine")
st.caption("Tabdeal-only • Spot • سفارش خودکار با فعال‌سازی صریح • بدون داده ساختگی")

try:
    markets = extract_markets(exchange_info())
except Exception as e:
    st.error(f"دریافت بازارهای تبدیل ناموفق بود: {e}")
    st.stop()

symbols = sorted(markets.keys())

if not symbols:
    st.error("هیچ بازار USDT فعالی از Tabdeal دریافت نشد.")
    st.stop()

if "auto_enabled" not in st.session_state:
    st.session_state.auto_enabled = False

control1, control2, control3 = st.columns(3)
with control1:
    auto_enabled = st.toggle("فعال‌سازی معامله خودکار", value=st.session_state.auto_enabled, key="auto_enabled")
with control2:
    auto_usdt = st.number_input("مبلغ هر معامله (USDT)", min_value=max(1.0, AUTO_MIN_USDT), value=max(AUTO_DEFAULT_USDT, AUTO_MIN_USDT), step=1.0)
with control3:
    st.metric("حالت", "فعال" if auto_enabled else "خاموش")

if auto_enabled:
    st.warning("معامله خودکار فعال است؛ با هر اجرای صفحه، در صورت تصمیم نهایی «معامله» سفارش واقعی ارسال می‌شود. Cooldown از تکرار سریع جلوگیری می‌کند.")

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

        st.subheader("معامله خودکار")
        confirm = st.checkbox("ارسال سفارش واقعی را تأیید می‌کنم.", key=f"confirm_{selected}")
        if result["decision"] == "معامله" and auto_enabled and confirm:
            with st.spinner("در حال بررسی و اجرای معامله خودکار..."):
                trade_result = auto_trade_cycle(selected, markets[selected], result, float(auto_usdt))
            if trade_result.get("ok"):
                st.success(trade_result["message"])
                st.session_state[f"last_trade_message_{selected}"] = trade_result["message"]
            elif "Cooldown" not in trade_result.get("message", ""):
                st.error(trade_result.get("message", "خطای نامشخص"))
        elif result["decision"] == "معامله" and not auto_enabled:
            st.info("سیگنال آماده است، اما معامله خودکار خاموش است.")
        else:
            st.info("تا وقتی تصمیم نهایی «معامله» نباشد، سفارش واقعی ارسال نمی‌شود.")

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
    st.write("**ارسال سفارش:** فقط با فعال‌سازی معامله خودکار و تأیید دستی")
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
