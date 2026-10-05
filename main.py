
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Crypto Analyzer Pro — Robust / Backtestable Edition
# ============================================================
# IMPORTANT:
# - Spot mode is long-only by default. Short is disabled unless Futures mode is selected.
# - "Confidence" is calibrated from historical out-of-sample-style rolling outcomes,
#   not from an arbitrary score-to-percent formula.
# - Forecast percentages are NOT fabricated from the score.
# - Signals are generated only from CLOSED candles.
# - Backtest enters on the NEXT candle open and includes configurable fee + slippage.
# - If TP and SL are both touched in the same candle, the conservative assumption is SL first.
# ============================================================

st.set_page_config(
    page_title="Crypto Analyzer Pro",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TIMEFRAMES = {
    "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h",
    "12H": "12h", "1D": "1d", "3D": "3d", "1W": "1w"
}

BINANCE = "https://api.binance.com"
BINANCE_DATA = "https://data-api.binance.vision"
TABDEAL = "https://api1.tabdeal.org"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoAnalyzerPro/2.0"})

# Trading assumptions. Change these in the sidebar.
DEFAULT_FEE_PCT = 0.10
DEFAULT_SLIPPAGE_PCT = 0.05
DEFAULT_HORIZON_BARS = 24

TABDEAL_MARKET_ENDPOINTS = [
    "/r/api/v1/exchangeInfo",
    "/api/v1/exchangeInfo",
    "/v1/market/symbols",
    "/v1/markets",
    "/api/v1/markets",
]


# ============================================================
# HTTP
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_json(url, params=None, timeout=12):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def flatten_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from flatten_dicts(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from flatten_dicts(x)


def normalize_symbol(s):
    return str(s).upper().replace("-", "").replace("_", "").replace("/", "")


def base_asset(symbol):
    s = normalize_symbol(symbol)
    return s[:-4] if s.endswith("USDT") else s


def money(value):
    try:
        v = float(value)
        if not np.isfinite(v):
            return "-"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        if abs(v) >= 1:
            return f"{v:,.4f}"
        if abs(v) >= 0.01:
            return f"{v:,.6f}"
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return "-"


# ============================================================
# MARKET UNIVERSE
# ============================================================
def tabdeal_markets():
    symbols = set()
    for ep in TABDEAL_MARKET_ENDPOINTS:
        data = get_json(TABDEAL + ep, timeout=15)
        if not data:
            continue
        for d in flatten_dicts(data):
            if not isinstance(d, dict):
                continue
            raw = None
            for key in ("symbol", "tabdealSymbol", "market", "pair"):
                if isinstance(d.get(key), str) and d.get(key).strip():
                    raw = d[key]
                    break
            if not raw:
                continue
            s = normalize_symbol(raw)
            status = str(d.get("status", "TRADING")).upper()
            quote = str(d.get("quoteAsset", "")).upper()
            if (quote == "USDT" or s.endswith("USDT")) and status in (
                "TRADING", "ACTIVE", "ENABLED", ""
            ):
                if s.endswith("USDT"):
                    symbols.add(s)
        if symbols and "exchangeInfo" in ep:
            break
    return sorted(symbols)


@st.cache_data(ttl=180, show_spinner=False)
def get_universe():
    syms = tabdeal_markets()
    if syms:
        return syms

    # Dynamic Binance fallback.
    data = get_json(BINANCE_DATA + "/api/v3/exchangeInfo", timeout=15)
    if isinstance(data, dict):
        for d in data.get("symbols", []):
            if (
                str(d.get("status", "")).upper() == "TRADING"
                and str(d.get("quoteAsset", "")).upper() == "USDT"
            ):
                s = normalize_symbol(d.get("symbol", ""))
                if s.endswith("USDT"):
                    syms.append(s)
    return sorted(set(syms))


# ============================================================
# KLINES
# ============================================================
def rows_to_df(rows):
    if not isinstance(rows, list) or len(rows) < 40:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(rows)
        df = df.iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        # Binance API normally returns ms. Public archive/API can evolve, so infer.
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


@st.cache_data(ttl=30, show_spinner=False)
def binance_klines(symbol, interval, limit=1000, end_time=None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(int(limit), 1000),
    }
    if end_time is not None:
        params["endTime"] = int(end_time)
    data = get_json(BINANCE_DATA + "/api/v3/klines", params, timeout=15)
    return rows_to_df(data)


def fetch_klines(symbol, interval, limit=1000):
    return binance_klines(symbol, interval, limit)


def resample_from_1h(df, rule):
    if df.empty:
        return df
    x = df.set_index("time").sort_index()
    out = x.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return out


def get_tf_data(symbol, tf, limit=1000):
    iv = TIMEFRAMES[tf]
    # Binance natively supports all requested intervals.
    df = fetch_klines(symbol, iv, limit)
    if len(df) >= 80:
        return df

    # Fallback aggregation for larger intervals if a provider rejects one.
    if tf in ("2H", "6H", "12H", "3D"):
        base = fetch_klines(symbol, "1h" if tf != "3D" else "1d", 1000)
        rules = {"2H": "2h", "6H": "6h", "12H": "12h", "3D": "3D"}
        if len(base) >= 80:
            return resample_from_1h(base, rules[tf])
    return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close):
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    return line, signal, line - signal


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - pc).abs(),
        (df.low - pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up = df.high.diff()
    down = -df.low.diff()
    plus = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )
    a = atr(df, n).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(20)


def ichimoku(df):
    h9, l9 = df.high.rolling(9).max(), df.low.rolling(9).min()
    h26, l26 = df.high.rolling(26).max(), df.low.rolling(26).min()
    h52, l52 = df.high.rolling(52).max(), df.low.rolling(52).min()
    tenkan = (h9 + l9) / 2
    kijun = (h26 + l26) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (h52 + l52) / 2
    return tenkan, kijun, span_a, span_b


def price_action_score(df):
    if len(df) < 20:
        return 0.0
    c, p = df.iloc[-1], df.iloc[-2]
    rng = max(float(c.high - c.low), 1e-12)
    body = abs(float(c.close - c.open))
    upper = float(c.high - max(c.open, c.close))
    lower = float(min(c.open, c.close) - c.low)
    score = 0
    if c.close > c.open:
        score += 7
    else:
        score -= 7
    if c.close > p.high:
        score += 12
    if c.close < p.low:
        score -= 12
    if lower > body * 1.5 and c.close > c.open:
        score += 7
    if upper > body * 1.5 and c.close < c.open:
        score -= 7
    if body / rng > 0.65:
        score += 5 if c.close > c.open else -5
    return float(np.clip(score, -25, 25))


def structure_levels(df, lookback=120):
    if df.empty:
        return [], []
    x = df.tail(lookback)
    price = float(x.close.iloc[-1])
    high_roll = x.high.rolling(7, center=True).max()
    low_roll = x.low.rolling(7, center=True).min()
    highs = x.high[high_roll.eq(x.high)].dropna().tolist()
    lows = x.low[low_roll.eq(x.low)].dropna().tolist()
    supports = sorted([v for v in lows if v < price], reverse=True)[:4]
    resistances = sorted([v for v in highs if v > price])[:4]
    return [float(v) for v in supports], [float(v) for v in resistances]


# ============================================================
# SINGLE-TF SIGNAL
# ============================================================
def tf_signal(df):
    if len(df) < 80:
        return None

    close = df.close
    r = float(rsi(close).iloc[-1])
    ml, ms, mh = macd(close)
    mac, sig, hist = float(ml.iloc[-1]), float(ms.iloc[-1]), float(mh.iloc[-1])
    at = float(atr(df).iloc[-1])
    ad = float(adx(df).iloc[-1])
    ten, kij, sa, sb = ichimoku(df)
    price = float(close.iloc[-1])

    cloud_top = max(float(sa.iloc[-1]), float(sb.iloc[-1]))
    cloud_bottom = min(float(sa.iloc[-1]), float(sb.iloc[-1]))

    score = 0.0
    reasons = []

    # Trend / Ichimoku
    if price > cloud_top:
        score += 20
        reasons.append("بالای کلود")
    elif price < cloud_bottom:
        score -= 20
        reasons.append("زیر کلود")
    else:
        reasons.append("داخل کلود")

    if float(ten.iloc[-1]) > float(kij.iloc[-1]):
        score += 9
        reasons.append("تنکن بالای کیجون")
    else:
        score -= 9

    # MACD
    if mac > sig:
        score += 13
        reasons.append("MACD صعودی")
    else:
        score -= 13
        reasons.append("MACD نزولی")

    if hist > 0 and hist > float(mh.iloc[-2]):
        score += 5
    elif hist < 0 and hist < float(mh.iloc[-2]):
        score -= 5

    # RSI: avoid buying an already extreme RSI.
    if 52 <= r <= 68:
        score += 9
        reasons.append("RSI مناسب")
    elif 68 < r < 75:
        score += 3
        reasons.append("RSI نسبتاً داغ")
    elif r >= 75:
        score -= 7
        reasons.append("RSI بسیار داغ")
    elif r <= 30:
        score += 2
        reasons.append("اشباع فروش")
    elif r < 45:
        score -= 6

    mom = float((close.iloc[-1] / close.iloc[-13] - 1) * 100)
    score += float(np.clip(mom * 2.0, -12, 12))

    if ad >= 25:
        score += 7 if price > float(kij.iloc[-1]) else -7
    else:
        # Low ADX = less confidence in trend-following signals.
        score *= 0.92

    score += price_action_score(df)

    return {
        "score": float(np.clip(score, -100, 100)),
        "rsi": r,
        "macd": mac,
        "signal": sig,
        "momentum": mom,
        "adx": ad,
        "atr": at,
        "price": price,
        "reasons": reasons,
    }


# ============================================================
# MULTI-TIMEFRAME CURRENT ANALYSIS
# ============================================================
TF_WEIGHTS = {
    "5m": 0.35, "15m": 0.60, "30m": 0.75,
    "1H": 1.10, "2H": 1.25, "4H": 1.55,
    "6H": 1.35, "12H": 1.25, "1D": 1.45,
    "3D": 1.05, "1W": 0.85,
}


def aggregate_score(analyses):
    if not analyses:
        return None
    total_w = sum(TF_WEIGHTS.get(tf, 1.0) for tf in analyses)
    weighted = sum(
        a["score"] * TF_WEIGHTS.get(tf, 1.0)
        for tf, a in analyses.items()
    ) / total_w

    # Agreement between the important timeframes.
    core = [
        analyses[tf]["score"]
        for tf in ("1H", "4H", "1D")
        if tf in analyses
    ]
    agreement_bonus = 0.0
    if len(core) >= 2:
        same_bull = all(x >= 10 for x in core)
        same_bear = all(x <= -10 for x in core)
        if same_bull or same_bear:
            agreement_bonus = 8
        elif any(x > 10 for x in core) and any(x < -10 for x in core):
            agreement_bonus = -8

    final = float(np.clip(weighted + agreement_bonus, -100, 100))
    return final


def position_from_score(score, mode="SPOT"):
    if mode == "SPOT":
        return "لانگ" if score >= 24 else "صبر"
    return "لانگ" if score >= 24 else ("شورت" if score <= -24 else "صبر")


# ============================================================
# HISTORICAL SIGNAL ENGINE FOR CALIBRATION
# ============================================================
def historical_signal_score(df, idx):
    """Signal using only rows <= idx. No future candles are used."""
    if idx < 100:
        return None

    x = df.iloc[:idx + 1].copy()
    a = tf_signal(x)
    if not a:
        return None
    return float(a["score"])


def simulate_trade(df, signal_idx, direction, fee_pct, slippage_pct,
                    horizon_bars=24, rr1=1.6, rr2=2.6):
    """
    Entry = next candle open.
    SL/TP calculated from information available at signal candle only.
    Conservative same-candle collision: SL first.
    """
    if signal_idx + 1 >= len(df):
        return None

    sig = df.iloc[signal_idx]
    entry_raw = float(df.iloc[signal_idx + 1].open)

    atrv = float(atr(df.iloc[:signal_idx + 1]).iloc[-1])
    if not np.isfinite(atrv) or atrv <= 0:
        return None

    supports, resistances = structure_levels(df.iloc[:signal_idx + 1])
    if direction == "LONG":
        entry = entry_raw * (1 + slippage_pct / 100)
        candidate_sl = supports[0] if supports and supports[0] < entry else entry - 1.35 * atrv
        sl = min(candidate_sl, entry - 0.006 * entry)
        risk = max(entry - sl, 0.006 * entry)
        tp1 = entry + rr1 * risk
        tp2 = entry + rr2 * risk
    else:
        entry = entry_raw * (1 - slippage_pct / 100)
        candidate_sl = resistances[0] if resistances and resistances[0] > entry else entry + 1.35 * atrv
        sl = max(candidate_sl, entry + 0.006 * entry)
        risk = max(sl - entry, 0.006 * entry)
        tp1 = entry - rr1 * risk
        tp2 = entry - rr2 * risk

    end = min(len(df), signal_idx + 1 + horizon_bars)
    result = "TIMEOUT"
    exit_price = float(df.iloc[end - 1].close)
    exit_idx = end - 1
    tp1_hit = False
    tp2_hit = False

    for j in range(signal_idx + 1, end):
        bar = df.iloc[j]
        hi, lo = float(bar.high), float(bar.low)

        if direction == "LONG":
            hit_sl = lo <= sl
            hit_tp1 = hi >= tp1
            hit_tp2 = hi >= tp2

            # Conservative ordering when both occur in one candle.
            if hit_sl:
                result = "SL"
                exit_price = sl
                exit_idx = j
                break
            if hit_tp2:
                result = "TP2"
                exit_price = tp2
                exit_idx = j
                tp1_hit = True
                tp2_hit = True
                break
            if hit_tp1:
                result = "TP1"
                exit_price = tp1
                exit_idx = j
                tp1_hit = True
                break
        else:
            hit_sl = hi >= sl
            hit_tp1 = lo <= tp1
            hit_tp2 = lo <= tp2
            if hit_sl:
                result = "SL"
                exit_price = sl
                exit_idx = j
                break
            if hit_tp2:
                result = "TP2"
                exit_price = tp2
                exit_idx = j
                tp1_hit = True
                tp2_hit = True
                break
            if hit_tp1:
                result = "TP1"
                exit_price = tp1
                exit_idx = j
                tp1_hit = True
                break

    # Approximate round-trip costs.
    cost = 2 * (fee_pct + slippage_pct) / 100
    gross_return = (
        (exit_price / entry - 1) if direction == "LONG"
        else (entry / exit_price - 1)
    )
    net_return = gross_return - cost

    return {
        "signal_idx": signal_idx,
        "exit_idx": exit_idx,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "result": result,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "net_return": float(net_return),
        "bars_held": int(exit_idx - signal_idx),
        "score": None,
    }


def backtest_1h(symbol, fee_pct=0.10, slippage_pct=0.05,
                horizon_bars=24, min_score=24, max_trades=180):
    """
    Rolling historical test on 1H data.
    This deliberately tests the same core signal on unseen future candles.
    """
    df = fetch_klines(symbol, "1h", 1000)
    if len(df) < 300:
        return None

    trades = []
    # Step through history; skip ahead after a trade to reduce overlapping signals.
    i = 120
    while i < len(df) - horizon_bars - 2 and len(trades) < max_trades:
        score = historical_signal_score(df, i)
        if score is None:
            i += 1
            continue

        # Calibration is for long-only Spot by default.
        if score >= min_score:
            tr = simulate_trade(
                df, i, "LONG",
                fee_pct, slippage_pct,
                horizon_bars=horizon_bars
            )
            if tr:
                tr["score"] = score
                trades.append(tr)
                # Avoid counting every consecutive candle as a separate position.
                i = tr["exit_idx"] + 1
                continue
        i += 1

    if not trades:
        return None

    t = pd.DataFrame(trades)
    wins = t["net_return"] > 0
    tp1 = t["tp1_hit"].mean()
    tp2 = t["tp2_hit"].mean()
    win_rate = wins.mean()

    gross_profit = t.loc[t.net_return > 0, "net_return"].sum()
    gross_loss = -t.loc[t.net_return < 0, "net_return"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    equity = (1 + t["net_return"]).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = float(dd.min())

    expectancy = float(t["net_return"].mean())

    # Empirical confidence: smoothed TP1 probability.
    # Beta(1,1) prior prevents 100% with tiny sample sizes.
    n = len(t)
    successes = int(t["tp1_hit"].sum())
    calibrated_tp1 = (successes + 1) / (n + 2)

    return {
        "symbol": symbol,
        "trades": n,
        "win_rate": float(win_rate),
        "tp1_rate": float(tp1),
        "tp2_rate": float(tp2),
        "calibrated_tp1": float(calibrated_tp1),
        "profit_factor": float(pf) if np.isfinite(pf) else 999.0,
        "max_drawdown": max_dd,
        "expectancy": expectancy,
        "net_return": float(equity.iloc[-1] - 1),
        "trades_df": t,
    }


# ============================================================
# LIVE ANALYSIS
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def analyze_symbol(symbol, mode="SPOT", fee_pct=0.10, slippage_pct=0.05):
    frames = {}
    for tf in TIMEFRAMES:
        df = get_tf_data(symbol, tf, 1000)
        if len(df) >= 80:
            frames[tf] = df

    if not frames:
        return None

    analyses = {
        tf: tf_signal(df)
        for tf, df in frames.items()
        if tf_signal(df) is not None
    }
    if not analyses:
        return None

    score = aggregate_score(analyses)
    position = position_from_score(score, mode)

    # Use 4H for risk structure where available.
    ref_tf = "4H" if "4H" in frames else ("1H" if "1H" in frames else next(iter(frames)))
    ref_df = frames[ref_tf]
    ref_a = analyses[ref_tf]
    entry = float(ref_df.close.iloc[-1])
    atrv = float(ref_a["atr"])

    supports, resistances = structure_levels(ref_df)

    if position == "لانگ":
        sl = supports[0] if supports and supports[0] < entry else entry - 1.35 * atrv
        sl = min(sl, entry - 0.006 * entry)
        risk = max(entry - sl, 0.006 * entry)
        tp1 = entry + 1.6 * risk
        tp2 = entry + 2.6 * risk
        tp3 = entry + 3.8 * risk
    elif position == "شورت":
        sl = resistances[0] if resistances and resistances[0] > entry else entry + 1.35 * atrv
        sl = max(sl, entry + 0.006 * entry)
        risk = max(sl - entry, 0.006 * entry)
        tp1 = entry - 1.6 * risk
        tp2 = entry - 2.6 * risk
        tp3 = entry - 3.8 * risk
    else:
        sl = entry - 1.35 * atrv
        risk = abs(entry - sl)
        tp1 = entry + 1.6 * risk
        tp2 = entry + 2.6 * risk
        tp3 = entry + 3.8 * risk

    # Historical calibration is separate from current technical score.
    bt = backtest_1h(
        symbol,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        horizon_bars=DEFAULT_HORIZON_BARS,
        min_score=24,
    )

    if bt:
        calibrated_conf = 100 * bt["calibrated_tp1"]
        sample_note = f"بر اساس {bt['trades']} معامله تاریخی 1H"
    else:
        calibrated_conf = np.nan
        sample_note = "داده تاریخی کافی برای کالیبراسیون موجود نیست"

    # No fake directional price forecast.
    # Instead, give historical outcome probabilities and expected return.
    forecast = {
        "4H": None,
        "24H": None,
        "72H": None,
    }

    return {
        "symbol": symbol,
        "frames": frames,
        "analyses": analyses,
        "score": float(score),
        "position": position,
        "decision": "معامله" if position in ("لانگ", "شورت") else "صبر",
        "entry": entry,
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "risk_pct": float(abs(entry - sl) / entry * 100),
        "supports": supports,
        "resistances": resistances,
        "backtest": bt,
        "calibrated_confidence": calibrated_conf,
        "confidence_note": sample_note,
        "forecast": forecast,
    }


# ============================================================
# UI
# ============================================================
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1100px}
.head{background:linear-gradient(135deg,#111318,#292d33);color:#fff;border-radius:22px;padding:20px;margin-bottom:14px}
.brand{font-size:25px;font-weight:900}.sub{font-size:12px;color:#cfd2d7;margin-top:6px}
.card{background:#fff;border:1px solid #e8e8e8;border-radius:20px;padding:16px;margin:12px 0;box-shadow:0 3px 14px rgba(0,0,0,.04)}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}
.box{background:#f7f7f8;border-radius:13px;padding:10px}.k{font-size:10px;color:#777}.v{font-size:16px;font-weight:850;margin-top:4px}
.buy{color:#087f3e}.sell{color:#b42318}.wait{color:#806000}
.note{font-size:11px;color:#777;line-height:1.8}
@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
<div class="head">
<div class="brand">₿ تحلیل‌گر حرفه‌ای رمزارز — نسخه قابل‌آزمون</div>
<div class="sub">سیگنال بر اساس کندل بسته‌شده • ورود روی کندل بعدی در بک‌تست • کارمزد و اسلیپیج • کالیبراسیون تاریخی • بدون پیش‌بینی ساختگی</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("تنظیمات تست")
    mode = st.selectbox("نوع بازار", ["SPOT", "FUTURES"])
    fee_pct = st.number_input("کارمزد هر سمت (%)", 0.0, 1.0, DEFAULT_FEE_PCT, 0.01)
    slip_pct = st.number_input("اسلیپیج هر سمت (%)", 0.0, 1.0, DEFAULT_SLIPPAGE_PCT, 0.01)
    horizon = st.number_input("افق بک‌تست (تعداد کندل 1H)", 4, 120, DEFAULT_HORIZON_BARS, 1)
    st.caption("در SPOT فقط لانگ فعال است. برای شورت باید واقعاً روی بازار Futures/Margin معامله شود.")

universe = get_universe()

st.markdown('<div class="card"><b>انتخاب بازار</b><div class="note">حداکثر ۵ ارز. تحلیل چندتایم‌فریمی انجام می‌شود، اما اطمینان فقط زمانی نمایش داده می‌شود که نتیجه تاریخی کافی داشته باشیم.</div></div>', unsafe_allow_html=True)

search = st.text_input("جستجو", placeholder="BTC / ETH / SOL ...")
filtered = [s for s in universe if search.upper() in s] if search else universe
selected = st.multiselect(
    "ارزها",
    filtered,
    max_selections=5,
    format_func=lambda x: x.replace("USDT", "/USDT"),
)

if selected:
    results = []
    with st.spinner("در حال تحلیل چندتایم‌فریمی و کالیبراسیون تاریخی..."):
        with ThreadPoolExecutor(max_workers=min(5, len(selected))) as ex:
            jobs = {
                ex.submit(
                    analyze_symbol,
                    s, mode, fee_pct, slip_pct
                ): s for s in selected
            }
            for job in as_completed(jobs):
                try:
                    r = job.result()
                    if r:
                        results.append(r)
                except Exception:
                    pass

    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        cls = "buy" if r["position"] == "لانگ" else ("sell" if r["position"] == "شورت" else "wait")
        bt = r["backtest"]

        if np.isfinite(r["calibrated_confidence"]):
            conf_txt = f'{r["calibrated_confidence"]:.1f}%'
        else:
            conf_txt = "نامشخص"

        if bt:
            bt_txt = (
                f'Win Rate: {bt["win_rate"]*100:.1f}% • '
                f'TP1: {bt["tp1_rate"]*100:.1f}% • '
                f'TP2: {bt["tp2_rate"]*100:.1f}% • '
                f'PF: {bt["profit_factor"]:.2f} • '
                f'DD: {bt["max_drawdown"]*100:.1f}% • '
                f'Expectancy: {bt["expectancy"]*100:.2f}%'
            )
        else:
            bt_txt = "داده کافی برای بک‌تست موجود نیست."

        st.markdown(f"""
        <div class="card">
          <h3>{r["symbol"].replace("USDT","/USDT")} —
            <span class="{cls}">{r["position"]}</span></h3>

          <div class="grid">
            <div class="box"><div class="k">امتیاز فعلی</div><div class="v">{r["score"]:.1f}</div></div>
            <div class="box"><div class="k">اطمینان کالیبره‌شده TP1</div><div class="v">{conf_txt}</div></div>
            <div class="box"><div class="k">ورود</div><div class="v">{money(r["entry"])}</div></div>
            <div class="box"><div class="k">حد ضرر</div><div class="v">{money(r["sl"])}</div></div>
            <div class="box"><div class="k">ریسک</div><div class="v">{r["risk_pct"]:.2f}%</div></div>
            <div class="box"><div class="k">TP1</div><div class="v">{money(r["tp1"])}</div></div>
            <div class="box"><div class="k">TP2</div><div class="v">{money(r["tp2"])}</div></div>
            <div class="box"><div class="k">TP3</div><div class="v">{money(r["tp3"])}</div></div>
            <div class="box"><div class="k">تصمیم</div><div class="v">{r["decision"]}</div></div>
            <div class="box"><div class="k">کالیبراسیون</div><div class="v">{r["confidence_note"]}</div></div>
          </div>

          <p class="note"><b>بک‌تست:</b> {bt_txt}</p>
          <p class="note"><b>نکته:</b> این درصد احتمال رسیدن تاریخی به TP1 در نمونه بک‌تست است، نه تضمین موفقیت معامله فعلی.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f'جزئیات تایم‌فریم‌ها — {r["symbol"].replace("USDT","/USDT")}'):
            rows = []
            for tf in TIMEFRAMES:
                a = r["analyses"].get(tf)
                if a:
                    rows.append({
                        "تایم‌فریم": tf,
                        "نوع": "لانگ" if a["score"] >= 24 else ("شورت" if a["score"] <= -24 else "صبر"),
                        "امتیاز": round(a["score"], 1),
                        "RSI": round(a["rsi"], 1),
                        "Momentum %": round(a["momentum"], 2),
                        "ADX": round(a["adx"], 1),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.write("دلایل تایم‌فریم‌های اصلی:")
            for tf in ("1H", "4H", "1D"):
                a = r["analyses"].get(tf)
                if a:
                    st.write(f"**{tf}:** " + "؛ ".join(a["reasons"][:6]))

            if bt:
                st.write("توزیع نتایج بک‌تست")
                counts = bt["trades_df"]["result"].value_counts().rename_axis("نتیجه").reset_index(name="تعداد")
                st.dataframe(counts, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="card note"><b>تغییر مهم:</b> دیگر «اطمینان ۹۰٪» از روی امتیاز ساخته نمی‌شود. '
        'اگر نمونه تاریخی کافی نباشد، برنامه صراحتاً «نامشخص» نشان می‌دهد.</div>',
        unsafe_allow_html=True
    )
else:
    st.info("یک یا چند ارز انتخاب کن.")

st.markdown(
    '<div class="note" style="text-align:center;margin-top:18px">'
    'این ابزار تحقیقاتی است و تضمین سود نمی‌دهد. بک‌تست گذشته تضمین آینده نیست. '
    'قبل از معامله واقعی، اجرای Paper Trading و سپس Forward Test توصیه می‌شود.'
    '</div>',
    unsafe_allow_html=True
)
