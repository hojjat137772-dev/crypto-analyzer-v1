import os
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

APP_VERSION = "1.0.0"
BINANCE_URLS = [
    "https://api.binance.com",
    "https://api.binance.us",
]
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"

app = FastAPI(title="تحلیل‌گر بازار کریپتو", version=APP_VERSION)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CryptoAnalyzer/1.0",
    "Accept": "application/json",
})

CACHE = {}
CACHE_TTL = 60
SYMBOLS_CACHE = {"time": 0, "rows": []}
SYMBOLS_CACHE_TTL = 600


def get_json(path, params=None, timeout=15):
    last_error = None
    for base in BINANCE_URLS:
        try:
            r = SESSION.get(base + path, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"دریافت اطلاعات بازار ناموفق بود: {last_error}")


def clean_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("/", "").replace("-", "").strip()


def interval_to_text(interval: str) -> str:
    allowed = {
        "15m": "15 دقیقه",
        "1h": "1 ساعت",
        "4h": "4 ساعت",
        "1d": "روزانه",
    }
    return allowed.get(interval, interval)


def get_all_symbols():
    now = time.time()
    if SYMBOLS_CACHE["rows"] and now - SYMBOLS_CACHE["time"] < SYMBOLS_CACHE_TTL:
        return SYMBOLS_CACHE["rows"]

    data = get_json("/api/v3/exchangeInfo")
    rows = []
    for s in data.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", True)
        ):
            rows.append({
                "symbol": s["symbol"],
                "base": s.get("baseAsset", ""),
                "quote": s.get("quoteAsset", "USDT"),
            })

    rows.sort(key=lambda x: x["symbol"])
    SYMBOLS_CACHE["rows"] = rows
    SYMBOLS_CACHE["time"] = now
    return rows


def get_history(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = 300):
    symbol = clean_symbol(symbol)
    limit = max(100, min(int(limit), 1000))
    key = f"{symbol}:{interval}:{limit}"
    now = time.time()

    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        return CACHE[key]["df"].copy()

    raw = get_json(
        "/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=20,
    )

    if not raw:
        raise ValueError("داده تاریخی برای این نماد پیدا نشد.")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
    df = df.reset_index(drop=True)

    CACHE[key] = {"time": now, "df": df.copy()}
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def support_resistance(df, window=40):
    recent = df.tail(window)
    support = float(recent["low"].min())
    resistance = float(recent["high"].max())
    return support, resistance


def score_trend(price, e20, e50, e200):
    score = 0
    if price > e20:
        score += 10
    if e20 > e50:
        score += 10
    if e50 > e200:
        score += 10
    if price > e200:
        score += 5
    return score


def build_signal(df):
    price = float(df["close"].iloc[-1])
    e20 = float(df["ema20"].iloc[-1])
    e50 = float(df["ema50"].iloc[-1])
    e200 = float(df["ema200"].iloc[-1])
    r = float(df["rsi"].iloc[-1])
    ml = float(df["macd"].iloc[-1])
    ms = float(df["macd_signal"].iloc[-1])
    mh = float(df["macd_hist"].iloc[-1])
    a = float(df["atr"].iloc[-1])

    support, resistance = support_resistance(df)

    score = score_trend(price, e20, e50, e200)

    # RSI: not treated as a standalone buy/sell trigger.
    if 45 <= r <= 68:
        score += 15
    elif r < 35:
        score += 8
    elif r > 75:
        score -= 8

    if ml > ms:
        score += 15
    if mh > 0:
        score += 5

    distance_to_res = max(0.0, resistance - price)
    if resistance > price and price > support:
        score += 4

    score = int(max(0, min(100, score)))

    if score >= 65 and price > e20 and ml >= ms:
        status = "صعودی"
        signal = "خرید مشروط / بررسی ورود"
    elif score <= 40 and price < e20 and ml < ms:
        status = "نزولی"
        signal = "فروش / عدم ورود"
    else:
        status = "خنثی"
        signal = "انتظار"

    # Risk levels are mechanical reference levels, not guaranteed targets.
    stop = max(0.0, min(support, price - 1.5 * a))
    if stop >= price:
        stop = max(0.0, price - 1.5 * a)

    risk = price - stop
    if risk <= 0:
        risk = max(a, price * 0.01)
        stop = max(0.0, price - risk)

    tp1 = price + 1.0 * risk
    tp2 = price + 2.0 * risk
    tp3 = price + 3.0 * risk

    return {
        "score": score,
        "signal": signal,
        "status": status,
        "price": price,
        "entry": price,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "support": support,
        "resistance": resistance,
        "rsi": r,
        "ema20": e20,
        "ema50": e50,
        "ema200": e200,
        "macd": ml,
        "macd_signal": ms,
        "atr": a,
        "risk_pct": abs(price - stop) / price * 100 if price else 0,
        "distance_to_resistance_pct": distance_to_res / price * 100 if price else 0,
    }


def analyze_symbol(symbol: str, interval: str):
    df = get_history(symbol, interval, 300)

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    ml, ms, mh = macd(df["close"])
    df["macd"] = ml
    df["macd_signal"] = ms
    df["macd_hist"] = mh
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        raise ValueError("داده کافی برای تحلیل وجود ندارد.")

    result = build_signal(df)
    result["symbol"] = clean_symbol(symbol)
    result["interval"] = interval
    result["interval_text"] = interval_to_text(interval)
    result["updated_at"] = datetime.now(timezone.utc).isoformat()
    return result


def run_backtest(symbol: str, interval: str, limit: int = 300):
    df = get_history(symbol, interval, limit)
    df["ema10"] = ema(df["close"], 10)
    df["ema20"] = ema(df["close"], 20)
    df["rsi"] = rsi(df["close"], 14)
    ml, ms, mh = macd(df["close"])
    df["macd"] = ml
    df["macd_signal"] = ms
    df["macd_hist"] = mh
    df = df.dropna().reset_index(drop=True)

    if len(df) < 60:
        raise ValueError("داده کافی برای بک‌تست وجود ندارد.")

    trades = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        buy = (
            prev["ema10"] <= prev["ema20"]
            and row["ema10"] > row["ema20"]
            and row["rsi"] < 70
            and row["macd_hist"] > 0
        )

        sell = (
            prev["ema10"] >= prev["ema20"]
            and row["ema10"] < row["ema20"]
        )

        if position is None and buy:
            position = {
                "entry": float(row["close"]),
                "entry_time": row["time"],
            }
        elif position is not None and sell:
            exit_price = float(row["close"])
            ret = (exit_price / position["entry"] - 1) * 100
            trades.append({
                "entry": position["entry"],
                "exit": exit_price,
                "return_pct": ret,
                "entry_time": str(position["entry_time"]),
                "exit_time": str(row["time"]),
            })
            position = None

    if position is not None:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        ret = (exit_price / position["entry"] - 1) * 100
        trades.append({
            "entry": position["entry"],
            "exit": exit_price,
            "return_pct": ret,
            "entry_time": str(position["entry_time"]),
            "exit_time": str(last["time"]),
        })

    if not trades:
        return {
            "symbol": clean_symbol(symbol),
            "interval": interval,
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "compound_return": 0,
            "max_drawdown": 0,
            "trades_detail": [],
            "note": "در این بازه معامله‌ای با قوانین بک‌تست ایجاد نشد.",
        }

    returns = np.array([t["return_pct"] / 100 for t in trades], dtype=float)
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = (equity - peaks) / peaks * 100

    win_rate = float((returns > 0).mean() * 100)
    avg_return = float(returns.mean() * 100)
    compound_return = float((equity[-1] - 1) * 100)
    max_drawdown = float(drawdown.min())

    return {
        "symbol": clean_symbol(symbol),
        "interval": interval,
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "compound_return": compound_return,
        "max_drawdown": max_drawdown,
        "trades_detail": trades[-30:],
        "note": "بک‌تست با کارمزد، اسلیپیج و محدودیت نقدشوندگی محاسبه نشده است.",
    }


def html_page():
    return """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>تحلیل‌گر بازار کریپتو</title>
<style>
*{box-sizing:border-box}
body{font-family:Tahoma,Arial,sans-serif;background:#eef3f8;color:#18212b;margin:0}
.wrap{max-width:980px;margin:auto;padding:18px}
.card{background:#fff;border-radius:22px;padding:20px;margin:14px 0;box-shadow:0 8px 30px #00000010}
h1{margin:0 0 8px;font-size:26px}
h2{margin:0 0 15px}
.sub{color:#68717b}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.item{background:#f8fafc;border:1px solid #e6ebf0;border-radius:16px;padding:15px}
.label{font-size:13px;color:#68717b}
.value{font-size:22px;font-weight:bold;margin-top:5px}
select,input,button{width:100%;box-sizing:border-box;padding:13px;border-radius:12px;border:1px solid #d7dee8;font-size:16px}
button{background:#2447a8;color:white;border:0;font-weight:bold;cursor:pointer;margin-top:10px}
button.secondary{background:#344256}
.row{display:grid;grid-template-columns:2fr 1fr;gap:10px}
.good{color:#16815c}.bad{color:#c33}.neutral{color:#555}
ul{line-height:2}
.small{font-size:12px;color:#6b7280}
pre{white-space:pre-wrap;direction:ltr;text-align:left}
@media(max-width:600px){.grid,.row{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<h1>تحلیل‌گر بازار کریپتو 🪙</h1>
<div class="sub">نسخه 1.0.0 — تحلیل تکنیکال، سیگنال مشروط، مدیریت ریسک و بک‌تست</div>
</div>

<div class="card">
<div class="row">
<input id="symbol" value="BTCUSDT" placeholder="مثلاً BTCUSDT یا ETHUSDT">
<select id="interval">
<option value="15m">15 دقیقه</option>
<option value="1h" selected>1 ساعت</option>
<option value="4h">4 ساعت</option>
<option value="1d">روزانه</option>
</select>
</div>
<button onclick="analyze()">🔎 تحلیل ارز</button>
<button class="secondary" onclick="backtest()">🧪 بک‌تست</button>
<button onclick="scan()">🔍 اسکن بازار</button>
</div>

<div id="out"></div>
</div>

<script>
function n(x,d=2){
  if(x===null||x===undefined||Number.isNaN(Number(x))) return "-";
  return Number(x).toLocaleString("en-US",{maximumFractionDigits:d});
}
function card(label,value){return `<div class="item"><div class="label">${label}</div><div class="value">${value}</div></div>`}
async function analyze(){
  const s=document.getElementById("symbol").value.trim();
  const i=document.getElementById("interval").value;
  out.innerHTML='<div class="card">در حال دریافت اطلاعات بازار...</div>';
  try{
    const r=await fetch(`/analyze?symbol=${encodeURIComponent(s)}&interval=${i}`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||"خطا");
    const cls=j.status==="صعودی"?"good":j.status==="نزولی"?"bad":"neutral";
    out.innerHTML=`
    <div class="card">
      <h2>${j.symbol} — ${j.interval_text}</h2>
      <div class="grid">
        ${card("وضعیت",`<span class="${cls}">${j.status}</span>`)}
        ${card("امتیاز سیگنال (0 تا 100)",n(j.score,0))}
        ${card("قیمت آخر",n(j.price))}
        ${card("ورود مرجع",n(j.entry))}
        ${card("حد ضرر",n(j.stop))}
        ${card("حد سود 1",n(j.tp1))}
        ${card("حد سود 2",n(j.tp2))}
        ${card("حد سود 3",n(j.tp3))}
        ${card("RSI",n(j.rsi))}
        ${card("EMA20",n(j.ema20))}
        ${card("EMA50",n(j.ema50))}
        ${card("EMA200",n(j.ema200))}
        ${card("حمایت",n(j.support))}
        ${card("مقاومت",n(j.resistance))}
        ${card("ریسک تا حد ضرر",n(j.risk_pct)+"%")}
      </div>
      <h2 style="margin-top:20px">نتیجه تحلیل</h2>
      <p>${j.signal}</p>
      <ul>
        <li>قیمت نسبت به EMA20: ${j.price>j.ema20?"بالاتر":"پایین‌تر"}</li>
        <li>EMA20 نسبت به EMA50: ${j.ema20>j.ema50?"بالاتر":"پایین‌تر"}</li>
        <li>EMA50 نسبت به EMA200: ${j.ema50>j.ema200?"بالاتر":"پایین‌تر"}</li>
        <li>MACD: ${j.macd>=j.macd_signal?"مثبت/صعودی":"منفی/نزولی"}</li>
        <li>فاصله تا مقاومت: ${n(j.distance_to_resistance_pct)}%</li>
      </ul>
      <div class="small">این خروجی ابزار تحلیل است و تضمین نتیجه معامله یا سود نیست.</div>
    </div>`;
  }catch(e){out.innerHTML=`<div class="card"><b>خطا:</b> ${e.message}</div>`}
}
async function backtest(){
  const s=document.getElementById("symbol").value.trim();
  const i=document.getElementById("interval").value;
  out.innerHTML='<div class="card">در حال اجرای بک‌تست...</div>';
  try{
    const r=await fetch(`/backtest?symbol=${encodeURIComponent(s)}&interval=${i}`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||"خطا");
    out.innerHTML=`
    <div class="card">
      <h2>نتیجه بک‌تست ${j.symbol}</h2>
      <div class="grid">
        ${card("تعداد معاملات",n(j.trades,0))}
        ${card("درصد برد",n(j.win_rate)+"%")}
        ${card("میانگین بازده هر معامله",n(j.avg_return)+"%")}
        ${card("بازده مرکب",n(j.compound_return)+"%")}
        ${card("حداکثر افت سرمایه",n(j.max_drawdown)+"%")}
      </div>
      <p class="small">${j.note}</p>
    </div>`;
  }catch(e){out.innerHTML=`<div class="card"><b>خطا:</b> ${e.message}</div>`}
}
async function scan(){
  out.innerHTML='<div class="card">در حال اسکن ارزهای USDT...</div>';
  try{
    const i=document.getElementById("interval").value;
    const r=await fetch(`/scan?interval=${i}&limit=20`);
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||"خطا");
    let rows=j.results.map(x=>`<div class="item"><b>${x.symbol}</b><br>امتیاز: ${x.score} — ${x.status}<br>قیمت: ${n(x.price)}</div>`).join("");
    out.innerHTML=`<div class="card"><h2>نتیجه اسکن بازار</h2><div class="grid">${rows}</div></div>`;
  }catch(e){out.innerHTML=`<div class="card"><b>خطا:</b> ${e.message}</div>`}
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return html_page()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "data_source": "Binance public market data",
        "trading_enabled": False,
    }


@app.get("/symbols")
def symbols():
    return {"count": len(get_all_symbols()), "symbols": get_all_symbols()}


@app.get("/analyze")
def analyze(
    symbol: str = Query(DEFAULT_SYMBOL),
    interval: str = Query(DEFAULT_INTERVAL),
):
    allowed = {"15m", "1h", "4h", "1d"}
    if interval not in allowed:
        return JSONResponse({"detail": "تایم‌فریم نامعتبر است."}, status_code=400)

    try:
        return analyze_symbol(symbol, interval)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/backtest")
def backtest(
    symbol: str = Query(DEFAULT_SYMBOL),
    interval: str = Query(DEFAULT_INTERVAL),
):
    allowed = {"15m", "1h", "4h", "1d"}
    if interval not in allowed:
        return JSONResponse({"detail": "تایم‌فریم نامعتبر است."}, status_code=400)

    try:
        return run_backtest(symbol, interval)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/scan")
def scan(
    interval: str = Query(DEFAULT_INTERVAL),
    limit: int = Query(20, ge=1, le=50),
):
    try:
        symbols = get_all_symbols()

        # A deterministic first group is used to avoid thousands of API calls.
        # The app remains analysis-only; it never places orders.
        preferred = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
            "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT",
            "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "NEARUSDT",
        ]
        available = {x["symbol"] for x in symbols}
        selected = [x for x in preferred if x in available][:limit]

        results = []
        for s in selected:
            try:
                r = analyze_symbol(s, interval)
                results.append({
                    "symbol": s,
                    "score": r["score"],
                    "status": r["status"],
                    "price": r["price"],
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "interval": interval,
            "count": len(results),
            "results": results,
            "note": "اسکن نمونه‌ای از ارزهای نقدشونده USDT است؛ برای اسکن گسترده‌تر می‌توان مرحله بعد رتبه‌بندی حجم بازار را اضافه کرد.",
        }
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=502)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
