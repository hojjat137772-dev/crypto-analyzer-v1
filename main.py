کدی که نوشتی میخوام معامله سل و import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ============================================================
# Crypto Analyzer Pro
# BUY / SELL / WAIT + multi-market scanner
# Public market data only - no order execution
# Dependencies: streamlit, pandas, numpy, requests
# ============================================================

st.set_page_config(page_title="ربات تحلیل حرفه‌ای کریپتو", page_icon="📈", layout="wide")

BINANCE = "https://data-api.binance.vision"
WALLEX = "https://api.wallex.ir"
NOBITEX = "https://api.nobitex.ir"
TABDEAL = "https://api1.tabdeal.org"
EXCHANGES = ["Binance", "والکس", "نوبیتکس", "تبدیل"]
TFS = ["15m", "1h", "4h", "1d"]

st.markdown("""
<style>
.block-container{padding-top:1rem;max-width:1450px}
.signal{padding:18px;border-radius:16px;text-align:center;font-size:25px;font-weight:800;margin:10px 0}
</style>
""", unsafe_allow_html=True)

st.title("📈 ربات تحلیل حرفه‌ای کریپتو")
st.caption("تحلیل تکنیکال چندتایم‌فریمی | BUY / SELL / WAIT | بدون ثبت سفارش")


def get_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": "CryptoAnalyzer/3.0"})
    r.raise_for_status()
    return r.json()


def fnum(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def clean_symbol(s):
    return str(s).upper().replace("-", "").replace("_", "").replace("/", "")


def quote_of(s):
    s = clean_symbol(s)
    for q in ("USDT", "USDC", "FDUSD", "BUSD", "IRT", "RLS", "BTC", "ETH"):
        if s.endswith(q) and len(s) > len(q):
            return q
    return ""


def base_of(s):
    q = quote_of(s)
    return clean_symbol(s)[:-len(q)] if q else clean_symbol(s)


# ---------------- Binance ----------------
@st.cache_data(ttl=180, show_spinner=False)
def binance_markets():
    d = get_json(f"{BINANCE}/api/v3/exchangeInfo")
    return sorted({str(x["symbol"]).upper() for x in d.get("symbols", [])
                   if x.get("status") == "TRADING" and str(x.get("symbol", "")).upper().endswith("USDT")})


@st.cache_data(ttl=30, show_spinner=False)
def binance_price(symbol):
    d = get_json(f"{BINANCE}/api/v3/ticker/price", {"symbol": clean_symbol(symbol)})
    return fnum(d.get("price"))


@st.cache_data(ttl=30, show_spinner=False)
def binance_candles(symbol, interval, limit=300):
    d = get_json(f"{BINANCE}/api/v3/klines", {
        "symbol": clean_symbol(symbol), "interval": interval, "limit": min(limit, 1000)
    })
    if not isinstance(d, list) or len(d) < 80:
        raise RuntimeError("کندل Binance کافی نیست")
    df = pd.DataFrame(d, columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","x"])
    for c in ["o","h","l","c","v","qv"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.rename(columns={"t":"open_time","o":"open","h":"high","l":"low","c":"close","v":"volume","qv":"quote_volume"})[["open_time","open","high","low","close","volume","quote_volume"]].dropna()


# ---------------- Wallex ----------------
@st.cache_data(ttl=180, show_spinner=False)
def wallex_markets():
    d = get_json(f"{WALLEX}/v1/markets")
    syms = d.get("result", {}).get("symbols", {})
    if isinstance(syms, dict):
        return sorted({clean_symbol(k) for k in syms.keys() if clean_symbol(k)})
    raise RuntimeError("لیست بازار والکس دریافت نشد")


@st.cache_data(ttl=30, show_spinner=False)
def wallex_price(symbol):
    d = get_json(f"{WALLEX}/v1/markets")
    item = d.get("result", {}).get("symbols", {}).get(str(symbol).upper())
    if not item:
        # try exact key from returned dictionary
        for k, v in d.get("result", {}).get("symbols", {}).items():
            if clean_symbol(k) == clean_symbol(symbol):
                item = v
                break
    p = (item or {}).get("stats", {}).get("lastPrice")
    if p is None:
        raise RuntimeError("قیمت والکس پیدا نشد")
    return fnum(p)


def udf_df(d):
    if not isinstance(d, dict) or d.get("s") not in ("ok", "no_data"):
        raise RuntimeError("پاسخ کندل معتبر نیست")
    n = min(*(len(d.get(k, [])) for k in ["t","o","h","l","c","v"]))
    if n < 80:
        raise RuntimeError("کندل کافی نیست")
    return pd.DataFrame({
        "open_time": pd.to_datetime(pd.to_numeric(d["t"][:n]), unit="s"),
        "open": pd.to_numeric(d["o"][:n], errors="coerce"),
        "high": pd.to_numeric(d["h"][:n], errors="coerce"),
        "low": pd.to_numeric(d["l"][:n], errors="coerce"),
        "close": pd.to_numeric(d["c"][:n], errors="coerce"),
        "volume": pd.to_numeric(d["v"][:n], errors="coerce"),
    }).dropna()


@st.cache_data(ttl=45, show_spinner=False)
def wallex_candles(symbol, interval, limit=300):
    res = {"15m":15, "1h":60, "4h":240, "1d":"D"}[interval]
    sec = {"15m":900, "1h":3600, "4h":14400, "1d":86400}[interval]
    now = int(time.time())
    d = get_json(f"{WALLEX}/v1/udf/history", {
        "symbol": str(symbol).upper(), "resolution":res,
        "from":now-sec*(limit+50), "to":now
    })
    return udf_df(d).tail(limit)


# ---------------- Nobitex ----------------
@st.cache_data(ttl=180, show_spinner=False)
def nobitex_markets():
    d = get_json(f"{NOBITEX}/market/stats")
    return sorted({clean_symbol(k) for k in d.get("stats", {}).keys() if "-" in str(k)})


def nobitex_parts(symbol):
    s = clean_symbol(symbol)
    if s.endswith("USDT"):
        return s[:-4].lower(), "usdt"
    if s.endswith("IRT"):
        return s[:-3].lower(), "rls"
    if s.endswith("RLS"):
        return s[:-3].lower(), "rls"
    return base_of(s).lower(), "usdt"


@st.cache_data(ttl=30, show_spinner=False)
def nobitex_price(symbol):
    src, dst = nobitex_parts(symbol)
    d = get_json(f"{NOBITEX}/market/stats", {"srcCurrency":src, "dstCurrency":dst})
    item = d.get("stats", {}).get(f"{src}-{dst}")
    if not item:
        raise RuntimeError("قیمت نوبیتکس پیدا نشد")
    return fnum(item.get("latest"))


@st.cache_data(ttl=45, show_spinner=False)
def nobitex_candles(symbol, interval, limit=300):
    src, dst = nobitex_parts(symbol)
    res = {"15m":15, "1h":60, "4h":240, "1d":"D"}[interval]
    sec = {"15m":900, "1h":3600, "4h":14400, "1d":86400}[interval]
    now = int(time.time())
    d = get_json(f"{NOBITEX}/market/udf/history", {
        "symbol":f"{src.upper()}{dst.upper()}", "resolution":res,
        "from":now-sec*(limit+50), "to":now
    })
    return udf_df(d).tail(limit)


# ---------------- Tabdeal ----------------
def tabdeal_name(symbol):
    s = clean_symbol(symbol)
    q = quote_of(s)
    return f"{s[:-len(q)]}_{q}" if q else s


def extract_symbols(obj):
    out=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            if str(k).lower() in ("symbol","tabdealsymbol") and isinstance(v,str):
                out.append(clean_symbol(v))
            else:
                out.extend(extract_symbols(v))
    elif isinstance(obj,list):
        for v in obj:
            out.extend(extract_symbols(v))
    return out


@st.cache_data(ttl=180, show_spinner=False)
def tabdeal_markets():
    d = get_json(f"{TABDEAL}/r/api/v1/exchangeInfo")
    out = [s for s in extract_symbols(d) if 5 <= len(s) <= 25 and s.isalnum()]
    if not out:
        raise RuntimeError("لیست بازارهای تبدیل دریافت نشد")
    return sorted(set(out))


@st.cache_data(ttl=30, show_spinner=False)
def tabdeal_price(symbol):
    ts = tabdeal_name(symbol)
    last = None
    for params in ({"tabdealSymbol":ts}, {"symbol":clean_symbol(symbol)}):
        try:
            d = get_json(f"{TABDEAL}/r/api/v1/depth", params)
            bids, asks = d.get("bids", []), d.get("asks", [])
            if bids and asks:
                return (fnum(bids[0][0]) + fnum(asks[0][0]))/2
            if bids: return fnum(bids[0][0])
            if asks: return fnum(asks[0][0])
        except Exception as e:
            last=e
    raise RuntimeError(f"قیمت تبدیل پیدا نشد: {ts}") from last


# IMPORTANT: For Tabdeal USDT markets use Binance OHLC as technical-history fallback.
# This prevents the known /trades?symbol=...&limit=2000 400 error from breaking analysis.
@st.cache_data(ttl=45, show_spinner=False)
def tabdeal_candles(symbol, interval):
    s = clean_symbol(symbol)
    if quote_of(s) == "USDT":
        return binance_candles(s, interval)
    raise RuntimeError("برای این بازار تبدیل تاریخچه کندلی عمومی کافی نیست")


# ---------------- Router ----------------
def markets(exchange):
    if exchange == "Binance": return binance_markets()
    if exchange == "والکس": return wallex_markets()
    if exchange == "نوبیتکس": return nobitex_markets()
    return tabdeal_markets()


def price(exchange, symbol):
    if exchange == "Binance": return binance_price(symbol)
    if exchange == "والکس": return wallex_price(symbol)
    if exchange == "نوبیتکس": return nobitex_price(symbol)
    return tabdeal_price(symbol)


def candles(exchange, symbol, tf):
    if exchange == "Binance": return binance_candles(symbol, tf)
    if exchange == "والکس": return wallex_candles(symbol, tf)
    if exchange == "نوبیتکس": return nobitex_candles(symbol, tf)
    return tabdeal_candles(symbol, tf)


# ---------------- Indicators ----------------
def add_indicators(df):
    x=df.copy()
    x["ema20"]=x.close.ewm(span=20,adjust=False).mean()
    x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
    x["ema200"]=x.close.ewm(span=200,adjust=False).mean()
    d=x.close.diff()
    gain=d.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi"]=(100-100/(1+rs)).fillna(50)
    x["macd"]=x.close.ewm(span=12,adjust=False).mean()-x.close.ewm(span=26,adjust=False).mean()
    x["macd_signal"]=x.macd.ewm(span=9,adjust=False).mean()
    prev=x.close.shift(1)
    tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x["vol_ratio"]=x.volume/x.volume.rolling(20).mean().replace(0,np.nan)
    return x.dropna().reset_index(drop=True)


def score_tf(df):
    x=add_indicators(df)
    if len(x)<20: raise RuntimeError("اندیکاتور کافی نیست")
    a=x.iloc[-1]; p=x.iloc[-2]
    score=50
    if a.ema20>a.ema50: score+=10
    else: score-=10
    if a.close>a.ema200: score+=10
    else: score-=10
    if a.macd>a.macd_signal: score+=8
    else: score-=8
    if a.macd>a.macd_signal and a.macd-p.macd>0: score+=4
    if a.macd<a.macd_signal and a.macd-p.macd<0: score-=4
    if 50<=a.rsi<=68: score+=8
    elif a.rsi>72: score-=7
    elif a.rsi<30: score+=2
    else: score-=1
    vr=fnum(a.vol_ratio,1)
    if vr>=1.5: score+=7
    elif vr>=1.1: score+=3
    return int(np.clip(score,0,100)),x


def support_resistance(x):
    r=x.tail(60)
    return float(r.low.min()),float(r.high.max())


def signal(scores, main):
    """BUY/SELL only with multi-timeframe confirmation; otherwise WAIT."""
    s1,s4=scores.get("1h"),scores.get("4h")
    a=main.iloc[-1]
    bull=(a.ema20>a.ema50 and a.close>a.ema200 and a.macd>a.macd_signal and 50<=a.rsi<=70)
    bear=(a.ema20<a.ema50 and a.close<a.ema200 and a.macd<a.macd_signal and 30<=a.rsi<=50)
    if s1 is not None and s4 is not None:
        bull = bull and s1>=60 and s4>=60
        bear = bear and s1<=40 and s4<=40
    vals=[v for v in scores.values() if v is not None]
    total=round(float(np.mean(vals))) if vals else 50
    if total>=68 and bull: return "BUY",total
    if total<=32 and bear: return "SELL",total
    return "WAIT",total


def levels(side, entry, atrv, sup, res):
    atrv=max(float(atrv) if np.isfinite(atrv) and atrv>0 else entry*.02, entry*.005)
    if side=="BUY":
        sl=min(entry-1.5*atrv, sup*.995 if sup<entry else entry-1.5*atrv)
        risk=max(entry-sl,entry*.005)
        return sl,entry+1.5*risk,entry+2.5*risk,entry+4*risk
    if side=="SELL":
        sl=max(entry+1.5*atrv,res*1.005 if res>entry else entry+1.5*atrv)
        risk=max(sl-entry,entry*.005)
        return sl,entry-1.5*risk,entry-2.5*risk,entry-4*risk
    return np.nan,np.nan,np.nan,np.nan


def fmt(v):
    if v is None or not np.isfinite(v): return "-"
    if abs(v)>=1000:return f"{v:,.2f}"
    if abs(v)>=1:return f"{v:.4f}"
    return f"{v:.8f}"


# ---------------- Scanner ----------------
def scanner_universe(ms, n):
    preferred=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","TRXUSDT","SHIBUSDT","DOTUSDT","LTCUSDT","SUIUSDT","PEPEUSDT","ARBUSDT","OPUSDT","APTUSDT","NEARUSDT"]
    clean=[]
    for m in ms:
        s=clean_symbol(m)
        if quote_of(s) in ("USDT","IRT","RLS") and base_of(s) not in ("USDT","USDC","IRT","RLS"):
            clean.append(s)
    out=[x for x in preferred if x in clean]
    out += [x for x in clean if x not in out]
    return out[:n]


def scan_market(exchange,symbol,tfs):
    scores={}; main=None; last=None
    for tf in tfs:
        try:
            sc,x=score_tf(candles(exchange,symbol,tf))
            scores[tf]=sc
            if tf=="4h" or main is None:
                main=x; last=x.iloc[-1]
        except Exception:
            scores[tf]=None
    valid=[v for v in scores.values() if v is not None]
    if len(valid)<2 or main is None:
        raise RuntimeError("داده حداقل دو تایم‌فریم موجود نیست")
    side,total=signal(scores,main)
    entry=price(exchange,symbol)
    sup,res=support_resistance(main)
    sl,tp1,tp2,tp3=levels(side,entry,last.atr,sup,res)
    return {"بازار":symbol,"امتیاز":total,"سیگنال":side,"قیمت":entry,"ورود":entry,"حدضرر":sl,"TP1":tp1,"TP2":tp2,"TP3":tp3,"حمایت":sup,"مقاومت":res,"1h":scores.get("1h"),"4h":scores.get("4h")}


def run_scan(exchange,ms,n,tfs):
    rows=[]
    bar=st.progress(0)
    syms=scanner_universe(ms,n)
    for i,s in enumerate(syms,1):
        try: rows.append(scan_market(exchange,s,tfs))
        except Exception as e: rows.append({"بازار":s,"امتیاز":np.nan,"سیگنال":"WAIT","خطا":str(e)[:45]})
        bar.progress(i/len(syms),text=f"اسکن {s} | {i}/{len(syms)}")
    bar.empty()
    return pd.DataFrame(rows).sort_values("امتیاز",ascending=False,na_position="last").reset_index(drop=True)


# ---------------- UI ----------------
with st.sidebar:
    st.header("🏦 صرافی")
    exchange=st.selectbox("انتخاب صرافی",EXCHANGES)
    try:
        ms=markets(exchange)
        st.success(f"{len(ms)} بازار دریافت شد")
    except Exception as e:
        ms=[]
        st.error("لیست بازارها دریافت نشد")
        st.caption(str(e))
    symbol=st.selectbox("🪙 ارز / بازار",ms if ms else ["BTCUSDT"])
    tf=st.selectbox("⏱ تایم‌فریم",TFS,index=2)
    analyze=st.button("🚀 تحلیل",use_container_width=True)
    st.divider()
    st.header("🔎 اسکن چند ارز")
    n=st.slider("تعداد ارز",5,30,15,5)
    mode=st.selectbox("نوع اسکن",["سریع: 1h + 4h","کامل: 15m + 1h + 4h + 1d"])
    scan=st.button("🔎 شروع اسکن",use_container_width=True)

if scan:
    tfs=["1h","4h"] if mode.startswith("سریع") else ["15m","1h","4h","1d"]
    st.subheader("🔎 نتیجه اسکن بازار")
    if not ms:
        st.error("لیست بازار صرافی موجود نیست")
    else:
        with st.spinner("در حال اسکن..."):
            result=run_scan(exchange,ms,n,tfs)
        valid=result[pd.to_numeric(result["امتیاز"],errors="coerce").notna()].head(3)
        if valid.empty:
            st.warning("هیچ بازار دارای داده کافی پیدا نشد")
        else:
            st.markdown("### 🏆 ۳ بازار با بالاترین امتیاز محاسباتی")
            cols=st.columns(len(valid))
            for col,(_,r) in zip(cols,valid.iterrows()):
                with col:
                    st.metric(r["بازار"],f"{int(r['امتیاز'])}/100")
                    if r["سیگنال"]=="BUY": st.success("🟢 BUY")
                    elif r["سیگنال"]=="SELL": st.error("🔴 SELL")
                    else: st.warning("🟡 WAIT")
                    st.write(f"ورود: **{fmt(r['ورود'])}**")
                    st.write(f"SL: **{fmt(r['حدضرر'])}**")
                    st.write(f"TP1: **{fmt(r['TP1'])}**")
                    st.write(f"TP2: **{fmt(r['TP2'])}**")
            show=result[[c for c in ["بازار","امتیاز","سیگنال","ورود","حدضرر","TP1","TP2","TP3","حمایت","مقاومت","1h","4h"] if c in result.columns]].copy()
            for c in ["ورود","حدضرر","TP1","TP2","TP3","حمایت","مقاومت"]:
                if c in show: show[c]=show[c].apply(lambda x:fmt(fnum(x)))
            st.dataframe(show,use_container_width=True,hide_index=True)

# ---------------- Single analysis ----------------
try:
    p=price(exchange,symbol)
    all_scores={}; data={}
    for x_tf in TFS:
        try:
            sc,x=score_tf(candles(exchange,symbol,x_tf))
            all_scores[x_tf]=sc; data[x_tf]=x
        except Exception:
            all_scores[x_tf]=None
    main=data.get(tf) or data.get("4h") or data.get("1h")
    if main is None: raise RuntimeError("داده کافی برای تحلیل وجود ندارد")
    side,total=signal(all_scores,main)
    last=main.iloc[-1]
    sup,res=support_resistance(main)
    sl,tp1,tp2,tp3=levels(side,p,last.atr,sup,res)

    st.subheader(f"{exchange} | {symbol}")
    if side=="BUY":
        st.markdown('<div class="signal">🟢 BUY — شرایط خرید تأیید شده</div>',unsafe_allow_html=True)
    elif side=="SELL":
        st.markdown('<div class="signal">🔴 SELL — شرایط فروش/خروج تأیید شده</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal">🟡 WAIT — فعلاً دست نگه دار</div>',unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("قیمت",fmt(p)); c2.metric("امتیاز",f"{total}/100"); c3.metric("RSI",f"{last.rsi:.1f}"); c4.metric("ATR",fmt(last.atr))
    st.markdown("### 🎯 سطوح معامله")
    a,b,c,d=st.columns(4)
    a.metric("ورود",fmt(p)); b.metric("حدضرر",fmt(sl)); c.metric("TP1",fmt(tp1)); d.metric("TP2",fmt(tp2))
    st.metric("TP3",fmt(tp3))

    st.markdown("### 🧭 تایم‌فریم‌ها")
    rows=[]
    for k in TFS:
        s=all_scores.get(k)
        rows.append({"تایم‌فریم":k,"امتیاز":"-" if s is None else s,"وضعیت":"داده ندارد" if s is None else ("صعودی" if s>=60 else "نزولی" if s<=40 else "خنثی")})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.markdown("### 📊 نمودار")
    chart=main.tail(150).set_index("open_time")[["close","ema20","ema50"]]
    st.line_chart(chart,use_container_width=True)

    st.caption(f"آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | تحلیل عمومی، بدون ثبت سفارش")
except Exception as e:
    st.error(f"تحلیل انجام نشد: {e}")
