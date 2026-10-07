import streamlit as st
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Crypto Analyzer Pro V5", page_icon="₿", layout="wide")

# ============================================================
# CONFIG
# ============================================================
TF = {"5m":"5m","15m":"15m","30m":"30m","1H":"1h","2H":"2h",
      "4H":"4h","6H":"6h","12H":"12h","1D":"1d","3D":"3d","1W":"1w"}

BINANCE = "https://api.binance.com"
OKX = "https://www.okx.com"
TABDEAL = "https://api.tabdeal.org"
TIMEOUT = 8
LIMIT = 260
S = requests.Session()
S.headers.update({"User-Agent":"CryptoAnalyzerPro-V5/1.0"})

# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def universe():
    # Tabdeal is the primary market universe.
    urls = [
        f"{TABDEAL}/r/api/v1/exchangeInfo", f"{TABDEAL}/api/v1/exchangeInfo",
        f"{TABDEAL}/v1/market/symbols", f"{TABDEAL}/v1/markets",
        f"{TABDEAL}/api/v1/markets"
    ]
    for u in urls:
        try:
            r = S.get(u, timeout=TIMEOUT)
            if not r.ok: continue
            d = r.json()
            items = d.get("symbols") if isinstance(d,dict) else d
            if items is None and isinstance(d,dict): items=d.get("data")
            out=[]
            for x in items if isinstance(items,list) else []:
                raw=str(x.get("symbol") or x.get("name") or x.get("code") or "")
                raw=raw.upper().replace("-","").replace("_","").replace("/","")
                if raw.endswith("USDT"): out.append(raw)
            if out: return sorted(set(out)), "Tabdeal"
        except Exception: pass

    try:
        d=S.get(f"{BINANCE}/api/v3/exchangeInfo",timeout=TIMEOUT).json()
        out=[x["symbol"] for x in d["symbols"]
             if x.get("quoteAsset")=="USDT" and x.get("status")=="TRADING"]
        return sorted(set(out)), "Binance fallback"
    except Exception:
        return [], "Unavailable"

def sym(x): return str(x).upper().replace("/","").replace("-","").replace("_","")

def binance_klines(symbol, interval):
    r=S.get(f"{BINANCE}/api/v3/klines",
            params={"symbol":sym(symbol),"interval":interval,"limit":LIMIT},
            timeout=TIMEOUT)
    r.raise_for_status()
    d=r.json()
    if len(d)<100: raise ValueError("insufficient")
    df=pd.DataFrame(d,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","i"])
    for c in ["o","h","l","c","v"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})[
        ["time","open","high","low","close","volume"]].dropna()

def okx_klines(symbol, interval):
    bars={"5m":"5m","15m":"15m","30m":"30m","1h":"1H","2h":"2H","4h":"4H",
          "6h":"6H","12h":"12H","1d":"1D","3d":"3D","1w":"1W"}
    inst=sym(symbol)[:-4]+"-USDT"
    r=S.get(f"{OKX}/api/v5/market/candles",
            params={"instId":inst,"bar":bars.get(interval,interval),"limit":"300"},
            timeout=TIMEOUT)
    r.raise_for_status()
    d=r.json().get("data",[])
    if len(d)<100: raise ValueError("insufficient")
    d=list(reversed(d))
    return pd.DataFrame([[pd.to_datetime(int(x[0]),unit="ms"),float(x[1]),float(x[2]),
                           float(x[3]),float(x[4]),float(x[5])] for x in d],
                        columns=["time","open","high","low","close","volume"])

def candles(symbol, interval):
    for f in (binance_klines, okx_klines):
        try: return f(symbol,interval), f.__name__
        except Exception: pass
    return None,None

# ============================================================
# INDICATORS
# ============================================================
def ema(x,n): return x.ewm(span=n,adjust=False).mean()

def rsi(x,n=14):
    d=x.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    a=up.ewm(alpha=1/n,adjust=False).mean()
    b=dn.ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+a/b.replace(0,np.nan))).fillna(50)

def atr(df,n=14):
    pc=df.close.shift()
    tr=pd.concat([df.high-df.low,(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def macd(x):
    m=ema(x,12)-ema(x,26); q=ema(m,9)
    return m,q,m-q

def adx(df,n=14):
    up=df.high.diff(); dn=-df.low.diff()
    plus=up.where((up>dn)&(up>0),0); minus=dn.where((dn>up)&(dn>0),0)
    pc=df.close.shift()
    tr=pd.concat([df.high-df.low,(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    av=tr.ewm(alpha=1/n,adjust=False).mean()
    p=100*plus.ewm(alpha=1/n,adjust=False).mean()/av.replace(0,np.nan)
    m=100*minus.ewm(alpha=1/n,adjust=False).mean()/av.replace(0,np.nan)
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean().fillna(0)

def trend(df):
    c=df.close
    e20,e50,e200=ema(c,20),ema(c,50),ema(c,200)
    if c.iloc[-1]>e20.iloc[-1]>e50.iloc[-1]>e200.iloc[-1]: return "صعودی"
    if c.iloc[-1]<e20.iloc[-1]<e50.iloc[-1]<e200.iloc[-1]: return "نزولی"
    return "خنثی"

# ============================================================
# HISTORICAL BACKTEST
# ============================================================
def backtest(df, direction, lookahead=12):
    if len(df)<150: return None
    c=df.close.values; a=atr(df).values
    wins=0; losses=0; trades=0
    start=60; end=len(df)-lookahead-1
    for i in range(start,end):
        e20=pd.Series(c[:i+1]).ewm(span=20,adjust=False).mean().iloc[-1]
        e50=pd.Series(c[:i+1]).ewm(span=50,adjust=False).mean().iloc[-1]
        rr=pd.Series(c[:i+1])
        rv=rsi(rr).iloc[-1]
        if direction=="LONG":
            cond=e20>e50 and rv>=52 and rv<=72
            if not cond: continue
            entry=c[i]; risk=max(a[i],entry*.008); sl=entry-1.5*risk; tp=entry+2*risk
            future=df.iloc[i+1:i+1+lookahead]
            hit=None
            for _,z in future.iterrows():
                if z.low<=sl: hit=False; break
                if z.high>=tp: hit=True; break
        else:
            cond=e20<e50 and rv>=28 and rv<=48
            if not cond: continue
            entry=c[i]; risk=max(a[i],entry*.008); sl=entry+1.5*risk; tp=entry-2*risk
            future=df.iloc[i+1:i+1+lookahead]
            hit=None
            for _,z in future.iterrows():
                if z.high>=sl: hit=False; break
                if z.low<=tp: hit=True; break
        if hit is not None:
            trades+=1; wins+=int(hit); losses+=int(not hit)
    if trades<8: return None
    return round(100*wins/trades,1), trades

# ============================================================
# ANALYSIS
# ============================================================
def analyze(symbol, primary_tf):
    frames={}
    for tf in ["15m","1H","4H","1D"]:
        df,src=candles(symbol,TF[tf])
        if df is None: return None
        frames[tf]=df
    df=frames[primary_tf] if primary_tf in frames else frames["1H"]
    c=df.close; last=float(c.iloc[-1])
    e20,e50,e200=ema(c,20),ema(c,50),ema(c,200)
    rv=rsi(c); m,ms,mh=macd(c); av=atr(df); ax=adx(df)
    mom=(last/float(c.iloc[-6])-1)*100
    vr=float(df.volume.iloc[-1]/df.volume.rolling(20).mean().iloc[-1])
    t15,t1,t4,tD=[trend(frames[x]) for x in ["15m","1H","4H","1D"]]

    bull=sum(x=="صعودی" for x in [t15,t1,t4,tD])
    bear=sum(x=="نزولی" for x in [t15,t1,t4,tD])
    if bull>=3: regime="صعودی"
    elif bear>=3: regime="نزولی"
    else: regime="رنج"

    score=50
    score += 10 if last>e20.iloc[-1] else -10
    score += 8 if e20.iloc[-1]>e50.iloc[-1] else -8
    score += 7 if e50.iloc[-1]>e200.iloc[-1] else -7
    score += 7 if m.iloc[-1]>ms.iloc[-1] else -7
    score += 5 if 52<=rv.iloc[-1]<=68 else (-5 if rv.iloc[-1]<35 or rv.iloc[-1]>78 else 0)
    score += 5 if mom>0 else -5
    score += 4 if ax.iloc[-1]>=20 else 0
    score += 4 if vr>=1.15 else 0
    score += 10 if regime=="صعودی" else (-10 if regime=="نزولی" else 0)
    score=float(np.clip(score,0,100))

    # Only issue a trade when higher timeframe agrees.
    long_ok=(regime=="صعودی" and t1=="صعودی" and
             m.iloc[-1]>ms.iloc[-1] and 50<rv.iloc[-1]<72 and mom>0)
    short_ok=(regime=="نزولی" and t1=="نزولی" and
              m.iloc[-1]<ms.iloc[-1] and 28<rv.iloc[-1]<50 and mom<0)

    if long_ok:
        direction="LONG"; trade="خرید"
    elif short_ok:
        direction="SHORT"; trade="فروش"
    else:
        direction="WAIT"; trade="صبر"

    bt=backtest(df,direction if direction!="WAIT" else ("LONG" if score>=50 else "SHORT"))
    historical=bt[0] if bt else None
    sample=bt[1] if bt else 0
    probability=round((score*.45)+(historical*.55),1) if historical is not None else round(score*.75,1)

    risk=max(float(av.iloc[-1]),last*.008)
    support=float(df.low.tail(40).min()); resistance=float(df.high.tail(40).max())
    if direction=="LONG":
        entry=last; sl=max(support*.995,entry-1.5*risk)
        tp1=entry+1.5*(entry-sl); tp2=entry+2.5*(entry-sl)
        rr=(tp2-entry)/(entry-sl) if entry>sl else 0
    elif direction=="SHORT":
        entry=last; sl=min(resistance*1.005,entry+1.5*risk)
        tp1=entry-1.5*(sl-entry); tp2=entry-2.5*(sl-entry)
        rr=(entry-tp2)/(sl-entry) if sl>entry else 0
    else:
        entry=last; sl=support; tp1=resistance; tp2=resistance; rr=0

    return {
        "symbol":symbol,"price":last,"trade":trade,"direction":direction,
        "probability":probability,"score":round(score,1),"historical":historical,
        "sample":sample,"regime":regime,"t15":t15,"t1":t1,"t4":t4,"tD":tD,
        "rsi":round(float(rv.iloc[-1]),1),"adx":round(float(ax.iloc[-1]),1),
        "momentum":round(float(mom),2),"volume":round(vr,2),
        "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":round(rr,2),
    }

def fmt(x):
    try:
        x=float(x)
        return f"{x:,.2f}" if x>=100 else (f"{x:,.5f}" if x>=.01 else f"{x:.9f}")
    except: return "-"

@st.cache_data(ttl=120,show_spinner=False)
def scan(symbols,tf):
    out=[]
    with ThreadPoolExecutor(max_workers=min(10,max(2,len(symbols)))) as ex:
        fs={ex.submit(analyze,s,tf):s for s in symbols}
        for f in as_completed(fs):
            try:
                x=f.result()
                if x: out.append(x)
            except: pass
    return pd.DataFrame(out)

# ============================================================
# UI
# ============================================================
st.title("₿ Crypto Analyzer Pro V5")
st.caption("Multi-timeframe • Backtest • Risk Management • Market Regime")

symbols,source=universe()
if not symbols:
    st.error("فهرست ارزها دریافت نشد.")
    st.stop()

with st.sidebar:
    st.write(f"منبع بازار: **{source}**")
    tf=st.selectbox("تایم‌فریم ورود",list(TF.keys()),index=3)
    q=st.text_input("جستجوی ارز","")
    max_scan=st.slider("تعداد ارز اسکن",10,min(250,len(symbols)),min(80,len(symbols)))
    risk_pct=st.slider("ریسک هر معامله (%)",0.25,2.0,1.0,0.25)
    capital=st.number_input("سرمایه (USDT)",100.0,1000000.0,1000.0,100.0)

filtered=[s for s in symbols if q.upper() in s] if q else symbols

c1,c2=st.columns(2)
with c1:
    selected=st.multiselect("حداکثر ۵ ارز",filtered[:500],max_selections=5)
with c2:
    run=st.button("اجرای تحلیل",use_container_width=True)

if run and selected:
    rows=[]
    with st.spinner("تحلیل چندتایم‌فریمی و بک‌تست..."):
        for s in selected:
            try:
                x=analyze(s,tf)
                if x: rows.append(x)
            except: pass
    if not rows: st.warning("داده کافی دریافت نشد.")
    for x in rows:
        st.subheader(f"{x['symbol']} — {x['trade']}")
        a,b,c,d,e,f,g=st.columns(7)
        a.metric("احتمال",f"{x['probability']}%")
        b.metric("رژیم",x["regime"])
        c.metric("ورود",fmt(x["entry"]))
        d.metric("SL",fmt(x["sl"]))
        e.metric("TP1",fmt(x["tp1"]))
        f.metric("TP2",fmt(x["tp2"]))
        g.metric("R:R",x["rr"])
        st.write(f"15m: {x['t15']} | 1H: {x['t1']} | 4H: {x['t4']} | 1D: {x['tD']} | "
                 f"RSI {x['rsi']} | ADX {x['adx']} | مومنتوم {x['momentum']}% | حجم {x['volume']}x")
        if x["direction"]!="WAIT":
            risk_money=capital*risk_pct/100
            distance=abs(x["entry"]-x["sl"])
            qty=risk_money/distance if distance else 0
            st.info(f"حجم پیشنهادی بر اساس ریسک {risk_pct}%: {qty:.6g} واحد | "
                    f"ریسک دلاری: {risk_money:.2f} USDT | بک‌تست: "
                    f"{x['historical']}% از {x['sample']} معامله" if x["historical"] else
                    f"حجم پیشنهادی: {qty:.6g} واحد | ریسک: {risk_money:.2f} USDT")

st.divider()
st.subheader("اسکنر بازار — فقط موقعیت‌های تأییدشده")
if st.button("اسکن بازار"):
    with st.spinner("در حال اسکن بازار..."):
        df=scan(tuple(filtered[:max_scan]),tf)
    if df.empty:
        st.warning("نتیجه‌ای با داده کافی دریافت نشد.")
    else:
        # Strict trade filter: no forced signals.
        trade_df=df[(df.direction!="WAIT") & (df.probability>=60) & (df.rr>=1.5)].copy()
        trade_df=trade_df.sort_values(["probability","rr"],ascending=False)
        if trade_df.empty:
            st.info("در این اسکن هیچ معامله‌ای شرایط ورود تأییدشده ندارد.")
        else:
            show=trade_df[["symbol","trade","probability","regime","price","entry","sl","tp1","tp2","rr","historical","sample"]].copy()
            show.columns=["ارز","نوع معامله","احتمال","رژیم","قیمت","ورود","حد ضرر","هدف ۱","هدف ۲","R:R","بک‌تست %","نمونه"]
            for col in ["قیمت","ورود","حد ضرر","هدف ۱","هدف ۲"]: show[col]=show[col].map(fmt)
            show["احتمال"]=show["احتمال"].map(lambda x:f"{x:.1f}%")
            show["بک‌تست %"]=show["بک‌تست %"].map(lambda x:f"{x:.1f}%" if pd.notna(x) else "-")
            st.dataframe(show,use_container_width=True,hide_index=True)

st.caption("این سیستم تصمیم‌یار است، نه تضمین سود. بک‌تست گذشته تضمین عملکرد آینده نیست. معامله خودکار در این نسخه فعال نیست.")
