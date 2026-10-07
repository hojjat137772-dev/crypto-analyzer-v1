import streamlit as st
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

st.set_page_config(page_title="Crypto Analyzer Pro V6", page_icon="₿", layout="wide")

# ============================================================
# V6 = V5 BASE + PROFESSIONAL DECISION ENGINE
# ============================================================
TF = {"5m":"5m","15m":"15m","30m":"30m","1H":"1h","2H":"2h",
      "4H":"4h","6H":"6h","12H":"12h","1D":"1d","3D":"3d","1W":"1w"}

BINANCE="https://api.binance.com"
OKX="https://www.okx.com"
TABDEAL="https://api.tabdeal.org"
TIMEOUT=8
LIMIT=320

S=requests.Session()
S.headers.update({"User-Agent":"CryptoAnalyzerPro-V6/1.0"})

# ---------------- DATA ----------------
@st.cache_data(ttl=900, show_spinner=False)
def universe():
    urls=[
        f"{TABDEAL}/r/api/v1/exchangeInfo",
        f"{TABDEAL}/api/v1/exchangeInfo",
        f"{TABDEAL}/v1/market/symbols",
        f"{TABDEAL}/v1/markets",
        f"{TABDEAL}/api/v1/markets"
    ]
    for u in urls:
        try:
            r=S.get(u,timeout=TIMEOUT)
            if not r.ok: continue
            d=r.json()
            items=d.get("symbols") if isinstance(d,dict) else d
            if items is None and isinstance(d,dict): items=d.get("data")
            out=[]
            for x in items if isinstance(items,list) else []:
                raw=str(x.get("symbol") or x.get("name") or x.get("code") or "")
                raw=raw.upper().replace("-","").replace("_","").replace("/","")
                if raw.endswith("USDT"): out.append(raw)
            if out: return sorted(set(out)),"Tabdeal"
        except: pass
    try:
        d=S.get(f"{BINANCE}/api/v3/exchangeInfo",timeout=TIMEOUT).json()
        out=[x["symbol"] for x in d["symbols"]
             if x.get("quoteAsset")=="USDT" and x.get("status")=="TRADING"]
        return sorted(set(out)),"Binance fallback"
    except: return [],"Unavailable"

def norm(x): return str(x).upper().replace("/","").replace("-","").replace("_","")

def binance(symbol,interval):
    r=S.get(f"{BINANCE}/api/v3/klines",
            params={"symbol":norm(symbol),"interval":interval,"limit":LIMIT},
            timeout=TIMEOUT)
    r.raise_for_status(); d=r.json()
    if len(d)<120: raise ValueError("insufficient")
    df=pd.DataFrame(d,columns=["t","o","h","l","c","v","ct","q","n","tb","tq","i"])
    for c in ["o","h","l","c","v"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})[
        ["time","open","high","low","close","volume"]].dropna()

def okx(symbol,interval):
    bars={"5m":"5m","15m":"15m","30m":"30m","1h":"1H","2h":"2H","4h":"4H",
          "6h":"6H","12h":"12H","1d":"1D","3d":"3D","1w":"1W"}
    inst=norm(symbol)[:-4]+"-USDT"
    r=S.get(f"{OKX}/api/v5/market/candles",
            params={"instId":inst,"bar":bars.get(interval,interval),"limit":"300"},
            timeout=TIMEOUT)
    r.raise_for_status(); d=r.json().get("data",[])
    if len(d)<120: raise ValueError("insufficient")
    d=list(reversed(d))
    return pd.DataFrame([[pd.to_datetime(int(x[0]),unit="ms"),float(x[1]),float(x[2]),
                           float(x[3]),float(x[4]),float(x[5])] for x in d],
                        columns=["time","open","high","low","close","volume"])

def candles(symbol,interval):
    for f in (binance,okx):
        try: return f(symbol,interval),f.__name__
        except: pass
    return None,None

# ---------------- INDICATORS ----------------
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
    m=ema(x,12)-ema(x,26); s=ema(m,9)
    return m,s,m-s

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

def bollinger(x,n=20,k=2):
    mid=x.rolling(n).mean(); sd=x.rolling(n).std()
    return mid,mid+k*sd,mid-k*sd

def trend(df):
    c=df.close; e20,e50,e200=ema(c,20),ema(c,50),ema(c,200)
    if c.iloc[-1]>e20.iloc[-1]>e50.iloc[-1]>e200.iloc[-1]: return "صعودی"
    if c.iloc[-1]<e20.iloc[-1]<e50.iloc[-1]<e200.iloc[-1]: return "نزولی"
    return "رنج"

# ---------------- PRICE ACTION ENGINE ----------------
def _swing_points(df, left=3, right=3):
    h=df.high.reset_index(drop=True); l=df.low.reset_index(drop=True)
    sh=[]; sl=[]
    for i in range(left, len(df)-right):
        if h.iloc[i] >= h.iloc[i-left:i].max() and h.iloc[i] > h.iloc[i+1:i+right+1].max(): sh.append(i)
        if l.iloc[i] <= l.iloc[i-left:i].min() and l.iloc[i] < l.iloc[i+1:i+right+1].min(): sl.append(i)
    return sh,sl

def price_action_engine(df):
    if len(df)<60:
        return {"structure":"نامشخص","bos":"—","choch":"—","sweep":"—","retest":"—","fake_breakout":"—","pa_long":45.0,"pa_short":45.0,"pa_score":45.0}
    sh,sl=_swing_points(df)
    c=float(df.close.iloc[-1]); hi=float(df.high.iloc[-1]); lo=float(df.low.iloc[-1])
    last_sh=[i for i in sh if i < len(df)-2][-3:]
    last_sl=[i for i in sl if i < len(df)-2][-3:]
    structure="رنج"
    if len(last_sh)>=2 and len(last_sl)>=2:
        hh=df.high.iloc[last_sh[-1]]>df.high.iloc[last_sh[-2]]
        hl=df.low.iloc[last_sl[-1]]>df.low.iloc[last_sl[-2]]
        lh=df.high.iloc[last_sh[-1]]<df.high.iloc[last_sh[-2]]
        ll=df.low.iloc[last_sl[-1]]<df.low.iloc[last_sl[-2]]
        if hh and hl: structure="HH/HL صعودی"
        elif lh and ll: structure="LH/LL نزولی"
    swing_hi=float(df.high.iloc[last_sh[-1]]) if last_sh else float(df.high.tail(20).max())
    swing_lo=float(df.low.iloc[last_sl[-1]]) if last_sl else float(df.low.tail(20).min())
    prev_hi=float(df.high.iloc[-2]); prev_lo=float(df.low.iloc[-2])
    bos_up=c>swing_hi and prev_hi<=swing_hi
    bos_down=c<swing_lo and prev_lo>=swing_lo
    choch_up=structure=="LH/LL نزولی" and c>swing_hi
    choch_down=structure=="HH/HL صعودی" and c<swing_lo
    recent_hi=float(df.high.tail(25).iloc[:-1].max()); recent_lo=float(df.low.tail(25).iloc[:-1].min())
    sweep_up=lo<recent_lo and c>recent_lo
    sweep_down=hi>recent_hi and c<recent_hi
    atrv=float(atr(df).iloc[-1])
    retest_up=(c>swing_hi and abs(c-swing_hi)<=max(atrv*.45,c*.004))
    retest_down=(c<swing_lo and abs(c-swing_lo)<=max(atrv*.45,c*.004))
    fake_up=(hi>recent_hi and c<recent_hi and not sweep_down)
    fake_down=(lo<recent_lo and c>recent_lo and not sweep_up)
    pl=45.0; ps=45.0
    if structure=="HH/HL صعودی": pl+=20
    if structure=="LH/LL نزولی": ps+=20
    if bos_up: pl+=18
    if bos_down: ps+=18
    if choch_up: pl+=12
    if choch_down: ps+=12
    if sweep_up: pl+=12
    if sweep_down: ps+=12
    if retest_up: pl+=10
    if retest_down: ps+=10
    if fake_up: ps+=15; pl-=15
    if fake_down: pl+=15; ps-=15
    pl=float(np.clip(pl,0,100)); ps=float(np.clip(ps,0,100))
    return {
        "structure":structure,
        "bos":"BOS صعودی" if bos_up else "BOS نزولی" if bos_down else "—",
        "choch":"CHOCH صعودی" if choch_up else "CHOCH نزولی" if choch_down else "—",
        "sweep":"Liquidity Sweep صعودی" if sweep_up else "Liquidity Sweep نزولی" if sweep_down else "—",
        "retest":"Retest صعودی" if retest_up else "Retest نزولی" if retest_down else "—",
        "fake_breakout":"Fake Breakout صعودی" if fake_up else "Fake Breakout نزولی" if fake_down else "—",
        "pa_long":pl,"pa_short":ps,"pa_score":max(pl,ps),
        "swing_high":swing_hi,"swing_low":swing_lo,
    }

# ---------------- MARKET CONTEXT ----------------
@st.cache_data(ttl=180,show_spinner=False)
def market_context():
    out={"btc_trend":"نامشخص","btc_momentum":0.0,"breadth":50.0,"breadth_long":50.0,"breadth_short":50.0,"regime":"نامشخص"}
    try:
        b,_=candles("BTCUSDT","1h")
        if b is not None:
            out["btc_trend"]=trend(b)
            out["btc_momentum"]=round((b.close.iloc[-1]/b.close.iloc[-25]-1)*100,2)
    except: pass
    return out

def breadth_from_results(rows):
    if not rows: return {"breadth":50.0,"breadth_long":50.0,"breadth_short":50.0,"regime":"خنثی"}
    n=len(rows); bull=sum(r.get("market_regime")=="صعودی" for r in rows); bear=sum(r.get("market_regime")=="نزولی" for r in rows)
    long_pct=100*bull/n; short_pct=100*bear/n
    regime="Bull Market" if long_pct>=60 else "Bear Market" if short_pct>=60 else "Range Market"
    return {"breadth":round(long_pct-short_pct+50,1),"breadth_long":round(long_pct,1),"breadth_short":round(short_pct,1),"regime":regime}

def relative_strength(df,bdf):
    try:
        n=min(24,len(df)-1,len(bdf)-1)
        r1=float(df.close.iloc[-1]/df.close.iloc[-1-n]-1)
        r2=float(bdf.close.iloc[-1]/bdf.close.iloc[-1-n]-1)
        return round((r1-r2)*100,2)
    except: return 0.0

# ---------------- PROFESSIONAL BACKTEST ----------------
def backtest(df,direction,lookahead=18,fee_pct=0.10,slippage_pct=0.05):
    if len(df)<180: return None
    wins=losses=trades=0; profits=[]
    c=df.close; av=atr(df)
    start=80; end=len(df)-lookahead-1
    for i in range(start,end):
        hist=df.iloc[:i+1]; hc=hist.close
        e20=ema(hc,20).iloc[-1]; e50=ema(hc,50).iloc[-1]; rv=rsi(hc).iloc[-1]
        mm,ss,_=macd(hc); vol=hist.volume.iloc[-1]/max(hist.volume.rolling(20).mean().iloc[-1],1e-12)
        entry=float(c.iloc[i]); risk=max(float(av.iloc[i]),entry*.008); future=df.iloc[i+1:i+1+lookahead]
        if direction=="LONG":
            condition=e20>e50 and 50<rv<72 and mm.iloc[-1]>ss.iloc[-1] and vol>=.8
            sl=entry-1.5*risk; tp=entry+2*risk
        else:
            condition=e20<e50 and 28<rv<50 and mm.iloc[-1]<ss.iloc[-1] and vol>=.8
            sl=entry+1.5*risk; tp=entry-2*risk
        if not condition: continue
        result=None
        for _,z in future.iterrows():
            if direction=="LONG":
                if z.low<=sl: result=-1; break
                if z.high>=tp: result=2; break
            else:
                if z.high>=sl: result=-1; break
                if z.low<=tp: result=2; break
        if result is not None:
            # Round-trip fee + slippage converted approximately to R.
            cost_pct=2*(fee_pct+slippage_pct)/100
            cost_r=cost_pct*entry/risk
            net=result-cost_r
            trades+=1; wins+=net>0; losses+=net<=0; profits.append(net)
    if trades<10: return None
    wr=100*wins/trades; gp=sum(x for x in profits if x>0); gl=abs(sum(x for x in profits if x<0)); pf=gp/gl if gl else 99
    eq=np.cumsum(profits); peak=np.maximum.accumulate(eq); dd=float(np.max(peak-eq)) if len(eq) else 0
    return {"winrate":round(wr,1),"trades":trades,"profit_factor":round(pf,2),"drawdown_r":round(dd,2),"expectancy_r":round(float(np.mean(profits)),2),"fees_slippage_pct":round(2*(fee_pct+slippage_pct),3)}

def walk_forward(df,direction):
    if len(df)<240: return None
    cut=int(len(df)*.70); oos=backtest(df.iloc[cut:].reset_index(drop=True),direction)
    if not oos or oos["trades"]<10: return None
    return {"oos_winrate":oos["winrate"],"oos_trades":oos["trades"],"oos_pf":oos["profit_factor"],"oos_expectancy":oos["expectancy_r"]}

# ---------------- SIGNAL ENGINE ----------------
def analyze(symbol,entry_tf,capital,risk_pct):
    frames={}
    for tf in ["15m","1H","4H","1D"]:
        df,src=candles(symbol,TF[tf])
        if df is None: return None
        frames[tf]=df

    ctx=market_context()
    btc_df,_=candles("BTCUSDT","1h")
    df=frames.get(entry_tf,frames["1H"])
    c=df.close; last=float(c.iloc[-1])
    e20,e50,e200=ema(c,20),ema(c,50),ema(c,200)
    rv=rsi(c); mm,ms,mh=macd(c); av=atr(df); ax=adx(df)
    mid,up,lo=bollinger(c)
    mom=(last/float(c.iloc[-6])-1)*100
    rs_btc=relative_strength(df,btc_df) if btc_df is not None else 0.0
    dollar_volume=float((df.close*df.volume).tail(20).mean())
    liquidity_score=float(np.clip(50+math.log10(max(dollar_volume,1)/100000)*15,0,100))
    pa=price_action_engine(df)
    vol_ratio=float(df.volume.iloc[-1]/df.volume.rolling(20).mean().iloc[-1])

    t15,t1,t4,tD=[trend(frames[x]) for x in ["15m","1H","4H","1D"]]
    bull=sum(t=="صعودی" for t in [t15,t1,t4,tD])
    bear=sum(t=="نزولی" for t in [t15,t1,t4,tD])
    regime="صعودی" if bull>=3 else ("نزولی" if bear>=3 else "رنج")

    # Component scores, 0..100
    trend_score=100 if t1==t4==tD=="صعودی" else 75 if bull>=3 else 50 if bull==2 else 25 if bear>=3 else 40
    momentum_score=float(np.clip(50+mom*8,0,100))
    volume_score=float(np.clip(50+(vol_ratio-1)*45,0,100))
    rsi_score=85 if 52<=rv.iloc[-1]<=68 else 65 if 45<=rv.iloc[-1]<=75 else 30
    macd_score=80 if mm.iloc[-1]>ms.iloc[-1] else 25
    adx_score=float(np.clip(40+ax.iloc[-1]*2,0,100))

    recent_high=float(df.high.tail(30).iloc[:-1].max())
    recent_low=float(df.low.tail(30).iloc[:-1].min())
    breakout_up=last>=recent_high*.998
    breakout_down=last<=recent_low*1.002
    pullback_long=last>e50.iloc[-1] and abs(last-e20.iloc[-1])/last<.015
    pullback_short=last<e50.iloc[-1] and abs(last-e20.iloc[-1])/last<.015
    pa_long=float(max(pa["pa_long"],90 if breakout_up else 78 if pullback_long else 0))
    pa_short=float(max(pa["pa_short"],90 if breakout_down else 78 if pullback_short else 0))

    # Relative strength, breadth and liquidity are additional decision layers.
    rs_long=float(np.clip(50+rs_btc*8,0,100)); rs_short=float(np.clip(50-rs_btc*8,0,100))
    breadth_long=ctx.get("breadth_long",50.0); breadth_short=ctx.get("breadth_short",50.0)
    long_raw=(trend_score*.20 + momentum_score*.12 + volume_score*.10 +
              rsi_score*.08 + macd_score*.08 + adx_score*.07 + pa_long*.17 +
              rs_long*.07 + breadth_long*.06 + liquidity_score*.05)
    short_raw=((100-trend_score)*.20 + (100-momentum_score)*.12 + volume_score*.10 +
              (100-rsi_score)*.08 + (100-macd_score)*.08 + adx_score*.07 + pa_short*.17 +
              rs_short*.07 + breadth_short*.06 + liquidity_score*.05)

    if ctx["btc_trend"]=="نزولی":
        long_raw-=10
    if ctx["btc_trend"]=="صعودی":
        short_raw-=10

    long_ok=(regime=="صعودی" and t1=="صعودی" and mm.iloc[-1]>ms.iloc[-1] and
             48<rv.iloc[-1]<73 and mom>0 and (breakout_up or pullback_long or pa["bos"]=="BOS صعودی" or pa["retest"]=="Retest صعودی") and pa["fake_breakout"]!="Fake Breakout صعودی" and rs_btc>-2)
    short_ok=(regime=="نزولی" and t1=="نزولی" and mm.iloc[-1]<ms.iloc[-1] and
              27<rv.iloc[-1]<52 and mom<0 and (breakout_down or pullback_short or pa["bos"]=="BOS نزولی" or pa["retest"]=="Retest نزولی") and pa["fake_breakout"]!="Fake Breakout نزولی" and rs_btc<2)

    if long_ok and long_raw>=62:
        direction="LONG"; trade="خرید"
    elif short_ok and short_raw>=62:
        direction="SHORT"; trade="فروش"
    else:
        direction="WAIT"; trade="صبر"

    test_direction="LONG" if direction=="WAIT" and long_raw>=short_raw else "SHORT" if direction=="WAIT" else direction
    bt=backtest(df,test_direction)
    wf=walk_forward(df,test_direction)
    hist_prob=bt["winrate"] if bt else 50.0
    oos_prob=wf["oos_winrate"] if wf else hist_prob
    score=float(np.clip(max(long_raw,short_raw),0,100))
    mtf_agreement=100*max(bull,bear)/4
    fake_penalty=18 if pa["fake_breakout"]!="—" else 0
    data_quality=100.0
    if any(len(frames[t])<200 for t in frames): data_quality-=12
    confidence=float(np.clip(score*.25 + hist_prob*.25 + oos_prob*.20 + mtf_agreement*.10 + data_quality*.10 + liquidity_score*.05 + max(breadth_long,breadth_short)*.05 - fake_penalty,0,100))

    risk=max(float(av.iloc[-1]),last*.008)
    support=float(df.low.tail(50).min()); resistance=float(df.high.tail(50).max())

    if direction=="LONG":
        entry=last; sl=max(support*.995,entry-1.5*risk)
        tp1=entry+1.5*(entry-sl); tp2=entry+2.5*(entry-sl); tp3=entry+3.5*(entry-sl)
        rr=(tp2-entry)/(entry-sl) if entry>sl else 0
    elif direction=="SHORT":
        entry=last; sl=min(resistance*1.005,entry+1.5*risk)
        tp1=entry-1.5*(sl-entry); tp2=entry-2.5*(sl-entry); tp3=entry-3.5*(sl-entry)
        rr=(entry-tp2)/(sl-entry) if sl>entry else 0
    else:
        entry=last; sl=support; tp1=resistance; tp2=resistance; tp3=resistance; rr=0

    risk_money=capital*risk_pct/100
    distance=abs(entry-sl)
    qty=risk_money/distance if distance else 0

    # Pump detector
    pump_score=0
    pump_score += 25 if vol_ratio>=1.8 else 15 if vol_ratio>=1.35 else 0
    pump_score += 25 if mom>=4 else 15 if mom>=2 else 0
    pump_score += 25 if breakout_up else 12 if pullback_long else 0
    pump_score += 15 if t15=="صعودی" and t1=="صعودی" else 0
    pump_score += 10 if rv.iloc[-1]<72 else 0
    pump_label="🚀 پامپ تأییدشده" if pump_score>=70 else "⚡ آماده حرکت" if pump_score>=45 else "—"

    setup=("Breakout" if breakout_up or breakout_down else
           "Pullback" if pullback_long or pullback_short else "No setup")

    probability=float(np.clip(score*.45 + hist_prob*.55,0,100))
    return {
        "symbol":symbol,"trade":trade,"direction":direction,"probability":round(probability,1),
        "score":round(score,1),"confidence":round(confidence,1),"trend_score":round(trend_score,1),"momentum_score":round(momentum_score,1),
        "volume_score":round(volume_score,1),"price_action":round(max(pa_long,pa_short),1),"structure":pa["structure"],"bos":pa["bos"],"choch":pa["choch"],"sweep":pa["sweep"],"retest":pa["retest"],"fake_breakout":pa["fake_breakout"],"relative_strength":rs_btc,"liquidity_score":round(liquidity_score,1),"dollar_volume":dollar_volume,"breadth_long":breadth_long,"breadth_short":breadth_short,
        "market_regime":regime,"btc_trend":ctx["btc_trend"],"btc_momentum":ctx["btc_momentum"],
        "t15":t15,"t1":t1,"t4":t4,"tD":tD,"rsi":round(float(rv.iloc[-1]),1),
        "adx":round(float(ax.iloc[-1]),1),"momentum":round(mom,2),"volume":round(vol_ratio,2),
        "setup":setup,"pump_score":round(pump_score,1),"pump":pump_label,
        "price":last,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,"rr":round(rr,2),
        "qty":qty,"risk_money":risk_money,"backtest":bt,"walk_forward":wf
    }

def fmt(x):
    try:
        x=float(x)
        return f"{x:,.2f}" if x>=100 else (f"{x:,.5f}" if x>=.01 else f"{x:.9f}")
    except: return "-"

@st.cache_data(ttl=120,show_spinner=False)
def scan(symbols,tf,capital,risk_pct):
    out=[]
    with ThreadPoolExecutor(max_workers=min(10,max(2,len(symbols)))) as ex:
        fs={ex.submit(analyze,s,tf,capital,risk_pct):s for s in symbols}
        for f in as_completed(fs):
            try:
                x=f.result()
                if x: out.append(x)
            except: pass
    bctx=breadth_from_results(out)
    for x in out:
        # Breadth is computed from the same scan universe, then blended into displayed confidence.
        x["breadth_long"]=bctx["breadth_long"]; x["breadth_short"]=bctx["breadth_short"]
        x["market_breadth_regime"]=bctx["regime"]
        x["confidence"]=round(float(np.clip(x["confidence"] + (bctx["breadth_long"]-50)*0.08 if x["direction"]=="LONG" else x["confidence"] + (bctx["breadth_short"]-50)*0.08 if x["direction"]=="SHORT" else x["confidence"],0,100)),1)
    return pd.DataFrame(out)

# ---------------- UI ----------------
st.title("₿ Crypto Analyzer Pro V6")
st.caption("Professional Multi-Timeframe Decision Engine")

symbols,source=universe()
if not symbols:
    st.error("فهرست ارزها دریافت نشد."); st.stop()

with st.sidebar:
    st.write(f"منبع فهرست: **{source}**")
    tf=st.selectbox("تایم‌فریم ورود",list(TF.keys()),index=3)
    query=st.text_input("جستجو","")
    max_scan=st.slider("تعداد ارز اسکن",10,min(250,len(symbols)),min(80,len(symbols)))
    capital=st.number_input("سرمایه USDT",100.0,1000000.0,1000.0,100.0)
    risk_pct=st.slider("ریسک هر معامله %",0.25,2.0,1.0,0.25)

filtered=[x for x in symbols if query.upper() in x] if query else symbols
selected=st.multiselect("انتخاب تا ۵ ارز",filtered[:500],max_selections=5)

if st.button("تحلیل حرفه‌ای",use_container_width=True) and selected:
    for s in selected:
        try:
            x=analyze(s,tf,capital,risk_pct)
            if not x: continue
            st.markdown("""
            <style>
            .trade-card{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:18px;margin:8px 0 18px 0;background:rgba(128,128,128,.055);box-shadow:0 4px 14px rgba(0,0,0,.08)}
            .trade-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
            .trade-title{font-size:22px;font-weight:700}.trade-badge{padding:5px 10px;border-radius:999px;background:rgba(0,128,0,.14);font-weight:700}
            .trade-grid{display:grid;grid-template-columns:repeat(8,minmax(90px,1fr));gap:8px}.trade-cell{padding:10px;border-radius:10px;background:rgba(128,128,128,.08)}
            .trade-label{font-size:12px;opacity:.72}.trade-value{font-size:16px;font-weight:650;margin-top:3px}
            @media(max-width:900px){.trade-grid{grid-template-columns:repeat(2,minmax(120px,1fr))}}
            </style>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="trade-card">
              <div class="trade-head"><div class="trade-title">{x['symbol']} — {x['trade']}</div><div class="trade-badge">{x['setup']} | Confidence {x['confidence']}%</div></div>
              <div class="trade-grid">
                <div class="trade-cell"><div class="trade-label">Score</div><div class="trade-value">{x['score']:.1f}</div></div>
                <div class="trade-cell"><div class="trade-label">Entry</div><div class="trade-value">{fmt(x['entry'])}</div></div>
                <div class="trade-cell"><div class="trade-label">Stop Loss</div><div class="trade-value">{fmt(x['sl'])}</div></div>
                <div class="trade-cell"><div class="trade-label">Target 1</div><div class="trade-value">{fmt(x['tp1'])}</div></div>
                <div class="trade-cell"><div class="trade-label">Target 2</div><div class="trade-value">{fmt(x['tp2'])}</div></div>
                <div class="trade-cell"><div class="trade-label">Target 3</div><div class="trade-value">{fmt(x['tp3'])}</div></div>
                <div class="trade-cell"><div class="trade-label">R:R</div><div class="trade-value">{x['rr']}</div></div>
                <div class="trade-cell"><div class="trade-label">Pump Score</div><div class="trade-value">{x['pump_score']}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.write(
                f"15m: {x['t15']} | 1H: {x['t1']} | 4H: {x['t4']} | 1D: {x['tD']} | "
                f"BTC: {x['btc_trend']} ({x['btc_momentum']}%) | RSI: {x['rsi']} | "
                f"ADX: {x['adx']} | Momentum: {x['momentum']}% | Volume: {x['volume']}x"
            )
            st.write(
                f"Trend: {x['trend_score']} | Momentum: {x['momentum_score']} | Volume: {x['volume_score']} | "
                f"Price Action: {x['price_action']} | Structure: {x['structure']} | {x['bos']} | {x['choch']} | "
                f"{x['sweep']} | {x['retest']} | {x['fake_breakout']} | RS/BTC: {x['relative_strength']}% | {x['pump']}"
            )
            if x["direction"]!="WAIT":
                st.success(f"حجم پیشنهادی: {x['qty']:.6g} واحد | ریسک: {x['risk_money']:.2f} USDT")
            else:
                st.warning("شرایط ورود کامل تأیید نشده؛ برنامه معامله را اجباری نمی‌کند.")
            if x["backtest"]:
                b=x["backtest"]
                st.caption(f"Backtest: Win Rate {b['winrate']}% | Trades {b['trades']} | PF {b['profit_factor']} | Max DD {b['drawdown_r']}R | Expectancy {b['expectancy_r']}R | Cost {b['fees_slippage_pct']}%")
            if x.get("walk_forward"):
                w=x["walk_forward"]; st.caption(f"Walk-Forward OOS: {w['oos_winrate']}% | Trades {w['oos_trades']} | PF {w['oos_pf']} | Expectancy {w['oos_expectancy']}R")
        except Exception as e:
            st.warning(f"{s}: داده کافی یا معتبر دریافت نشد.")

st.divider()
st.subheader("اسکنر حرفه‌ای بازار")

if st.button("اسکن بازار"):
    with st.spinner("در حال اسکن چندتایم‌فریمی..."):
        df=scan(tuple(filtered[:max_scan]),tf,capital,risk_pct)

    if df.empty:
        st.warning("نتیجه‌ای دریافت نشد.")
    else:
        # Strict scanner: only confirmed setups.
        trade=df[(df.direction!="WAIT") & (df.confidence>=60) & (df.rr>=1.5) & (df.fake_breakout=="—") & (df.liquidity_score>=35)].copy()
        trade=trade.sort_values(["confidence","score","rr"],ascending=False)

        if trade.empty:
            st.info("در این اسکن هیچ موقعیت تأییدشده‌ای پیدا نشد.")
        else:
            show=trade[["symbol","trade","confidence","score","market_regime","setup","structure","bos","relative_strength","price",
                        "entry","sl","tp1","tp2","rr","pump_score","btc_trend","liquidity_score"]].copy()
            show.columns=["ارز","نوع معامله","Confidence","Score","رژیم","ستاپ","ساختار","BOS","قدرت نسبت به BTC %","قیمت",
                          "ورود","حد ضرر","هدف ۱","هدف ۲","R:R","Pump Score","BTC","نقدشوندگی"]
            for c in ["قیمت","ورود","حد ضرر","هدف ۱","هدف ۲"]: show[c]=show[c].map(fmt)
            show["Confidence"]=show["Confidence"].map(lambda x:f"{x:.1f}%")
            st.dataframe(show,use_container_width=True,hide_index=True)
            bc=breadth_from_results(df.to_dict("records"))
            st.caption(f"Market Breadth: Long {bc['breadth_long']}% | Short {bc['breadth_short']}% | وضعیت: {bc['regime']}")

        st.subheader("🚀 Pump Candidates")
        pump=df[df.pump_score>=45].sort_values("pump_score",ascending=False).head(20)
        if pump.empty: st.info("کاندید پامپ پیدا نشد.")
        else:
            p=pump[["symbol","pump_score","pump","momentum","volume","setup","probability"]].copy()
            p.columns=["ارز","Pump Score","وضعیت","مومنتوم %","حجم x","ستاپ","امتیاز"]
            p["مومنتوم %"]=p["مومنتوم %"].map(lambda x:f"{x:.2f}%")
            p["امتیاز"]=p["امتیاز"].map(lambda x:f"{x:.1f}%")
            st.dataframe(p,use_container_width=True,hide_index=True)

st.caption("این برنامه تصمیم‌یار است؛ احتمال و بک‌تست گذشته تضمین سود آینده نیست. معامله خودکار در V6 فعال نیست.")
