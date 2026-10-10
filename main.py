import streamlit as st
import pandas as pd
import numpy as np
import requests, math, time
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor, as_completed


TF={'5m':'5m','15m':'15m','30m':'30m','1H':'1h','2H':'2h','4H':'4h','6H':'6h','12H':'12h','1D':'1d','3D':'3d','1W':'1w'}
BINANCE='https://api.binance.com'; OKX='https://www.okx.com'; TABDEAL='https://api.tabdeal.org'
S=requests.Session(); S.headers.update({'User-Agent':'CryptoAnalyzerPro-V6.8/1.0'})
TIMEOUT=8; LIMIT=320

# ---------- DATA ----------
def norm(s): return str(s).upper().replace('/','').replace('-','').replace('_','')

def get_json(url, params=None, timeout=TIMEOUT):
    try:
        r=S.get(url,params=params,timeout=timeout); r.raise_for_status(); return r.json()
    except Exception: return None

@st.cache_data(ttl=600, show_spinner=False)
def universe():
    urls=[f'{TABDEAL}{x}' for x in ['/r/api/v1/exchangeInfo','/api/v1/exchangeInfo','/v1/market/symbols','/v1/markets','/api/v1/markets']]
    for u in urls:
        d=get_json(u,timeout=12)
        if not d: continue
        items=d.get('symbols') if isinstance(d,dict) else d
        if items is None and isinstance(d,dict): items=d.get('data')
        out=[]
        if isinstance(items,list):
            for x in items:
                if isinstance(x,dict):
                    s=norm(x.get('symbol') or x.get('name') or x.get('code') or '')
                    if s.endswith('USDT'): out.append(s)
        if out: return sorted(set(out)),'Tabdeal'
    d=get_json(f'{BINANCE}/api/v3/exchangeInfo',timeout=12)
    try:
        out=[x['symbol'] for x in d['symbols'] if x.get('quoteAsset')=='USDT' and x.get('status')=='TRADING']
        return sorted(set(out)),'Binance fallback'
    except Exception: return [],'Unavailable'

def binance(symbol, interval):
    d=get_json(f'{BINANCE}/api/v3/klines',{'symbol':norm(symbol),'interval':interval,'limit':LIMIT})
    if not isinstance(d,list) or len(d)<80: raise ValueError('insufficient')
    df=pd.DataFrame(d,columns=['t','o','h','l','c','v','ct','q','n','tb','tq','i'])
    for c in ['o','h','l','c','v']: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.rename(columns={'t':'time','o':'open','h':'high','l':'low','c':'close','v':'volume'})[['time','open','high','low','close','volume']].dropna()

def okx(symbol, interval):
    bars={'5m':'5m','15m':'15m','30m':'30m','1h':'1H','2h':'2H','4h':'4H','6h':'6H','12h':'12H','1d':'1D','3d':'3D','1w':'1W'}
    inst=norm(symbol)[:-4]+'-USDT'; d=get_json(f'{OKX}/api/v5/market/candles',{'instId':inst,'bar':bars.get(interval,interval),'limit':'300'})
    rows=d.get('data',[]) if isinstance(d,dict) else []
    if len(rows)<80: raise ValueError('insufficient')
    rows=list(reversed(rows))
    return pd.DataFrame([[pd.to_datetime(int(x[0]),unit='ms'),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])] for x in rows],columns=['time','open','high','low','close','volume'])

def candles(symbol, interval):
    for f in (binance,okx):
        try: return f(symbol,interval),f.__name__
        except Exception: pass
    return None,None

# ---------- LIVE MARKET TICKER ----------
def live_tickers(symbols):
    """Fetch uncached 24h prices; use OKX's full SPOT ticker list if Binance fails."""
    wanted=set(norm(x) for x in symbols)
    rows=[]
    data=get_json(f'{BINANCE}/api/v3/ticker/24hr', timeout=8)
    if isinstance(data,list):
        for x in data:
            sym=norm(x.get('symbol',''))
            if sym not in wanted:
                continue
            try:
                price=float(x.get('lastPrice',0))
                if price > 0:
                    rows.append({'symbol':sym,'price':price,
                        'change_pct':float(x.get('priceChangePercent',0)),
                        'quote_volume':float(x.get('quoteVolume',0)),
                        'high_24h':float(x.get('highPrice',0)),
                        'low_24h':float(x.get('lowPrice',0)),'source':'Binance'})
            except (TypeError,ValueError):
                continue
    # If Binance is unavailable OR doesn't list the selected Tabdeal market, try OKX.
    if not rows:
        alt=get_json(f'{OKX}/api/v5/market/tickers', {'instType':'SPOT'}, timeout=10)
        items=alt.get('data',[]) if isinstance(alt,dict) and str(alt.get('code','0'))=='0' else []
        for x in items:
            inst=str(x.get('instId',''))
            if not inst.endswith('-USDT'):
                continue
            sym=norm(inst.replace('-',''))
            if sym not in wanted:
                continue
            try:
                price=float(x.get('last',0)); op=float(x.get('open24h',0))
                if price <= 0:
                    continue
                rows.append({'symbol':sym,'price':price,
                    'change_pct':((price/op)-1)*100 if op>0 else 0.0,
                    'quote_volume':float(x.get('volCcy24h',0) or 0),
                    'high_24h':float(x.get('high24h',0) or 0),
                    'low_24h':float(x.get('low24h',0) or 0),'source':'OKX'})
            except (TypeError,ValueError):
                continue
    return pd.DataFrame(rows)

# ---------- INDICATORS ----------
def ema(x,n): return x.ewm(span=n,adjust=False).mean()
def rsi(x,n=14):
    d=x.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0); a=up.ewm(alpha=1/n,adjust=False).mean(); b=dn.ewm(alpha=1/n,adjust=False).mean(); return (100-100/(1+a/b.replace(0,np.nan))).fillna(50)
def atr(df,n=14):
    pc=df.close.shift(); tr=pd.concat([df.high-df.low,(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean()
def macd(x):
    m=ema(x,12)-ema(x,26); s=ema(m,9); return m,s,m-s
def adx(df,n=14):
    up=df.high.diff(); dn=-df.low.diff(); plus=up.where((up>dn)&(up>0),0); minus=dn.where((dn>up)&(dn>0),0); pc=df.close.shift(); tr=pd.concat([df.high-df.low,(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1); av=tr.ewm(alpha=1/n,adjust=False).mean(); p=100*plus.ewm(alpha=1/n,adjust=False).mean()/av.replace(0,np.nan); m=100*minus.ewm(alpha=1/n,adjust=False).mean()/av.replace(0,np.nan); dx=100*(p-m).abs()/(p+m).replace(0,np.nan); return dx.ewm(alpha=1/n,adjust=False).mean().fillna(0)
def trend(df):
    c=df.close; e20,e50,e200=ema(c,20),ema(c,50),ema(c,200)
    if c.iloc[-1]>e20.iloc[-1]>e50.iloc[-1]>e200.iloc[-1]: return 'صعودی'
    if c.iloc[-1]<e20.iloc[-1]<e50.iloc[-1]<e200.iloc[-1]: return 'نزولی'
    return 'رنج'

def pa(df):
    hi=float(df.high.tail(30).iloc[:-1].max()); lo=float(df.low.tail(30).iloc[:-1].min()); c=float(df.close.iloc[-1]); a=float(atr(df).iloc[-1]);
    breakout_up=c>=hi*.998; breakout_down=c<=lo*1.002
    e20,e50=ema(df.close,20),ema(df.close,50)
    pull_long=c>e50.iloc[-1] and abs(c-e20.iloc[-1])/c<.015
    pull_short=c<e50.iloc[-1] and abs(c-e20.iloc[-1])/c<.015
    return breakout_up,breakout_down,pull_long,pull_short,hi,lo


# ---------- CANDLESTICK / HEIKIN-ASHI ----------
def candle_features(df):
    o,h,l,c = [df[x].astype(float) for x in ['open','high','low','close']]
    body=(c-o).abs()
    rng=(h-l).replace(0,np.nan)
    upper=h-np.maximum(o,c)
    lower=np.minimum(o,c)-l
    doji=(body/rng <= 0.12)
    hammer=(lower >= body*2) & (upper <= body*0.8) & (body/rng <= 0.45)
    shooting=(upper >= body*2) & (lower <= body*0.8) & (body/rng <= 0.45)
    bull_engulf=(c>o) & (c.shift(1)<o.shift(1)) & (c>=o.shift(1)) & (o<=c.shift(1))
    bear_engulf=(c<o) & (c.shift(1)>o.shift(1)) & (o>=c.shift(1)) & (c<=o.shift(1))
    is_hammer=bool(hammer.iloc[-1]); is_shooting=bool(shooting.iloc[-1])
    is_bull_engulf=bool(bull_engulf.iloc[-1]); is_bear_engulf=bool(bear_engulf.iloc[-1])
    is_doji=bool(doji.iloc[-1])
    if is_bull_engulf: pattern='پوشای صعودی'
    elif is_bear_engulf: pattern='پوشای نزولی'
    elif is_hammer: pattern='چکش صعودی احتمالی'
    elif is_shooting: pattern='ستاره ثاقب نزولی احتمالی'
    elif is_doji: pattern='دوجی؛ بلاتکلیفی'
    else: pattern='الگوی کندلی تأییدشده‌ای نیست'
    pattern_score=2 if (is_bull_engulf or is_hammer) else -2 if (is_bear_engulf or is_shooting) else 0
    return {
        'bullish': bool(is_hammer or is_bull_engulf),
        'bearish': bool(is_shooting or is_bear_engulf),
        'doji': is_doji,
        'hammer': is_hammer,
        'shooting': is_shooting,
        'bull_engulf': is_bull_engulf,
        'bear_engulf': is_bear_engulf,
        'pattern_name': pattern,
        'pattern_score': pattern_score,
        'body_pct': float((body.iloc[-1]/rng.iloc[-1])*100) if pd.notna(rng.iloc[-1]) else 0
    }

def heikin_ashi(df):
    ha=pd.DataFrame(index=df.index)
    ha['close']=(df.open+df.high+df.low+df.close)/4
    ha['open']=0.0
    ha.iloc[0,ha.columns.get_loc('open')]=(df.open.iloc[0]+df.close.iloc[0])/2
    for i in range(1,len(df)):
        ha.iloc[i,ha.columns.get_loc('open')]=(ha.open.iloc[i-1]+ha.close.iloc[i-1])/2
    ha['high']=pd.concat([df.high,ha.open,ha.close],axis=1).max(axis=1)
    ha['low']=pd.concat([df.low,ha.open,ha.close],axis=1).min(axis=1)
    ha['body']=(ha.close-ha.open).abs()
    ha['range']=(ha.high-ha.low).replace(0,np.nan)
    ha['upper']=ha.high-np.maximum(ha.open,ha.close)
    ha['lower']=np.minimum(ha.open,ha.close)-ha.low
    ha['bull']=ha.close>ha.open
    ha['bear']=ha.close<ha.open
    ha['strong_bull']=ha.bull & (ha.lower/ha.range<0.20)
    ha['strong_bear']=ha.bear & (ha.upper/ha.range<0.20)
    return ha

def candle_decision(df):
    cf=candle_features(df)
    ha=heikin_ashi(df)
    ha_bull=int(ha.bull.tail(3).sum())
    ha_bear=int(ha.bear.tail(3).sum())
    ha_strong_bull=bool(ha.strong_bull.iloc[-1])
    ha_strong_bear=bool(ha.strong_bear.iloc[-1])
    # +2 strong, +1 ordinary; opposite candles subtract.
    candle_score=int(cf['pattern_score'])
    ha_score=(2 if ha_strong_bull else 0) - (2 if ha_strong_bear else 0)
    ha_score += 1 if ha_bull>=2 else -1 if ha_bear>=2 else 0
    return {
        **cf,
        'ha_bull':ha_bull,'ha_bear':ha_bear,
        'ha_strong_bull':ha_strong_bull,'ha_strong_bear':ha_strong_bear,
        'candle_score':candle_score,'ha_score':ha_score,
        'combined_score':candle_score+ha_score,
        'ha':ha
    }


# ---------- CLASSIC CHART PATTERNS ----------
def _pivots(df, left=3, right=3, lookback=140):
    x=df.tail(lookback).reset_index(drop=True)
    highs=[]; lows=[]
    for i in range(left, len(x)-right):
        h=float(x.high.iloc[i]); l=float(x.low.iloc[i])
        if h>=float(x.high.iloc[i-left:i+right+1].max()): highs.append((i,h))
        if l<=float(x.low.iloc[i-left:i+right+1].min()): lows.append((i,l))
    return x, highs, lows

def classic_patterns(df):
    """Heuristic classical chart-pattern detector; only closed OHLC candles are used."""
    x, highs, lows=_pivots(df)
    price=float(x.close.iloc[-1])
    result={
        'name':'بدون الگوی کلاسیک معتبر','type':'NONE','score':0,
        'bullish':False,'bearish':False,'confidence':0.0,'breakout':False,
        'neckline':None,'target':None,'detail':'',
    }
    if len(highs)<2 or len(lows)<2: return result
    H=highs[-8:]; L=lows[-8:]
    def rel(a,b): return abs(a-b)/max(abs((a+b)/2),1e-12)
    # Head & Shoulders: three peaks with middle higher, shoulders similar, troughs form neckline.
    if len(H)>=3:
        a,b,c=H[-3:]
        if a[0]<b[0]<c[0] and b[1]>a[1]*1.015 and b[1]>c[1]*1.015 and rel(a[1],c[1])<0.045:
            lows_between=[z for z in L if a[0]<z[0]<b[0] or b[0]<z[0]<c[0]]
            if len(lows_between)>=2:
                n1,n2=lows_between[-2:]; neckline=(n1[1]+n2[1])/2
                broken=price<neckline*0.997
                target=neckline-(b[1]-neckline)
                result.update(name='سر و شانه',type='HEAD_SHOULDERS',score=-10 if broken else -6,bearish=True,confidence=78 if broken else 66,breakout=broken,neckline=neckline,target=target,detail='شانه چپ/سر/شانه راست شناسایی شد؛ شکست خط گردن تأیید نزولی است.')
    # Inverse H&S
    if len(L)>=3:
        a,b,c=L[-3:]
        if a[0]<b[0]<c[0] and b[1]<a[1]*0.985 and b[1]<c[1]*0.985 and rel(a[1],c[1])<0.045:
            highs_between=[z for z in H if a[0]<z[0]<b[0] or b[0]<z[0]<c[0]]
            if len(highs_between)>=2:
                n1,n2=highs_between[-2:]; neckline=(n1[1]+n2[1])/2
                broken=price>neckline*1.003
                target=neckline+(neckline-b[1])
                result.update(name='سر و شانه معکوس',type='INV_HEAD_SHOULDERS',score=10 if broken else 6,bullish=True,confidence=78 if broken else 66,breakout=broken,neckline=neckline,target=target,detail='سر و شانه معکوس شناسایی شد؛ شکست خط گردن تأیید صعودی است.')
    # Double top / bottom
    if len(H)>=2:
        a,b=H[-2:]
        troughs=[z for z in L if a[0]<z[0]<b[0]]
        if rel(a[1],b[1])<0.025 and troughs:
            neck=min(z[1] for z in troughs); broken=price<neck*0.997
            result.update(name='دو قله',type='DOUBLE_TOP',score=-9 if broken else -5,bearish=True,confidence=74 if broken else 62,breakout=broken,neckline=neck,target=neck-(max(a[1],b[1])-neck),detail='دو سقف تقریباً هم‌سطح؛ شکست کف بین دو قله تأیید نزولی است.')
    if len(L)>=2:
        a,b=L[-2:]
        peaks=[z for z in H if a[0]<z[0]<b[0]]
        if rel(a[1],b[1])<0.025 and peaks:
            neck=max(z[1] for z in peaks); broken=price>neck*1.003
            result.update(name='دو کف',type='DOUBLE_BOTTOM',score=9 if broken else 5,bullish=True,confidence=74 if broken else 62,breakout=broken,neckline=neck,target=neck+(neck-min(a[1],b[1])),detail='دو کف تقریباً هم‌سطح؛ شکست سقف بین دو کف تأیید صعودی است.')
    # Triangle detection from recent pivot trendlines.
    if len(H)>=3 and len(L)>=3:
        hh=np.array(H[-4:],float); ll=np.array(L[-4:],float)
        hs=np.polyfit(hh[:,0],hh[:,1],1)[0]; ls=np.polyfit(ll[:,0],ll[:,1],1)[0]
        hflat=abs(hs)/max(np.mean(hh[:,1]),1e-12)<0.0009
        lflat=abs(ls)/max(np.mean(ll[:,1]),1e-12)<0.0009
        narrowing=(hh[-1,1]-ll[-1,1]) < (hh[0,1]-ll[0,1])
        if narrowing and hs<0 and ls>0:
            # Symmetrical triangle; direction only after breakout, otherwise neutral.
            up=price>hh[-1,1]*0.997; down=price<ll[-1,1]*1.003
            if up or down:
                bull=up and not down; sc=8 if bull else -8
                result.update(name='مثلث متقارن',type='SYMMETRIC_TRIANGLE',score=sc,bullish=bull,bearish=not bull,confidence=72,breakout=True,target=price+(1 if bull else -1)*(hh[-1,1]-ll[-1,1]),detail='مثلث متقارن با شکست اخیر شناسایی شد.')
            else:
                result.update(name='مثلث متقارن',type='SYMMETRIC_TRIANGLE',score=2,confidence=58,detail='مثلث متقارن در حال فشردگی؛ جهت پس از شکست مشخص می‌شود.')
        elif narrowing and hflat and ls>0:
            broken=price>hh[-1,1]*0.997
            result.update(name='مثلث صعودی',type='ASCENDING_TRIANGLE',score=8 if broken else 4,bullish=True,confidence=75 if broken else 60,breakout=broken,neckline=hh[-1,1],target=hh[-1,1]+(hh[-1,1]-ll[-1,1]),detail='سقف تقریباً افقی و کف‌های بالارونده؛ شکست سقف تأیید صعودی است.')
        elif narrowing and hs<0 and lflat:
            broken=price<ll[-1,1]*1.003
            result.update(name='مثلث نزولی',type='DESCENDING_TRIANGLE',score=-8 if broken else -4,bearish=True,confidence=75 if broken else 60,breakout=broken,neckline=ll[-1,1],target=ll[-1,1]-(hh[-1,1]-ll[-1,1]),detail='کف تقریباً افقی و سقف‌های پایین‌رونده؛ شکست کف تأیید نزولی است.')
    return result

# ---------- ANALYSIS ----------
def analyze(symbol, entry_tf, capital, risk_pct):
    # Whole-market scan is resilient: only the decision TF is mandatory.
    # Context TFs are optional; missing context lowers coverage, never deletes the coin.
    frames={}; coverage=[]
    for name in [entry_tf,'15m','1H','4H','1D']:
        if name in frames: continue
        df,_=candles(symbol,TF[name])
        if df is not None and len(df)>=80: frames[name]=df; coverage.append(name)
    if entry_tf not in frames: return {'symbol':symbol,'status':'NO_DATA','reason':f'{entry_tf} unavailable','coverage':0}

    df=frames[entry_tf]; c=df.close; price=float(c.iloc[-1]); e20,e50,e200=ema(c,20),ema(c,50),ema(c,200); rv=float(rsi(c).iloc[-1]); mm,ms,_=macd(c); av=float(atr(df).iloc[-1]); ax=float(adx(df).iloc[-1]); mom=float((price/c.iloc[-6]-1)*100); vol=float(df.volume.iloc[-1]/max(df.volume.rolling(20).mean().iloc[-1],1e-12))
    bu,bd,pl,ps,rh,rl=pa(df)
    cs=candle_decision(df)
    cp=classic_patterns(df)
    # Multi-timeframe context uses whatever is actually available.
    ts=[trend(frames[x]) for x in ['15m','1H','4H','1D'] if x in frames]
    bull=ts.count('صعودی'); bear=ts.count('نزولی');
    long_score=50 + (20 if e20.iloc[-1]>e50.iloc[-1] else -10) + (15 if mm.iloc[-1]>ms.iloc[-1] else -10) + np.clip(mom*5,-15,15) + (12 if 48<=rv<=72 else -5) + (10 if vol>=1.15 else 0) + (15 if bu else 8 if pl else 0) + bull*4 - bear*3 + cs['candle_score']*3 + cs['ha_score']*3 + cp['score']*2
    short_score=50 + (20 if e20.iloc[-1]<e50.iloc[-1] else -10) + (15 if mm.iloc[-1]<ms.iloc[-1] else -10) - np.clip(mom*5,-15,15) + (12 if 28<=rv<=52 else -5) + (10 if vol>=1.15 else 0) + (15 if bd else 8 if ps else 0) + bear*4 - bull*3 - cs['candle_score']*3 - cs['ha_score']*3 - cp['score']*2
    long_score=float(np.clip(long_score,0,100)); short_score=float(np.clip(short_score,0,100)); score=max(long_score,short_score)
    # Balanced trade gate: 4 confirmations out of 7, not all conditions simultaneously.
    lc=sum([e20.iloc[-1]>e50.iloc[-1], mm.iloc[-1]>ms.iloc[-1], mom>0, 48<=rv<=72, vol>=1.05, bu or pl or bull>=2, bull>=bear, cs['combined_score']>0, cs['candle_score']>0, cp['score']>0])
    sc=sum([e20.iloc[-1]<e50.iloc[-1], mm.iloc[-1]<ms.iloc[-1], mom<0, 28<=rv<=52, vol>=1.05, bd or ps or bear>=2, bear>=bull, cs['combined_score']<0, cs['candle_score']<0, cp['score']<0])
    direction='LONG' if lc>=4 and long_score>=52 and long_score>=short_score+2 else 'SHORT' if sc>=4 and short_score>=52 and short_score>=long_score+2 else 'WAIT'
    trade='خرید' if direction=='LONG' else 'فروش' if direction=='SHORT' else 'صبر'
    if direction=='LONG':
        entry=price; sl=max(rl*.995,entry-1.5*max(av,entry*.008)); tp1=entry+1.5*(entry-sl); tp2=entry+2.5*(entry-sl); tp3=entry+3.5*(entry-sl); rr=(tp2-entry)/(entry-sl) if entry>sl else 0
    elif direction=='SHORT':
        entry=price; sl=min(rh*1.005,entry+1.5*max(av,entry*.008)); tp1=entry-1.5*(sl-entry); tp2=entry-2.5*(sl-entry); tp3=entry-3.5*(sl-entry); rr=(entry-tp2)/(sl-entry) if sl>entry else 0
    else:
        entry=price; sl=rl; tp1=rh; tp2=rh; tp3=rh; rr=0
    dollar_vol=float((df.close*df.volume).tail(20).mean()); liq=float(np.clip(50+math.log10(max(dollar_vol,1)/100000)*15,0,100))
    confirmations=max(lc,sc); confidence=float(np.clip(score*.72 + (confirmations/7)*28 - (12 if len(coverage)<3 else 0),0,100))
    pump=float((25 if vol>=1.8 else 15 if vol>=1.35 else 5 if vol>=1.1 else 0)+(25 if mom>=4 else 18 if mom>=2.5 else 10 if mom>=1 else 0)+(25 if bu else 12 if pl else 0)+(15 if 'صعودی' in ts[:2] else 0)+(10 if rv<72 else 0))
    return {'symbol':symbol,'status':'OK','direction':direction,'trade':trade,'score':round(score,1),'confidence':round(confidence,1),'long_score':round(long_score,1),'short_score':round(short_score,1),'confirmations':confirmations,'coverage':len(coverage),'coverage_tfs':','.join(coverage),'price':price,'entry':entry,'sl':sl,'tp1':tp1,'tp2':tp2,'tp3':tp3,'rr':round(rr,2),'rsi':round(rv,1),'adx':round(ax,1),'momentum':round(mom,2),'volume':round(vol,2),'setup':'Breakout' if bu or bd else 'Pullback' if pl or ps else 'No setup','pump_score':round(pump,1),'liquidity':round(liq,1),'regime':'صعودی' if bull>bear else 'نزولی' if bear>bull else 'رنج','t15':trend(frames['15m']) if '15m' in frames else 'N/A','t1':trend(frames['1H']) if '1H' in frames else 'N/A','t4':trend(frames['4H']) if '4H' in frames else 'N/A','tD':trend(frames['1D']) if '1D' in frames else 'N/A','risk_money':capital*risk_pct/100,'qty':(capital*risk_pct/100)/abs(entry-sl) if abs(entry-sl)>0 else 0,'candle_signal':'صعودی' if cs['combined_score']>0 else 'نزولی' if cs['combined_score']<0 else 'خنثی','candle_pattern':cs['pattern_name'],'candle_score':cs['candle_score'],'ha_score':cs['ha_score'],'ha_bull':cs['ha_bull'],'ha_bear':cs['ha_bear'],'classic_pattern':cp['name'],'classic_type':cp['type'],'classic_score':cp['score'],'classic_confidence':cp['confidence'],'classic_breakout':cp['breakout']}

# ---------- WHOLE MARKET SCANNER ----------
@st.cache_data(ttl=90,show_spinner=False)
def scan_all(symbols,tf,capital,risk_pct):
    rows=[]; total=len(symbols); done=0
    progress=st.progress(0,text=f'اسکن 0/{total}')
    lock_rows=[]
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(analyze,s,tf,capital,risk_pct):s for s in symbols}
        for f in as_completed(fs):
            done+=1
            try: rows.append(f.result())
            except Exception as e: rows.append({'symbol':fs[f],'status':'ERROR','reason':str(e),'coverage':0})
            if done%5==0 or done==total: progress.progress(done/total,text=f'اسکن {done}/{total}')
    progress.empty()
    return pd.DataFrame(rows)

def fmt(x):
    try:
        x=float(x)
        if x>=1000:return f'{x:,.2f}'
        if x>=1:return f'{x:,.4f}'
        if x>=.01:return f'{x:,.6f}'
        return f'{x:.10f}'.rstrip('0').rstrip('.')
    except: return '-'

# ---------- DETAILED TRADE PLAN ----------
def market24(symbol):
    s=norm(symbol)
    try:
        d=get_json(f'{BINANCE}/api/v3/ticker/24hr',{'symbol':s},timeout=6)
        if isinstance(d,dict) and d.get('lastPrice'):
            return {'price':float(d['lastPrice']),'change24':float(d.get('priceChangePercent',0)),'high24':float(d.get('highPrice',0)),'low24':float(d.get('lowPrice',0)),'volume24':float(d.get('quoteVolume',0)),'source':'Binance'}
    except Exception: pass
    try:
        inst=s[:-4]+'-USDT'
        d=get_json(f'{OKX}/api/v5/market/ticker',{'instId':inst},timeout=6)
        row=(d.get('data') or [None])[0]
        if row:
            last=float(row[1]); open24=float(row[14]) if row[14] else last
            return {'price':last,'change24':(last/open24-1)*100 if open24 else 0,'high24':float(row[6]),'low24':float(row[5]),'volume24':float(row[7]) if row[7] else 0,'source':'OKX'}
    except Exception: pass
    return None

def levels(df, n=4):
    c=df.close
    hi=df.high.rolling(5,center=True).max()
    lo=df.low.rolling(5,center=True).min()
    highs=sorted([float(x) for x in hi.dropna()[hi.dropna()==df.high[hi.dropna().index]].tail(120).tolist() if np.isfinite(x)], reverse=True)
    lows=sorted([float(x) for x in lo.dropna()[lo.dropna()==df.low[lo.dropna().index]].tail(120).tolist() if np.isfinite(x)])
    price=float(c.iloc[-1])
    def unique_near(vals, side, count):
        out=[]
        for x in vals:
            if side=='above' and x<=price*1.002: continue
            if side=='below' and x>=price*.998: continue
            if not out or abs(x-out[-1])/max(abs(out[-1]),1e-12)>.012: out.append(x)
            if len(out)>=count: break
        return out
    rs=unique_near(highs,'above',n)
    ss=unique_near(list(reversed(lows)),'below',n)
    # Fill with recent extrema if local pivots are sparse.
    for x in sorted([float(df.high.tail(60).max()), float(df.high.tail(120).max())]):
        if x>price*1.002 and all(abs(x-y)/x>.012 for y in rs): rs.append(x)
    for x in sorted([float(df.low.tail(60).min()), float(df.low.tail(120).min())], reverse=True):
        if x<price*.998 and all(abs(x-y)/max(x,1e-12)>.012 for y in ss): ss.append(x)
    return sorted(rs)[:n], sorted(ss, reverse=True)[:n]



def fibonacci(df, lookback=120):
    """محاسبه فیبوناچی از سقف/کف سوئینگ اخیر و تبدیل آن به سطوح قابل استفاده در تصمیم."""
    x=df.tail(lookback).copy()
    hi=float(x.high.max()); lo=float(x.low.min()); span=max(hi-lo,1e-12); price=float(df.close.iloc[-1])
    hi_i=x.high.idxmax(); lo_i=x.low.idxmin()
    # Swing direction: low قبل از high => حرکت صعودی، در غیر این صورت نزولی
    bullish=lo_i < hi_i
    retr=[0.236,0.382,0.5,0.618,0.786]
    ext=[1.0,1.272,1.618,2.0]
    if bullish:
        levels={f'R{int(r*1000)}':hi-span*r for r in retr}
        extensions={f'E{int(r*1000)}':lo+span*r for r in ext}
    else:
        levels={f'R{int(r*1000)}':lo+span*r for r in retr}
        extensions={f'E{int(r*1000)}':hi-span*r for r in ext}
    # نزدیک‌ترین فیبوناچی به قیمت فعلی
    all_levels={**levels,**extensions}
    nearest_name,nearest=min(all_levels.items(),key=lambda kv:abs(kv[1]-price))
    distance=abs(price-nearest)/max(price,1e-12)*100
    return {
        'swing_high':hi,'swing_low':lo,'direction':'صعودی' if bullish else 'نزولی',
        'retracement':levels,'extensions':extensions,'nearest_name':nearest_name,
        'nearest':nearest,'distance_pct':distance
    }

def fmt_pct(x):
    try: return f'{float(x):+.2f}%'
    except: return '-'

def detailed_plan(symbol, tf, row):
    df,_=candles(symbol,TF[tf])
    if df is None or len(df)<80: return None
    t24=market24(symbol)
    price=float(df.close.iloc[-1])
    if t24 and t24.get('price'): price=t24['price']
    rs,ss=levels(df,4)
    fib=fibonacci(df,120)
    av=float(atr(df).iloc[-1]); e20=float(ema(df.close,20).iloc[-1]); e50=float(ema(df.close,50).iloc[-1]); rv=float(rsi(df.close).iloc[-1]); mm,ms,_=macd(df.close); mom=float((price/df.close.iloc[-6]-1)*100); vol=float(df.volume.iloc[-1]/max(df.volume.rolling(20).mean().iloc[-1],1e-12)); cs=candle_decision(df)
    cp=classic_patterns(df)
    direction=row.get('direction','WAIT')
    if direction=='LONG':
        sl=float(row.get('sl',max((ss[-1] if ss else price-av*2),price-av*2)))
        risk=abs(price-sl)
        resistance=[x for x in rs if x>price]
        fib_long=[v for v in fib['retracement'].values() if v<price]
        fib_ext=[v for v in fib['extensions'].values() if v>price]
        fib_targets=sorted(fib_ext)
        tps=(sorted(set((resistance[:3] if len(resistance)>=3 else []) + fib_targets))[:3] if (resistance or fib_targets) else [price+1.5*risk,price+2.5*risk,price+3.5*risk])
        fib_entry=sorted(fib_long)[-2:] if len(fib_long)>=2 else fib_long
        low_entry=max(ss[0] if ss else price-av, price-1.2*av)
        high_entry=min(e20 if e20<price else price, price)
        if fib_entry:
            low_entry=max(low_entry,min(fib_entry))
            high_entry=min(high_entry,max(fib_entry))
        safe=(min(low_entry,price), max(high_entry,price)) if low_entry<price else (price,price)
        aggressive=(max([resistance[0]*1.003 if resistance else price+0.8*av] + ([fib_targets[0]*1.001] if fib_targets else [])))
        decision='خرید / تأیید' if price>e20 and mom>0 and vol>=1.05 else 'صبر برای پولبک / تأیید شکست'
        scenario='اگر مقاومت نزدیک شکسته و بالای آن تثبیت شود، سناریوی صعودی فعال‌تر می‌شود.'
        downside='از دست رفتن حمایت نزدیک و شکست SL سناریوی صعودی را باطل می‌کند.'
    elif direction=='SHORT':
        sl=float(row.get('sl',min((rs[0] if rs else price+av*2),price+av*2)))
        risk=abs(sl-price)
        support=[x for x in ss if x<price]
        fib_short=[v for v in fib['retracement'].values() if v>price]
        fib_ext=[v for v in fib['extensions'].values() if v<price]
        fib_targets=sorted(fib_ext, reverse=True)
        tps=(sorted(set((support[:3] if len(support)>=3 else []) + fib_targets), reverse=True)[:3] if (support or fib_targets) else [price-1.5*risk,price-2.5*risk,price-3.5*risk])
        low_entry=max(price, e20 if e20>price else price)
        high_entry=min(rs[0] if rs else price+av, price+1.2*av)
        if fib_short:
            low_entry=max(low_entry,min(fib_short))
            high_entry=min(high_entry,max(fib_short))
        safe=(price, max(low_entry,price))
        aggressive=(min([support[0]*0.997 if support else price-0.8*av] + ([fib_targets[0]*0.999] if fib_targets else [])))
        decision='فروش / تأیید' if price<e20 and mom<0 and vol>=1.05 else 'صبر برای پولبک / تأیید شکست'
        scenario='اگر حمایت نزدیک شکسته و زیر آن تثبیت شود، سناریوی نزولی فعال‌تر می‌شود.'
        downside='بازپس‌گیری مقاومت و شکست SL سناریوی نزولی را باطل می‌کند.'
    else:
        # WAIT still gets a complete conditional plan.
        nearest_r=rs[0] if rs else price+av
        nearest_s=ss[0] if ss else price-av
        bullish_entry=nearest_r*1.003
        bearish_entry=nearest_s*0.997
        risk=abs(price-nearest_s)
        fib_up=sorted([v for v in fib['extensions'].values() if v>price])
        tps=sorted(set(rs[:3] + fib_up))[:3] or [price+1.5*max(av,price*.01),price+2.5*max(av,price*.01),price+3.5*max(av,price*.01)]
        sl=nearest_s
        fib_below=sorted([v for v in fib['retracement'].values() if v<price])
        safe=(max(nearest_s, fib_below[-1] if fib_below else nearest_s), price)
        aggressive=bullish_entry
        decision='صبر / تأیید شکست'
        scenario=f'عبور و تثبیت بالای {fmt(bullish_entry)} سناریوی صعودی را فعال می‌کند.'
        downside=f'شکست {fmt(bearish_entry)} به پایین، سناریوی نزولی را تقویت می‌کند.'
    # Heuristic scenario confidence, explicitly treated as an estimate.
    base=float(row.get('confidence',0) or 0)
    fib_bonus=8 if fib['distance_pct']<=0.8 else 5 if fib['distance_pct']<=1.5 else 0
    fib_alignment=1 if ((direction=='LONG' and fib['direction']=='صعودی') or (direction=='SHORT' and fib['direction']=='نزولی')) else 0
    quality=(10 if vol>=1.15 else 0)+(8 if abs(mom)>=1 else 0)+(5 if 35<=rv<=70 else 0)+fib_bonus+(4 if fib_alignment else 0)+(6 if ((direction=='LONG' and cs['combined_score']>0) or (direction=='SHORT' and cs['combined_score']<0)) else -4 if ((direction=='LONG' and cs['combined_score']<0) or (direction=='SHORT' and cs['combined_score']>0)) else 0)+(6 if ((direction=='LONG' and cs['candle_score']>0) or (direction=='SHORT' and cs['candle_score']<0)) else -6 if ((direction=='LONG' and cs['candle_score']<0) or (direction=='SHORT' and cs['candle_score']>0)) else 0)+(6 if ((direction=='LONG' and cp['score']>0) or (direction=='SHORT' and cp['score']<0)) else -6 if ((direction=='LONG' and cp['score']<0) or (direction=='SHORT' and cp['score']>0)) else 0)
    if direction=='LONG' and fib['distance_pct']<=1.5 and fib['direction']=='صعودی':
        decision='خرید / تأیید فیبوناچی' if decision=='خرید / تأیید' else 'صبر برای پولبک به فیبوناچی / تأیید شکست'
    elif direction=='SHORT' and fib['distance_pct']<=1.5 and fib['direction']=='نزولی':
        decision='فروش / تأیید فیبوناچی' if decision=='فروش / تأیید' else 'صبر برای پولبک به فیبوناچی / تأیید شکست'
    # A clear opposite candlestick pattern weakens an otherwise directional plan.
    if direction=='LONG' and cs['candle_score']<0:
        decision='احتیاط / کندل نزولی مخالف؛ منتظر تأیید مجدد'
        scenario += ' الگوی کندلی نزولی مخالف دیده شده؛ ورود تا تأیید مجدد ریسک بیشتری دارد.'
    elif direction=='SHORT' and cs['candle_score']>0:
        decision='احتیاط / کندل صعودی مخالف؛ منتظر تأیید مجدد'
        scenario += ' الگوی کندلی صعودی مخالف دیده شده؛ ورود تا تأیید مجدد ریسک بیشتری دارد.'
    est=float(np.clip(base+quality-(8 if len(rs)<2 or len(ss)<2 else 0),0,95))
    outlook=float(np.clip(mom*1.6 + (4 if direction=='LONG' else -4 if direction=='SHORT' else 0),-20,20))
    return {'price':price,'t24':t24,'supports':ss,'resistances':rs,'rsi':rv,'candle':cs,'momentum':mom,'volume':vol,'e20':e20,'e50':e50,'sl':sl,'tps':tps[:3],'safe_entry':safe,'aggressive_entry':aggressive,'decision':decision,'scenario':scenario,'downside':downside,'estimated_success':est,'outlook':outlook,'risk':risk,'fib':fib,'fib_bonus':fib_bonus,'fib_alignment':fib_alignment,'classic':cp}


# ---------- SHORT-TERM SCENARIOS (ADDITIVE) ----------
def short_term_scenarios(symbol):
    """Build conditional 1-6 hour scenarios from 1H context and 15M levels; not a guaranteed forecast."""
    d15,_=candles(symbol,TF['15m']); d1,_=candles(symbol,TF['1H'])
    if d15 is None or d1 is None or len(d15)<40 or len(d1)<40:
        return {'status':'NO_DATA','reason':'برای سناریوی کوتاه‌مدت، داده کافی 15M و 1H در دسترس نیست.'}
    price=float(d15.close.iloc[-1]); a=float(atr(d15).iloc[-1]);
    if not np.isfinite(a) or a<=0: return {'status':'NO_DATA','reason':'محاسبه نوسان برای سناریوی کوتاه‌مدت ممکن نشد.'}
    support=float(d15.low.tail(24).min()); resistance=float(d15.high.tail(24).max())
    e20=float(ema(d1.close,20).iloc[-1]); e50=float(ema(d1.close,50).iloc[-1]); rv=float(rsi(d1.close).iloc[-1])
    m,ms,_=macd(d1.close); macd_up=bool(m.iloc[-1]>ms.iloc[-1])
    vol_ratio=float(d15.volume.iloc[-1]/max(float(d15.volume.rolling(20).mean().iloc[-1]),1e-12))
    trend_up=price>e20 and e20>e50 and macd_up
    trend_down=price<e20 and e20<e50 and not macd_up
    # ATR buffers avoid treating tiny price touches as confirmed breakouts.
    long_trigger=resistance+0.15*a; short_trigger=support-0.15*a
    long_sl=min(support, price-1.0*a); short_sl=max(resistance, price+1.0*a)
    long_risk=max(long_trigger-min(support,long_trigger-a),0.5*a)
    short_risk=max(max(resistance,short_trigger+a)-short_trigger,0.5*a)
    return {'status':'OK','price':price,'support':support,'resistance':resistance,'atr':a,'rsi1h':rv,'volume15':vol_ratio,
      'trend_up':trend_up,'trend_down':trend_down,'macd_up':macd_up,'e20':e20,'e50':e50,
      'long_trigger':long_trigger,'long_sl':long_trigger-long_risk,'long_tp1':long_trigger+long_risk,'long_tp2':long_trigger+2*long_risk,
      'short_trigger':short_trigger,'short_sl':short_trigger+short_risk,'short_tp1':short_trigger-short_risk,'short_tp2':short_trigger-2*short_risk,
      'bull_confirm':price>long_trigger and vol_ratio>=1.1 and trend_up and 45<=rv<=72,
      'bear_confirm':price<short_trigger and vol_ratio>=1.1 and trend_down and 28<=rv<=55,
      'range_low':support,'range_high':resistance}

# ---------- LEVERAGED TRADING MODULE (ADDITIVE / DOES NOT CHANGE CORE ENGINE) ----------
def structure_hh_hl(df, lookback=40):
    x=df.tail(lookback).reset_index(drop=True)
    highs=[]; lows=[]
    for i in range(2,len(x)-2):
        if x.high.iloc[i] >= x.high.iloc[i-2:i+3].max(): highs.append(float(x.high.iloc[i]))
        if x.low.iloc[i] <= x.low.iloc[i-2:i+3].min(): lows.append(float(x.low.iloc[i]))
    if len(highs)<2 or len(lows)<2: return 'نامشخص'
    hh=highs[-1]>highs[-2]; hl=lows[-1]>lows[-2]
    lh=highs[-1]<highs[-2]; ll=lows[-1]<lows[-2]
    if hh and hl: return 'HH / HL'
    if lh and ll: return 'LH / LL'
    return 'مختلط'

def leverage_module(symbol, main_fib=None):
    """Strict multi-timeframe leveraged setup: 4H direction -> 1H confirmation -> 15M entry -> 5M trigger."""
    frames={}
    for name in ['4H','1H','15m','5m']:
        d,_=candles(symbol,TF[name])
        if d is not None and len(d)>=80: frames[name]=d
    required=['4H','1H','15m','5m']
    if any(x not in frames for x in required):
        return {'status':'NO_DATA','reason':'برای معامله اهرمی هر چهار تایم‌فریم 4H / 1H / 15M / 5M لازم است.'}

    d4,d1,d15,d5=[frames[x] for x in required]
    price=float(d5.close.iloc[-1])
    e20_4,e50_4,e200_4=[ema(d4.close,n).iloc[-1] for n in (20,50,200)]
    e20_1,e50_1=[ema(d1.close,n).iloc[-1] for n in (20,50)]
    r1=float(rsi(d1.close).iloc[-1]); m1,s1,_=macd(d1.close)
    vol1=float(d1.volume.iloc[-1]/max(d1.volume.rolling(20).mean().iloc[-1],1e-12))
    p4=float(d4.close.iloc[-1])
    trend4='صعودی' if p4>e20_4 and e20_4>e50_4>e200_4 else 'نزولی' if p4<e20_4 and e20_4<e50_4<e200_4 else 'خنثی'
    struct1=structure_hh_hl(d1)
    ema1=(e20_1>e50_1)
    macd1=(m1.iloc[-1]>s1.iloc[-1])
    volume1=vol1>=1.05

    # 15M entry area / price action
    r15,s15=levels(d15,3)
    c15=candle_decision(d15)
    p15=pa(d15)
    breakout_up,breakout_down,pull_long,pull_short,rh15,rl15=p15
    pa_long=bool(breakout_up or pull_long or c15['combined_score']>0)
    pa_short=bool(breakout_down or pull_short or c15['combined_score']<0)
    near_support=bool(s15 and abs(price-s15[0])/max(price,1e-12)<=0.012)
    near_resistance=bool(r15 and abs(price-r15[0])/max(price,1e-12)<=0.012)

    # 5M trigger: latest short-term breakout + volume + entry candle confirmation.
    c5=candle_decision(d5)
    last_hi=float(d5.high.tail(12).iloc[:-1].max()); last_lo=float(d5.low.tail(12).iloc[:-1].min())
    vol5=float(d5.volume.iloc[-1]/max(d5.volume.rolling(20).mean().iloc[-1],1e-12))
    trigger_long=price>last_hi*1.001 and vol5>=1.10 and c5['combined_score']>0
    trigger_short=price<last_lo*0.999 and vol5>=1.10 and c5['combined_score']<0

    # Risk plan is always >= 2R for an actionable setup; otherwise no leveraged entry.
    av15=float(atr(d15).iloc[-1]);
    if trend4=='صعودی':
        sl=min((s15[0] if s15 else price-av15),price-av15*0.8)
        risk=price-sl
        tp=price+2*risk
        rr=(tp-price)/risk if risk>0 else 0
    elif trend4=='نزولی':
        sl=max((r15[0] if r15 else price+av15),price+av15*0.8)
        risk=sl-price
        tp=price-2*risk
        rr=(price-tp)/risk if risk>0 else 0
    else:
        sl=tp=rr=0

    # Strict gate: higher timeframe direction is mandatory; 1H confirms; 15M gives setup; 5M fires trigger.
    setup15=False
    if trend4=='خنثی':
        decision='NO TRADE — 4H خنثی'
        side='NONE'
    elif trend4=='صعودی':
        confirm1=(struct1=='HH / HL' and ema1 and macd1 and volume1)
        setup15=bool(near_support or pull_long or breakout_up) and pa_long
        side='LONG'
        decision='LONG' if confirm1 and setup15 and trigger_long and rr>=2 else 'WAIT LONG'
    else:
        confirm1=(struct1=='LH / LL' and (not ema1) and (not macd1) and volume1)
        setup15=bool(near_resistance or pull_short or breakout_down) and pa_short
        side='SHORT'
        decision='SHORT' if confirm1 and setup15 and trigger_short and rr>=2 else 'WAIT SHORT'

    # Main-section Fibonacci levels are shown exactly as calculated by detailed_plan.
    fib_display=main_fib if main_fib else fibonacci(d15,120)
    fib_retr=fib_display.get('retracement',{})
    fib_ext=fib_display.get('extensions',{})
    relevant=[]
    for name,val in {**fib_retr,**fib_ext}.items():
        if np.isfinite(float(val)):
            relevant.append((name,float(val),abs(price-float(val))/max(price,1e-12)*100))
    relevant=sorted(relevant,key=lambda x:x[2])[:6]
    confidence=0
    if trend4!='خنثی': confidence+=25
    if (trend4=='صعودی' and struct1=='HH / HL') or (trend4=='نزولی' and struct1=='LH / LL'): confidence+=20
    if (trend4=='صعودی' and ema1 and macd1) or (trend4=='نزولی' and not ema1 and not macd1): confidence+=20
    if volume1: confidence+=10
    if (trend4=='صعودی' and setup15) or (trend4=='نزولی' and setup15): confidence+=15
    if (trigger_long if side=='LONG' else trigger_short if side=='SHORT' else False): confidence+=10
    return {
        'status':'OK','trend4':trend4,'structure1':struct1,'ema1':ema1,'macd1':macd1,'volume1':volume1,'volume5':vol5,
        'near_support':near_support,'near_resistance':near_resistance,'setup15':setup15,'trigger_long':trigger_long,'trigger_short':trigger_short,
        'last_hi5':last_hi,'last_lo5':last_lo,'entry':price,'sl':sl,'tp':tp,'rr':rr,'side':side,'decision':decision,'confidence':min(confidence,100),
        'fib':fib_display,'fib_near':relevant,'rsi1':r1,'candle5':c5,'atr15':av15,
        'checklist':{'4H':trend4,'1H structure':struct1,'1H EMA': 'تأیید' if (trend4=='صعودی' and ema1) or (trend4=='نزولی' and not ema1) else 'عدم تأیید','1H MACD':'تأیید' if (trend4=='صعودی' and macd1) or (trend4=='نزولی' and not macd1) else 'عدم تأیید','1H Volume':'تأیید' if volume1 else 'عدم تأیید','15M setup':'تأیید' if setup15 else 'عدم تأیید','5M trigger':'تأیید' if ((side=='LONG' and trigger_long) or (side=='SHORT' and trigger_short)) else 'منتظر','R:R':f'{rr:.2f}'}
    }

# ---------- UI ----------
# V6.8 compact modern dashboard UI — analysis engine unchanged.
st.set_page_config(page_title='Crypto Analyzer Pro V6.8', page_icon='₿', layout='wide', initial_sidebar_state='collapsed')

st.markdown("""
<style>
:root{--bg:#06101f;--card:#0b1a2f;--card2:#0e2138;--line:#173b62;--txt:#eef5ff;--muted:#8ea6c2;--blue:#2563eb;--green:#16c784;--red:#ef476f;--yellow:#f5b942;}
.stApp{background:radial-gradient(circle at 20% 0%,#102b4b 0,#06101f 42%,#040b15 100%);color:var(--txt)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07152a,#06101f);border-right:1px solid #143555;width:220px!important}
[data-testid="stSidebar"]>div:first-child{padding-top:1rem}
.block-container{max-width:1500px;padding-top:1.1rem;padding-bottom:2rem}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:12px}.brandcoin{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#ffb21a,#ff7a00);font-size:25px;color:white;box-shadow:0 8px 24px #ff8b0028}.brandtitle{font-size:23px;font-weight:800;line-height:1.05}.brandsub{font-size:11px;color:var(--muted);margin-top:4px}
.topbar{background:rgba(10,27,48,.82);border:1px solid #173d63;border-radius:16px;padding:10px 14px;margin-bottom:12px;backdrop-filter:blur(12px)}
.card{background:linear-gradient(145deg,rgba(13,34,58,.97),rgba(7,22,39,.97));border:1px solid var(--line);border-radius:16px;padding:14px;box-shadow:0 10px 28px #00000020;margin-bottom:12px}
.metriccard{min-height:92px;padding:13px 14px}.metriclabel{font-size:12px;color:var(--muted);display:flex;justify-content:space-between}.metricvalue{font-size:25px;font-weight:800;margin-top:8px}.metricnote{font-size:11px;color:var(--muted);margin-top:3px}.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}.blue{color:#62a0ff}
.coinhead{display:flex;align-items:center;justify-content:space-between;gap:10px}.coinname{font-size:22px;font-weight:800}.badge{display:inline-block;padding:5px 10px;border-radius:999px;font-size:12px;font-weight:800;border:1px solid}.badge-long{color:#5ff0b0;background:#0c3a2a;border-color:#146b4a}.badge-short{color:#ff8da7;background:#3a1020;border-color:#70213a}.badge-wait{color:#ffd66b;background:#3a2c0b;border-color:#6f5612}
.sectiontitle{font-size:16px;font-weight:800;margin:4px 0 10px}.mini{font-size:11px;color:var(--muted)}
.level{display:flex;justify-content:space-between;padding:8px 10px;border-radius:10px;margin:5px 0;background:#09192b;border:1px solid #153652;font-size:12px}.level b{font-size:13px}.level-res{border-color:#5a2232}.level-sup{border-color:#174e40}
.scenario{padding:10px 12px;border-radius:12px;border:1px solid #194a39;background:#09281f;margin:6px 0}.scenario-bear{border-color:#5a2435;background:#2a101a}.scenario-wait{border-color:#4e431e;background:#251f0b}
.planrow{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #14314d;font-size:12px}.planrow:last-child{border-bottom:0}.planrow b{font-size:13px}
.smalltag{display:inline-block;padding:4px 7px;border-radius:8px;background:#0a2035;border:1px solid #173d5f;font-size:11px;margin:2px;color:#b9cee4}
[data-testid="stMetric"]{background:transparent}.stButton>button{border-radius:11px;border:1px solid #20548a;background:#0d2b4a;font-weight:700}.stButton>button:hover{border-color:#3d83d8}
div[data-baseweb="select"]>div{border-radius:11px;background:#081a2d;border-color:#214b74}
[data-testid="stDataFrame"]{border:1px solid #173b62;border-radius:12px;overflow:hidden}
hr{border-color:#153452}
</style>
""", unsafe_allow_html=True)

def process_entry_alerts(scan_result, tf_value, alert_on):
    """Track fresh qualifying setups after either manual or scheduled scans."""
    if scan_result is None or scan_result.empty:
        current_alerts = set()
        eligible = pd.DataFrame()
    else:
        eligible = scan_result[(scan_result.status == 'OK') &
            (scan_result.direction.isin(['LONG', 'SHORT'])) &
            (scan_result.confidence >= 50) & (scan_result.rr >= 1.2) &
            (scan_result.liquidity >= 20)].copy()
        if not eligible.empty:
            eligible = eligible.sort_values(['confidence', 'score', 'rr'], ascending=False)
        current_alerts = {f"{r.symbol}:{r.direction}:{tf_value}" for r in eligible.itertuples()}
    # A signal is considered fresh if it wasn't present at the previous scan.
    previous = st.session_state.get('active_entry_alerts', set())
    if st.session_state.get('entry_alert_tf') != tf_value:
        previous = set()
    fresh = current_alerts - previous
    st.session_state['active_entry_alerts'] = current_alerts
    st.session_state['new_entry_alerts'] = sorted(fresh)
    st.session_state['entry_alert_tf'] = tf_value
    st.session_state['entry_alert_enabled'] = bool(alert_on)
    fresh_rows = eligible[[f"{r.symbol}:{r.direction}:{tf_value}" in fresh for r in eligible.itertuples()]] if not eligible.empty else eligible
    if not fresh_rows.empty:
        best = fresh_rows.iloc[0]
        st.session_state['best_live_setup'] = {
            'symbol': str(best.symbol), 'direction': str(best.direction),
            'confidence': float(best.confidence), 'entry': float(best.entry),
            'price': float(best.price), 'rr': float(best.rr),
            'updated': time.strftime('%H:%M:%S')}
    else:
        st.session_state['best_live_setup'] = None


st.markdown("""
<div class='brand'><div class='brandcoin'>₿</div><div><div class='brandtitle'>Crypto Analyzer Pro</div><div class='brandsub'>Smart Analysis &nbsp;•&nbsp; Better Trades</div></div></div>
""", unsafe_allow_html=True)

symbols,source=universe()
if not symbols:
    st.error('فهرست بازار دریافت نشد.'); st.stop()

with st.sidebar:
    st.markdown("### ₿ Crypto Analyzer")
    st.caption('پنل کنترل')
    live_enabled=st.checkbox('دریافت زنده داده‌ها', value=True, help='قیمت بازار تقریباً هر ۵ ثانیه تازه می‌شود؛ نیازمند باز بودن صفحه است.')
    auto_scan=st.checkbox('اسکن خودکار ارز انتخاب‌شده', value=False, help='فقط ارز انتخاب‌شده را در بازه زمانی تعیین‌شده دوباره تحلیل می‌کند.')
    scan_interval=st.selectbox('فاصله اسکن خودکار', [30,60,120,300], index=1, format_func=lambda x:f'هر {x} ثانیه')
    if live_enabled:
        st_autorefresh(interval=5000, key='live_market_autorefresh')
    tf=st.selectbox('تایم‌فریم ورود',list(TF.keys()),index=3)
    q=st.text_input('جستجوی ارز','')
    capital=st.number_input('سرمایه USDT',100.0,1000000.0,1000.0,100.0)
    risk=st.slider('ریسک هر معامله %',0.25,2.0,1.0,0.25)
    st.divider()
    alert_enabled=st.checkbox('هشدار صوتی ورود', value=True, help='برای ارز انتخاب‌شده، هنگام ظاهرشدن سیگنال تازه و تأییدشده هشدار می‌دهد.')
    st.caption('برای شنیدن صدا، صدای مرورگر/گوشی روشن باشد؛ بعضی مرورگرها پخش خودکار را مسدود می‌کنند.')
    if st.button('تست بوق هشدار', use_container_width=True):
        components.html("""<button id='testtone' style='font-size:14px;padding:8px 14px'>پخش بوق آزمایشی</button><script>document.getElementById('testtone').addEventListener('click',()=>{try{const C=window.AudioContext||window.webkitAudioContext;const c=new C();const o=c.createOscillator(),g=c.createGain();o.frequency.value=880;g.gain.value=.18;o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+.35);o.onended=()=>c.close()}catch(e){}})</script>""",height=45)
    st.caption(f'منبع: {source}')
    st.caption(f'جفت‌های USDT: {len(symbols):,}')
    if st.button('↻ پاک‌سازی کش داده',use_container_width=True):
        st.cache_data.clear(); st.rerun()

filtered=[s for s in symbols if q.upper() in s] if q else symbols

st.markdown("<div class='topbar'><b>داشبورد بازار</b> <span class='mini'> &nbsp; • &nbsp; تحلیل چندلایه با Fibonacci + Candlestick + Heikin-Ashi + Classic Patterns</span></div>",unsafe_allow_html=True)

colA,colB=st.columns([4,1])
with colA:
    selected_preview=st.selectbox('ارز برای تحلیل',filtered,index=0 if filtered else None,key='preview_symbol')
with colB:
    st.write('')
    if st.button('تحلیل / اسکن همین ارز',type='primary',use_container_width=True):
        try:
            result = analyze(selected_preview, tf, capital, risk)
            st.session_state['scan_df'] = pd.DataFrame([result])
            st.session_state['scan_symbol'] = selected_preview
            process_entry_alerts(st.session_state['scan_df'], tf, alert_enabled)
            st.session_state['auto_scan_last'] = time.time()
            st.session_state['auto_scan_time'] = time.strftime('%H:%M:%S')
        except Exception as e:
            st.error(f'خطا در تحلیل {selected_preview}: {e}')

st.caption(f'ارز انتخاب‌شده برای اسکن: **{selected_preview}**')

if live_enabled:
    st.markdown('### 🟢 قیمت‌های زنده بازار')
    live_df=live_tickers(tuple(filtered))
    if live_df.empty:
        st.warning('قیمت زنده از Binance و OKX دریافت نشد. تحلیل قبلی حفظ شده است؛ اتصال اینترنت یا دسترسی صرافی‌ها را بررسی کن.')
    else:
        live_show=live_df.sort_values('quote_volume',ascending=False).head(100).copy()
        live_show.columns=['ارز','قیمت لحظه‌ای','تغییر ۲۴ساعته %','حجم ۲۴ساعته USDT','بیشترین ۲۴ساعت','کمترین ۲۴ساعت','منبع داده']
        live_show['قیمت لحظه‌ای']=live_show['قیمت لحظه‌ای'].map(fmt)
        for col in ['بیشترین ۲۴ساعت','کمترین ۲۴ساعت','حجم ۲۴ساعته USDT']:
            live_show[col]=live_show[col].map(fmt)
        live_show['تغییر ۲۴ساعته %']=live_show['تغییر ۲۴ساعته %'].map(lambda x:f'{x:+.2f}%')
        st.dataframe(live_show,use_container_width=True,hide_index=True)
        st.caption('نمایش حداکثر ۱۰۰ جفت برتر بر اساس حجم. منبع هر ردیف در داده‌ها مشخص شده است؛ قیمت ممکن است با تبدیل اختلاف داشته باشد.')

# Timed rescan: only the currently selected coin is analyzed.
if auto_scan:
    now=time.time()
    last=float(st.session_state.get('auto_scan_last',0))
    selected_changed = st.session_state.get('scan_symbol') != selected_preview
    if now-last >= scan_interval or selected_changed or 'scan_df' not in st.session_state:
        try:
            result = analyze(selected_preview, tf, capital, risk)
            st.session_state['scan_df'] = pd.DataFrame([result])
            st.session_state['scan_symbol'] = selected_preview
            process_entry_alerts(st.session_state['scan_df'], tf, alert_enabled)
            st.session_state['auto_scan_last'] = now
            st.session_state['auto_scan_time'] = time.strftime('%H:%M:%S')
        except Exception as e:
            st.warning(f'اسکن {selected_preview} ناموفق بود: {e}')
    st.caption(f"اسکن خودکار فقط برای {selected_preview} | آخرین اسکن: {st.session_state.get('auto_scan_time', 'هنوز انجام نشده')}")

if 'scan_df' in st.session_state:
    df=st.session_state['scan_df'].copy()
    total=len(df); analyzed=int((df.status=='OK').sum()); nodata=int((df.status!='OK').sum())
    tradable=df[(df.status=='OK')&(df.direction!='WAIT')&(df.confidence>=50)&(df.rr>=1.2)&(df.liquidity>=20)].copy()
    # Entry alerts are triggered only when a qualifying signal first appears, not on every rerun.
    fresh_alerts=st.session_state.get('new_entry_alerts',[])
    if fresh_alerts and st.session_state.get('entry_alert_tf')==tf:
        alert_items=[]
        for sig in fresh_alerts:
            parts=sig.split(':')
            if len(parts)>=2:
                sym,side=parts[0],parts[1]
                alert_items.append(f"{sym} — {'خرید (LONG)' if side=='LONG' else 'فروش (SHORT)'}")
        if alert_items:
            best = st.session_state.get('best_live_setup')
            if best:
                side_fa = 'خرید / LONG' if best['direction'] == 'LONG' else 'فروش / SHORT'
                st.success(f"🔔 بهترین فرصت تازه: {best['symbol']} | {side_fa} | اطمینان {best['confidence']:.1f}% | ورود حدود {fmt(best['entry'])} | R:R {best['rr']:.2f}")
            st.info('سایر سیگنال‌های تازه: ' + ' | '.join(alert_items[:12]) + (f' و {len(alert_items)-12} ارز دیگر' if len(alert_items)>12 else ''))
            if st.session_state.get('entry_alert_enabled',True):
                # Browser audio can be blocked by autoplay policy; the visual alert always remains.
                components.html("""<div></div><script>(()=>{try{const C=window.AudioContext||window.webkitAudioContext;if(!C)return;const c=new C();const play=(f,t)=>{const o=c.createOscillator(),g=c.createGain();o.type='sine';o.frequency.value=f;g.gain.setValueAtTime(0.0001,c.currentTime+t);g.gain.exponentialRampToValueAtTime(0.18,c.currentTime+t+0.025);g.gain.exponentialRampToValueAtTime(0.0001,c.currentTime+t+0.22);o.connect(g);g.connect(c.destination);o.start(c.currentTime+t);o.stop(c.currentTime+t+0.24)};const go=()=>{play(880,0);play(1175,0.28);setTimeout(()=>c.close(),800)};if(c.state==='suspended')c.resume().then(go).catch(()=>{});else go()}catch(e){}})();</script>""",height=1)
        st.session_state['new_entry_alerts']=[]
    elif fresh_alerts:
        st.session_state['new_entry_alerts']=[]
    longs=int((tradable.direction=='LONG').sum()); shorts=int((tradable.direction=='SHORT').sum())
    waits=max(analyzed-longs-shorts,0)

    cards=st.columns(6)
    metrics=[('ارز انتخاب‌شده',st.session_state.get('scan_symbol', selected_preview),'برای تحلیل','blue'),('وضعیت داده', 'OK' if analyzed else 'خطا/ناقص','وضعیت دریافت','green' if analyzed else 'yellow'),('سیگنال LONG',longs,'فرصت خرید','green'),('سیگنال SHORT',shorts,'فرصت فروش','red'),('WAIT',waits,'صبر','yellow'),('داده ناقص',nodata,'NO DATA','yellow')]
    for c,(lab,val,note,cl) in zip(cards,metrics):
        with c:
            st.markdown(f"<div class='card metriccard'><div class='metriclabel'><span>{lab}</span><span class='{cl}'>●</span></div><div class='metricvalue {cl}'>{val:,}</div><div class='metricnote'>{note}</div></div>",unsafe_allow_html=True)

    ok_symbols=df.loc[df.status=='OK','symbol'].dropna().astype(str).tolist()
    if ok_symbols:
        default_symbol=(tradable.sort_values(['confidence','score'],ascending=False).iloc[0]['symbol'] if not tradable.empty else (selected_preview if selected_preview in ok_symbols else ok_symbols[0]))
        choices=[default_symbol]+[x for x in sorted(ok_symbols) if x!=default_symbol]
        selected=st.selectbox('تحلیل کامل ارز',choices,index=0,key='analysis_symbol')
        row=df[df.symbol==selected].iloc[0].to_dict()
        plan=detailed_plan(selected,tf,row)
        if plan:
            direction=row.get('direction','WAIT')
            badge='badge-long' if direction=='LONG' else 'badge-short' if direction=='SHORT' else 'badge-wait'
            badge_txt='LONG / خرید' if direction=='LONG' else 'SHORT / فروش' if direction=='SHORT' else 'WAIT / صبر'
            cp=plan.get('classic',{})
            pattern=cp.get('name','بدون الگوی واضح')
            pscore=float(cp.get('score',0) or 0)
            pconf=float(cp.get('confidence',0) or 0)
            cndl=plan['candle']
            candle_txt='صعودی' if cndl['combined_score']>0 else 'نزولی' if cndl['combined_score']<0 else 'خنثی'
            ha_txt='صعودی' if cndl['ha_bull']>=2 else 'نزولی' if cndl['ha_bear']>=2 else 'خنثی'

            st.markdown(f"<div class='card'><div class='coinhead'><div><span class='coinname'>{selected}</span><span class='mini'> &nbsp; • &nbsp; {tf}</span></div><div><span class='badge {badge}'>{badge_txt}</span> <b class='green' style='font-size:20px'>{plan['estimated_success']:.0f}%</b><span class='mini'> احتمال تخمینی</span></div></div></div>",unsafe_allow_html=True)

            # نمودارهای نمایشی Candlestick و Heikin-Ashi حذف شده‌اند؛ تحلیل آن‌ها در موتور فعال است.

            # Indicator cards
            ic=st.columns(6)
            indicator_cards=[('الگوی کلاسیک',pattern,f'قدرت {pconf:.0f}%','blue'),('کندل‌استیک',cndl.get('pattern_name','بدون الگوی واضح'),f"{candle_txt} • امتیاز {cndl['candle_score']:+.0f}",'green' if cndl['candle_score']>0 else 'red' if cndl['candle_score']<0 else 'yellow'),('Heikin-Ashi',ha_txt,f"{cndl['ha_bull']} صعودی / {cndl['ha_bear']} نزولی",'green' if cndl['ha_bull']>=2 else 'red' if cndl['ha_bear']>=2 else 'yellow'),('Fibonacci',plan['fib']['nearest_name'],f"فاصله {plan['fib']['distance_pct']:.2f}%",'blue'),('RSI',f"{plan['rsi']:.1f}",'Momentum / RSI','yellow'),('Volume',f"{plan['volume']:.2f}x",'نسبت به میانگین','green' if plan['volume']>=1.05 else 'yellow')]
            for c,(lab,val,note,cl) in zip(ic,indicator_cards):
                with c: st.markdown(f"<div class='card metriccard'><div class='metriclabel'><span>{lab}</span></div><div class='metricvalue {cl}' style='font-size:18px'>{val}</div><div class='metricnote'>{note}</div></div>",unsafe_allow_html=True)

            left,mid,right=st.columns([1,1.15,1])
            with left:
                st.markdown("<div class='card'><div class='sectiontitle'>سطوح کلیدی</div>",unsafe_allow_html=True)
                st.markdown('<div class="mini">مقاومت‌ها</div>',unsafe_allow_html=True)
                for i,x in enumerate(plan['resistances'][:4],1): st.markdown(f"<div class='level level-res'><span>R{i}</span><b class='red'>${fmt(x)}</b></div>",unsafe_allow_html=True)
                st.markdown('<div class="mini">حمایت‌ها</div>',unsafe_allow_html=True)
                for i,x in enumerate(plan['supports'][:4],1): st.markdown(f"<div class='level level-sup'><span>S{i}</span><b class='green'>${fmt(x)}</b></div>",unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)
            with mid:
                st.markdown("<div class='card'><div class='sectiontitle'>الگوهای شناسایی‌شده</div>",unsafe_allow_html=True)
                st.markdown(f"<div class='scenario'><b>{pattern}</b><br><span class='mini'>{cp.get('detail','الگوی معتبر کافی شناسایی نشد.')}</span><br><span class='smalltag'>امتیاز {pscore:+.0f}</span><span class='smalltag'>اعتماد {pconf:.0f}%</span>{'<span class=\"smalltag\">Breakout تأیید شد</span>' if cp.get('breakout') else ''}</div>",unsafe_allow_html=True)
                fib=plan['fib']; st.markdown(f"<div class='scenario scenario-wait'><b>Fibonacci</b><br><span class='mini'>سوئینگ {fib['direction']} • نزدیک‌ترین سطح {fib['nearest_name']}</span><br><span class='smalltag'>${fmt(fib['nearest'])}</span><span class='smalltag'>فاصله {fib['distance_pct']:.2f}%</span></div>",unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)
            with right:
                st.markdown("<div class='card'><div class='sectiontitle'>طرح معامله</div>",unsafe_allow_html=True)
                se=plan['safe_entry']; se_txt=f"${fmt(se[0])} — ${fmt(se[1])}" if isinstance(se,tuple) else f"${fmt(se)}"
                rows=[('نوع معامله',badge_txt),('ورود کم‌ریسک',se_txt),('ورود تهاجمی',f"${fmt(plan['aggressive_entry'])}"),('Stop Loss',f"${fmt(plan['sl'])}")]
                for lab,val in rows: st.markdown(f"<div class='planrow'><span>{lab}</span><b>{val}</b></div>",unsafe_allow_html=True)
                for i,x in enumerate(plan['tps'],1): st.markdown(f"<div class='planrow'><span>Target {i}</span><b class='green'>${fmt(x)}</b></div>",unsafe_allow_html=True)
                st.markdown(f"<div class='planrow'><span>موفقیت تخمینی</span><b class='green'>{plan['estimated_success']:.0f}%</b></div>",unsafe_allow_html=True)
                st.markdown(f"<div class='planrow'><span>چشم‌انداز</span><b>{plan['outlook']:+.1f}%</b></div>",unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)

            st.markdown(f"<div class='card'><b>تصمیم نهایی:</b> <span class='badge {badge}'>{plan['decision']}</span><div class='mini' style='margin-top:8px'>{plan['scenario']}</div><div class='mini' style='margin-top:4px;color:#ff9bb0'>{plan['downside']}</div></div>",unsafe_allow_html=True)


            # ---------- SHORT-TERM SCENARIOS ----------
            sts=short_term_scenarios(selected)
            st.markdown("<div class='card'><div class='sectiontitle'>⏱ سناریوهای کوتاه‌مدت | حدود ۱ تا ۶ ساعت آینده</div><div class='mini'>سناریوها شرطی‌اند؛ تا تأیید شکست و حجم، ورود قطعی محسوب نمی‌شوند.</div>",unsafe_allow_html=True)
            if sts.get('status')!='OK':
                st.info(sts.get('reason','داده کافی برای سناریوها موجود نیست.'))
            else:
                scc=st.columns(3)
                with scc[0]:
                    st.markdown("<div class='scenario'><b class='green'>سناریوی صعودی</b>",unsafe_allow_html=True)
                    st.markdown(f"<div class='planrow'><span>شرط فعال‌شدن</span><b>عبور از ${fmt(sts['long_trigger'])}</b></div><div class='planrow'><span>ورود پس از تأیید</span><b>${fmt(sts['long_trigger'])}</b></div><div class='planrow'><span>حد ضرر</span><b>${fmt(sts['long_sl'])}</b></div><div class='planrow'><span>هدف ۱ / ۲</span><b class='green'>${fmt(sts['long_tp1'])} / ${fmt(sts['long_tp2'])}</b></div><div class='mini'>{'تأیید شده؛ بررسی ورود با مدیریت ریسک' if sts['bull_confirm'] else 'هنوز تأیید نشده؛ منتظر شکست معتبر و حجم بالاتر'}</div></div>",unsafe_allow_html=True)
                with scc[1]:
                    st.markdown("<div class='scenario'><b class='red'>سناریوی نزولی</b>",unsafe_allow_html=True)
                    st.markdown(f"<div class='planrow'><span>شرط فعال‌شدن</span><b>شکست ${fmt(sts['short_trigger'])}</b></div><div class='planrow'><span>ورود پس از تأیید</span><b>${fmt(sts['short_trigger'])}</b></div><div class='planrow'><span>حد ضرر</span><b>${fmt(sts['short_sl'])}</b></div><div class='planrow'><span>هدف ۱ / ۲</span><b class='green'>${fmt(sts['short_tp1'])} / ${fmt(sts['short_tp2'])}</b></div><div class='mini'>{'تأیید شده؛ بررسی ورود با مدیریت ریسک' if sts['bear_confirm'] else 'هنوز تأیید نشده؛ منتظر شکست معتبر و حجم بالاتر'}</div></div>",unsafe_allow_html=True)
                with scc[2]:
                    st.markdown("<div class='scenario scenario-wait'><b>سناریوی خنثی / نوسانی</b>",unsafe_allow_html=True)
                    st.markdown(f"<div class='planrow'><span>حمایت محدوده</span><b>${fmt(sts['range_low'])}</b></div><div class='planrow'><span>مقاومت محدوده</span><b>${fmt(sts['range_high'])}</b></div><div class='planrow'><span>RSI تایم 1H</span><b>{sts['rsi1h']:.1f}</b></div><div class='planrow'><span>حجم 15M</span><b>{sts['volume15']:.2f}x</b></div><div class='mini'>اگر قیمت بین حمایت و مقاومت بماند، از تعقیب قیمت خودداری کن و منتظر شکست معتبر بمان.</div></div>",unsafe_allow_html=True)
                st.caption(f"قیمت مرجع: ${fmt(sts['price'])} • حمایت/مقاومت از آخرین ۲۴ کندل 15M • جهت 1H: {'صعودی' if sts['trend_up'] else 'نزولی' if sts['trend_down'] else 'نامشخص/خنثی'}")
            st.markdown('</div>',unsafe_allow_html=True)

            # ---------- LEVERAGED TRADING CARD ----------
            lev=leverage_module(selected, plan.get('fib'))
            st.markdown("<div class='card'><div class='sectiontitle'>⚡ سیستم پیشنهادی معاملات اهرمی</div><div class='mini'>4H جهت اصلی → 1H تأیید → 15M نقطه ورود → 5M ماشه ورود</div>",unsafe_allow_html=True)
            if lev.get('status')!='OK':
                st.warning(lev.get('reason','داده کافی برای سیستم اهرمی وجود ندارد.'))
            else:
                lc=st.columns(8)
                lcards=[
                    ('4H',lev['trend4'],'جهت اصلی','green' if lev['trend4']=='صعودی' else 'red' if lev['trend4']=='نزولی' else 'yellow'),
                    ('1H Structure',lev['structure1'],'HH/HL یا LH/LL','green' if lev['structure1'] in ('HH / HL','LH / LL') else 'yellow'),
                    ('EMA',lev['checklist']['1H EMA'],'1H','green' if lev['checklist']['1H EMA']=='تأیید' else 'red'),
                    ('MACD',lev['checklist']['1H MACD'],'1H','green' if lev['checklist']['1H MACD']=='تأیید' else 'red'),
                    ('Volume',lev['volume5'].__format__('.2f')+'x','5M','green' if lev['volume5']>=1.10 else 'yellow'),
                    ('15M Setup',lev['checklist']['15M setup'],'ورود','green' if lev['setup15'] else 'yellow'),
                    ('5M Trigger',lev['checklist']['5M trigger'],'ماشه','green' if ((lev['side']=='LONG' and lev['trigger_long']) or (lev['side']=='SHORT' and lev['trigger_short'])) else 'yellow'),
                    ('R:R',f"1:{lev['rr']:.1f}" if lev['rr'] else '—','حداقل 1:2','green' if lev['rr']>=2 else 'red')]
                for c,(lab,val,note,cl) in zip(lc,lcards):
                    with c: st.markdown(f"<div class='card metriccard'><div class='metriclabel'><span>{lab}</span></div><div class='metricvalue {cl}' style='font-size:16px'>{val}</div><div class='metricnote'>{note}</div></div>",unsafe_allow_html=True)

                lev_left,lev_mid,lev_right=st.columns([1,1.25,1])
                with lev_left:
                    st.markdown("<div class='card'><div class='sectiontitle'>چک‌لیست ورود</div>",unsafe_allow_html=True)
                    for k,v in lev['checklist'].items():
                        cl='green' if v in ('تأیید','صعودی','نزولی') or (k=='R:R' and lev['rr']>=2) else 'yellow' if v in ('منتظر','خنثی') else 'red'
                        st.markdown(f"<div class='planrow'><span>{k}</span><b class='{cl}'>{v}</b></div>",unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)
                with lev_mid:
                    st.markdown("<div class='card'><div class='sectiontitle'>Fibonacci — سطوح استفاده‌شده در بخش اصلی</div>",unsafe_allow_html=True)
                    st.markdown(f"<div class='mini'>جهت سوئینگ: <b>{lev['fib'].get('direction','-')}</b> • نزدیک‌ترین سطح: <b>{lev['fib'].get('nearest_name','-')}</b> در ${fmt(lev['fib'].get('nearest',0))}</div>",unsafe_allow_html=True)
                    for name,val,dist in lev['fib_near']:
                        st.markdown(f"<div class='level'><span>{name}</span><b>${fmt(val)}</b><span class='mini'>{dist:.2f}%</span></div>",unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)
                with lev_right:
                    st.markdown("<div class='card'><div class='sectiontitle'>تصمیم اهرمی</div>",unsafe_allow_html=True)
                    d=lev['decision']; dcl='green' if d in ('LONG','SHORT') else 'yellow' if d.startswith('WAIT') else 'red'
                    st.markdown(f"<div class='metricvalue {dcl}' style='font-size:24px'>{d}</div>",unsafe_allow_html=True)
                    st.markdown(f"<div class='planrow'><span>جهت</span><b>{lev['side']}</b></div><div class='planrow'><span>Confidence</span><b>{lev['confidence']}%</b></div><div class='planrow'><span>Entry</span><b>${fmt(lev['entry'])}</b></div><div class='planrow'><span>SL</span><b>${fmt(lev['sl'])}</b></div><div class='planrow'><span>TP 2R</span><b class='green'>${fmt(lev['tp'])}</b></div>",unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='sectiontitle'>فرصت‌های معاملاتی</div>",unsafe_allow_html=True)
    if tradable.empty:
        st.warning('در کل بازار موقعیت با شرایط فعلی پیدا نشد؛ جدول پایین وضعیت کامل بازار را نشان می‌دهد.')
    else:
        show=tradable.sort_values(['confidence','score','pump_score'],ascending=False)[['symbol','trade','direction','confidence','score','confirmations','classic_pattern','classic_score','setup','price','entry','sl','tp1','tp2','rr','pump_score','liquidity','t15','t1','t4','tD']].copy()
        show.columns=['ارز','معامله','جهت','Confidence','Score','تأییدها','الگوی کلاسیک','امتیاز الگو','ستاپ','قیمت','Entry','SL','TP1','TP2','R:R','Pump','Liquidity','15m','1H','4H','1D']
        for c in ['قیمت','Entry','SL','TP1','TP2']: show[c]=show[c].map(fmt)
        show['Confidence']=show['Confidence'].map(lambda x:f'{x:.1f}%')
        st.dataframe(show,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='sectiontitle'>نتیجه آخرین اسکن ارز انتخاب‌شده</div>",unsafe_allow_html=True)
    allshow=df.copy().sort_values(['status','confidence','score'],ascending=[True,False,False])
    cols=['symbol','status','direction','trade','confidence','score','confirmations','coverage','coverage_tfs','classic_pattern','classic_score','regime','setup','price','momentum','volume','pump_score','reason']
    for c in cols:
        if c not in allshow.columns: allshow[c]='-'
    allshow=allshow[cols]; allshow.columns=['ارز','وضعیت داده','جهت','معامله','Confidence','Score','تأییدها','پوشش TF','TFهای موجود','الگوی کلاسیک','امتیاز الگو','رژیم','ستاپ','قیمت','Momentum %','Volume x','Pump','علت']
    allshow['قیمت']=allshow['قیمت'].map(fmt)
    allshow['Confidence']=allshow['Confidence'].apply(lambda x:f'{float(x):.1f}%' if pd.notna(x) and str(x) not in ('-','nan') else '-')
    st.dataframe(allshow,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='sectiontitle'>🚀 Pump Candidates</div>",unsafe_allow_html=True)
    pump=df[(df.status=='OK')&(df.pump_score>=28)].sort_values(['pump_score','confidence'],ascending=False).head(30)
    if pump.empty: st.info('کاندید پامپ پیدا نشد.')
    else:
        p=pump[['symbol','pump_score','confidence','momentum','volume','classic_pattern','setup','direction','price']].copy(); p.columns=['ارز','Pump Score','Confidence','Momentum %','Volume x','الگوی کلاسیک','ستاپ','جهت','قیمت']; p['قیمت']=p['قیمت'].map(fmt); p['Confidence']=p['Confidence'].map(lambda x:f'{x:.1f}%'); st.dataframe(p,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)
else:
    st.markdown("<div class='card' style='text-align:center;padding:35px'><div style='font-size:28px'>₿</div><h3>برای شروع، یک ارز انتخاب کنید و اسکن همان ارز را اجرا کنید یا اسکن خودکار را روشن کنید</h3><div class='mini'>پس از اسکن، کارت‌های LONG / SHORT / WAIT و تحلیل کامل ارز در همین صفحه نمایش داده می‌شوند.</div></div>",unsafe_allow_html=True)

st.caption('Crypto Analyzer Pro V6.9: دریافت زنده قیمت، اسکن دوره‌ای ارز انتخاب‌شده و هشدار دیداری/صوتی؛ صدای خودکار ممکن است با سیاست مرورگر مسدود شود.')
