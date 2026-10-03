import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title='Crypto Analyzer Pro', page_icon='₿', layout='wide', initial_sidebar_state='collapsed')

# ============================================================
# CONFIG
# ============================================================
TIMEFRAMES = {
    '5m':'5m', '15m':'15m', '30m':'30m',
    '1H':'1h', '2H':'2h', '4H':'4h', '6H':'6h', '12H':'12h',
    '1D':'1d', '3D':'3d', '1W':'1w'
}
BINANCE = 'https://api.binance.com'
OKX = 'https://www.okx.com'
BYBIT = 'https://api.bybit.com'
KUCOIN = 'https://api.kucoin.com'
WALLEX = 'https://api.wallex.ir'
TABDEAL = 'https://api1.tabdeal.org'
SESSION = requests.Session()
SESSION.headers.update({'User-Agent':'CryptoAnalyzerPro/1.0'})

# Official public Tabdeal market endpoint. It returns the complete spot market list.
TABDEAL_MARKET_ENDPOINTS = [
    '/r/api/v1/exchangeInfo',
    '/api/v1/exchangeInfo',
    '/v1/market/symbols',
    '/v1/markets',
    '/api/v1/markets',
]

# ============================================================
# HTTP / JSON HELPERS
# ============================================================
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


def symbol_from_any(x):
    if not isinstance(x, dict): return None
    for k in ('symbol','market','pair','ticker','code'):
        v = x.get(k)
        if isinstance(v, str) and v:
            return v.upper().replace('-', '').replace('_','').replace('/','')
    return None


def normalize_symbol(s):
    return str(s).upper().replace('-', '').replace('_','').replace('/','')


def base_asset(symbol):
    s = normalize_symbol(symbol)
    return s[:-4] if s.endswith('USDT') else s


def money(value):
    """Format a price/value safely for display."""
    try:
        v = float(value)
        if not np.isfinite(v):
            return '-'
        if abs(v) >= 1000:
            return f'{v:,.2f}'
        if abs(v) >= 1:
            return f'{v:,.4f}'
        if abs(v) >= 0.01:
            return f'{v:,.6f}'
        return f'{v:.10f}'.rstrip('0').rstrip('.')
    except (TypeError, ValueError):
        return '-'

# ============================================================
# TABDEAL — COMPLETE USDT MARKET UNIVERSE
# ============================================================
def tabdeal_markets():
    """Return all active Tabdeal spot markets quoted in USDT.

    The official exchangeInfo endpoint returns the complete market universe;
    no fixed coin list is used.
    """
    symbols = set()

    for ep in TABDEAL_MARKET_ENDPOINTS:
        data = get_json(TABDEAL + ep, timeout=15)
        if not data:
            continue

        # Official response is normally a list of market dictionaries.
        for d in flatten_dicts(data):
            if not isinstance(d, dict):
                continue

            raw = None
            for key in ('symbol', 'tabdealSymbol', 'market', 'pair'):
                v = d.get(key)
                if isinstance(v, str) and v.strip():
                    raw = v
                    break

            if not raw:
                continue

            s = normalize_symbol(raw)
            status = str(d.get('status', 'TRADING')).upper()
            quote = str(d.get('quoteAsset', '')).upper()

            # Accept explicit USDT markets, and also symbols ending in USDT
            # when quoteAsset is absent in an alternative endpoint.
            if (quote == 'USDT' or s.endswith('USDT')) and status in ('TRADING', 'ACTIVE', 'ENABLED', ''):
                if s.endswith('USDT'):
                    symbols.add(s)

        # exchangeInfo is authoritative; stop after obtaining a real universe.
        if len(symbols) > 0 and ('exchangeInfo' in ep):
            break

    return sorted(symbols)


@st.cache_data(ttl=120, show_spinner=False)
def get_tabdeal_universe():
    syms = tabdeal_markets()

    # Dynamic fallback: if Tabdeal is temporarily unavailable, use Binance's
    # live USDT spot universe instead of a small hard-coded coin list.
    if not syms:
        data = get_json(BINANCE + '/api/v3/exchangeInfo', timeout=15)
        if isinstance(data, dict):
            for d in data.get('symbols', []):
                if not isinstance(d, dict):
                    continue
                if str(d.get('status', '')).upper() == 'TRADING' and str(d.get('quoteAsset', '')).upper() == 'USDT':
                    s = normalize_symbol(d.get('symbol', ''))
                    if s.endswith('USDT'):
                        syms.append(s)

    return sorted(set(syms))

# ============================================================
# BINANCE MARKET DATA
# ============================================================
def _df_from_rows(rows, source):
    try:
        if source == 'binance':
            cols=['open_time','open','high','low','close','volume','close_time','qv','trades','tbv','tqv','ignore']
            df=pd.DataFrame(rows, columns=cols)
            for c in ['open','high','low','close','volume']:
                df[c]=pd.to_numeric(df[c], errors='coerce')
            df['time']=pd.to_datetime(df['open_time'], unit='ms', utc=True)
            return df[['time','open','high','low','close','volume']].dropna().sort_values('time').reset_index(drop=True)
        if source == 'okx':
            # OKX: ts, open, high, low, close, volume, ...
            df=pd.DataFrame(rows)
            if len(df.columns)<6: return pd.DataFrame()
            df=df.iloc[:,:6]
            df.columns=['time','open','high','low','close','volume']
            df['time']=pd.to_datetime(pd.to_numeric(df['time']), unit='ms', utc=True)
            for c in ['open','high','low','close','volume']:
                df[c]=pd.to_numeric(df[c], errors='coerce')
            return df.dropna().sort_values('time').reset_index(drop=True)
        if source == 'bybit':
            # Bybit: startTime, open, high, low, close, volume, turnover
            df=pd.DataFrame(rows)
            if len(df.columns)<6: return pd.DataFrame()
            df=df.iloc[:,:6]
            df.columns=['time','open','high','low','close','volume']
            df['time']=pd.to_datetime(pd.to_numeric(df['time']), unit='ms', utc=True)
            for c in ['open','high','low','close','volume']:
                df[c]=pd.to_numeric(df[c], errors='coerce')
            return df.dropna().sort_values('time').reset_index(drop=True)
        if source == 'kucoin':
            # KuCoin: time, open, close, high, low, volume, turnover
            df=pd.DataFrame(rows)
            if len(df.columns)<6: return pd.DataFrame()
            df=df.iloc[:,:6]
            df.columns=['time','open','close','high','low','volume']
            df['time']=pd.to_datetime(pd.to_numeric(df['time']), unit='s', utc=True)
            df['high']=pd.to_numeric(df['high'], errors='coerce')
            df['low']=pd.to_numeric(df['low'], errors='coerce')
            for c in ['open','close','volume']:
                df[c]=pd.to_numeric(df[c], errors='coerce')
            df=df[['time','open','high','low','close','volume']]
            return df.dropna().sort_values('time').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def binance_klines(symbol, interval, limit=500):
    data=get_json(BINANCE + '/api/v3/klines', {'symbol':symbol, 'interval':interval, 'limit':limit})
    if isinstance(data,list) and len(data)>=40:
        df=_df_from_rows(data,'binance')
        if len(df)>=40: return df
    return pd.DataFrame()


def okx_klines(symbol, interval, limit=300):
    bar={'5m':'5m','15m':'15m','30m':'30m','1h':'1H','2h':'2H','4h':'4H','6h':'6H','12h':'12H','1d':'1D','3d':'3D','1w':'1W'}.get(interval,interval)
    inst=base_asset(symbol)+'-USDT'
    d=get_json(OKX+'/api/v5/market/candles', {'instId':inst,'bar':bar,'limit':str(min(limit,300))})
    rows=d.get('data',[]) if isinstance(d,dict) else []
    return _df_from_rows(rows,'okx') if len(rows)>=40 else pd.DataFrame()


def bybit_klines(symbol, interval, limit=200):
    iv={'5m':'5','15m':'15','30m':'30','1h':'60','2h':'120','4h':'240','6h':'360','12h':'720','1d':'D','3d':'3D','1w':'W'}.get(interval,interval)
    d=get_json(BYBIT+'/v5/market/kline', {'category':'spot','symbol':symbol,'interval':iv,'limit':str(min(limit,200))})
    rows=d.get('result',{}).get('list',[]) if isinstance(d,dict) else []
    return _df_from_rows(rows,'bybit') if len(rows)>=40 else pd.DataFrame()


def kucoin_klines(symbol, interval, limit=500):
    typ={'5m':'5min','15m':'15min','30m':'30min','1h':'1hour','2h':'2hour','4h':'4hour','6h':'6hour','12h':'12hour','1d':'1day','3d':'3day','1w':'1week'}.get(interval,interval)
    pair=base_asset(symbol)+'-USDT'
    d=get_json(KUCOIN+'/api/v1/market/candles', {'symbol':pair,'type':typ})
    rows=d.get('data',[]) if isinstance(d,dict) else []
    df=_df_from_rows(rows,'kucoin') if len(rows)>=40 else pd.DataFrame()
    return df.tail(limit).reset_index(drop=True) if not df.empty else df


def market_klines(symbol, interval, limit=500):
    # Primary + fallbacks. This is what lets Tabdeal-listed USDT coins
    # still receive historical data when they are absent from Binance.
    for fn in (
        lambda: binance_klines(symbol,interval,limit),
        lambda: okx_klines(symbol,interval,limit),
        lambda: bybit_klines(symbol,interval,min(limit,200)),
        lambda: kucoin_klines(symbol,interval,limit),
    ):
        try:
            df=fn()
            if not df.empty and len(df)>=40:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def binance_price(symbol):
    d=get_json(BINANCE + '/api/v3/ticker/price', {'symbol':symbol})
    try: return float(d['price'])
    except Exception: return None

# ============================================================
# WALLEX PUBLIC DATA
# ============================================================
def wallex_markets():
    data=get_json(WALLEX + '/v1/markets')
    out={}
    try:
        syms=data['result']['symbols']
        for s,d in syms.items():
            if normalize_symbol(s).endswith('USDT'): out[normalize_symbol(s)]=d
    except Exception: pass
    return out

@st.cache_data(ttl=120, show_spinner=False)
def wallex_currency_stats():
    data=get_json(WALLEX + '/v1/currencies/stats')
    rows={}
    if isinstance(data,dict):
        arr=data.get('result', data.get('data', []))
        if isinstance(arr,list):
            for x in arr:
                if isinstance(x,dict) and x.get('key'): rows[str(x['key']).upper()]=x
    return rows


def wallex_klines(symbol, resolution='60', days=12):
    end=int(time.time())
    start=end-days*86400
    d=get_json(WALLEX + '/v1/udf/history', {'symbol':symbol,'resolution':resolution,'from':start,'to':end})
    if not isinstance(d,dict) or d.get('s')!='ok': return pd.DataFrame()
    try:
        df=pd.DataFrame({'time':pd.to_datetime(d['t'],unit='s',utc=True),'open':pd.to_numeric(d['o']),'high':pd.to_numeric(d['h']),'low':pd.to_numeric(d['l']),'close':pd.to_numeric(d['c']),'volume':pd.to_numeric(d['v'])})
        return df.dropna().reset_index(drop=True)
    except Exception: return pd.DataFrame()


def alternate_price(symbol):
    checks=[
        (OKX+'/api/v5/market/ticker', {'instId':base_asset(symbol)+'-USDT'}, lambda d: d.get('data',[{}])[0].get('last')),
        (BYBIT+'/v5/market/tickers', {'category':'spot','symbol':symbol}, lambda d: d.get('result',{}).get('list',[{}])[0].get('lastPrice')),
        (KUCOIN+'/api/v1/market/orderbook/level1', {'symbol':base_asset(symbol)+'-USDT'}, lambda d: d.get('data',{}).get('price')),
    ]
    for url,params,getter in checks:
        try:
            d=get_json(url,params,timeout=8)
            v=getter(d) if isinstance(d,dict) else None
            if v is not None: return float(v)
        except Exception: pass
    return None

def wallex_price(symbol):
    markets=wallex_markets()
    d=markets.get(symbol)
    try: return float(d['stats']['lastPrice'])
    except Exception: return None

# ============================================================
# MULTI-SOURCE PRICE CONSENSUS
# ============================================================
def source_prices(symbol):
    vals={}
    p=binance_price(symbol)
    if p: vals['بایننس']=p
    p=wallex_price(symbol)
    if p: vals['والکس']=p
    p=alternate_price(symbol)
    if p: vals['بازارهای جایگزین']=p
    # Tabdeal current price: try ticker-like endpoints.
    base=base_asset(symbol)
    for ep in ['/v1/ticker/24hr','/v1/market/stats','/v1/market/ticker','/v1/markets']:
        d=get_json(TABDEAL+ep, {'symbol':symbol})
        found=None
        for x in flatten_dicts(d):
            s=symbol_from_any(x)
            if s==symbol:
                for k in ('lastPrice','last_price','price','last','close'):
                    try:
                        if x.get(k) is not None: found=float(x[k]); break
                    except Exception: pass
        if found:
            vals['تبدیل']=found; break
    return vals

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def sma(s,n): return s.rolling(n).mean()

def rsi(close,n=14):
    delta=close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=gain/(loss.replace(0,np.nan))
    return (100-(100/(1+rs))).fillna(50)

def macd(close):
    line=ema(close,12)-ema(close,26)
    signal=ema(line,9)
    return line,signal,line-signal

def atr(df,n=14):
    pc=df.close.shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def stochastic(df,n=14):
    lo=df.low.rolling(n).min(); hi=df.high.rolling(n).max()
    k=100*(df.close-lo)/(hi-lo).replace(0,np.nan)
    return k.fillna(50)

def adx(df,n=14):
    up=df.high.diff(); down=-df.low.diff()
    plus=np.where((up>down)&(up>0),up,0.0); minus=np.where((down>up)&(down>0),down,0.0)
    a=atr(df,n)
    pdi=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    mdi=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/a.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean().fillna(20)

def ichimoku(df):
    h9=df.high.rolling(9).max(); l9=df.low.rolling(9).min()
    h26=df.high.rolling(26).max(); l26=df.low.rolling(26).min()
    h52=df.high.rolling(52).max(); l52=df.low.rolling(52).min()
    ten=(h9+l9)/2; kij=(h26+l26)/2
    span_a=(ten+kij)/2; span_b=(h52+l52)/2
    return ten,kij,span_a,span_b

# ============================================================
# PRICE ACTION / SUPPORT / RESISTANCE
# ============================================================
def pivots(df, window=3):
    highs=df.high[(df.high==df.high.rolling(window*2+1,center=True).max())]

    lows=df.low[(df.low==df.low.rolling(window*2+1,center=True).min())]
    return highs.dropna(), lows.dropna()


def levels(df):
    highs,lows=pivots(df.tail(180),3)
    price=float(df.close.iloc[-1])
    supports=[float(x) for x in lows if x<price]
    resistances=[float(x) for x in highs if x>price]
    supports=sorted(supports,reverse=True)[:4]
    resistances=sorted(resistances)[:4]
    return supports,resistances


def price_action_score(df):
    if len(df)<20:return 0
    c=df.iloc[-1]; p=df.iloc[-2]
    rng=max(c.high-c.low,1e-12)
    body=abs(c.close-c.open)
    upper=c.high-max(c.open,c.close); lower=min(c.open,c.close)-c.low
    score=0
    if c.close>c.open: score+=8
    if c.close>p.high: score+=12
    if c.close<p.low: score-=12
    if lower>body*1.5 and c.close>c.open: score+=8
    if upper>body*1.5 and c.close<c.open: score-=8
    if body/rng>0.65 and c.close>c.open: score+=6
    if body/rng>0.65 and c.close<c.open: score-=6
    return float(np.clip(score,-25,25))

# ============================================================
# ANALYSIS ENGINE
# ============================================================
def timeframe_analysis(df):
    if len(df)<80: return None
    close=df.close
    r=rsi(close).iloc[-1]
    ml,ms,mh=macd(close)
    mac=ml.iloc[-1]; sig=ms.iloc[-1]; hist=mh.iloc[-1]
    at=atr(df).iloc[-1]
    ad=adx(df).iloc[-1]
    ten,kij,sa,sb=ichimoku(df)
    price=close.iloc[-1]
    cloud=max(sa.iloc[-1],sb.iloc[-1]); cloud_low=min(sa.iloc[-1],sb.iloc[-1])
    score=0; reasons=[]
    if price>cloud: score+=18; reasons.append('قیمت بالای ایچیموکو')
    elif price<cloud_low: score-=18; reasons.append('قیمت زیر ایچیموکو')
    else: reasons.append('قیمت داخل کلود')
    if ten.iloc[-1]>kij.iloc[-1]: score+=10; reasons.append('تنکن بالای کیجون')
    else: score-=10
    if mac>sig: score+=14; reasons.append('MACD صعودی')
    else: score-=14
    if hist>0 and hist>mh.iloc[-2]: score+=5
    elif hist<0 and hist<mh.iloc[-2]: score-=5
    if 50<r<70: score+=10; reasons.append('RSI سالم صعودی')
    elif r>=75: score-=5; reasons.append('RSI داغ')
    elif r<30: score+=4; reasons.append('RSI اشباع فروش')
    elif r<45: score-=7
    mom=(close.iloc[-1]/close.iloc[-min(13,len(close))]-1)*100
    score+=float(np.clip(mom*2,-15,15))
    if mom>0: reasons.append('مومنتوم مثبت')
    if ad>25:
        score += 8 if price>kij.iloc[-1] else -8
    score+=price_action_score(df)
    return {'score':float(np.clip(score,-100,100)),'rsi':float(r),'macd':float(mac),'signal':float(sig),'momentum':float(mom),'adx':float(ad),'atr':float(at),'price':float(price),'reasons':reasons,'tenkan':float(ten.iloc[-1]),'kijun':float(kij.iloc[-1]),'cloud_top':float(cloud),'cloud_bottom':float(cloud_low)}


def aggregate(symbol):
    frames={}
    # Analyze the full practical multi-timeframe set. The core decision still gives
    # extra weight to 1H/4H/1D, while every available timeframe is displayed separately.
    for tf,iv in TIMEFRAMES.items():
        df=market_klines(symbol,iv,500)
        if df.empty and tf in ('1H','4H'):
            df=wallex_klines(symbol,'60',14)
            if tf=='4H' and not df.empty:
                x=df.set_index('time').resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
                df=x
        frames[tf]=df
    analyses={tf:timeframe_analysis(df) for tf,df in frames.items() if not df.empty}
    analyses={k:v for k,v in analyses.items() if v}
    if not analyses: return None
    weights={'5m':0.45,'15m':0.75,'30m':0.9,'1H':1.15,'2H':1.25,'4H':1.5,'6H':1.35,'12H':1.25,'1D':1.35,'3D':1.0,'1W':0.9}
    total=sum(weights.get(k,1.0) for k in analyses)
    tech=sum(v['score']*weights.get(k,1.0) for k,v in analyses.items())/total
    p=analyses.get('15m',analyses.get('1H'))['price']
    source=source_prices(symbol)
    consensus=np.mean(list(source.values())) if source else p
    # Fundamental/market-context score from Wallex public currency stats when available.
    stats=wallex_currency_stats().get(base_asset(symbol),{})
    fscore=0
    if stats:
        rank=stats.get('rank'); dom=stats.get('dominance'); vol=stats.get('volume_24h'); mcap=stats.get('market_cap')
        try:
            if rank and float(rank)<=20: fscore+=10
            elif rank and float(rank)<=100: fscore+=5
            if dom and float(dom)>0.01: fscore+=3
            if vol and mcap and float(mcap)>0 and float(vol)/float(mcap)>0.05: fscore+=5
        except Exception: pass
    # Market-source agreement bonus.
    agreement=0
    if len(source)>=2:
        arr=np.array(list(source.values()),dtype=float)
        dispersion=np.std(arr)/max(np.mean(arr),1e-12)*100
        if dispersion<0.3: agreement=8
        elif dispersion<0.8: agreement=4
        elif dispersion>2: agreement=-5
    final_score=float(np.clip(tech*0.78+fscore*0.22+agreement,-100,100))
    confidence=float(np.clip(50+abs(final_score)*0.42 + (6 if len(analyses)>=8 else 3 if len(analyses)>=5 else 0) + max(0,agreement),50,96))
    bullish=final_score>=18
    bearish=final_score<=-18
    decision='معامله کن' if bullish or bearish else 'صبر کن'
    position='لانگ' if bullish else ('شورت' if bearish else 'بدون پوزیشن')
    atrv=analyses.get('4H',analyses.get('1H',next(iter(analyses.values()))))['atr']
    # Conservative levels based on ATR and recent structure.
    all_df=frames.get('4H') if not frames.get('4H',pd.DataFrame()).empty else frames.get('1H')
    supports,resistances=levels(all_df) if all_df is not None and not all_df.empty else ([],[])
    sr_by_tf={}
    for tf,df in frames.items():
        if df is not None and not df.empty:
            ss,rr=levels(df)
            sr_by_tf[tf]={'supports':ss,'resistances':rr}
    entry=consensus
    if bullish:
        sl=(supports[0] if supports and supports[0]<entry else entry-1.35*atrv)
        if sl>=entry: sl=entry-1.35*atrv
        risk=max(entry-sl,entry*0.006)
        t1=entry+1.6*risk; t2=entry+2.6*risk; t3=entry+3.8*risk
    elif bearish:
        sl=(resistances[0] if resistances and resistances[0]>entry else entry+1.35*atrv)
        if sl<=entry: sl=entry+1.35*atrv
        risk=max(sl-entry,entry*0.006)
        t1=entry-1.6*risk; t2=entry-2.6*risk; t3=entry-3.8*risk
    else:
        sl=entry-1.2*atrv; risk=abs(entry-sl); t1=entry+1.5*risk; t2=entry+2.5*risk; t3=entry+3.5*risk
    expected_4h=float(np.clip(final_score*0.055, -9, 9))
    expected_24h=float(np.clip(final_score*0.11, -18, 18))
    expected_72h=float(np.clip(final_score*0.18, -30, 30))
    forecast={
        '4H':entry*(1+expected_4h/100),
        '24H':entry*(1+expected_24h/100),
        '72H':entry*(1+expected_72h/100),
    }
    return {'symbol':symbol,'price':entry,'sources':source,'analyses':analyses,'technical':tech,'fundamental':fscore,'agreement':agreement,'score':final_score,'confidence':confidence,'decision':decision,'position':position,'entry':entry,'sl':sl,'t1':t1,'t2':t2,'t3':t3,'risk_pct':abs(entry-sl)/entry*100,'supports':supports,'resistances':resistances,'sr_by_tf':sr_by_tf,'forecast':forecast,'forecast_pct':{'4H':expected_4h,'24H':expected_24h,'72H':expected_72h}}

# ============================================================
# UI — MOBILE / CARD LAYOUT
# ============================================================
st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1050px}
.app-head{background:linear-gradient(135deg,#111318,#24272d);color:white;border-radius:24px;padding:20px 22px;margin-bottom:14px;box-shadow:0 8px 24px rgba(0,0,0,.12)}
.app-brand{font-size:26px;font-weight:900;line-height:1.2}.app-sub{font-size:12px;color:#cfd2d7;margin-top:6px}
.menu-wrap{background:#fff;border:1px solid #ececec;border-radius:20px;padding:8px;margin-bottom:14px;box-shadow:0 3px 14px rgba(0,0,0,.05)}
.menu-wrap [role="radiogroup"]{gap:6px;flex-wrap:wrap}.menu-wrap label{border-radius:14px!important;padding:8px 12px!important}
.section-card{background:#fff;border:1px solid #ececec;border-radius:20px;padding:16px;margin-bottom:14px;box-shadow:0 3px 14px rgba(0,0,0,.045)}
.section-title{font-size:17px;font-weight:850;margin-bottom:4px}.section-sub{font-size:12px;color:#777;margin-bottom:12px}
.stat-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.stat{flex:1;min-width:120px;background:#f7f7f8;border-radius:14px;padding:10px 12px}
.stat-k{font-size:11px;color:#777}.stat-v{font-size:17px;font-weight:850;margin-top:3px}
.result-card{background:#fff;border:1px solid #e9e9e9;border-radius:22px;padding:17px;margin:14px 0;box-shadow:0 5px 18px rgba(0,0,0,.055)}
.result-head{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.coin-name{font-size:23px;font-weight:900}.decision{padding:7px 12px;border-radius:14px;font-size:13px;font-weight:850}
.decision-buy{background:#e9f8ef;color:#138a45}.decision-sell{background:#fdecec;color:#c62828}.decision-wait{background:#fff7d9;color:#8a6900}
.price-box{background:#f7f8fa;border-radius:17px;padding:13px;margin:12px 0}.price-label{font-size:11px;color:#777}
.price-main{font-size:26px;font-weight:900;margin-top:3px}.level-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:10px}
.level{background:#fafafa;border:1px solid #ededed;border-radius:14px;padding:10px}.level.entry{border-top:4px solid #eab308}.level.sl{border-top:4px solid #dc2626}.level.tp{border-top:4px solid #16a34a}
.level-k{font-size:10px;color:#777}.level-v{font-size:14px;font-weight:850;margin-top:4px;word-break:break-word}
.reason-box{background:#f8f8f9;border-radius:15px;padding:11px 13px;font-size:12px;line-height:2;margin-top:11px}
.tf-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px}.tf{background:#f7f8fa;border-radius:14px;padding:10px}
.tf-k{font-size:10px;color:#777}.tf-v{font-size:15px;font-weight:850;margin-top:3px}
.footer-note{color:#777;font-size:11px;text-align:center;line-height:1.9;padding:12px}.tf-detail{border:1px solid #ededed;border-radius:16px;padding:12px;margin:8px 0;background:#fafafa}.tf-detail-title{font-size:15px;font-weight:900;margin-bottom:7px}.sr-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.sr-box{background:#fff;border:1px solid #eee;border-radius:12px;padding:9px}.sr-k{font-size:10px;color:#777}.sr-v{font-size:12px;font-weight:800;margin-top:4px;line-height:1.8}
@media(max-width:700px){.block-container{padding-left:.65rem;padding-right:.65rem}.app-brand{font-size:22px}.level-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.level:last-child{grid-column:span 2}.tf-grid{grid-template-columns:1fr}.coin-name{font-size:20px}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-head">
  <div class="app-brand">₿ تحلیل‌گر حرفه‌ای رمزارز</div>
  <div class="app-sub">بازار USDT صرافی تبدیل • تأیید با Binance و Wallex • تکنیکال + شرایط بازار • بدون اجرای سفارش</div>
</div>
""", unsafe_allow_html=True)

universe=get_tabdeal_universe()
if "watch" not in st.session_state: st.session_state.watch=[]

st.markdown('<div class="menu-wrap">', unsafe_allow_html=True)
menu=st.radio("منوی اصلی",["بازارها","تحلیل انتخابی","⭐ نشانه‌گذاری","راهنما"],horizontal=True,label_visibility="collapsed",key="main_menu")
st.markdown('</div>', unsafe_allow_html=True)

if menu=="راهنما":
    st.markdown("""
    <div class="section-card">
      <div class="section-title">راهنمای برنامه</div>
      <div class="section-sub">منو و کارت‌های تحلیل از هم جدا شده‌اند تا روی موبایل مرتب‌تر دیده شوند.</div>
      <b>بازارها:</b> انتخاب تا ۵ ارز از بازار USDT تبدیل.<br>
      <b>تحلیل انتخابی:</b> نمایش کارت تحلیل ارزهای انتخاب‌شده.<br>
      <b>⭐ نشانه‌گذاری:</b> نگهداری ارزهای موردنظر برای دسترسی سریع.<br>
      <b>کارت تحلیل:</b> ورود، حد ضرر، تارگت‌ها، نوع پوزیشن، اطمینان مدل و دلایل اصلی.
    </div>
    """,unsafe_allow_html=True)
    st.markdown('<div class="footer-note">درصد اطمینان، شاخص اطمینان مدل است و تضمین سود یا موفقیت معامله نیست.</div>',unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="section-card">',unsafe_allow_html=True)
st.markdown('<div class="section-title">بازار و انتخاب ارز</div>',unsafe_allow_html=True)
st.markdown('<div class="section-sub">از میان تمام بازارهای USDT تبدیل جستجو کن و حداکثر ۵ ارز را همزمان انتخاب کن.</div>',unsafe_allow_html=True)

search=st.text_input("جستجو",placeholder="مثلاً BTC، ETH، SOL یا BTCUSDT",label_visibility="collapsed",key="coin_search")
filtered=[s for s in universe if search.upper() in s] if search else universe
if menu=="⭐ نشانه‌گذاری": filtered=[s for s in filtered if s in st.session_state.watch]
if menu=="تحلیل انتخابی" and st.session_state.watch: filtered=[s for s in filtered if s in st.session_state.watch]

default_selected=[x for x in st.session_state.watch if x in filtered][:5]
selected=st.multiselect("انتخاب حداکثر ۵ ارز",options=filtered,default=default_selected,max_selections=5,format_func=lambda x:x.replace("USDT","/USDT"),key="selected_coins")
st.session_state.watch=list(dict.fromkeys(selected))

st.markdown(f"""
<div class="stat-row">
  <div class="stat"><div class="stat-k">بازارهای USDT</div><div class="stat-v">{len(universe):,}</div></div>
  <div class="stat"><div class="stat-k">انتخاب‌شده</div><div class="stat-v">{len(selected)}/5</div></div>
  <div class="stat"><div class="stat-k">نشانه‌گذاری</div><div class="stat-v">⭐ {len(st.session_state.watch)}</div></div>
</div>
""",unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)

if not selected:
    st.markdown('<div class="section-card"><div class="section-title">برای شروع یک یا چند ارز انتخاب کن</div><div class="section-sub">پس از انتخاب، کارت تحلیل هر ارز جداگانه نمایش داده می‌شود.</div></div>',unsafe_allow_html=True)
    st.stop()

results=[]
with st.spinner("در حال دریافت داده و اجرای تحلیل چندمنبعی..."):
    with ThreadPoolExecutor(max_workers=5) as ex:
        jobs={ex.submit(aggregate,s):s for s in selected}
        for job in as_completed(jobs):
            try:
                r=job.result()
                if r: results.append(r)
            except Exception: pass

if not results:
    st.error("برای ارزهای انتخاب‌شده داده کافی دریافت نشد. چند لحظه بعد دوباره امتحان کن.")
    st.stop()

st.markdown(f'<div class="section-card"><div class="section-title">نتیجه تحلیل</div><div class="section-sub">{len(results)} کارت تحلیل آماده شد؛ هر کارت مستقل از کارت‌های دیگر است.</div></div>',unsafe_allow_html=True)

for r in sorted(results,key=lambda x:x["score"],reverse=True):
    if r["position"]=="لانگ":
        decision_cls="decision-buy"; decision_icon="🟢"
    elif r["position"]=="شورت":
        decision_cls="decision-sell"; decision_icon="🔴"
    else:
        decision_cls="decision-wait"; decision_icon="🟡"

    reason="؛ ".join(reason_text for tf in ["15m","1H","4H","1D"] if tf in r["analyses"] for reason_text in r["analyses"][tf]["reasons"][:3])
    src=" | ".join(f"{k}: {money(v)}" for k,v in r["sources"].items()) or "-"

    st.markdown(f"""
    <div class="result-card">
      <div class="result-head">
        <div class="coin-name">{decision_icon} {r["symbol"].replace("USDT","/USDT")}</div>
        <div class="decision {decision_cls}">{r["decision"]}</div>
      </div>
      <div class="price-box">
        <div class="price-label">قیمت مرجع چندمنبعی</div>
        <div class="price-main">{money(r["price"])}</div>
        <div class="muted">نوع پوزیشن: <b>{r["position"]}</b> &nbsp; • &nbsp; امتیاز مدل: <b>{r["score"]:.0f}/100</b> &nbsp; • &nbsp; اطمینان تحلیل: <b>{r["confidence"]:.0f}%</b></div>
      </div>
      <div class="level-grid">
        <div class="level entry"><div class="level-k">ورود</div><div class="level-v">{money(r["entry"])}</div></div>
        <div class="level sl"><div class="level-k">حد ضرر</div><div class="level-v">{money(r["sl"])}</div></div>
        <div class="level tp"><div class="level-k">Target 1</div><div class="level-v">{money(r["t1"])}</div></div>
        <div class="level tp"><div class="level-k">Target 2</div><div class="level-v">{money(r["t2"])}</div></div>
        <div class="level tp"><div class="level-k">Target 3</div><div class="level-v">{money(r["t3"])}</div></div>
      </div>
      <div class="tf-grid">
        <div class="tf"><div class="tf-k">پیش‌بینی 4H</div><div class="tf-v">{r["forecast_pct"]["4H"]:+.1f}% → {money(r["forecast"]["4H"])}</div></div>
        <div class="tf"><div class="tf-k">پیش‌بینی 24H</div><div class="tf-v">{r["forecast_pct"]["24H"]:+.1f}% → {money(r["forecast"]["24H"])}</div></div>
        <div class="tf"><div class="tf-k">پیش‌بینی 72H</div><div class="tf-v">{r["forecast_pct"]["72H"]:+.1f}% → {money(r["forecast"]["72H"])}</div></div>
      </div>
      <div class="reason-box"><b>دلایل کلیدی:</b> {reason or "-"}<br><b>ریسک تا حد ضرر:</b> {r["risk_pct"]:.2f}%<br><span class="muted">منابع قیمت: {src}</span></div>
    </div>
    """,unsafe_allow_html=True)

    with st.expander(f"تایم‌فریم‌ها و اندیکاتورها — {r['symbol'].replace('USDT','/USDT')}"):
        st.markdown('**تمام تایم‌فریم‌های درخواستی**')
        rows=[]
        for tf in TIMEFRAMES:
            a=r["analyses"].get(tf)
            if a:
                rows.append({"تایم‌فریم":tf,"امتیاز":round(a["score"],1),"RSI":round(a["rsi"],1),"MACD":round(a["macd"],6),"Momentum %":round(a["momentum"],2),"ADX":round(a["adx"],1)})
            else:
                rows.append({"تایم‌فریم":tf,"امتیاز":"داده نیست","RSI":"-","MACD":"-","Momentum %":"-","ADX":"-"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        st.markdown('**حمایت و مقاومت به تفکیک تایم‌فریم**')
        for tf in TIMEFRAMES:
            a=r["analyses"].get(tf)
            sr=r["sr_by_tf"].get(tf,{"supports":[],"resistances":[]})
            if not a and not sr["supports"] and not sr["resistances"]:
                continue
            supports_txt=", ".join(money(x) for x in sr["supports"]) or "-"
            resistances_txt=", ".join(money(x) for x in sr["resistances"]) or "-"
            st.markdown(f'''<div class="tf-detail"><div class="tf-detail-title">{tf}</div><div class="sr-grid"><div class="sr-box"><div class="sr-k">حمایت‌ها</div><div class="sr-v">{supports_txt}</div></div><div class="sr-box"><div class="sr-k">مقاومت‌ها</div><div class="sr-v">{resistances_txt}</div></div></div></div>''',unsafe_allow_html=True)
        st.caption("اندیکاتورها: ایچیموکو، MACD، RSI، Momentum، ATR، ADX و پرایس‌اکشن. حمایت/مقاومت بر اساس ساختار سقف و کف‌های اخیر هر تایم‌فریم محاسبه می‌شود.")

st.markdown('<div class="footer-note">هشدار: این برنامه ابزار تحقیق و تحلیل است. «اطمینان تحلیل» احتمال سود قطعی نیست. پیش‌بینی بازار قطعی نیست و قبل از معامله باید نقدشوندگی، کارمزد، اخبار و ریسک شخصی بررسی شود.</div>',unsafe_allow_html=True)
