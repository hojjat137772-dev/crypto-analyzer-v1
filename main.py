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
TIMEFRAMES = {'15m':'15m', '1H':'1h', '4H':'4h', '1D':'1d'}
BINANCE = 'https://api.binance.com'
WALLEX = 'https://api.wallex.ir'
TABDEAL = 'https://api.tabdeal.org'
SESSION = requests.Session()
SESSION.headers.update({'User-Agent':'CryptoAnalyzerPro/1.0'})

# Tabdeal endpoints can change. The adapter tries the public variants below.
TABDEAL_MARKET_ENDPOINTS = [
    '/v1/market/symbols', '/v1/markets', '/api/v1/markets',
    '/v1/market/stats', '/api/v1/market/stats', '/v1/ticker/24hr'
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

# ============================================================
# TABDEAL — MARKET UNIVERSE
# ============================================================
def tabdeal_markets():
    symbols = {}
    for ep in TABDEAL_MARKET_ENDPOINTS:
        data = get_json(TABDEAL + ep)
        if not data: continue
        for d in flatten_dicts(data):
            s = symbol_from_any(d)
            if s and s.endswith('USDT'):
                symbols[s] = d
        if len(symbols) > 20:
            break
    # Some public APIs use nested symbol metadata.
    return sorted(symbols.keys())


@st.cache_data(ttl=120, show_spinner=False)
def get_tabdeal_universe():
    syms = tabdeal_markets()
    # Conservative fallback for installation/testing when Tabdeal changes its public endpoint.
    if not syms:
        syms = ['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','TRXUSDT','AVAXUSDT','LINKUSDT','DOTUSDT','LTCUSDT','SHIBUSDT','ATOMUSDT','NEARUSDT','APTUSDT','SUIUSDT','ARBUSDT','OPUSDT','PEPEUSDT']
    return syms

# ============================================================
# BINANCE MARKET DATA
# ============================================================
def binance_klines(symbol, interval, limit=500):
    data = get_json(BINANCE + '/api/v3/klines', {'symbol':symbol, 'interval':interval, 'limit':limit})
    if not isinstance(data, list) or len(data) < 40:
        return pd.DataFrame()
    cols=['open_time','open','high','low','close','volume','close_time','qv','trades','tbv','tqv','ignore']
    df=pd.DataFrame(data, columns=cols)
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c], errors='coerce')
    df['time']=pd.to_datetime(df['open_time'], unit='ms', utc=True)
    return df[['time','open','high','low','close','volume']].dropna().reset_index(drop=True)


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
    # Binance is the primary OHLC source because it has stable public historical candles.
    mapping={'15m':'15m','1H':'1h','4H':'4h','1D':'1d'}
    for tf,iv in mapping.items():
        df=binance_klines(symbol,iv,500)
        if df.empty and tf in ('1H','4H'):
            # Wallex fallback
            df=wallex_klines(symbol,'60',14)
            if tf=='4H' and not df.empty:
                x=df.set_index('time').resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
                df=x
        frames[tf]=df
    analyses={tf:timeframe_analysis(df) for tf,df in frames.items() if not df.empty}
    analyses={k:v for k,v in analyses.items() if v}
    if not analyses: return None
    weights={'15m':1.0,'1H':1.2,'4H':1.5,'1D':1.3}
    total=sum(weights[k] for k in analyses)
    tech=sum(v['score']*weights[k] for k,v in analyses.items())/total
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
    confidence=float(np.clip(50+abs(final_score)*0.42 + (5 if len(analyses)>=4 else 0) + max(0,agreement),50,96))
    bullish=final_score>=18
    bearish=final_score<=-18
    decision='معامله کن' if bullish or bearish else 'صبر کن'
    position='لانگ' if bullish else ('شورت' if bearish else 'بدون پوزیشن')
    atrv=analyses.get('4H',analyses.get('1H',next(iter(analyses.values()))))['atr']
    # Conservative levels based on ATR and recent structure.
    all_df=frames.get('4H') if not frames.get('4H',pd.DataFrame()).empty else frames.get('1H')
    supports,resistances=levels(all_df) if all_df is not None and not all_df.empty else ([],[])
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
    return {'symbol':symbol,'price':entry,'sources':source,'analyses':analyses,'technical':tech,'fundamental':fscore,'agreement':agreement,'score':final_score,'confidence':confidence,'decision':decision,'position':position,'entry':entry,'sl':sl,'t1':t1,'t2':t2,'t3':t3,'risk_pct':abs(entry-sl)/entry*100,'supports':supports,'resistances':resistances,'forecast':forecast,'forecast_pct':{'4H':expected_4h,'24H':expected_24h,'72H':expected_72h}}

# ============================================================
# UI
# ============================================================
def money(x):
    if x is None or not np.isfinite(x): return '-'
    if abs(x)>=1000:return f'{x:,.2f}'
    if abs(x)>=1:return f'{x:,.4f}'
    return f'{x:.8f}'.rstrip('0').rstrip('.')

st.markdown('''<style>
.block-container{padding-top:1.1rem;max-width:1250px}
.card{border:1px solid #e7e7e7;border-radius:18px;padding:18px;background:#fff;box-shadow:0 3px 14px rgba(0,0,0,.05);margin-bottom:16px}
.buy{border-right:6px solid #16a34a}.sell{border-right:6px solid #dc2626}.wait{border-right:6px solid #eab308}
.big{font-size:30px;font-weight:800}.muted{color:#666;font-size:13px}.pill{display:inline-block;padding:5px 10px;border-radius:20px;background:#f5f5f5;margin-left:5px;font-size:12px}
</style>''',unsafe_allow_html=True)

st.title('₿ تحلیل‌گر حرفه‌ای رمزارز')
st.caption('بازار USDT صرافی تبدیل • تأیید چندمنبعی با بایننس و والکس • تکنیکال + شرایط بازار • بدون اجرای سفارش')

universe=get_tabdeal_universe()

if 'watch' not in st.session_state: st.session_state.watch=[]

with st.sidebar:
    st.subheader('تنظیمات')
    search=st.text_input('جستجوی ارز',placeholder='BTC / ETH / SOL ...')
    filtered=[s for s in universe if search.upper() in s] if search else universe
    watch_only=st.checkbox('فقط نشانه‌گذاری‌شده‌ها')
    options=[s for s in (st.session_state.watch if watch_only else filtered)]
    if not options: options=filtered
    selected=st.multiselect('انتخاب حداکثر ۵ ارز',options=options,default=[x for x in st.session_state.watch if x in options][:5],max_selections=5)
    st.session_state.watch=selected
    st.divider()
    st.write(f'تعداد بازارهای USDT تبدیل: **{len(universe)}**')
    st.caption('تحلیل با داده‌های عمومی بازار انجام می‌شود و سیگنال سود قطعی نیست.')

# Main selection also visible for phone users.
search2=st.text_input('🔎 جستجوی ارز از میان بازارهای تبدیل',placeholder='مثلاً BTCUSDT یا BTC')
view=[s for s in universe if search2.upper() in s] if search2 else universe
if st.session_state.watch:
    pinned=[s for s in st.session_state.watch if s in universe]
    view=list(dict.fromkeys(pinned+view))

selected=st.multiselect('⭐ حداکثر ۵ ارز برای تحلیل همزمان',view,default=st.session_state.watch[:5],max_selections=5)
st.session_state.watch=selected

if not selected:
    st.info('از منوی بالا تا ۵ ارز را انتخاب کنید.')
    st.stop()

results=[]
with st.spinner('در حال جمع‌آوری داده از منابع و اجرای تحلیل...'):
    with ThreadPoolExecutor(max_workers=5) as ex:
        jobs={ex.submit(aggregate,s):s for s in selected}
        for job in as_completed(jobs):
            try:
                r=job.result()
                if r: results.append(r)
            except Exception: pass

for r in sorted(results,key=lambda x:x['score'],reverse=True):
    cls='buy' if r['decision']=='معامله کن' and r['position']=='لانگ' else ('sell' if r['position']=='شورت' else 'wait')
    icon='🟢' if r['position']=='لانگ' else ('🔴' if r['position']=='شورت' else '🟡')
    src=' | '.join(f'{k}: {money(v)}' for k,v in r['sources'].items()) or 'منبع قیمت در دسترس نیست'
    reason='؛ '.join(reason_text for k in ['15m','1H','4H','1D'] if k in r['analyses'] for reason_text in r['analyses'][k]['reasons'][:3])
    st.markdown(f'''<div class="card {cls}"><div class="big">{icon} {r['symbol']} — {r['decision']}</div>
    <div class="muted">نوع پوزیشن: <b>{r['position']}</b> &nbsp; | &nbsp; امتیاز مدل: <b>{r['score']:.0f}/100</b> &nbsp; | &nbsp; اطمینان تحلیل: <b>{r['confidence']:.0f}%</b></div>
    <hr><b>قیمت مرجع:</b> {money(r['price'])}<br>
    <b>ورود:</b> {money(r['entry'])} &nbsp; <b>حد ضرر:</b> {money(r['sl'])} &nbsp; <b>ریسک:</b> {r['risk_pct']:.2f}%<br>
    <b>Target 1:</b> {money(r['t1'])} &nbsp; <b>Target 2:</b> {money(r['t2'])} &nbsp; <b>Target 3:</b> {money(r['t3'])}<br>
    <b>پیش‌بینی مدل:</b> 4H {r['forecast_pct']['4H']:+.1f}% → {money(r['forecast']['4H'])} &nbsp; | &nbsp; 24H {r['forecast_pct']['24H']:+.1f}% → {money(r['forecast']['24H'])} &nbsp; | &nbsp; 72H {r['forecast_pct']['72H']:+.1f}% → {money(r['forecast']['72H'])}<br>
    <b>دلایل کلیدی:</b> {reason}<br><span class="muted">منابع قیمت: {src}</span></div>''',unsafe_allow_html=True)
    with st.expander(f'جزئیات تحلیل {r["symbol"]}'):
        rows=[]
        for tf,a in r['analyses'].items():
            rows.append({'تایم‌فریم':tf,'امتیاز':round(a['score'],1),'RSI':round(a['rsi'],1),'MACD':round(a['macd'],6),'Momentum %':round(a['momentum'],2),'ADX':round(a['adx'],1)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        c1,c2=st.columns(2)
        with c1:
            st.write('**حمایت‌ها:**', ', '.join(money(x) for x in r['supports']) or '-')
        with c2:
            st.write('**مقاومت‌ها:**', ', '.join(money(x) for x in r['resistances']) or '-')
        st.caption('مدل از ایچیموکو، MACD، RSI، Momentum، ATR، ADX، ساختار قیمت و پرایس‌اکشن استفاده می‌کند. بخش بنیادی/بازاری از داده‌های عمومی رتبه، ارزش بازار و حجم والکس به‌عنوان فیلتر کمکی استفاده می‌کند.')

st.divider()
st.caption('هشدار: این برنامه ابزار تحقیق و تحلیل است. درصد اطمینان، احتمال سود قطعی نیست و پیش‌بینی بازار ذاتاً نامطمئن است. قبل از معامله شرایط نقدشوندگی، کارمزد، اخبار و ریسک شخصی را بررسی کنید.')
