import streamlit as st
import pandas as pd
import numpy as np
import requests, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title='Crypto Analyzer Pro V6.3', page_icon='₿', layout='wide', initial_sidebar_state='collapsed')

TF={'5m':'5m','15m':'15m','30m':'30m','1H':'1h','2H':'2h','4H':'4h','6H':'6h','12H':'12h','1D':'1d','3D':'3d','1W':'1w'}
BINANCE='https://api.binance.com'; OKX='https://www.okx.com'; TABDEAL='https://api.tabdeal.org'
S=requests.Session(); S.headers.update({'User-Agent':'CryptoAnalyzerPro-V6.3/1.0'})
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
    # Multi-timeframe context uses whatever is actually available.
    ts=[trend(frames[x]) for x in ['15m','1H','4H','1D'] if x in frames]
    bull=ts.count('صعودی'); bear=ts.count('نزولی');
    long_score=50 + (20 if e20.iloc[-1]>e50.iloc[-1] else -10) + (15 if mm.iloc[-1]>ms.iloc[-1] else -10) + np.clip(mom*5,-15,15) + (12 if 48<=rv<=72 else -5) + (10 if vol>=1.15 else 0) + (15 if bu else 8 if pl else 0) + bull*4 - bear*3
    short_score=50 + (20 if e20.iloc[-1]<e50.iloc[-1] else -10) + (15 if mm.iloc[-1]<ms.iloc[-1] else -10) - np.clip(mom*5,-15,15) + (12 if 28<=rv<=52 else -5) + (10 if vol>=1.15 else 0) + (15 if bd else 8 if ps else 0) + bear*4 - bull*3
    long_score=float(np.clip(long_score,0,100)); short_score=float(np.clip(short_score,0,100)); score=max(long_score,short_score)
    # Balanced trade gate: 4 confirmations out of 7, not all conditions simultaneously.
    lc=sum([e20.iloc[-1]>e50.iloc[-1], mm.iloc[-1]>ms.iloc[-1], mom>0, 48<=rv<=72, vol>=1.05, bu or pl or bull>=2, bull>=bear])
    sc=sum([e20.iloc[-1]<e50.iloc[-1], mm.iloc[-1]<ms.iloc[-1], mom<0, 28<=rv<=52, vol>=1.05, bd or ps or bear>=2, bear>=bull])
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
    return {'symbol':symbol,'status':'OK','direction':direction,'trade':trade,'score':round(score,1),'confidence':round(confidence,1),'long_score':round(long_score,1),'short_score':round(short_score,1),'confirmations':confirmations,'coverage':len(coverage),'coverage_tfs':','.join(coverage),'price':price,'entry':entry,'sl':sl,'tp1':tp1,'tp2':tp2,'tp3':tp3,'rr':round(rr,2),'rsi':round(rv,1),'adx':round(ax,1),'momentum':round(mom,2),'volume':round(vol,2),'setup':'Breakout' if bu or bd else 'Pullback' if pl or ps else 'No setup','pump_score':round(pump,1),'liquidity':round(liq,1),'regime':'صعودی' if bull>bear else 'نزولی' if bear>bull else 'رنج','t15':trend(frames['15m']) if '15m' in frames else 'N/A','t1':trend(frames['1H']) if '1H' in frames else 'N/A','t4':trend(frames['4H']) if '4H' in frames else 'N/A','tD':trend(frames['1D']) if '1D' in frames else 'N/A','risk_money':capital*risk_pct/100,'qty':(capital*risk_pct/100)/abs(entry-sl) if abs(entry-sl)>0 else 0}

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
    av=float(atr(df).iloc[-1]); e20=float(ema(df.close,20).iloc[-1]); e50=float(ema(df.close,50).iloc[-1]); rv=float(rsi(df.close).iloc[-1]); mm,ms,_=macd(df.close); mom=float((price/df.close.iloc[-6]-1)*100); vol=float(df.volume.iloc[-1]/max(df.volume.rolling(20).mean().iloc[-1],1e-12))
    direction=row.get('direction','WAIT')
    if direction=='LONG':
        sl=float(row.get('sl',max((ss[-1] if ss else price-av*2),price-av*2)))
        risk=abs(price-sl)
        resistance=[x for x in rs if x>price]
        tps=(resistance[:3] if len(resistance)>=3 else [price+1.5*risk,price+2.5*risk,price+3.5*risk])
        low_entry=max(ss[0] if ss else price-av, price-1.2*av)
        high_entry=min(e20 if e20<price else price, price)
        safe=(min(low_entry,price), max(high_entry,price)) if low_entry<price else (price,price)
        aggressive=(resistance[0]*1.003 if resistance else price+0.8*av)
        decision='خرید / تأیید' if price>e20 and mom>0 and vol>=1.05 else 'صبر برای پولبک / تأیید شکست'
        scenario='اگر مقاومت نزدیک شکسته و بالای آن تثبیت شود، سناریوی صعودی فعال‌تر می‌شود.'
        downside='از دست رفتن حمایت نزدیک و شکست SL سناریوی صعودی را باطل می‌کند.'
    elif direction=='SHORT':
        sl=float(row.get('sl',min((rs[0] if rs else price+av*2),price+av*2)))
        risk=abs(sl-price)
        support=[x for x in ss if x<price]
        tps=(support[:3] if len(support)>=3 else [price-1.5*risk,price-2.5*risk,price-3.5*risk])
        low_entry=max(price, e20 if e20>price else price)
        high_entry=min(rs[0] if rs else price+av, price+1.2*av)
        safe=(price, max(low_entry,price))
        aggressive=(support[0]*0.997 if support else price-0.8*av)
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
        tps=[x for x in rs[:3]] or [price+1.5*max(av,price*.01),price+2.5*max(av,price*.01),price+3.5*max(av,price*.01)]
        sl=nearest_s
        safe=(nearest_s, price)
        aggressive=bullish_entry
        decision='صبر / تأیید شکست'
        scenario=f'عبور و تثبیت بالای {fmt(bullish_entry)} سناریوی صعودی را فعال می‌کند.'
        downside=f'شکست {fmt(bearish_entry)} به پایین، سناریوی نزولی را تقویت می‌کند.'
    # Heuristic scenario confidence, explicitly treated as an estimate.
    base=float(row.get('confidence',0) or 0)
    quality=(10 if vol>=1.15 else 0)+(8 if abs(mom)>=1 else 0)+(5 if 35<=rv<=70 else 0)
    est=float(np.clip(base+quality-(8 if len(rs)<2 or len(ss)<2 else 0),0,95))
    outlook=float(np.clip(mom*1.6 + (4 if direction=='LONG' else -4 if direction=='SHORT' else 0),-20,20))
    return {'price':price,'t24':t24,'supports':ss,'resistances':rs,'rsi':rv,'momentum':mom,'volume':vol,'e20':e20,'e50':e50,'sl':sl,'tps':tps[:3],'safe_entry':safe,'aggressive_entry':aggressive,'decision':decision,'scenario':scenario,'downside':downside,'estimated_success':est,'outlook':outlook,'risk':risk}

# ---------- UI ----------
st.title('₿ Crypto Analyzer Pro V6.4')
st.caption('Whole-Market Scanner — بدون سقف مصنوعی تعداد ارز')
symbols,source=universe()
if not symbols: st.error('فهرست بازار دریافت نشد.'); st.stop()

with st.sidebar:
    st.write(f'منبع Universe: **{source}**')
    st.write(f'تعداد جفت‌های USDT: **{len(symbols)}**')
    tf=st.selectbox('تایم‌فریم ورود',list(TF.keys()),index=3)
    q=st.text_input('فیلتر نماد اختیاری','')
    capital=st.number_input('سرمایه USDT',100.0,1000000.0,1000.0,100.0)
    risk=st.slider('ریسک هر معامله %',0.25,2.0,1.0,0.25)
    st.info('اسکن بازار همیشه کل Universe فیلترشده را بررسی می‌کند. هیچ max_scan=250 وجود ندارد.')

filtered=[s for s in symbols if q.upper() in s] if q else symbols
st.metric('Universe قابل اسکن',len(filtered))

# Full-market coin dropdown: available even before scanning.
st.subheader('انتخاب ارز')
coin_search=st.text_input('جستجوی سریع ارز', '', placeholder='مثلاً BTC یا ETH', key='coin_search')
coin_options=sorted([s for s in symbols if coin_search.upper() in s] if coin_search else symbols)
if coin_options:
    selected_coin=st.selectbox('همه ارزهای USDT', coin_options, key='selected_coin_dropdown', help='تمام جفت‌های USDT موجود در Universe')
    if 'selected_list' not in st.session_state:
        st.session_state.selected_list=[]
    if st.button('➕ افزودن ارز به تحلیل', key='add_coin'):
        if selected_coin not in st.session_state.selected_list and len(st.session_state.selected_list)<5:
            st.session_state.selected_list.append(selected_coin)
    if st.session_state.get('selected_list'):
        st.caption('ارزهای انتخاب‌شده: ' + ' | '.join(st.session_state.selected_list))
        if st.button('پاک کردن انتخاب‌ها', key='clear_coins'):
            st.session_state.selected_list=[]

if st.button('اسکن کل بازار',type='primary',use_container_width=True):
    df=scan_all(tuple(filtered),tf,capital,risk)
    st.session_state['scan_df']=df

if 'scan_df' in st.session_state:
    df=st.session_state['scan_df'].copy(); total=len(filtered); analyzed=int((df.status=='OK').sum()); nodata=int((df.status!='OK').sum())
    tradable=df[(df.status=='OK')&(df.direction!='WAIT')&(df.confidence>=50)&(df.rr>=1.2)&(df.liquidity>=20)].copy()
    longs=int((tradable.direction=='LONG').sum()); shorts=int((tradable.direction=='SHORT').sum())
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric('کل بازار',total); c2.metric('بررسی‌شده',analyzed); c3.metric('داده ناقص/خطا',nodata); c4.metric('LONG',longs); c5.metric('SHORT',shorts)

    # Detailed analysis uses the full-market dropdown above.
    selected=st.session_state.get('selected_coin_dropdown', coin_options[0] if coin_options else symbols[0])
    matched=df[df.symbol==selected]
    if not matched.empty and matched.iloc[0].get('status')=='OK':
        row=matched.iloc[0].to_dict()
    else:
        # If the selected coin was not part of the current scan result, analyze it on demand.
        row=analyze(selected,tf,capital,risk)
    plan=detailed_plan(selected,tf,row)
    if plan:
        st.subheader(f'تحلیل کامل {selected}')
        m=plan['t24'] or {}
        trend_txt=row.get('regime','رنج')
        st.markdown(f"**روند کوتاه‌مدت:** {'🟢 صعودی' if trend_txt=='صعودی' else '🔴 نزولی' if trend_txt=='نزولی' else '🟡 رنج'}")
        a,b,c,d,e=st.columns(5)
        a.metric('قیمت فعلی',fmt(plan['price']))
        b.metric('رشد 24H',fmt_pct(m.get('change24',row.get('momentum',0))))
        c.metric('سقف 24H',fmt(m.get('high24',0)))
        d.metric('کف 24H',fmt(m.get('low24',0)))
        e.metric('حجم 24H',f"${m.get('volume24',0)/1e6:.2f}M" if m.get('volume24') else '-')
        st.markdown('**مقاومت‌ها**')
        st.write(' • '.join('$'+fmt(x) for x in plan['resistances']) if plan['resistances'] else 'سطح مقاومت کافی شناسایی نشد')
        st.markdown('**حمایت‌ها**')
        st.write(' • '.join('$'+fmt(x) for x in plan['supports']) if plan['supports'] else 'سطح حمایت کافی شناسایی نشد')
        st.markdown('### پوزیشن پیشنهادی من')
        pos=st.columns(2)
        with pos[0]:
            st.markdown(f"**نوع:** {'Long / خرید' if row.get('direction')=='LONG' else 'Short / فروش' if row.get('direction')=='SHORT' else 'WAIT / صبر'}")
            se=plan['safe_entry']
            if isinstance(se,tuple): st.write(f"**ورود کم‌ریسک:** ${fmt(se[0])} — ${fmt(se[1])}")
            else: st.write(f"**ورود کم‌ریسک:** ${fmt(se)}")
            st.write(f"**ورود تهاجمی:** ${fmt(plan['aggressive_entry'])}")
            st.write(f"**Stop Loss:** ${fmt(plan['sl'])}")
        with pos[1]:
            for i,x in enumerate(plan['tps'],1): st.write(f"**Target {i}:** ${fmt(x)}")
            st.write(f"**احتمال موفقیت تخمینی:** {plan['estimated_success']:.0f}%")
            st.write(f"**چشم‌انداز سناریویی:** {plan['outlook']:+.1f}%")
            st.write(f"**تصمیم فعلی:** **{plan['decision']}**")
        st.info(plan['scenario'])
        st.warning(plan['downside'])
        st.caption('احتمال موفقیت و چشم‌انداز، برآورد الگوریتمی بر اساس داده بازار هستند و تضمین نتیجه معامله نیستند.')
    st.subheader('فرصت‌های معاملاتی')
    if tradable.empty: st.warning('در کل بازار موقعیت با شرایط فعلی پیدا نشد؛ اما همه ارزها اسکن شده‌اند و جدول پایین وضعیت کامل را نشان می‌دهد.')
    else:
        show=tradable.sort_values(['confidence','score','pump_score'],ascending=False)[['symbol','trade','direction','confidence','score','confirmations','coverage','setup','price','entry','sl','tp1','tp2','rr','pump_score','liquidity','t15','t1','t4','tD']].copy()
        show.columns=['ارز','معامله','جهت','Confidence','Score','تأییدها','پوشش TF','ستاپ','قیمت','Entry','SL','TP1','TP2','R:R','Pump','Liquidity','15m','1H','4H','1D']
        for c in ['قیمت','Entry','SL','TP1','TP2']: show[c]=show[c].map(fmt)
        show['Confidence']=show['Confidence'].map(lambda x:f'{x:.1f}%')
        st.dataframe(show,use_container_width=True,hide_index=True)

    st.subheader('کل بازار — بدون حذف نتایج')
    allshow=df.copy()
    allshow=allshow.sort_values(['status','confidence','score'],ascending=[True,False,False])
    cols=['symbol','status','direction','trade','confidence','score','confirmations','coverage','coverage_tfs','regime','setup','price','momentum','volume','pump_score','reason']
    for c in cols:
        if c not in allshow.columns: allshow[c]='-'
    allshow=allshow[cols]
    allshow.columns=['ارز','وضعیت داده','جهت','معامله','Confidence','Score','تأییدها','پوشش TF','TFهای موجود','رژیم','ستاپ','قیمت','Momentum %','Volume x','Pump','علت']
    allshow['قیمت']=allshow['قیمت'].map(fmt)
    allshow['Confidence']=allshow['Confidence'].apply(lambda x:f'{float(x):.1f}%' if pd.notna(x) and str(x) not in ('-','nan') else '-')
    st.dataframe(allshow,use_container_width=True,hide_index=True)

    st.subheader('🚀 Pump Candidates')
    pump=df[(df.status=='OK')&(df.pump_score>=28)].sort_values(['pump_score','confidence'],ascending=False).head(30)
    if pump.empty: st.info('کاندید پامپ پیدا نشد.')
    else:
        p=pump[['symbol','pump_score','confidence','momentum','volume','setup','direction','price']].copy(); p.columns=['ارز','Pump Score','Confidence','Momentum %','Volume x','ستاپ','جهت','قیمت']; p['قیمت']=p['قیمت'].map(fmt); p['Confidence']=p['Confidence'].map(lambda x:f'{x:.1f}%'); st.dataframe(p,use_container_width=True,hide_index=True)

st.caption('V6.4: تحلیل سناریومحور، حمایت/مقاومت، دو نوع ورود، اهداف و تصمیم نهایی؛ کل Universe اسکن می‌شود.')
