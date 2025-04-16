import praw
import pandas as pd
import re
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config
from datetime import datetime

sbn = "IndianStreetBets"
hlm = 25
tlm = 50
clm = 20

tgt = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR',
    'ITC', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'KOTAKBANK', 'LT',
    'ASIANPAINT', 'AXISBANK', 'MARUTI', 'TITAN', 'WIPRO', 'ULTRACEMCO',
    'ADANIENT', 'ADANIPORTS', 'NESTLEIND', 'POWERGRID', 'NTPC',
    'TATAMOTORS', 'TATASTEEL', 'JSWSTEEL', 'SUNPHARMA', 'ONGC',
    'INDUSINDBK', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'HCLTECH',
    'HDFCLIFE', 'SBILIFE', 'BAJAJFINSV', 'TECHM', 'GRASIM', 'BPCL',
    'SHREECEM', 'HEROMOTOCO', 'EICHERMOT', 'COALINDIA', 'BRITANNIA',
    'UPL', 'APOLLOHOSP', 'HINDALCO', 'BAJAJ-AUTO',
    'PARAG PARIKH FLEXI CAP', 'AXIS BLUECHIP', 'MIRAE ASSET LARGE CAP',
    'NIFTY', 'BANKNIFTY', 'SENSEX', 'IPO', 'FPO'
]
tgt = [t.upper() for t in tgt]

urp = re.compile(r"http\S+|www\S+|https\S+", re.MULTILINE)
mnp = re.compile(r"@\w+|#")
nap = re.compile(r"[^a-zA-Z\s\$]")
spp = re.compile(r"\s+")

def ptx(txt):
    if not isinstance(txt, str): return ""
    txt = txt.lower()
    txt = urp.sub('', txt)
    txt = mnp.sub('', txt)
    txt = nap.sub('', txt)
    return spp.sub(' ', txt).strip()

def fnd(txt, ens):
    fds = set()
    up = txt.upper()
    wds = up.split()
    for e in ens:
        if e in wds or f'${e}' in up:
            fds.add(e)
    return list(fds)

def gsn(scr):
    if scr >= 0.5: return "Very Positive"
    if scr >= 0.05: return "Positive"
    if scr <= -0.5: return "Very Negative"
    if scr <= -0.05: return "Negative"
    return "Neutral"

def smd(df, fn="isb_report.md", tot=0, st=None):
    if df.empty: return
    now = datetime.now()
    dur = (now - st).total_seconds() if st else 0
    df['ovs'] = df['avs'].apply(gsn)
    rep = df[[
        'ent', 'cnt', 'avs', 'ovs',
        'pos', 'neg', 'neu'
    ]].copy()
    rep.rename(columns={
        'ent': 'Entity',
        'cnt': 'Mentions',
        'avs': 'AvgScore',
        'ovs': 'Sentiment',
        'pos': 'Positive',
        'neg': 'Negative',
        'neu': 'Neutral'
    }, inplace=True)
    rep['AvgScore'] = rep['AvgScore'].map('{:.2f}'.format)
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(f"# r/{sbn} Sentiment Analysis\n\n")
        f.write(f"Generated: {now:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Duration: {dur:.2f} secs\n")
        f.write(f"Total Items: {tot}\n\n")
        f.write("## Summary by Entity\n\n")
        f.write(rep.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("Disclaimer: automated analysis, not financial advice\n")

st = datetime.now()
try:
    r = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        username=config.REDDIT_USERNAME,
        password=config.REDDIT_PASSWORD,
        check_for_async=False
    )
except Exception as e:
    exit()
    
sub = r.subreddit(sbn)
dt = []
seen = set()

def psb(sbs):
    global dt, seen
    for p in sbs:
        if p.id in seen: continue
        seen.add(p.id)
        dt.append({
            'id': p.id,
            'typ': 'post',
            'ttl': p.title,
            'txt': p.selftext,
            'scr': p.score,
            'url': p.url,
            'utc': p.created_utc
        })
        try:
            p.comments.replace_more(limit=0)
            n = 0
            for c in p.comments.list():
                if n >= clm: break
                if c.id in seen: continue
                seen.add(c.id)
                dt.append({
                    'id': c.id,
                    'typ': 'comment',
                    'ttl': None,
                    'txt': c.body,
                    'scr': c.score,
                    'url': f"https://reddit.com{c.permalink}",
                    'utc': c.created_utc
                })
                n += 1
        except:
            pass

psb(sub.hot(limit=hlm))
psb(sub.top(time_filter='week', limit=tlm))

if not dt: exit()
    
df = pd.DataFrame(dt)
df['ftx'] = df['ttl'].fillna('') + ' ' + df['txt'].fillna('')
df['cln'] = df['ftx'].apply(ptx)
df['ent'] = df['cln'].apply(lambda t: fnd(t, tgt))
df = df[df['ent'].map(len) > 0]

if df.empty: exit()
    
viz = SentimentIntensityAnalyzer()
df['ssc'] = df['cln'].apply(lambda t: viz.polarity_scores(t)['compound'])
df['slb'] = df['ssc'].apply(lambda s: 'Positive' if s >= 0.05 else ('Negative' if s <= -0.05 else 'Neutral'))

exp = df.explode('ent')
smr = exp.groupby('ent').agg(
    cnt=('id', 'count'),
    avs=('ssc', 'mean'),
    pos=('slb', lambda x: (x == 'Positive').sum()),
    neg=('slb', lambda x: (x == 'Negative').sum()),
    neu=('slb', lambda x: (x == 'Neutral').sum())
).reset_index().sort_values('cnt', ascending=False)

smr.to_csv('sentiment_summary.csv', index=False)
smd(smr, 'sentiment_report.md', len(dt), st)
