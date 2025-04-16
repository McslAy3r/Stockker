
import praw
import pandas as pd
import re
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config
from datetime import datetime

subredditName = "IndianStreetBets"
hotLimit = 25
topLimit = 50
commentLimit = 20

targets = [
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
targets = [t.upper() for t in targets]

urlPattern = re.compile(r"http\S+|www\S+|https\S+", re.MULTILINE)
mentionPattern = re.compile(r"@\w+|#")
nonAlphaPattern = re.compile(r"[^a-zA-Z\s\$]")
spacePattern = re.compile(r"\s+")

def preprocessText(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = urlPattern.sub('', text)
    text = mentionPattern.sub('', text)
    text = nonAlphaPattern.sub('', text)
    return spacePattern.sub(' ', text).strip()

def findEntities(text, entityList):
    found = set()
    up = text.upper()
    words = up.split()
    for e in entityList:
        if e in words or f'${e}' in up:
            found.add(e)
    return list(found)

def getQualSent(score):
    if score >= 0.5: return "Very Positive"
    if score >= 0.05: return "Positive"
    if score <= -0.5: return "Very Negative"
    if score <= -0.05: return "Negative"
    return "Neutral"

def saveMarkdown(df, filename="isb_report.md", total=0, start=None):
    if df.empty:
        print("No data to report")
        return
    now = datetime.now()
    duration = (now - start).total_seconds() if start else 0
    df['overallSent'] = df['average_sentiment'].apply(getQualSent)
    rep = df[[
        'mentioned_entities', 'mention_count', 'average_sentiment', 'overallSent',
        'positive_mentions', 'negative_mentions', 'neutral_mentions'
    ]].copy()
    rep.rename(columns={
        'mentioned_entities': 'Entity',
        'mention_count': 'Mentions',
        'average_sentiment': 'AvgScore',
        'overallSent': 'Sentiment',
        'positive_mentions': 'Positive',
        'negative_mentions': 'Negative',
        'neutral_mentions': 'Neutral'
    }, inplace=True)
    rep['AvgScore'] = rep['AvgScore'].map('{:.2f}'.format)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# r/{subredditName} Sentiment Analysis\n\n")
        f.write(f"Generated: {now:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Duration: {duration:.2f} secs\n")
        f.write(f"Total Items: {total}\n\n")
        f.write("## Summary by Entity\n\n")
        f.write(rep.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("Disclaimer: automated analysis, not financial advice\n")
    print(f"Report saved to {filename}")

start = datetime.now()
print("Connecting to Reddit...")
try:
    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        username=config.REDDIT_USERNAME,
        password=config.REDDIT_PASSWORD,
        check_for_async=False
    )
except Exception as e:
    print(f"Error connecting to Reddit: {e}")
    exit()
sub = reddit.subreddit(subredditName)
data = []
seen = set()

def processSubs(subs):
    global data, seen
    for post in subs:
        if post.id in seen: continue
        seen.add(post.id)
        data.append({
            'id': post.id,
            'type': 'post',
            'title': post.title,
            'text': post.selftext,
            'score': post.score,
            'url': post.url,
            'created_utc': post.created_utc
        })
        try:
            post.comments.replace_more(limit=0)
            cnt = 0
            for c in post.comments.list():
                if cnt >= commentLimit: break
                if c.id in seen: continue
                seen.add(c.id)
                data.append({
                    'id': c.id,
                    'type': 'comment',
                    'title': None,
                    'text': c.body,
                    'score': c.score,
                    'url': f"https://reddit.com{c.permalink}",
                    'created_utc': c.created_utc
                })
                cnt += 1
        except:
            pass

processSubs(sub.hot(limit=hotLimit))
processSubs(sub.top(time_filter='week', limit=topLimit))

if not data:
    print("No data scraped")
    exit()
df = pd.DataFrame(data)
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['cleaned'] = df['full_text'].apply(preprocessText)
df['mentioned_entities'] = df['cleaned'].apply(lambda t: findEntities(t, targets))
df = df[df['mentioned_entities'].map(len) > 0]
if df.empty:
    print("No mentions found")
    exit()
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['cleaned'].apply(lambda t: analyzer.polarity_scores(t)['compound'])
df['sent_label'] = df['sentiment_score'].apply(lambda s: 'Positive' if s >= 0.05 else ('Negative' if s <= -0.05 else 'Neutral'))
exp = df.explode('mentioned_entities')
summary = exp.groupby('mentioned_entities').agg(
    mention_count=('id', 'count'),
    average_sentiment=('sentiment_score', 'mean'),
    positive_mentions=('sent_label', lambda x: (x == 'Positive').sum()),
    negative_mentions=('sent_label', lambda x: (x == 'Negative').sum()),
    neutral_mentions=('sent_label', lambda x: (x == 'Neutral').sum())
).reset_index().sort_values('mention_count', ascending=False)
summary.to_csv('sentiment_summary.csv', index=False)
saveMarkdown(summary, 'sentiment_report.md', len(data), start)
print("Done.")


