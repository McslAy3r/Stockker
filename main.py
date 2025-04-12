

import praw
import pandas as pd
import re
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config
from datetime import datetime


SUBREDDIT_NAME = "IndianStreetBets"
HOT_POST_LIMIT = 25
TOP_POST_LIMIT = 50
COMMENT_LIMIT_PER_POST = 20

TARGET_ENTITIES = [
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
TARGET_ENTITIES_UPPER = [entity.upper() for entity in TARGET_ENTITIES]




def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z\s\$]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_mentioned_entities(text, entity_list):
    mentioned = set()
    text_upper = text.upper()
    words = text_upper.split()
    for entity in entity_list:
        if entity in words:
             mentioned.add(entity)
        elif f"${entity}" in text_upper:
             mentioned.add(entity)
    return list(mentioned)

def get_qualitative_sentiment(score):
    if score >= 0.5:
        return "Very Positive"
    elif score >= 0.05:
        return "Positive"
    elif score <= -0.5:
        return "Very Negative"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def save_markdown_report(summary_df, filename="isb_sentiment_report.md", total_items=0, start_time=None):
    if summary_df.empty:
      
        print("No data to generate report.")
        return

    now = datetime.now()
    duration = (now - start_time).total_seconds() if start_time else 0

    summary_df['sentiment_label_qualitative'] = summary_df['average_sentiment'].apply(get_qualitative_sentiment)
    report_df = summary_df[[
        'mentioned_entities', 'mention_count', 'average_sentiment', 'sentiment_label_qualitative',
        'positive_mentions', 'negative_mentions', 'neutral_mentions'
    ]].copy()
    report_df.rename(columns={
        'mentioned_entities': 'Entity/Term',
        'mention_count': 'Mentions',
        'average_sentiment': 'Avg. Score',
        'sentiment_label_qualitative': 'Overall Sentiment',
        'positive_mentions': 'Positive',
        'negative_mentions': 'Negative',
        'neutral_mentions': 'Neutral'
    }, inplace=True)
    report_df['Avg. Score'] = report_df['Avg. Score'].map('{:.2f}'.format)

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# r/{SUBREDDIT_NAME} Sentiment Analysis Report\n\n")
            f.write(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if start_time:
                 f.write(f"**Analysis Duration:** {duration:.2f} seconds\n")
            f.write(f"**Data Source:** Combined 'Hot' ({HOT_POST_LIMIT}) and 'Top (Week)' ({TOP_POST_LIMIT}) posts\n")
            f.write(f"**Total Posts/Comments Processed (Unique):** {total_items}\n")
            f.write(f"**Total Mentions Analyzed:** {report_df['Mentions'].sum()}\n\n")
            f.write(f"## Sentiment Summary for Tracked Entities/Terms\n\n")
            f.write(report_df.to_markdown(index=False))
            f.write("\n\n")
            f.write("---\n\n")
            f.write("### Notes:\n")
            f.write("- **Avg. Score:** Average VADER compound sentiment score (-1.0 to +1.0).\n")
            f.write("- **Overall Sentiment:** Qualitative label based on Avg. Score.\n")
            f.write("- Sentiment analysis performed using VADER on post titles, text, and comments.\n\n")
            f.write("************************************************************\n")
            f.write("**DISCLAIMER:** This analysis is based on public Reddit data using automated tools. Social media sentiment is volatile, potentially biased, and NOT reliable financial advice. Make investment decisions based on thorough personal research and professional advice.\n")
            f.write("************************************************************\n")
        
        print(f"User-readable report saved to {filename}")
    except Exception as e:
        
        print(f"Error writing Markdown report: {e}")



start_time = datetime.now()


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
subreddit = reddit.subreddit(SUBREDDIT_NAME)

scraped_data = []
processed_ids = set()

def process_submissions(submissions):
    global scraped_data, processed_ids
    for submission in submissions:
        if submission.id not in processed_ids:
            scraped_data.append({
                'id': submission.id,
                'type': 'post',
                'title': submission.title,
                'text': submission.selftext,
                'score': submission.score,
                'url': submission.url,
                'created_utc': submission.created_utc
            })
            processed_ids.add(submission.id)

            try:
                submission.comments.replace_more(limit=0)
                comment_count = 0
                for comment in submission.comments.list():
                    if comment_count >= COMMENT_LIMIT_PER_POST:
                        break
                    if comment.id not in processed_ids:
                        scraped_data.append({
                            'id': comment.id,
                            'type': 'comment',
                            'title': None,
                            'text': comment.body,
                            'score': comment.score,
                            'url': f"https://reddit.com{comment.permalink}",
                            'created_utc': comment.created_utc
                        })
                        processed_ids.add(comment.id)
                        comment_count += 1
            except Exception as comment_error:
                 
                 pass 


try:
    hot_submissions = subreddit.hot(limit=HOT_POST_LIMIT)
    process_submissions(hot_submissions)

    top_submissions = subreddit.top(time_filter='week', limit=TOP_POST_LIMIT)
    process_submissions(top_submissions)

except praw.exceptions.PRAWException as praw_error:
    
    print(f"\nA PRAW error occurred during scraping: {praw_error}")
    exit() 
except Exception as e:
    
    print(f"\nAn unexpected error occurred during scraping: {e}")
    exit() # Exit on unexpected errors


if not scraped_data:

    print("No unique data scraped. Exiting.")
    exit()

df = pd.DataFrame(scraped_data)
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['cleaned_text'] = df['full_text'].apply(preprocess_text)
df['mentioned_entities'] = df['cleaned_text'].apply(lambda text: find_mentioned_entities(text, TARGET_ENTITIES_UPPER))

df_filtered = df[df['mentioned_entities'].apply(len) > 0].copy()

if df_filtered.empty:
    
    print("No mentions of target entities found in the scraped data.")
    exit()

analyzer = SentimentIntensityAnalyzer()
df_filtered['sentiment_score'] = df_filtered['cleaned_text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
df_filtered['sentiment_label'] = df_filtered['sentiment_score'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))

df_exploded = df_filtered.explode('mentioned_entities')
sentiment_summary = df_exploded.groupby('mentioned_entities').agg(
    mention_count=('id', 'count'),
    average_sentiment=('sentiment_score', 'mean'),
    positive_mentions=('sentiment_label', lambda x: (x == 'Positive').sum()),
    negative_mentions=('sentiment_label', lambda x: (x == 'Negative').sum()),
    neutral_mentions=('sentiment_label', lambda x: (x == 'Neutral').sum())
).reset_index()
sentiment_summary = sentiment_summary.sort_values(by='mention_count', ascending=False)


output_csv_filename = "isb_sentiment_summary.csv"
try:
    sentiment_summary.to_csv(output_csv_filename, index=False, encoding='utf-8')
    
    print(f"Detailed results saved to {output_csv_filename}")
except Exception as e:
    
    print(f"\nError saving results to CSV: {e}")

output_md_filename = "isb_sentiment_report.md"
save_markdown_report(sentiment_summary, output_md_filename, total_items=len(scraped_data), start_time=start_time)

end_time = datetime.now()

print(f"\nScript finished. Total execution time: {(end_time - start_time).total_seconds():.2f} seconds.")


print("\n************************************************************")
print("Remember: This is NOT financial advice. Please read the")
print(f"full disclaimer in the generated report '{output_md_filename}'.")
print("************************************************************")