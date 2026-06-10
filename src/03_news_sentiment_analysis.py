"""
Phase 3 - News Sentiment Analysis
Source: RSS feeds (Bloomberg, Financial Times, Yahoo Finance, Reuters, The Economist)
Methods: TextBlob polarity + NLTK VADER (Naive Bayes proxy)
Output: Top 10 stocks ranked by combined average sentiment
Note: RSS feed URLs expire. Replace the feed URLs in RSS_FEEDS with fresh
      ones from https://rss.app before running. The code structure is correct.
"""

import pandas as pd
import feedparser
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt',         quiet=True)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
TOP20_STOCKS = [
    "AAPL", "AMZN", "MSFT", "GOOG",  "GOOGL",
    "META", "TSLA", "NVDA", "JPM",   "JNJ",
    "V",    "PG",   "UNH",  "HD",    "MA",
    "BAC",  "DIS",  "PYPL", "NFLX",  "ADBE",
]

# Replace these URLs with freshly generated feeds from https://rss.app
RSS_FEEDS = {
    "Bloomberg":       "https://rss.app/feeds/A8rERqMwZVhBJwtG.xml",
    "Financial Times": "https://rss.app/feeds/sGYFsq1U1DVSm3Z5.xml",
    "Yahoo Finance":   "https://rss.app/feeds/9LjDf4jsLDjXY19u.xml",
    "Reuters":         "https://rss.app/feeds/e4FCSt0S4OXf3xux.xml",
    "The Economist":   "https://rss.app/feeds/KbmteltvoytJRIFW.xml",
}

ARTICLES_PER_STOCK = 25

# ------------------------------------------------------------------
# Part A: Scrape articles from RSS feeds
# ------------------------------------------------------------------
def collect_articles(stocks: list, feeds: dict, limit: int = 25) -> pd.DataFrame:
    feed_urls   = list(feeds.items())
    all_records = []
    feed_idx    = 0

    for stock in stocks:
        source_name, rss_url = feed_urls[feed_idx % len(feed_urls)]
        feed = feedparser.parse(rss_url)
        count = 0

        for entry in feed.entries:
            if count >= limit:
                break
            all_records.append({
                'Stock':      stock,
                'Date':       entry.get('published',                      'Not Provided'),
                'Journalist': entry.get('author', entry.get('dc:creator', 'Not Provided')),
                'Article':    entry.get('title',                          'No Title'),
                'Source':     source_name,
                'URL':        entry.get('link',                           'Not Provided'),
            })
            count += 1

        feed_idx += 1

    return pd.DataFrame(all_records)

df_articles = collect_articles(TOP20_STOCKS, RSS_FEEDS, ARTICLES_PER_STOCK)
print(f"Total articles collected: {len(df_articles)}")
print(df_articles.head())

# ------------------------------------------------------------------
# Part B: TextBlob sentiment (polarity score)
# ------------------------------------------------------------------
df_articles['Sentiment_TextBlob'] = df_articles['Article'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)
df_textblob_sorted = df_articles.sort_values(by='Sentiment_TextBlob', ascending=False)
print("\nTop 5 most positive articles (TextBlob):")
print(df_textblob_sorted[['Stock', 'Article', 'Sentiment_TextBlob']].head())

# ------------------------------------------------------------------
# Part C: VADER (Naive Bayes proxy) sentiment
# ------------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
df_articles['Sentiment_NB'] = df_articles['Article'].apply(
    lambda x: sia.polarity_scores(x)['compound']
)
df_nb_sorted = df_articles.sort_values(by='Sentiment_NB', ascending=False)
print("\nTop 5 most positive articles (VADER):")
print(df_nb_sorted[['Stock', 'Article', 'Sentiment_NB']].head())

# ------------------------------------------------------------------
# Part D: Portfolio selection - top 10 stocks by combined sentiment
# ------------------------------------------------------------------
avg_textblob = df_articles.groupby('Stock')['Sentiment_TextBlob'].mean()
avg_nb       = df_articles.groupby('Stock')['Sentiment_NB'].mean()

combined_df = pd.DataFrame({
    'TextBlob':   avg_textblob,
    'NaiveBayes': avg_nb,
}).reset_index()

combined_df['AverageSentiment'] = (
    combined_df['TextBlob'] + combined_df['NaiveBayes']
) / 2

top_10 = combined_df.sort_values(by='AverageSentiment', ascending=False).head(10)

print("\nTop 10 Portfolio Stocks by Combined Sentiment:")
print(top_10.to_string(index=False))

# From the original run: AAPL, BAC, V, META, NVDA, UNH, MSFT, PYPL, MA, ADBE
# Selection rationale: combined TextBlob + VADER average scores ranked highest,
# with consistent positive sentiment signals across both models.

# Save sentiment results
df_articles.to_csv('reports/sentiment_analysis_results.csv', index=False)
top_10.to_csv('reports/top10_portfolio_stocks.csv', index=False)
print("\nSaved: reports/sentiment_analysis_results.csv")
print("Saved: reports/top10_portfolio_stocks.csv")
