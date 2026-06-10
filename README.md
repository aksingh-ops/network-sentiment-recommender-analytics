# Network Analysis, Sentiment Analytics, and Recommender Systems

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![NetworkX](https://img.shields.io/badge/NetworkX-3.1%2B-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green?style=flat-square)
![yfinance](https://img.shields.io/badge/yfinance-0.2%2B-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

Four-problem analytical pipeline covering graph-based network ranking, financial data visualization, NLP-driven sentiment analysis, and SVD-based collaborative filtering. Built on two real-world datasets spanning social network edges and movie ratings.

---

## Project Overview

<table>
  <thead>
    <tr>
      <th>Phase</th>
      <th>Scope</th>
      <th>Dataset</th>
      <th>Key Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1 &mdash; Network Analysis</strong></td>
      <td>PageRank and HITS (Hubs + Authorities) on a directed social graph</td>
      <td>sn_ids.csv</td>
      <td>Node 31890 ranked #1 by PageRank (0.019) and Authority score (0.010)</td>
    </tr>
    <tr>
      <td><strong>2 &mdash; Institutional Holders Network</strong></td>
      <td>Yahoo Finance API &rarr; directed bipartite graph (holder &rarr; ticker)</td>
      <td>yfinance (live)</td>
      <td>Vanguard and BlackRock identified as highest-degree hub nodes across all 20 tickers</td>
    </tr>
    <tr>
      <td><strong>3 &mdash; News Sentiment Analysis</strong></td>
      <td>RSS feed scraping + TextBlob + VADER (Naive Bayes proxy)</td>
      <td>500 articles across 20 stocks</td>
      <td>Top 10 portfolio: AAPL, BAC, V, META, NVDA, UNH, MSFT, PYPL, MA, ADBE</td>
    </tr>
    <tr>
      <td><strong>4 &mdash; SVD Recommender System</strong></td>
      <td>Collaborative filtering via TruncatedSVD on a sparse user-movie utility matrix</td>
      <td>ratings.csv</td>
      <td>RMSE 1.04 | 75.50% matrix sparsity | top-5 recommendations per user</td>
    </tr>
  </tbody>
</table>

---

## Datasets

| File | Records | Columns | Description |
|---|---|---|---|
| `data/sn_ids.csv` | 289,003 | 2 | Directed social network edges (id_1 &rarr; id_2) across 37,700 unique nodes |
| `data/ratings.csv` | 100,836 | 3 | MovieLens ratings (userId, movieId, rating) &mdash; 610 users, 9,724 movies |

---

## Repository Structure

```
network-sentiment-recommender-analytics/
├── data/
│   ├── sn_ids.csv
│   └── ratings.csv
├── notebooks/
│   └── MIS546_HW4_Analysis.html          # Rendered notebook with all outputs
├── reports/
│   ├── network_pagerank_top5.png
│   ├── network_hits_authorities_top5.png
│   ├── network_hits_hubs_top5.png
│   ├── holder_ticker_network_base.png
│   ├── holder_ticker_network_enhanced.png
│   ├── sentiment_analysis_results.csv
│   └── top10_portfolio_stocks.csv
├── src/
│   ├── 01_network_analysis_pagerank_hits.py
│   ├── 02_institutional_holders_network.py
│   ├── 03_news_sentiment_analysis.py
│   └── 04_svd_recommender_system.py
├── requirements.txt
└── README.md
```

---

## Methodology Highlights

### Network Analysis (Phase 1)
- Directed graph built from 289,003 social network edges using NetworkX
- **PageRank:** Propagates authority through outgoing links; identifies globally influential nodes. Top node: ID 31890 (score: 0.019)
- **HITS:** Separates hubs (nodes with strong outgoing links) from authorities (highly cited nodes). Top hub: 27803; top authority: 31890
- Network graphs plotted at figsize(50, 50) with differentiated colors for id_1 (source) and id_2 (target) nodes

| Algorithm | Top Node | Score |
|---|---|---|
| PageRank | 31890 | 0.019 |
| HITS &mdash; Authority | 31890 | 0.010 |
| HITS &mdash; Hub | 27803 | 0.005 |

### Institutional Holders Network (Phase 2)
- Pulled top 20 institutional holders per ticker from Yahoo Finance (200 rows, 20 tickers)
- Directed bipartite graph: holder &rarr; ticker, edge width normalized by holding dollar amount
- Ticker node size scaled by degree (number of institutional connections)
- Enhanced graph uses Kamada-Kawai layout and highlights top-5 holders by total AUM in orange

### Sentiment Analysis (Phase 3)
- 500 articles distributed across 20 stocks from 5 RSS feeds (Bloomberg, FT, Yahoo Finance, Reuters, The Economist)
- **TextBlob:** Pattern-based polarity scoring (&minus;1.0 to +1.0)
- **VADER:** Lexicon-based compound score, more robust for financial and news text
- Combined average sentiment used to rank and select top 10 portfolio stocks

| Stock | TextBlob | VADER | Combined |
|---|---|---|---|
| AAPL | 0.080 | 0.082 | 0.081 |
| BAC  | 0.080 | 0.082 | 0.081 |
| V    | 0.080 | 0.082 | 0.081 |
| META | 0.080 | 0.082 | 0.081 |
| NVDA | 0.065 | 0.049 | 0.057 |

> Note: RSS feeds expire. Replace feed URLs in `src/03_news_sentiment_analysis.py` with fresh ones from [rss.app](https://rss.app) before running. Code logic is feed-agnostic.

### SVD Recommender System (Phase 4)
- Filtered to 138 movies rated 100+ times, reducing the 100,836-row dataset to 20,188 ratings
- Utility matrix: 597 users &times; 138 movies, filled with 0 for missing entries
- TruncatedSVD (20 latent components) factorizes the sparse matrix and reconstructs predicted ratings
- Recommendations exclude already-rated movies for the queried user

| Metric | Value |
|---|---|
| Ratings after filter | 20,188 |
| Matrix dimensions | 597 &times; 138 |
| Missing values | 75.50% |
| SVD components | 20 |
| RMSE | 1.04 |

---

## Setup and Usage

```bash
# Clone the repository
git clone https://github.com/aksingh-ops/network-sentiment-recommender-analytics.git
cd network-sentiment-recommender-analytics

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Run phases independently
python src/01_network_analysis_pagerank_hits.py
python src/02_institutional_holders_network.py
python src/03_news_sentiment_analysis.py
python src/04_svd_recommender_system.py
```

Open the fully annotated notebook output:
```
notebooks/MIS546_HW4_Analysis.html
```

---

## Tech Stack

Python &bull; NetworkX &bull; pandas &bull; NumPy &bull; scikit-learn &bull; SciPy &bull; yfinance &bull; TextBlob &bull; NLTK (VADER) &bull; feedparser &bull; Matplotlib

---

## Author

**Akash Singh**  
M.S. Business Analytics &mdash; Iowa State University  
[github.com/aksingh-ops](https://github.com/aksingh-ops)
