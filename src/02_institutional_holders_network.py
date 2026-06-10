"""
Phase 2 - Institutional Holders Network Analysis
Source: Yahoo Finance API (yfinance)
Tickers: Top 20 large-cap US equities
Graph: Holder (source) -> Ticker (target), edge width = normalized holding amount
Node size for tickers scaled by degree; holders uniform size
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf

# ------------------------------------------------------------------
# Ticker list
# ------------------------------------------------------------------
TOP20_TICKERS = [
    "AAPL", "AMZN", "MSFT", "GOOG", "GOOGL",
    "META", "TSLA", "NVDA", "JPM",  "JNJ",
    "V",    "PG",   "UNH",  "HD",   "MA",
    "BAC",  "DIS",  "PYPL", "NFLX", "ADBE",
]

# ------------------------------------------------------------------
# Fetch institutional holders via Yahoo Finance
# ------------------------------------------------------------------
def get_institutional_holders(ticker: str) -> pd.DataFrame | None:
    try:
        obj     = yf.Ticker(ticker)
        holders = obj.institutional_holders
        if holders is None:
            return None
        holders = holders.reset_index()
        if 'Holder' in holders.columns and 'Value' in holders.columns:
            return holders[['Holder', 'Value']].head(20)
        print(f"Unexpected columns for {ticker}: {holders.columns.tolist()}")
        return None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

records = []
for ticker in TOP20_TICKERS:
    df = get_institutional_holders(ticker)
    if df is not None:
        for _, row in df.iterrows():
            records.append({
                'Ticker Symbol':             ticker,
                'Top 20 Holders':            row['Holder'],
                'Holdings in Dollar Amount': row['Value'],
            })

holders_data = pd.DataFrame(records)
print(f"Total records fetched: {len(holders_data)}")
print(holders_data.head())

# ------------------------------------------------------------------
# Build directed graph: holder -> ticker
# ------------------------------------------------------------------
G = nx.DiGraph()

for _, row in holders_data.iterrows():
    holder = row['Top 20 Holders']
    ticker = row['Ticker Symbol']
    amount = row['Holdings in Dollar Amount']
    G.add_node(holder, node_type='holder')
    G.add_node(ticker, node_type='ticker')
    G.add_edge(holder, ticker, weight=amount)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ------------------------------------------------------------------
# Part C: Base network graph - holder=blue, ticker=green
#         Edge width = normalized holding; ticker size = degree
# ------------------------------------------------------------------
def build_graph(G: nx.DiGraph, title: str, filename: str,
                k: float = 0.15, iterations: int = 20,
                edge_alpha: float = 0.5, font_size: int = 9):

    pos = nx.spring_layout(G, k=k, iterations=iterations, seed=42)

    # Node colors
    node_colors = [
        'steelblue' if G.nodes[n]['node_type'] == 'holder' else 'mediumseagreen'
        for n in G.nodes()
    ]

    # Node sizes: holders uniform; tickers scaled by degree
    holder_base = 3000
    ticker_scale = 100
    node_sizes = [
        holder_base if G.nodes[n]['node_type'] == 'holder'
        else max(300, ticker_scale * G.degree(n))
        for n in G.nodes()
    ]

    # Edge widths: normalize holding amounts to 0.5-5 range
    weights   = [G[u][v]['weight'] for u, v in G.edges()]
    w_min, w_max = min(weights), max(weights)
    norm_widths  = [0.5 + 4.5 * (w - w_min) / (w_max - w_min + 1) for w in weights]

    fig, ax = plt.subplots(figsize=(20, 15))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold',
                            bbox=dict(facecolor='white', alpha=0.5), ax=ax)
    nx.draw_networkx_edges(G, pos, width=norm_widths, edge_color='gray',
                           alpha=edge_alpha, arrows=True, arrowsize=12, ax=ax)

    legend_patches = [
        mpatches.Patch(color='steelblue',      label='Institutional Holder (source)'),
        mpatches.Patch(color='mediumseagreen', label='Ticker Symbol (target)'),
    ]
    ax.legend(handles=legend_patches, fontsize=11, loc='upper left')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'reports/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: reports/{filename}")

# Base graph (Part C)
build_graph(
    G,
    title='Institutional Holders and Ticker Symbols - Network Graph',
    filename='holder_ticker_network_base.png',
)

# ------------------------------------------------------------------
# Part D: Enhanced graph - kamada_kawai layout, edge color gradient,
#         increased font contrast, top holders highlighted in orange
# ------------------------------------------------------------------
top_holders = holders_data.groupby('Top 20 Holders')['Holdings in Dollar Amount'] \
                           .sum().nlargest(5).index.tolist()

pos_kk = nx.kamada_kawai_layout(G)

node_colors_enhanced = []
for n in G.nodes():
    if G.nodes[n]['node_type'] == 'ticker':
        node_colors_enhanced.append('mediumseagreen')
    elif n in top_holders:
        node_colors_enhanced.append('darkorange')
    else:
        node_colors_enhanced.append('steelblue')

node_sizes_enh = [
    4000 if G.nodes[n]['node_type'] == 'holder' and n in top_holders
    else 3000 if G.nodes[n]['node_type'] == 'holder'
    else max(400, 120 * G.degree(n))
    for n in G.nodes()
]

weights_enh  = [G[u][v]['weight'] for u, v in G.edges()]
w_min, w_max = min(weights_enh), max(weights_enh)
norm_w_enh   = [0.5 + 4.5 * (w - w_min) / (w_max - w_min + 1) for w in weights_enh]

fig, ax = plt.subplots(figsize=(22, 17))
nx.draw_networkx_nodes(G, pos_kk, node_color=node_colors_enhanced,
                       node_size=node_sizes_enh, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G, pos_kk, font_size=8, font_weight='bold',
                        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7), ax=ax)
nx.draw_networkx_edges(G, pos_kk, width=norm_w_enh, edge_color='slategray',
                       alpha=0.45, arrows=True, arrowsize=14, ax=ax)

legend_patches = [
    mpatches.Patch(color='darkorange',     label='Top 5 Holders by Total Holdings'),
    mpatches.Patch(color='steelblue',      label='Other Institutional Holders'),
    mpatches.Patch(color='mediumseagreen', label='Ticker Symbol (size = degree)'),
]
ax.legend(handles=legend_patches, fontsize=11, loc='upper left')
ax.set_title(
    'Enhanced Network: Institutional Holders vs Ticker Symbols\n'
    '(Kamada-Kawai layout | Edge width = holding amount | Orange = top 5 holders)',
    fontsize=14, fontweight='bold'
)
ax.axis('off')
plt.tight_layout()
plt.savefig('reports/holder_ticker_network_enhanced.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/holder_ticker_network_enhanced.png")

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
summary = holders_data.groupby('Ticker Symbol')['Holdings in Dollar Amount'] \
                       .sum().sort_values(ascending=False).reset_index()
summary.columns = ['Ticker', 'Total Holdings ($)']
print("\nTotal institutional holdings by ticker (top 10):")
print(summary.head(10).to_string(index=False))
