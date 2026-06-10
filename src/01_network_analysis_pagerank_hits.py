"""
Phase 1 - Network Analysis: PageRank and HITS
Dataset: sn_ids.csv (289,004 directed edges across 37,700 unique nodes)
Algorithms: PageRank, HITS (Hubs and Authorities)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
data = pd.read_csv('data/sn_ids.csv')
print("Dataset shape:", data.shape)
print(data.head())

# ------------------------------------------------------------------
# Build directed graph
# ------------------------------------------------------------------
G = nx.from_pandas_edgelist(data, 'id_1', 'id_2', create_using=nx.DiGraph())
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ==================================================================
# PART A: PageRank - rank all IDs in decreasing order
# ==================================================================
pagerank_values = nx.pagerank(G)

pagerank_df = pd.DataFrame(list(pagerank_values.items()), columns=['ids', 'PageRank'])
pagerank_df['PageRank'] = pagerank_df['PageRank'].round(3)
pagerank_df_sorted = pagerank_df.sort_values(by='PageRank', ascending=False).reset_index(drop=True)

print("\nTop 10 PageRank nodes:")
print(pagerank_df_sorted.head(10).to_string(index=False))

# ==================================================================
# PART B: Network graph of top-5 PageRank IDs (id_1=blue, id_2=green)
# ==================================================================
top_ids_pr = pagerank_df_sorted['ids'].head(5).tolist()
print("\nTop 5 PageRank IDs:", top_ids_pr)

filtered_pr = data[(data['id_1'].isin(top_ids_pr)) | (data['id_2'].isin(top_ids_pr))]
G_filtered_pr = nx.from_pandas_edgelist(filtered_pr, 'id_1', 'id_2', create_using=nx.DiGraph())

# Node colors: id_1 sources = steelblue, id_2 targets = coral
id1_nodes = set(filtered_pr['id_1'].unique())
id2_nodes = set(filtered_pr['id_2'].unique()) - id1_nodes
node_colors_pr = [
    'steelblue' if n in id1_nodes else 'coral'
    for n in G_filtered_pr.nodes()
]

pos_pr = nx.spring_layout(G_filtered_pr, iterations=20, seed=42)

plt.figure(figsize=(50, 50))
nx.draw_networkx_nodes(G_filtered_pr, pos_pr, node_color=node_colors_pr, node_size=500, alpha=0.85)
nx.draw_networkx_edges(G_filtered_pr, pos_pr, edge_color='gray', width=0.8, alpha=0.5,
                       arrows=True, arrowsize=10)

legend_patches = [
    mpatches.Patch(color='steelblue', label='id_1 (source)'),
    mpatches.Patch(color='coral',     label='id_2 (target)'),
]
plt.legend(handles=legend_patches, fontsize=30, loc='upper left')
plt.title('Network Graph - Top 5 PageRank IDs', fontsize=40, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('reports/network_pagerank_top5.png', dpi=80, bbox_inches='tight')
plt.close()
print("Saved: reports/network_pagerank_top5.png")

# ==================================================================
# PART C: HITS algorithm - hubs and authorities
# ==================================================================
hubs, authorities = nx.hits(G)

hubs_df = pd.DataFrame(list(hubs.items()), columns=['ids', 'Hubs'])
hubs_df['Hubs'] = hubs_df['Hubs'].round(3)
hubs_df_sorted = hubs_df.sort_values(by='Hubs', ascending=False).reset_index(drop=True)

authorities_df = pd.DataFrame(list(authorities.items()), columns=['ids', 'Authorities'])
authorities_df['Authorities'] = authorities_df['Authorities'].round(3)
authorities_df_sorted = authorities_df.sort_values(by='Authorities', ascending=False).reset_index(drop=True)

print("\nTop 5 Hubs:")
print(hubs_df_sorted.head(5).to_string(index=False))
print("\nTop 5 Authorities:")
print(authorities_df_sorted.head(5).to_string(index=False))

# ==================================================================
# PART D: Network graph for top-5 HITS authorities (id_1=blue, id_2=green)
# ==================================================================
top_authorities = authorities_df_sorted['ids'].head(5).tolist()
top_hubs        = hubs_df_sorted['ids'].head(5).tolist()
print("\nTop 5 HITS Authorities:", top_authorities)
print("Top 5 HITS Hubs:", top_hubs)

# Authorities graph
filtered_auth = data[(data['id_1'].isin(top_authorities)) | (data['id_2'].isin(top_authorities))]
G_auth = nx.from_pandas_edgelist(filtered_auth, 'id_1', 'id_2', create_using=nx.DiGraph())

id1_auth = set(filtered_auth['id_1'].unique())
id2_auth = set(filtered_auth['id_2'].unique()) - id1_auth
node_colors_auth = ['steelblue' if n in id1_auth else 'coral' for n in G_auth.nodes()]

pos_auth = nx.spring_layout(G_auth, iterations=50, seed=42)

plt.figure(figsize=(50, 50))
nx.draw_networkx_nodes(G_auth, pos_auth, node_color=node_colors_auth, node_size=500, alpha=0.85)
nx.draw_networkx_edges(G_auth, pos_auth, edge_color='gray', width=0.8, alpha=0.5,
                       arrows=True, arrowsize=10)
plt.legend(handles=legend_patches, fontsize=30, loc='upper left')
plt.title('Network Graph - Top 5 HITS Authorities', fontsize=40, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('reports/network_hits_authorities_top5.png', dpi=80, bbox_inches='tight')
plt.close()
print("Saved: reports/network_hits_authorities_top5.png")

# Hubs graph
filtered_hubs = data[(data['id_1'].isin(top_hubs)) | (data['id_2'].isin(top_hubs))]
G_hubs = nx.from_pandas_edgelist(filtered_hubs, 'id_1', 'id_2', create_using=nx.DiGraph())

id1_hubs = set(filtered_hubs['id_1'].unique())
id2_hubs = set(filtered_hubs['id_2'].unique()) - id1_hubs
node_colors_hubs = ['steelblue' if n in id1_hubs else 'coral' for n in G_hubs.nodes()]

pos_hubs = nx.spring_layout(G_hubs, iterations=50, seed=42)

plt.figure(figsize=(50, 50))
nx.draw_networkx_nodes(G_hubs, pos_hubs, node_color=node_colors_hubs, node_size=500, alpha=0.85)
nx.draw_networkx_edges(G_hubs, pos_hubs, edge_color='gray', width=0.8, alpha=0.5,
                       arrows=True, arrowsize=10)
plt.legend(handles=legend_patches, fontsize=30, loc='upper left')
plt.title('Network Graph - Top 5 HITS Hubs', fontsize=40, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('reports/network_hits_hubs_top5.png', dpi=80, bbox_inches='tight')
plt.close()
print("Saved: reports/network_hits_hubs_top5.png")

# ==================================================================
# Summary
# ==================================================================
print("\n--- Summary ---")
print(f"Top PageRank node: {pagerank_df_sorted.iloc[0]['ids']} (score: {pagerank_df_sorted.iloc[0]['PageRank']})")
print(f"Top Hub node:      {hubs_df_sorted.iloc[0]['ids']} (score: {hubs_df_sorted.iloc[0]['Hubs']})")
print(f"Top Authority node:{authorities_df_sorted.iloc[0]['ids']} (score: {authorities_df_sorted.iloc[0]['Authorities']})")
print("\nPageRank identifies globally influential nodes by propagating link authority.")
print("HITS separates hub nodes (strong outgoing linkers) from authority nodes (well-cited targets).")
