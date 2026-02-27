"""
FraudFlow v0.1.0

engineer.py — Feature engineering with graph topology features.

DVC stage: featurize
Usage: python -m src.features.engineer --config configs/pipeline.yaml

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("fraudflow.features")


def load_raw_elliptic(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw Elliptic CSVs."""
    raw_dir = Path(raw_dir)
    features = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    return features, classes, edges


def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from the edgelist."""
    G = nx.from_pandas_edgelist(edges_df, source="txId1", target="txId2", create_using=nx.DiGraph)
    logger.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def compute_graph_features(
    df: pd.DataFrame,
    G: nx.DiGraph,
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Compute graph topology features and merge onto the feature dataframe.

    Args:
        df: Transaction features with 'txId' column.
        G: NetworkX graph of transactions.
        feature_list: Which graph features to compute.

    Returns:
        DataFrame with new graph feature columns added.
    """
    tx_ids = set(df["txId"].values)
    G_sub = G.subgraph(tx_ids)

    for feat in feature_list:
        logger.info(f"  Computing: {feat}")
        if feat == "degree":
            deg = dict(G_sub.degree())
            df["graph_degree"] = df["txId"].map(deg).fillna(0).astype(int)
            df["graph_in_degree"] = df["txId"].map(dict(G_sub.in_degree())).fillna(0).astype(int)
            df["graph_out_degree"] = df["txId"].map(dict(G_sub.out_degree())).fillna(0).astype(int)

        elif feat == "clustering_coefficient":
            # Undirected clustering on the underlying undirected graph
            G_und = G_sub.to_undirected()
            cc = nx.clustering(G_und)
            df["graph_clustering_coeff"] = df["txId"].map(cc).fillna(0.0)

        elif feat == "pagerank":
            try:
                pr = nx.pagerank(G_sub, max_iter=50, tol=1e-4)
                df["graph_pagerank"] = df["txId"].map(pr).fillna(0.0)
            except nx.PowerIterationFailedConvergence:
                logger.warning("PageRank did not converge — filling zeros")
                df["graph_pagerank"] = 0.0

        elif feat == "betweenness_centrality_approx":
            # Approximate betweenness (sample k nodes)
            n_nodes = G_sub.number_of_nodes()
            k = min(500, max(50, n_nodes // 10))
            bc = nx.betweenness_centrality(G_sub, k=k)
            df["graph_betweenness"] = df["txId"].map(bc).fillna(0.0)

    return df


def compute_neighbor_aggregations(
    df: pd.DataFrame,
    G: nx.DiGraph,
    hops: int = 2,
    agg_functions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate feature statistics from k-hop neighbors.

    Args:
        df: Feature DataFrame with 'txId' column.
        G: Transaction graph.
        hops: Number of hops to aggregate.
        agg_functions: Aggregation functions to apply.

    Returns:
        DataFrame with neighbor aggregation columns.
    """
    if agg_functions is None:
        agg_functions = ["mean", "std", "max"]

    # Use only numeric feature columns (exclude txId, timestep, label)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feature_cols:
        logger.warning("No feat_* columns found — skipping neighbor aggregation")
        return df

    # Build lookup: txId → feature vector
    tx_to_idx = {tx: i for i, tx in enumerate(df["txId"].values)}
    feat_matrix = df[feature_cols].values

    agg_results = {f"neigh_{h}hop_{fn}": [] for h in range(1, hops + 1) for fn in agg_functions}

    for tx_id in df["txId"].values:
        for hop in range(1, hops + 1):
            # Get k-hop neighbors via BFS
            if tx_id in G:
                neighbors = set()
                frontier = {tx_id}
                for _ in range(hop):
                    next_frontier = set()
                    for n in frontier:
                        next_frontier.update(G.predecessors(n))
                        next_frontier.update(G.successors(n))
                    frontier = next_frontier - {tx_id}
                    neighbors.update(frontier)

                neigh_idxs = [tx_to_idx[n] for n in neighbors if n in tx_to_idx]
            else:
                neigh_idxs = []

            if neigh_idxs:
                neigh_feats = feat_matrix[neigh_idxs]
                for fn in agg_functions:
                    if fn == "mean":
                        val = float(np.mean(neigh_feats))
                    elif fn == "std":
                        val = float(np.std(neigh_feats))
                    elif fn == "max":
                        val = float(np.max(neigh_feats))
                    else:
                        val = 0.0
                    agg_results[f"neigh_{hop}hop_{fn}"].append(val)
            else:
                for fn in agg_functions:
                    agg_results[f"neigh_{hop}hop_{fn}"].append(0.0)

    for col_name, values in agg_results.items():
        df[col_name] = values

    return df


def engineer_features(
    raw_dir: Path,
    processed_dir: Path,
    graph_features: list[str],
    neighbor_cfg: dict[str, Any],
) -> dict:
    """
    Run the full feature engineering pipeline.

    Args:
        raw_dir: Directory containing raw CSVs.
        processed_dir: Output directory for features.parquet.
        graph_features: List of graph features to compute.
        neighbor_cfg: Config for neighbor aggregation (hops, functions).

    Returns:
        Dict of feature statistics.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    logger.info("Loading raw data...")
    features_df, classes_df, edges_df = load_raw_elliptic(raw_dir)

    # Name columns: txId, timestep, feat_0 .. feat_164
    n_cols = features_df.shape[1]
    col_names = ["txId", "timestep"] + [f"feat_{i}" for i in range(n_cols - 2)]
    features_df.columns = col_names

    # Merge labels: 1=illicit, 0=licit, -1=unknown
    classes_df.columns = ["txId", "class"]
    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes_df["label"] = classes_df["class"].astype(str).map(label_map)
    df = features_df.merge(classes_df[["txId", "label"]], on="txId", how="left")
    df["label"] = df["label"].fillna(-1).astype(int)

    logger.info(f"Loaded {len(df):,} transactions, {n_cols - 2} raw features")

    # Build graph
    G = build_graph(edges_df)

    # Add graph topology features
    if graph_features:
        logger.info(f"Computing graph features: {graph_features}")
        df = compute_graph_features(df, G, graph_features)

    # Add neighbor aggregations
    if neighbor_cfg.get("hops", 0) > 0:
        logger.info(f"Computing neighbor aggregations ({neighbor_cfg['hops']} hops)...")
        df = compute_neighbor_aggregations(
            df,
            G,
            hops=neighbor_cfg["hops"],
            agg_functions=neighbor_cfg.get("functions", ["mean", "std", "max"]),
        )

    # Save
    out_path = processed_dir / "features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"✓ Saved {out_path} ({len(df):,} rows × {len(df.columns)} cols)")

    # Stats
    n_labelled = (df["label"] != -1).sum()
    n_illicit = (df["label"] == 1).sum()
    n_licit = (df["label"] == 0).sum()

    stats = {
        "n_transactions": len(df),
        "n_features": len(df.columns),
        "n_labelled": int(n_labelled),
        "n_illicit": int(n_illicit),
        "n_licit": int(n_licit),
        "illicit_ratio": float(n_illicit / n_labelled) if n_labelled else 0.0,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "graph_features_added": graph_features,
    }

    with open(processed_dir / "feature_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--config", "-c", required=True, help="Pipeline YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stats = engineer_features(
        raw_dir=Path(cfg["data"]["raw_dir"]),
        processed_dir=Path(cfg["data"]["processed_dir"]),
        graph_features=cfg["features"]["graph_features"],
        neighbor_cfg=cfg["features"]["neighbor_agg"],
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
