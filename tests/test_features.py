"""Tests for src.features.engineer — graph feature engineering."""

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from src.features.engineer import build_graph, compute_graph_features


# ── build_graph ─────────────────────────────────────────────────────


class TestBuildGraph:
    def test_simple_graph(self):
        edges_df = pd.DataFrame({"txId1": [1, 2, 3], "txId2": [2, 3, 1]})
        G = build_graph(edges_df)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3

    def test_empty_graph(self):
        edges_df = pd.DataFrame({"txId1": pd.Series(dtype=int), "txId2": pd.Series(dtype=int)})
        G = build_graph(edges_df)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


# ── compute_graph_features ──────────────────────────────────────────


class TestComputeGraphFeatures:
    @pytest.fixture
    def triangle_data(self):
        """Create a directed triangle graph and matching DataFrame."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        df = pd.DataFrame({
            "txId": [1, 2, 3],
            "feat_0": [0.1, 0.2, 0.3],
            "label": [0, 1, 0],
        })
        return df, G

    def test_degree_features(self, triangle_data):
        df, G = triangle_data
        result = compute_graph_features(df.copy(), G, ["degree"])
        # In a directed triangle, each node has degree=2, in=1, out=1
        assert "graph_degree" in result.columns
        assert "graph_in_degree" in result.columns
        assert "graph_out_degree" in result.columns
        assert list(result["graph_degree"]) == [2, 2, 2]
        assert list(result["graph_in_degree"]) == [1, 1, 1]
        assert list(result["graph_out_degree"]) == [1, 1, 1]

    def test_clustering(self, triangle_data):
        df, G = triangle_data
        result = compute_graph_features(df.copy(), G, ["clustering_coefficient"])
        assert "graph_clustering_coeff" in result.columns
        for val in result["graph_clustering_coeff"]:
            assert isinstance(val, float)

    def test_pagerank(self, triangle_data):
        df, G = triangle_data
        result = compute_graph_features(df.copy(), G, ["pagerank"])
        assert "graph_pagerank" in result.columns
        total = result["graph_pagerank"].sum()
        assert abs(total - 1.0) < 1e-4

    def test_unknown_feature_skipped(self, triangle_data):
        df, G = triangle_data
        result = compute_graph_features(df.copy(), G, ["degree", "nonexistent_metric"])
        # Should still compute degree even if unknown feature is requested
        assert "graph_degree" in result.columns
