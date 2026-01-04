"""Unit tests for retrieval metrics."""

import pytest

from evals.metrics.retrieval import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    """Test recall@k metric."""

    def test_perfect_recall(self):
        """All ground truth items retrieved in top-k."""
        assert recall_at_k([1, 2, 3, 4], [2, 3], k=4) == 1.0

    def test_partial_recall(self):
        """Some ground truth items retrieved."""
        assert recall_at_k([1, 2, 3, 4], [2, 5], k=3) == 0.5

    def test_zero_recall(self):
        """No ground truth items retrieved."""
        assert recall_at_k([1, 2, 3], [4, 5, 6], k=3) == 0.0

    def test_k_larger_than_retrieved(self):
        """k exceeds number of retrieved items."""
        result = recall_at_k([1, 2], [1, 2, 3], k=10)
        assert abs(result - 2 / 3) < 0.001

    def test_empty_retrieved(self):
        """No items retrieved."""
        assert recall_at_k([], [1, 2], k=3) == 0.0

    def test_empty_ground_truth(self):
        """No ground truth items."""
        assert recall_at_k([1, 2, 3], [], k=3) == 0.0

    def test_both_empty(self):
        """Both lists empty."""
        assert recall_at_k([], [], k=3) == 0.0

    def test_k_smaller_than_retrieved(self):
        """Relevant item beyond k is not counted."""
        assert recall_at_k([1, 2, 3, 4], [4], k=2) == 0.0


class TestPrecisionAtK:
    """Test precision@k metric."""

    def test_perfect_precision(self):
        """All retrieved items are relevant."""
        assert precision_at_k([1, 2, 3], [1, 2, 3, 4], k=3) == 1.0

    def test_partial_precision(self):
        """Some retrieved items are relevant."""
        result = precision_at_k([1, 2, 3, 4], [2, 3], k=3)
        assert abs(result - 2 / 3) < 0.001

    def test_zero_precision(self):
        """No retrieved items are relevant."""
        assert precision_at_k([1, 2, 3], [4, 5], k=3) == 0.0

    def test_k_larger_than_retrieved(self):
        """k exceeds retrieved length - uses actual length."""
        assert precision_at_k([1, 2], [1], k=10) == 0.5

    def test_empty_retrieved(self):
        """No items retrieved."""
        assert precision_at_k([], [1, 2], k=3) == 0.0

    def test_empty_ground_truth(self):
        """No ground truth items."""
        assert precision_at_k([1, 2, 3], [], k=3) == 0.0


class TestHitRate:
    """Test hit rate metric."""

    def test_hit_first_position(self):
        """Relevant item at first position."""
        assert hit_rate([1, 2, 3], [1], k=3) == 1.0

    def test_hit_last_position(self):
        """Relevant item at last position within k."""
        assert hit_rate([1, 2, 3], [3], k=3) == 1.0

    def test_no_hit(self):
        """No relevant items in top-k."""
        assert hit_rate([1, 2, 3], [4, 5], k=3) == 0.0

    def test_hit_beyond_k(self):
        """Relevant item exists but beyond k."""
        assert hit_rate([1, 2, 3, 4], [4], k=2) == 0.0

    def test_multiple_hits(self):
        """Multiple relevant items - still returns 1.0."""
        assert hit_rate([1, 2, 3], [1, 2, 3], k=3) == 1.0

    def test_empty_lists(self):
        """Edge cases with empty lists."""
        assert hit_rate([], [1], k=3) == 0.0
        assert hit_rate([1], [], k=3) == 0.0


class TestMRR:
    """Test Mean Reciprocal Rank metric."""

    def test_first_position(self):
        """Relevant item at rank 1."""
        assert mrr([1, 2, 3], [1]) == 1.0

    def test_second_position(self):
        """Relevant item at rank 2."""
        assert mrr([1, 2, 3], [2]) == 0.5

    def test_third_position(self):
        """Relevant item at rank 3."""
        result = mrr([1, 2, 3], [3])
        assert result is not None
        assert abs(result - 1 / 3) < 0.001

    def test_multiple_relevant_uses_first(self):
        """Multiple relevant items - uses first occurrence."""
        assert mrr([1, 2, 3, 4], [3, 2]) == 0.5  # First hit at rank 2

    def test_no_relevant_items(self):
        """No relevant items found - returns None."""
        assert mrr([1, 2, 3], [4, 5]) is None

    def test_empty_retrieved(self):
        """No items retrieved."""
        assert mrr([], [1, 2]) is None

    def test_empty_ground_truth(self):
        """No ground truth items."""
        assert mrr([1, 2, 3], []) is None


class TestNDCGAtK:
    """Test NDCG@k metric."""

    def test_perfect_ranking(self):
        """All relevant items at top positions."""
        assert ndcg_at_k([1, 2, 3, 4], [1, 2], k=4) == 1.0

    def test_imperfect_ranking(self):
        """Relevant items not optimally ranked."""
        # [1, 2, 3] with ground truth [2, 3]
        # DCG = 0 + 1/log2(3) + 1/log2(4) ≈ 0.630 + 0.5 = 1.130
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.630 = 1.630
        # NDCG = 1.130 / 1.630 ≈ 0.693
        result = ndcg_at_k([1, 2, 3], [2, 3], k=3)
        assert 0.69 < result < 0.70

    def test_no_relevant_items(self):
        """No relevant items in top-k."""
        assert ndcg_at_k([1, 2, 3], [4, 5], k=3) == 0.0

    def test_single_relevant_at_top(self):
        """Single relevant item at rank 1."""
        assert ndcg_at_k([1, 2, 3], [1], k=3) == 1.0

    def test_k_larger_than_ground_truth(self):
        """k exceeds number of ground truth items."""
        # Perfect ranking: ground truth [1, 2] at positions 1, 2
        assert ndcg_at_k([1, 2, 3, 4, 5], [1, 2], k=10) == 1.0

    def test_empty_lists(self):
        """Edge cases with empty lists."""
        assert ndcg_at_k([], [1, 2], k=3) == 0.0
        assert ndcg_at_k([1, 2], [], k=3) == 0.0

    def test_relevant_beyond_k(self):
        """Relevant items exist but beyond k."""
        assert ndcg_at_k([1, 2, 3, 4], [4], k=2) == 0.0


class TestComputeRetrievalMetrics:
    """Test main compute function."""

    def test_perfect_retrieval(self):
        """Perfect retrieval scenario."""
        metrics = compute_retrieval_metrics(
            retrieved=[1, 2, 3],
            ground_truth=[1, 2, 3],
            k=3,
        )
        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 1.0
        assert metrics.hit_rate == 1.0
        assert metrics.mrr == 1.0
        assert metrics.ndcg_at_k == 1.0
        assert metrics.k == 3
        assert metrics.num_retrieved == 3
        assert metrics.num_ground_truth == 3
        assert metrics.num_relevant_retrieved == 3

    def test_partial_retrieval(self):
        """Partial overlap scenario."""
        metrics = compute_retrieval_metrics(
            retrieved=[10, 20, 30, 40, 50],
            ground_truth=[20, 30, 60],
            k=3,
        )
        assert abs(metrics.recall_at_k - 2 / 3) < 0.001  # 2 of 3 found
        assert abs(metrics.precision_at_k - 2 / 3) < 0.001  # 2 of 3 retrieved
        assert metrics.hit_rate == 1.0
        assert metrics.mrr == 0.5  # First hit at rank 2
        assert metrics.num_retrieved == 5
        assert metrics.num_ground_truth == 3
        assert metrics.num_relevant_retrieved == 2

    def test_no_overlap(self):
        """No overlap between retrieved and ground truth."""
        metrics = compute_retrieval_metrics(
            retrieved=[1, 2, 3],
            ground_truth=[4, 5, 6],
            k=3,
        )
        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.mrr is None
        assert metrics.ndcg_at_k == 0.0
        assert metrics.num_relevant_retrieved == 0

    def test_k_validation(self):
        """Raises error for invalid k."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            compute_retrieval_metrics([1, 2, 3], [1], k=0)

        with pytest.raises(ValueError, match="k must be >= 1"):
            compute_retrieval_metrics([1, 2, 3], [1], k=-1)

    def test_empty_retrieved(self):
        """Handle empty retrieved list."""
        metrics = compute_retrieval_metrics(
            retrieved=[],
            ground_truth=[1, 2, 3],
            k=5,
        )
        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.mrr is None
        assert metrics.num_retrieved == 0

    def test_empty_ground_truth(self):
        """Handle empty ground truth list."""
        metrics = compute_retrieval_metrics(
            retrieved=[1, 2, 3],
            ground_truth=[],
            k=3,
        )
        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.mrr is None
        assert metrics.num_ground_truth == 0

    def test_dataclass_attributes(self):
        """Verify all dataclass fields are populated."""
        metrics = compute_retrieval_metrics(
            retrieved=[1, 2, 3, 4, 5],
            ground_truth=[2, 4],
            k=3,
        )

        # Check all attributes exist and have correct types
        assert isinstance(metrics.recall_at_k, float)
        assert isinstance(metrics.precision_at_k, float)
        assert isinstance(metrics.hit_rate, float)
        assert metrics.mrr is None or isinstance(metrics.mrr, float)
        assert isinstance(metrics.ndcg_at_k, float)
        assert isinstance(metrics.k, int)
        assert isinstance(metrics.num_retrieved, int)
        assert isinstance(metrics.num_ground_truth, int)
        assert isinstance(metrics.num_relevant_retrieved, int)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_duplicate_chunk_ids_in_retrieved(self):
        """Handle duplicates in retrieved list."""
        metrics = compute_retrieval_metrics(
            retrieved=[1, 1, 2, 2, 3],
            ground_truth=[1, 2],
            k=5,
        )
        # Set conversion handles duplicates in ground truth check
        assert metrics.num_relevant_retrieved == 2

    def test_very_large_k(self):
        """k much larger than retrieved list."""
        metrics = compute_retrieval_metrics(
            retrieved=[1, 2],
            ground_truth=[1],
            k=1000,
        )
        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 0.5

    def test_k_equals_one(self):
        """Minimum valid k value."""
        metrics = compute_retrieval_metrics(
            retrieved=[1, 2, 3],
            ground_truth=[2],
            k=1,
        )
        assert metrics.recall_at_k == 0.0  # Item 2 not in top-1
        assert metrics.precision_at_k == 0.0
        assert metrics.mrr == 0.5  # MRR looks at full list
