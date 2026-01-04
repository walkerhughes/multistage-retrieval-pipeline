"""Retrieval evaluation metrics.

Provides standard IR metrics for evaluating retrieval quality:
- Recall@k: Fraction of ground truth items retrieved in top-k
- Precision@k: Fraction of retrieved items (top-k) that are ground truth
- Hit Rate: Binary indicator if ANY ground truth item is in top-k
- MRR (Mean Reciprocal Rank): 1 / rank of first ground truth item
- NDCG@k (Normalized Discounted Cumulative Gain): Ranking quality metric

All metrics handle edge cases gracefully:
- Empty retrieved/ground truth lists → 0.0 (except MRR which returns None)
- k > len(retrieved) → uses actual retrieved length
- No overlap between lists → 0.0 scores
"""

import math
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Complete set of retrieval evaluation metrics.

    All scores are in [0.0, 1.0] range except MRR which can be None.

    Attributes:
        recall_at_k: Fraction of ground truth items found in top-k results
        precision_at_k: Fraction of top-k results that are ground truth
        hit_rate: 1.0 if any ground truth item in top-k, else 0.0
        mrr: 1/rank of first ground truth item (None if no matches)
        ndcg_at_k: Normalized DCG score accounting for ranking quality
        k: The k value used for @k metrics
        num_retrieved: Actual number of items retrieved
        num_ground_truth: Number of ground truth items
        num_relevant_retrieved: Number of ground truth items found in top-k
    """

    recall_at_k: float
    precision_at_k: float
    hit_rate: float
    mrr: float | None
    ndcg_at_k: float
    k: int
    num_retrieved: int
    num_ground_truth: int
    num_relevant_retrieved: int


def recall_at_k(
    retrieved: list[int],
    ground_truth: list[int],
    k: int,
) -> float:
    """Calculate Recall@k: fraction of ground truth items in top-k results.

    Recall@k = |ground_truth ∩ retrieved[:k]| / |ground_truth|

    Args:
        retrieved: Ordered list of retrieved chunk IDs (rank order)
        ground_truth: List of ground truth chunk IDs (unordered)
        k: Number of top results to consider

    Returns:
        Recall score in [0.0, 1.0]. Returns 0.0 if ground_truth is empty.

    Examples:
        >>> recall_at_k([1, 2, 3, 4], [2, 5], k=3)
        0.5  # Found 1 of 2 ground truth items
        >>> recall_at_k([1, 2, 3], [4, 5], k=3)
        0.0  # No overlap
    """
    if not ground_truth:
        return 0.0

    if not retrieved:
        return 0.0

    top_k = retrieved[:k]
    ground_truth_set = set(ground_truth)

    relevant_retrieved = sum(1 for item in top_k if item in ground_truth_set)

    return relevant_retrieved / len(ground_truth)


def precision_at_k(
    retrieved: list[int],
    ground_truth: list[int],
    k: int,
) -> float:
    """Calculate Precision@k: fraction of top-k results that are ground truth.

    Precision@k = |ground_truth ∩ retrieved[:k]| / min(k, |retrieved|)

    Args:
        retrieved: Ordered list of retrieved chunk IDs (rank order)
        ground_truth: List of ground truth chunk IDs (unordered)
        k: Number of top results to consider

    Returns:
        Precision score in [0.0, 1.0]. Returns 0.0 if retrieved is empty.

    Examples:
        >>> precision_at_k([1, 2, 3, 4], [2, 3], k=3)
        0.667  # 2 of 3 retrieved are relevant
    """
    if not retrieved:
        return 0.0

    if not ground_truth:
        return 0.0

    top_k = retrieved[:k]
    ground_truth_set = set(ground_truth)

    relevant_retrieved = sum(1 for item in top_k if item in ground_truth_set)

    return relevant_retrieved / len(top_k)


def hit_rate(
    retrieved: list[int],
    ground_truth: list[int],
    k: int,
) -> float:
    """Calculate Hit Rate: binary indicator if ANY ground truth item in top-k.

    Hit Rate = 1.0 if |ground_truth ∩ retrieved[:k]| > 0, else 0.0

    Args:
        retrieved: Ordered list of retrieved chunk IDs (rank order)
        ground_truth: List of ground truth chunk IDs (unordered)
        k: Number of top results to consider

    Returns:
        1.0 if at least one ground truth item found, else 0.0

    Examples:
        >>> hit_rate([1, 2, 3], [2], k=3)
        1.0
        >>> hit_rate([1, 2, 3], [4, 5], k=3)
        0.0
    """
    if not retrieved or not ground_truth:
        return 0.0

    top_k = retrieved[:k]
    ground_truth_set = set(ground_truth)

    return 1.0 if any(item in ground_truth_set for item in top_k) else 0.0


def mrr(
    retrieved: list[int],
    ground_truth: list[int],
) -> float | None:
    """Calculate Mean Reciprocal Rank: 1 / rank of first ground truth item.

    MRR = 1 / rank(first_relevant_item)

    Note: Rank is 1-indexed (first position = rank 1).
    This is a "single query MRR" - for multiple queries, average the results.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (rank order)
        ground_truth: List of ground truth chunk IDs (unordered)

    Returns:
        Reciprocal rank in (0.0, 1.0] if match found, else None.
        Returns None (not 0.0) to distinguish "no match" from "match at infinity".

    Examples:
        >>> mrr([1, 2, 3, 4], [2, 5])
        0.5  # First match at rank 2
        >>> mrr([1, 2, 3], [4, 5])
        None  # No match
    """
    if not retrieved or not ground_truth:
        return None

    ground_truth_set = set(ground_truth)

    for rank, item in enumerate(retrieved, start=1):
        if item in ground_truth_set:
            return 1.0 / rank

    return None


def ndcg_at_k(
    retrieved: list[int],
    ground_truth: list[int],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain@k.

    NDCG measures ranking quality by giving higher weight to relevant items
    appearing earlier in the ranking. Binary relevance: item is either
    relevant (1) or not (0).

    DCG@k = Σ(i=1 to k) relevance[i] / log2(i + 1)
    IDCG@k = DCG for ideal ranking (all relevant items first)
    NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved: Ordered list of retrieved chunk IDs (rank order)
        ground_truth: List of ground truth chunk IDs (unordered)
        k: Number of top results to consider

    Returns:
        NDCG score in [0.0, 1.0]. Returns 0.0 if no ground truth items or
        no retrieved items. Returns 1.0 for perfect ranking.

    Examples:
        >>> ndcg_at_k([1, 2, 3], [1, 2], k=3)
        1.0  # Perfect ranking
        >>> ndcg_at_k([1, 2, 3], [4, 5], k=3)
        0.0  # No relevant items
    """
    if not retrieved or not ground_truth:
        return 0.0

    ground_truth_set = set(ground_truth)
    top_k = retrieved[:k]

    # Calculate DCG
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in ground_truth_set:
            dcg += 1.0 / math.log2(rank + 1)

    if dcg == 0.0:
        return 0.0

    # Calculate IDCG (ideal DCG - all relevant items ranked first)
    num_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_retrieval_metrics(
    retrieved: list[int],
    ground_truth: list[int],
    k: int,
) -> RetrievalMetrics:
    """Compute all retrieval metrics for a single query.

    This is the main entry point for metric computation. It calculates all
    standard IR metrics and returns them in a single dataclass.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (rank order matters)
        ground_truth: List of ground truth chunk IDs (order doesn't matter)
        k: Number of top results to consider for @k metrics

    Returns:
        RetrievalMetrics dataclass containing all computed metrics

    Raises:
        ValueError: If k < 1

    Examples:
        >>> metrics = compute_retrieval_metrics(
        ...     retrieved=[10, 20, 30, 40, 50],
        ...     ground_truth=[20, 30, 60],
        ...     k=3
        ... )
        >>> metrics.recall_at_k
        0.667  # Found 2 of 3 ground truth items
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    ground_truth_set = set(ground_truth)
    top_k_set = set(retrieved[:k])
    num_relevant_retrieved = len(ground_truth_set & top_k_set)

    recall = recall_at_k(retrieved, ground_truth, k)
    precision = precision_at_k(retrieved, ground_truth, k)
    hit = hit_rate(retrieved, ground_truth, k)
    reciprocal_rank = mrr(retrieved, ground_truth)
    ndcg = ndcg_at_k(retrieved, ground_truth, k)

    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        hit_rate=hit,
        mrr=reciprocal_rank,
        ndcg_at_k=ndcg,
        k=k,
        num_retrieved=len(retrieved),
        num_ground_truth=len(ground_truth),
        num_relevant_retrieved=num_relevant_retrieved,
    )
