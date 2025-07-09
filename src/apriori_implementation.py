import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import time
from typing import List, Set, FrozenSet, Dict, Tuple, Union


def calculate_support(itemset: FrozenSet[str], transactions: List[List[str]],
                      cache: Dict[FrozenSet[str], float] = None) -> float:
    """
    Calculate support for an itemset with optional caching for performance

    Support = (Number of transactions containing itemset) / (Total number of transactions)

    Args:
        itemset (frozenset): Set of items to check
        transactions (list): List of transactions (lists of items)
        cache (dict, optional): Cache to store previously calculated supports

    Returns:
        float: Support value between 0 and 1

    Examples:
        >>> transactions = [['milk', 'bread'], ['milk', 'eggs'], ['bread', 'butter']]
        >>> calculate_support(frozenset(['milk']), transactions)
        0.6667
    """
    if not transactions:
        return 0.0

    if not itemset:
        return 0.0

    # Check the cache first if provided
    if cache is not None and itemset in cache:
        return cache[itemset]

    # Count transactions containing the itemset
    count = 0
    for transaction in transactions:
        transaction_set = set(transaction)
        if itemset.issubset(transaction_set):
            count += 1

    support = count / len(transactions)

    # Store in a cache if provided
    if cache is not None:
        cache[itemset] = support

    return support


def calculate_support_batch(itemsets: List[FrozenSet[str]],
                            transactions: List[List[str]]) -> Dict[FrozenSet[str], float]:
    """
    Calculate support for multiple itemsets efficiently

    Args:
        itemsets (list): List of itemsets to calculate support for
        transactions (list): List of transactions

    Returns:
        dict: Dictionary mapping itemsets to their support values
    """
    if not transactions:
        return {itemset: 0.0 for itemset in itemsets}

    support_dict = {}
    total_transactions = len(transactions)

    # Convert transactions to sets for faster subset checking
    transaction_sets = [set(transaction) for transaction in transactions]

    for itemset in itemsets:
        if not itemset:
            support_dict[itemset] = 0.0
            continue

        count = sum(1 for transaction_set in transaction_sets
                    if itemset.issubset(transaction_set))
        support_dict[itemset] = count / total_transactions

    return support_dict


def get_unique_items(transactions: List[List[str]]) -> Set[str]:
    """
    Extract all unique items from transactions

    Args:
        transactions (list): List of transactions

    Returns:
        set: Set of all unique items
    """
    unique_items = set()
    for transaction in transactions:
        unique_items.update(transaction)
    return unique_items


def validate_transactions(transactions: List[List[str]]) -> Tuple[bool, str]:
    """
    Validate transaction data format and content

    Args:
        transactions (list): List of transactions to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(transactions, list):
        return False, "Transactions must be a list"

    if not transactions:
        return False, "Transactions list is empty"

    for i, transaction in enumerate(transactions):
        if not isinstance(transaction, list):
            return False, f"Transaction {i} is not a list"

        for j, item in enumerate(transaction):
            if not isinstance(item, str):
                return False, f"Item {j} in transaction {i} is not a string"
            if not item.strip():
                return False, f"Empty item found in transaction {i}"

    return True, "Valid"


def print_support_analysis(itemsets_with_support: Dict[FrozenSet[str], float],
                           min_support: float = 0.0, top_n: int = 10):
    """
    Print detailed analysis of itemset supports

    Args:
        itemsets_with_support (dict): Dictionary of itemsets and their supports
        min_support (float): Minimum support threshold for filtering
        top_n (int): Number of top itemsets to display
    """
    # Filter by minimum support
    filtered_itemsets = {itemset: support for itemset, support in itemsets_with_support.items()
                         if support >= min_support}

    if not filtered_itemsets:
        print(f"No itemsets found with support >= {min_support}")
        return

    # Sort by support (descending)
    sorted_itemsets = sorted(filtered_itemsets.items(), key=lambda x: x[1], reverse=True)

    print(f"\n=== SUPPORT ANALYSIS ===")
    print(f"Total itemsets: {len(itemsets_with_support)}")
    print(f"Itemsets with support >= {min_support}: {len(filtered_itemsets)}")
    print(f"\nTop {min(top_n, len(sorted_itemsets))} itemsets by support:")
    print("-" * 60)

    for i, (itemset, support) in enumerate(sorted_itemsets[:top_n]):
        items_str = ', '.join(sorted(list(itemset)))
        print(f"{i + 1:2d}. {{{items_str:<40}}} | Support: {support:.4f}")


def generate_candidates(frequent_itemsets: Set[FrozenSet[str]], k: int) -> Set[FrozenSet[str]]:
    """
    Generate candidate itemsets of size k from frequent itemsets of size k-1
    Uses the join step of the Apriori algorithm

    Args:
        frequent_itemsets (set): Set of frequent itemsets of size k-1
        k (int): Size of candidates to generate

    Returns:
        set: Set of candidate itemsets of size k

    Example:
        >>> frequent_1 = {frozenset(['A']), frozenset(['B']), frozenset(['C'])}
        >>> generate_candidates(frequent_1, 2)
        {frozenset(['A', 'B']), frozenset(['A', 'C']), frozenset(['B', 'C'])}
    """
    if k <= 1:
        return set()

    if not frequent_itemsets:
        return set()

    candidates = set()
    frequent_list = list(frequent_itemsets)

    # Generate candidates by joining frequent itemsets
    for i in range(len(frequent_list)):
        for j in range(i + 1, len(frequent_list)):
            # Union of two frequent itemsets
            candidate = frequent_list[i].union(frequent_list[j])

            # Only keep if the union has exactly k items
            if len(candidate) == k:
                candidates.add(candidate)

    return candidates


def generate_candidates_optimized(frequent_itemsets: Set[FrozenSet[str]], k: int) -> Set[FrozenSet[str]]:
    """
    Optimized candidate generation using lexicographic ordering
    More efficient for larger itemsets

    Args:
        frequent_itemsets (set): Set of frequent itemsets of size k-1
        k (int): Size of candidates to generate

    Returns:
        set: Set of candidate itemsets of size k
    """
    if k <= 1:
        return set()

    if not frequent_itemsets:
        return set()

    candidates = set()

    # Convert to sorted lists for lexicographic comparison
    frequent_sorted = [sorted(list(itemset)) for itemset in frequent_itemsets]
    frequent_sorted.sort()

    for i in range(len(frequent_sorted)):
        for j in range(i + 1, len(frequent_sorted)):
            itemset1 = frequent_sorted[i]
            itemset2 = frequent_sorted[j]

            # Check if the first k-2 items are the same (lexicographic join condition)
            if itemset1[:-1] == itemset2[:-1]:
                # Create a candidate by combining the two itemsets
                candidate = frozenset(itemset1 + [itemset2[-1]])
                candidates.add(candidate)

    return candidates


def prune_candidates(candidates: Set[FrozenSet[str]],
                     frequent_itemsets: Set[FrozenSet[str]]) -> Set[FrozenSet[str]]:
    """
    Prune candidates using the Apriori principle
    If any (k-1)-subset of a candidate is not frequent, remove the candidate

    Args:
        candidates (set): Set of candidate itemsets
        frequent_itemsets (set): Set of frequent (k-1)-itemsets

    Returns:
        set: The pruned set of candidates
    """
    pruned_candidates = set()

    for candidate in candidates:
        # Generate all (k-1)-subsets of the candidate
        items = list(candidate)
        k = len(items)

        if k <= 1:
            pruned_candidates.add(candidate)
            continue

        # Check if all (k-1)-subsets are frequent
        all_subsets_frequent = True
        for i in range(k):
            subset = frozenset(items[:i] + items[i + 1:])
            if subset not in frequent_itemsets:
                all_subsets_frequent = False
                break

        if all_subsets_frequent:
            pruned_candidates.add(candidate)

    return pruned_candidates


def print_candidate_analysis(candidates: Set[FrozenSet[str]], k: int,
                             pruned_count: int = None):
    """
    Print analysis of the candidate generation step

    Args:
        candidates (set): Generated candidates
        k (int): Size of candidates
        pruned_count (int, optional): Number of candidates pruned
    """
    print(f"\n=== CANDIDATE GENERATION (k={k}) ===")
    print(f"Generated candidates: {len(candidates)}")

    if pruned_count is not None:
        print(f"Candidates after pruning: {len(candidates)}")
        print(f"Candidates pruned: {pruned_count}")

    if candidates and len(candidates) <= 10:
        print("Candidates:")
        for i, candidate in enumerate(sorted(candidates, key=lambda x: sorted(list(x)))):
            items_str = ', '.join(sorted(list(candidate)))
            print(f"  {i + 1}. {{{items_str}}}")
    elif candidates:
        print(f"Sample of first 5 candidates:")
        sample_candidates = list(candidates)[:5]
        for i, candidate in enumerate(sample_candidates):
            items_str = ', '.join(sorted(list(candidate)))
            print(f"  {i + 1}. {{{items_str}}}")


def find_frequent_itemsets(transactions: List[List[str]],
                           min_support: float,
                           max_itemset_size: int = None,
                           use_pruning: bool = True,
                           verbose: bool = True) -> Dict[int, Set[FrozenSet[str]]]:
    """
    Find all frequent itemsets using the Apriori algorithm

    Args:
        transactions (list): List of transactions
        min_support (float): Minimum support threshold (0 to 1)
        max_itemset_size (int, optional): Maximum size of itemsets to generate
        use_pruning (bool): Whether to use candidate pruning
        verbose (bool): Whether to print progress information

    Returns:
        dict: Dictionary mapping itemset size to the set of frequent itemsets

    Algorithm Steps:
        1. Find frequent 1-itemsets
        2. For k = 2, 3, 4, ... until no more frequent itemsets:
           a. Generate the candidate k-itemsets from frequent (k-1)-itemsets
           b. Optionally prune candidates using Apriori principle
           c. Test each candidate for minimum support
           d. Keep candidates that meet the minimum support threshold
        3. Return all frequent itemsets
    """
    start_time = time.time()

    # Validate input
    is_valid, error_msg = validate_transactions(transactions)
    if not is_valid:
        raise ValueError(f"Invalid transactions: {error_msg}")

    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be between 0 and 1")

    if verbose:
        print("=" * 60)
        print("           APRIORI ALGORITHM EXECUTION")
        print("=" * 60)
        print(f"Transactions: {len(transactions)}")
        print(f"Minimum support: {min_support} ({min_support * len(transactions):.1f} transactions)")
        print(f"Maximum itemset size: {max_itemset_size or 'unlimited'}")
        print(f"Use pruning: {use_pruning}")
        print("-" * 60)

    frequent_itemsets = {}
    support_cache = {}

    # Step 1: Find frequent 1-itemsets
    if verbose:
        print("STEP 1: Finding frequent 1-itemsets...")

    # Get all unique items
    all_items = get_unique_items(transactions)
    if verbose:
        print(f"  Total unique items: {len(all_items)}")

    # Test each item for frequency
    frequent_1 = set()
    item_supports = {}

    for item in all_items:
        itemset = frozenset([item])
        support = calculate_support(itemset, transactions, support_cache)
        item_supports[itemset] = support

        if support >= min_support:
            frequent_1.add(itemset)

    frequent_itemsets[1] = frequent_1

    if verbose:
        print(f"  Frequent 1-itemsets found: {len(frequent_1)}")
        print(f"  Items eliminated: {len(all_items) - len(frequent_1)}")

    # Step 2: Generate frequent k-itemsets for k > 1
    k = 2
    while frequent_itemsets[k - 1] and (max_itemset_size is None or k <= max_itemset_size):
        if verbose:
            print(f"\nSTEP {k}: Finding frequent {k}-itemsets...")

        step_start_time = time.time()

        # Generate candidates
        candidates = generate_candidates_optimized(frequent_itemsets[k - 1], k)

        if verbose:
            print(f"  Initial candidates generated: {len(candidates)}")

        # Prune candidates using Apriori principle
        if use_pruning and k > 2:
            initial_candidate_count = len(candidates)
            candidates = prune_candidates(candidates, frequent_itemsets[k - 1])
            pruned_count = initial_candidate_count - len(candidates)

            if verbose:
                print(f"  Candidates after pruning: {len(candidates)} (pruned: {pruned_count})")

        # Test candidates for frequency
        frequent_k = set()
        candidate_supports = calculate_support_batch(list(candidates), transactions)

        for candidate, support in candidate_supports.items():
            if support >= min_support:
                frequent_k.add(candidate)

        frequent_itemsets[k] = frequent_k

        step_time = time.time() - step_start_time
        if verbose:
            print(f"  Frequent {k}-itemsets found: {len(frequent_k)}")
            print(f"  Step execution time: {step_time:.2f} seconds")

        # Stop if no frequent itemsets found
        if not frequent_k:
            if verbose:
                print(f"  No frequent {k}-itemsets found. Algorithm terminated.")
            break

        k += 1

    total_time = time.time() - start_time

    if verbose:
        print("-" * 60)
        print("ALGORITHM COMPLETED")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Total frequent itemsets found: {sum(len(itemsets) for itemsets in frequent_itemsets.values())}")
        print("=" * 60)

    return frequent_itemsets


def analyze_frequent_itemsets(frequent_itemsets: Dict[int, Set[FrozenSet[str]]],
                              transactions: List[List[str]],
                              detailed: bool = True):
    """
    Analyze and summarize frequent itemsets results

    Args:
        frequent_itemsets (dict): Dictionary of frequent itemsets by size
        transactions (list): Original transactions for support calculation
        detailed (bool): Whether to show detailed analysis
    """
    print("\n" + "=" * 60)
    print("           FREQUENT ITEMSETS ANALYSIS")
    print("=" * 60)

    total_itemsets = sum(len(itemsets) for itemsets in frequent_itemsets.values())
    print(f"Total frequent itemsets: {total_itemsets}")

    print(f"\nFrequent itemsets by size:")
    for k in sorted(frequent_itemsets.keys()):
        count = len(frequent_itemsets[k])
        print(f"  {k}-itemsets: {count}")

    if detailed:
        print(f"\nDetailed analysis by size:")
        for k in sorted(frequent_itemsets.keys()):
            if frequent_itemsets[k]:
                print(f"\n--- {k}-ITEMSETS ---")

                # Calculate supports for this level
                itemsets_with_support = {}
                for itemset in frequent_itemsets[k]:
                    support = calculate_support(itemset, transactions)
                    itemsets_with_support[itemset] = support

                # Sort by support
                sorted_itemsets = sorted(itemsets_with_support.items(),
                                         key=lambda x: x[1], reverse=True)

                # Show the top 10 or all if fewer
                display_count = min(10, len(sorted_itemsets))
                print(f"Top {display_count} by support:")

                for i, (itemset, support) in enumerate(sorted_itemsets[:display_count]):
                    items_str = ', '.join(sorted(list(itemset)))
                    print(f"  {i + 1:2d}. {{{items_str:<30}}} | Support: {support:.4f}")

                if len(sorted_itemsets) > 10:
                    print(f"  ... and {len(sorted_itemsets) - 10} more")

    print("=" * 60)
