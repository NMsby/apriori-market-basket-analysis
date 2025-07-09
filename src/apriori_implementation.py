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


def generate_association_rules(frequent_itemsets: Dict[int, Set[FrozenSet[str]]],
                               transactions: List[List[str]],
                               min_confidence: float) -> List[Dict]:
    """
    Generate association rules from frequent itemsets

    Rule: X → Y where X ∪ Y is a frequent itemset

    Args:
        frequent_itemsets (dict): Dictionary of frequent itemsets by size
        transactions (list): List of transactions for support calculation
        min_confidence (float): Minimum confidence threshold (0 to 1)

    Returns:
        list: List of association rules with metrics

    For each frequent itemset of size ≥ 2:
    1. Generate all possible ways to split it into antecedent → consequent
    2. Calculate confidence = support(antecedent ∪ consequent) / support(antecedent)
    3. Keep rules with confidence ≥ min_confidence
    4. Calculate lift = confidence / support(consequent)
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")

    rules = []
    support_cache = {}

    # Generate rules from itemsets of size 2 and above
    for k in range(2, len(frequent_itemsets) + 1):
        if k not in frequent_itemsets:
            continue

        for itemset in frequent_itemsets[k]:
            items = list(itemset)

            # Generate all possible antecedent-consequent pairs
            for i in range(1, len(items)):  # i is the size of antecedent
                for antecedent_items in combinations(items, i):
                    antecedent = frozenset(antecedent_items)
                    consequent = itemset - antecedent

                    # Calculate metrics
                    support_itemset = calculate_support(itemset, transactions, support_cache)
                    support_antecedent = calculate_support(antecedent, transactions, support_cache)

                    if support_antecedent > 0:
                        confidence = support_itemset / support_antecedent

                        if confidence >= min_confidence:
                            support_consequent = calculate_support(consequent, transactions, support_cache)
                            lift = confidence / support_consequent if support_consequent > 0 else 0

                            rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support_itemset,
                                'confidence': confidence,
                                'lift': lift,
                                'antecedent_support': support_antecedent,
                                'consequent_support': support_consequent
                            })

    return rules


def calculate_rule_metrics(antecedent: FrozenSet[str],
                           consequent: FrozenSet[str],
                           transactions: List[List[str]]) -> Dict[str, float]:
    """
    Calculate all metrics for a single association rule

    Args:
        antecedent (frozenset): Antecedent itemset
        consequent (frozenset): Consequent itemset
        transactions (list): List of transactions

    Returns:
        dict: Dictionary with support, confidence, lift, and other metrics
    """
    itemset = antecedent.union(consequent)

    support_itemset = calculate_support(itemset, transactions)
    support_antecedent = calculate_support(antecedent, transactions)
    support_consequent = calculate_support(consequent, transactions)

    confidence = support_itemset / support_antecedent if support_antecedent > 0 else 0
    lift = confidence / support_consequent if support_consequent > 0 else 0

    # Additional metrics
    conviction = (1 - support_consequent) / (1 - confidence) if confidence < 1 else float('inf')
    leverage = support_itemset - (support_antecedent * support_consequent)

    return {
        'support': support_itemset,
        'confidence': confidence,
        'lift': lift,
        'antecedent_support': support_antecedent,
        'consequent_support': support_consequent,
        'conviction': conviction,
        'leverage': leverage
    }


def filter_rules_by_metrics(rules: List[Dict],
                            min_support: float = 0.0,
                            min_confidence: float = 0.0,
                            min_lift: float = 0.0,
                            max_antecedent_size: int = None) -> List[Dict]:
    """
    Filter association rules based on multiple criteria

    Args:
        rules (list): List of association rules
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold
        max_antecedent_size (int): Maximum antecedent size

    Returns:
        list: Filtered list of rules
    """
    filtered_rules = []

    for rule in rules:
        # Apply filters
        if rule['support'] < min_support:
            continue
        if rule['confidence'] < min_confidence:
            continue
        if rule['lift'] < min_lift:
            continue
        if max_antecedent_size and len(rule['antecedent']) > max_antecedent_size:
            continue

        filtered_rules.append(rule)

    return filtered_rules


def sort_rules(rules: List[Dict], sort_by: str = 'confidence', ascending: bool = False) -> List[Dict]:
    """
    Sort association rules by a specified metric

    Args:
        rules (list): List of association rules
        sort_by (str): Metric to sort by ('confidence', 'lift', 'support')
        ascending (bool): Sort order

    Returns:
        list: Sorted list of rules
    """
    if sort_by not in ['confidence', 'lift', 'support']:
        raise ValueError("sort_by must be 'confidence', 'lift', or 'support'")

    return sorted(rules, key=lambda x: x[sort_by], reverse=not ascending)


def analyze_rules(rules: List[Dict]) -> None:
    """
    Analyze and filter association rules based on different criteria
    """
    if not rules:
        print("No rules to analyze!")
        return

    print("=" * 60)
    print("           ASSOCIATION RULES ANALYSIS")
    print("=" * 60)

    # Basic statistics
    print(f"Total rules generated: {len(rules)}")

    # Support distribution
    supports = [rule['support'] for rule in rules]
    confidences = [rule['confidence'] for rule in rules]
    lifts = [rule['lift'] for rule in rules]

    print(f"\nRule metrics distribution:")
    print(f"  Support   - Min: {min(supports):.4f}, Max: {max(supports):.4f}, Avg: {np.mean(supports):.4f}")
    print(f"  Confidence - Min: {min(confidences):.4f}, Max: {max(confidences):.4f}, Avg: {np.mean(confidences):.4f}")
    print(f"  Lift      - Min: {min(lifts):.4f}, Max: {max(lifts):.4f}, Avg: {np.mean(lifts):.4f}")

    # TOP 10 RULES BY LIFT (Most surprising associations)
    print(f"\n🚀 TOP 10 RULES BY LIFT (Most surprising associations):")
    print("-" * 60)
    lift_sorted = sort_rules(rules, 'lift', ascending=False)

    for i, rule in enumerate(lift_sorted[:10]):
        antecedent_str = ', '.join(sorted(rule['antecedent']))
        consequent_str = ', '.join(sorted(rule['consequent']))
        print(f"  {i + 1:2d}. {{{antecedent_str}}} → {{{consequent_str}}}")
        print(f"      Lift: {rule['lift']:.3f} | Confidence: {rule['confidence']:.3f} | Support: {rule['support']:.4f}")

    # TOP 10 RULES BY SUPPORT (Most frequent associations)
    print(f"\n📊 TOP 10 RULES BY SUPPORT (Most frequent associations):")
    print("-" * 60)
    support_sorted = sort_rules(rules, 'support', ascending=False)

    for i, rule in enumerate(support_sorted[:10]):
        antecedent_str = ', '.join(sorted(rule['antecedent']))
        consequent_str = ', '.join(sorted(rule['consequent']))
        print(f"  {i + 1:2d}. {{{antecedent_str}}} → {{{consequent_str}}}")
        print(f"      Support: {rule['support']:.4f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}")

    # BALANCED RULES (good confidence AND lift)
    print(f"\n⚖️ BALANCED RULES (Confidence > 0.5 and Lift > 1.5):")
    print("-" * 60)
    balanced_rules = filter_rules_by_metrics(rules, min_confidence=0.5, min_lift=1.5)
    balanced_sorted = sort_rules(balanced_rules, 'confidence', ascending=False)

    if balanced_rules:
        for i, rule in enumerate(balanced_sorted[:10]):
            antecedent_str = ', '.join(sorted(rule['antecedent']))
            consequent_str = ', '.join(sorted(rule['consequent']))
            quality_score = rule['confidence'] * rule['lift']
            print(f"  {i + 1:2d}. {{{antecedent_str}}} → {{{consequent_str}}}")
            print(
                f"      Quality: {quality_score:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}")
    else:
        print("  No rules meet the balanced criteria")

    # RULES INVOLVING 'whole milk'
    print(f"\n🥛 RULES INVOLVING 'whole milk':")
    print("-" * 60)
    milk_rules = [rule for rule in rules
                  if 'whole milk' in rule['antecedent'] or 'whole milk' in rule['consequent']]

    if milk_rules:
        milk_sorted = sort_rules(milk_rules, 'confidence', ascending=False)
        for i, rule in enumerate(milk_sorted[:10]):
            antecedent_str = ', '.join(sorted(rule['antecedent']))
            consequent_str = ', '.join(sorted(rule['consequent']))
            print(f"  {i + 1:2d}. {{{antecedent_str}}} → {{{consequent_str}}}")
            print(
                f"      Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f} | Support: {rule['support']:.4f}")
    else:
        print("  No rules involving 'whole milk' found")


def get_rules_for_item(rules: List[Dict], item: str, as_antecedent: bool = True) -> List[Dict]:
    """
    Get rules where a specific item appears as antecedent or consequent

    Args:
        rules (list): List of association rules
        item (str): Item to search for
        as_antecedent (bool): If True, find rules where the item is in antecedent

    Returns:
        list: Filtered rules containing the item
    """
    if as_antecedent:
        return [rule for rule in rules if item in rule['antecedent']]
    else:
        return [rule for rule in rules if item in rule['consequent']]


def print_rule_summary(rules: List[Dict], title: str = "Association Rules Summary") -> None:
    """
    Print a formatted summary of association rules

    Args:
        rules (list): List of association rules
        title (str): Title for the summary
    """
    print(f"\n{title}")
    print("-" * len(title))

    if not rules:
        print("No rules found.")
        return

    print(f"Total rules: {len(rules)}")

    # Group by antecedent size
    by_size = {}
    for rule in rules:
        size = len(rule['antecedent'])
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(rule)

    for size in sorted(by_size.keys()):
        count = len(by_size[size])
        avg_conf = np.mean([rule['confidence'] for rule in by_size[size]])
        avg_lift = np.mean([rule['lift'] for rule in by_size[size]])
        print(f"  {size}-item antecedent: {count} rules (avg conf: {avg_conf:.3f}, avg lift: {avg_lift:.3f})")
