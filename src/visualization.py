import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter


def plot_transaction_patterns(transactions, transaction_stats):
    """
    Create visualizations for transaction patterns

    Args:
        transactions (list): List of transaction lists
        transaction_stats (dict): Transaction statistics
    """
    if not transactions:
        print("No transactions to visualize")
        return

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Transaction length distribution
    axes[0, 0].hist(transaction_stats['transaction_lengths'],
                    bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Number of Items per Transaction')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Transaction Lengths')
    axes[0, 0].axvline(transaction_stats['avg_length'], color='red',
                       linestyle='--', label=f'Average: {transaction_stats["avg_length"]:.2f}')
    axes[0, 0].legend()

    # Plot 2: Top 20 most frequent items
    all_items = [item for transaction in transactions for item in transaction]
    item_counts = Counter(all_items)
    top_20_items = item_counts.most_common(20)

    items = [item[0] for item in top_20_items]
    counts = [item[1] for item in top_20_items]

    axes[0, 1].bar(range(len(items)), counts, color='lightcoral')
    axes[0, 1].set_xlabel('Items')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Top 20 Most Frequent Items')
    axes[0, 1].set_xticks(range(len(items)))
    axes[0, 1].set_xticklabels(items, rotation=45, ha='right')

    # Plot 3: Transaction length vs frequency (box plot style)
    length_counts = Counter(transaction_stats['transaction_lengths'])
    lengths = list(length_counts.keys())
    frequencies = list(length_counts.values())

    axes[1, 0].scatter(lengths, frequencies, alpha=0.6, color='green')
    axes[1, 0].set_xlabel('Transaction Length')
    axes[1, 0].set_ylabel('Number of Transactions')
    axes[1, 0].set_title('Transaction Length vs Frequency')

    # Plot 4: Cumulative percentage of items
    total_items = len(all_items)
    cumulative_percentages = []
    running_total = 0

    for _, count in top_20_items:
        running_total += count
        cumulative_percentages.append((running_total / total_items) * 100)

    axes[1, 1].plot(range(1, len(cumulative_percentages) + 1),
                    cumulative_percentages, marker='o', color='purple')
    axes[1, 1].set_xlabel('Top N Items')
    axes[1, 1].set_ylabel('Cumulative Percentage of Total Items')
    axes[1, 1].set_title('Pareto Analysis: Top Items Coverage')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print the top 10 items with percentages
    print("\n=== TOP 10 MOST FREQUENT ITEMS ===")
    for i, (item, count) in enumerate(top_20_items[:10]):
        percentage = (count / total_items) * 100
        print(f"{i + 1:2d}. {item:<25} | Count: {count:>5} | Percentage: {percentage:>5.2f}%")


def create_data_summary_report(df, transactions, transaction_stats):
    """
    Create a comprehensive data summary report

    Args:
        df (pd.DataFrame): Original dataset
        transactions (list): List of transactions
        transaction_stats (dict): Transaction statistics
    """
    print("=" * 60)
    print("                DATA SUMMARY REPORT")
    print("=" * 60)

    # Dataset overview
    print(f"📊 DATASET OVERVIEW:")
    print(f"   • Total records in dataset: {len(df):,}")
    print(f"   • Total unique customers: {df['Member_number'].nunique():,}")
    print(f"   • Total unique items: {df['itemDescription'].nunique():,}")
    print(f"   • Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

    # Transaction overview
    print(f"\n🛒 TRANSACTION OVERVIEW:")
    print(f"   • Total transactions: {transaction_stats['total_transactions']:,}")
    print(f"   • Average items per transaction: {transaction_stats['avg_length']:.2f}")
    print(f"   • Transaction size range: {transaction_stats['min_length']} - {transaction_stats['max_length']} items")

    # Calculate some interesting metrics
    all_items = [item for transaction in transactions for item in transaction]
    unique_items_in_transactions = len(set(all_items))

    print(f"\n📈 KEY METRICS:")
    print(f"   • Total item purchases: {len(all_items):,}")
    print(f"   • Unique items in transactions: {unique_items_in_transactions:,}")
    print(f"   • Average purchases per item: {len(all_items) / unique_items_in_transactions:.2f}")

    # Data quality
    print(f"\n✅ DATA QUALITY:")
    print(f"   • Missing values: {df.isnull().sum().sum()}")
    print(f"   • Empty transactions: {len([t for t in transactions if len(t) == 0])}")
    print(f"   • Single-item transactions: {len([t for t in transactions if len(t) == 1])}")

    print("=" * 60)
