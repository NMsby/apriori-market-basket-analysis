import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json


def load_groceries_dataset(filepath='data/raw/groceries.csv'):
    """
    Load the groceries dataset from CSV file

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please ensure the groceries.csv file is in the data/raw/ directory")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def explore_dataset(df):
    """
    Perform initial exploration of the dataset

    Args:
        df (pd.DataFrame): Dataset to explore
    """
    if df is None:
        print("Cannot explore dataset - data not loaded")
        return

    print("=== DATASET EXPLORATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nDataset info:")
    print(df.info())

    print(f"\nMissing values:")
    print(df.isnull().sum())

    # Check for unique values
    print(f"\nUnique members: {df['Member_number'].nunique()}")
    print(f"Unique items: {df['itemDescription'].nunique()}")

    # Date range analysis - fix the date parsing warning
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)

    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Sample of the most frequent items
    print(f"\nTop 10 most frequent items:")
    item_counts = df['itemDescription'].value_counts().head(10)
    for item, count in item_counts.items():
        print(f"  {item}: {count}")

    return df


def create_transaction_list(df):
    """
    Convert the dataframe into a list of transactions (itemsets)
    Each transaction is a list of items bought together by the same member on the same date

    Args:
        df (pd.DataFrame): DataFrame with columns Member_number, Date, itemDescription

    Returns:
        list: List of transactions, where each transaction is a list of items
    """
    if df is None:
        print("Cannot create transaction list - data not loaded")
        return []

    print("Converting data to transaction format...")

    # Convert Date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)

    # Group by Member_number and Date to get individual transactions
    transactions = []
    grouped = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)

    for transaction in grouped:
        # Remove duplicates within a transaction and filter out empty items
        unique_items = list(set([item.strip() for item in transaction if item.strip()]))
        if len(unique_items) > 0:  # Only keep non-empty transactions
            transactions.append(unique_items)

    print(f"Created {len(transactions)} transactions from {len(df)} records")
    return transactions


def analyze_transactions(transactions):
    """
    Analyze transaction statistics and patterns

    Args:
        transactions (list): List of transaction lists

    Returns:
        dict: Dictionary with transaction statistics
    """
    if not transactions:
        print("No transactions to analyze")
        return {}

    print("=== TRANSACTION ANALYSIS ===")

    # Calculate transaction lengths
    transaction_lengths = [len(transaction) for transaction in transactions]

    # Basic statistics
    avg_length = np.mean(transaction_lengths)
    min_length = min(transaction_lengths)
    max_length = max(transaction_lengths)

    print(f"Total transactions: {len(transactions)}")
    print(f"Average items per transaction: {avg_length:.2f}")
    print(f"Min items per transaction: {min_length}")
    print(f"Max items per transaction: {max_length}")

    # Show the first 5 transactions as examples
    print(f"\nFirst 5 transactions:")
    for i, transaction in enumerate(transactions[:5]):
        print(f"  Transaction {i + 1}: {transaction}")

    # Distribution of transaction lengths
    length_distribution = Counter(transaction_lengths)
    print(f"\nTransaction length distribution (top 10):")
    for length, count in sorted(length_distribution.items())[:10]:
        print(f"  {length} items: {count} transactions")

    return {
        'total_transactions': len(transactions),
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'transaction_lengths': transaction_lengths
    }


def save_processed_data(transactions, transaction_stats, output_dir='data/processed'):
    """
    Save processed transaction data and statistics

    Args:
        transactions (list): List of transactions
        transaction_stats (dict): Transaction statistics
        output_dir (str): Directory to save processed data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save transactions as pickle for efficient loading
    with open(os.path.join(output_dir, 'transactions.pkl'), 'wb') as f:
        pickle.dump(transactions, f)

    # Save statistics as JSON for readability
    with open(os.path.join(output_dir, 'transaction_stats.json'), 'w') as f:
        json.dump(transaction_stats, f, indent=2)

    print(f"Processed data saved to {output_dir}/")
    print(f"  - transactions.pkl: {len(transactions)} transactions")
    print(f"  - transaction_stats.json: Statistical summary")


def load_processed_data(data_dir='data/processed'):
    """
    Load processed transaction data

    Args:
        data_dir (str): Directory containing processed data

    Returns:
        tuple: (transactions, transaction_stats)
    """
    try:
        with open(os.path.join(data_dir, 'transactions.pkl'), 'rb') as f:
            transactions = pickle.load(f)

        with open(os.path.join(data_dir, 'transaction_stats.json'), 'r') as f:
            transaction_stats = json.load(f)

        print(f"Loaded {len(transactions)} transactions from {data_dir}/")
        return transactions, transaction_stats

    except FileNotFoundError:
        print(f"Processed data not found in {data_dir}/")
        return None, None
