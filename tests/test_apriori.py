import unittest
import sys
import os

sys.path.append('../src')

from data_preprocessing import load_groceries_dataset, create_transaction_list, analyze_transactions
import pandas as pd


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        # Create sample test data
        self.test_data = pd.DataFrame({
            'Member_number': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'Date': ['2023-01-01', '2023-01-01', '2023-01-02',
                     '2023-01-01', '2023-01-01', '2023-01-01',
                     '2023-01-01', '2023-01-01', '2023-01-01'],
            'itemDescription': ['milk', 'bread', 'eggs',
                                'milk', 'butter', 'milk',
                                'bread', 'eggs', 'cheese']
        })

    def test_create_transaction_list(self):
        """Test transaction list creation"""
        transactions = create_transaction_list(self.test_data)

        # Should have 3 transactions (member 1 has 2 different dates)
        self.assertEqual(len(transactions), 3)

        # Check that transactions contain expected items
        self.assertIn(['milk', 'bread'], transactions)
        self.assertIn(['eggs'], transactions)
        self.assertIn(['milk', 'butter'], transactions)

    def test_analyze_transactions(self):
        """Test transaction analysis"""
        transactions = [['milk', 'bread'], ['eggs'], ['milk', 'butter']]
        stats = analyze_transactions(transactions)

        self.assertEqual(stats['total_transactions'], 3)
        self.assertEqual(stats['min_length'], 1)
        self.assertEqual(stats['max_length'], 2)
        self.assertAlmostEqual(stats['avg_length'], 1.67, places=2)


if __name__ == '__main__':
    unittest.main()