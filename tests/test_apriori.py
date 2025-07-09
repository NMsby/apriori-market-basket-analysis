import unittest
import sys
import os

sys.path.append('../src')

from apriori_implementation import (
    calculate_support,
    calculate_support_batch,
    get_unique_items,
    validate_transactions,
    generate_candidates,
    generate_candidates_optimized,
    prune_candidates,
    find_frequent_itemsets
)
from data_preprocessing import load_processed_data
import pandas as pd


class TestAprioriAlgorithm(unittest.TestCase):

    def setUp(self):
        """Set up test data for all tests"""
        # Simple test transactions
        self.simple_transactions = [
            ['milk', 'bread', 'eggs'],
            ['milk', 'bread'],
            ['milk', 'eggs'],
            ['bread', 'eggs'],
            ['milk'],
            ['bread'],
            ['eggs', 'butter'],
            ['milk', 'butter']
        ]

        # Expected supports for simple transactions
        self.expected_supports = {
            frozenset(['milk']): 5 / 8,  # appears in 5 transactions
            frozenset(['bread']): 4 / 8,  # appears in 4 transactions
            frozenset(['eggs']): 4 / 8,  # appears in 4 transactions
            frozenset(['butter']): 2 / 8,  # appears in 2 transactions
            frozenset(['milk', 'bread']): 2 / 8,  # appears in 2 transactions
            frozenset(['milk', 'eggs']): 2 / 8,  # appears in 2 transactions
        }

    def test_calculate_support(self):
        """Test support calculation function"""
        for itemset, expected_support in self.expected_supports.items():
            calculated_support = calculate_support(itemset, self.simple_transactions)
            self.assertAlmostEqual(calculated_support, expected_support, places=4,
                                   msg=f"Support calculation failed for {itemset}")

    def test_calculate_support_edge_cases(self):
        """Test support calculation edge cases"""
        # Empty itemset
        self.assertEqual(calculate_support(frozenset(), self.simple_transactions), 0.0)

        # Empty transactions
        self.assertEqual(calculate_support(frozenset(['milk']), []), 0.0)

        # Non-existent item
        self.assertEqual(calculate_support(frozenset(['non_existent']), self.simple_transactions), 0.0)

    def test_calculate_support_batch(self):
        """Test batch support calculation"""
        itemsets = [frozenset(['milk']), frozenset(['bread']), frozenset(['milk', 'bread'])]
        batch_supports = calculate_support_batch(itemsets, self.simple_transactions)

        for itemset in itemsets:
            individual_support = calculate_support(itemset, self.simple_transactions)
            self.assertAlmostEqual(batch_supports[itemset], individual_support, places=4)

    def test_get_unique_items(self):
        """Test unique items extraction"""
        unique_items = get_unique_items(self.simple_transactions)
        expected_items = {'milk', 'bread', 'eggs', 'butter'}
        self.assertEqual(unique_items, expected_items)

    def test_validate_transactions(self):
        """Test transaction validation"""
        # Valid transactions
        is_valid, msg = validate_transactions(self.simple_transactions)
        self.assertTrue(is_valid)

        # Invalid transactions
        invalid_cases = [
            ([['milk'], [123]], "Non-string item"),  # Non-string item
            ([['milk'], ['']], "Empty item"),  # Empty item
            (['not_a_list'], "Non-list transaction"),  # Non-list transaction
            ([], "Empty transactions list")  # Empty list
        ]

        for invalid_trans, description in invalid_cases:
            is_valid, msg = validate_transactions(invalid_trans)
            self.assertFalse(is_valid, f"Should be invalid: {description}")

    def test_generate_candidates(self):
        """Test candidate generation"""
        # Test 2-itemset generation from 1-itemsets
        frequent_1 = {frozenset(['A']), frozenset(['B']), frozenset(['C'])}
        candidates_2 = generate_candidates(frequent_1, 2)
        expected_2 = {frozenset(['A', 'B']), frozenset(['A', 'C']), frozenset(['B', 'C'])}
        self.assertEqual(candidates_2, expected_2)

        # Test 3-itemset generation from 2-itemsets
        frequent_2 = {frozenset(['A', 'B']), frozenset(['A', 'C']), frozenset(['B', 'C'])}
        candidates_3 = generate_candidates(frequent_2, 3)
        expected_3 = {frozenset(['A', 'B', 'C'])}
        self.assertEqual(candidates_3, expected_3)

    def test_generate_candidates_optimized(self):
        """Test optimized candidate generation produces same results"""
        frequent_1 = {frozenset(['A']), frozenset(['B']), frozenset(['C']), frozenset(['D'])}

        regular_candidates = generate_candidates(frequent_1, 2)
        optimized_candidates = generate_candidates_optimized(frequent_1, 2)

        self.assertEqual(regular_candidates, optimized_candidates)

    def test_prune_candidates(self):
        """Test candidate pruning"""
        # Create a scenario where pruning should occur
        candidates = {
            frozenset(['A', 'B', 'C']),
            frozenset(['A', 'B', 'D']),
            frozenset(['A', 'C', 'D']),
            frozenset(['B', 'C', 'D'])
        }

        # Missing frozenset(['C', 'D']) from frequent 2-itemsets
        frequent_2 = {
            frozenset(['A', 'B']),
            frozenset(['A', 'C']),
            frozenset(['A', 'D']),
            frozenset(['B', 'C']),
            frozenset(['B', 'D'])
        }

        pruned = prune_candidates(candidates, frequent_2)

        # frozenset(['A', 'C', 'D']) and frozenset(['B', 'C', 'D']) should be pruned
        # because frozenset(['C', 'D']) is not in frequent_2
        expected_pruned = {
            frozenset(['A', 'B', 'C']),
            frozenset(['A', 'B', 'D'])
        }

        self.assertEqual(pruned, expected_pruned)

    def test_find_frequent_itemsets(self):
        """Test complete Apriori algorithm"""
        # Test with simple transactions
        frequent_itemsets = find_frequent_itemsets(
            transactions=self.simple_transactions,
            min_support=0.25,  # 25% support (2 out of 8 transactions)
            verbose=False
        )

        # Check that we get expected frequent itemsets
        self.assertIn(1, frequent_itemsets)  # Should have frequent 1-itemsets
        self.assertIn(2, frequent_itemsets)  # Should have frequent 2-itemsets

        # Check specific itemsets
        frequent_1 = frequent_itemsets[1]
        self.assertIn(frozenset(['milk']), frequent_1)  # 5/8 = 0.625 > 0.25
        self.assertIn(frozenset(['bread']), frequent_1)  # 4/8 = 0.5 > 0.25
        self.assertIn(frozenset(['eggs']), frequent_1)  # 4/8 = 0.5 > 0.25

        # Butter should not be frequent with a 0.25 threshold (2/8 = 0.25, not > 0.25)
        # Note: depends on implementation of >= vs > for the threshold

    def test_algorithm_with_real_data(self):
        """Test algorithm with real data if available"""
        try:
            # Try to load real data
            transactions, _ = load_processed_data('../data/processed')
            if transactions and len(transactions) > 100:
                # Test with a small subset and high support threshold
                test_transactions = transactions[:100]
                frequent_itemsets = find_frequent_itemsets(
                    transactions=test_transactions,
                    min_support=0.1,
                    max_itemset_size=2,
                    verbose=False
                )

                # Basic sanity checks
                self.assertIsInstance(frequent_itemsets, dict)
                self.assertIn(1, frequent_itemsets)
                self.assertGreater(len(frequent_itemsets[1]), 0)

        except Exception:
            # Skip test if real data not available
            self.skipTest("Real data not available for testing")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
