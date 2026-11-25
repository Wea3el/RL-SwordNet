import unittest

from sim.a1_triangle_screen import first_region_checks


class TriangleScreenTests(unittest.TestCase):
    def test_earlier_wrong_target_cannot_claim_first_hit_coverage(self):
        row = first_region_checks({'left': 1., 'upper': 1.2}, 'upper', (.5,2.))
        self.assertTrue(row['targeted_entry_pass'])
        self.assertFalse(row['unrestricted_first_region_pass'])

    def test_coarse_tie_is_not_a_unique_first_region(self):
        row = first_region_checks({'left': 1., 'right': 1.}, 'left', (.5,2.))
        self.assertFalse(row['unrestricted_first_region_pass'])
        self.assertEqual(row['first_regions_at_sample_resolution'], ['left','right'])

    def test_entry_during_preparation_is_not_a_strike(self):
        row = first_region_checks({'left': .1}, 'left', (.5,2.))
        self.assertFalse(row['targeted_entry_pass'])

    def test_unique_target_in_window(self):
        row = first_region_checks({'upper': 1.}, 'upper', (.5,2.))
        self.assertTrue(row['unrestricted_first_region_pass'])


if __name__ == '__main__':
    unittest.main()
