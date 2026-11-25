import unittest

from sim.a1_coverage_run import resolve_defences


class DefenceVariantSelectionTests(unittest.TestCase):
    def setUp(self):
        self.bank={'actions':{
            'parry_high':{'kind':'defence','family':'parry_high'},
            'parry_right_thrust':{'kind':'defence','family':'parry_right'},
            'parry_right_descending':{'kind':'defence','family':'parry_right'},
            'attack':{'kind':'attack','family':'cut_down_center'},
            'unknown':{'kind':'defence','family':'invented'}}}

    def test_resolves_only_declared_variants(self):
        self.assertEqual(resolve_defences(self.bank),
            ('parry_high','parry_right_thrust','parry_right_descending'))
        self.assertEqual(resolve_defences(self.bank,['parry_right_thrust']),('parry_right_thrust',))

    def test_rejects_unknown_attack_and_duplicate_names(self):
        for names in (['missing'],['attack'],['unknown'],['parry_high','parry_high']):
            with self.subTest(names=names),self.assertRaises(ValueError):
                resolve_defences(self.bank,names)

    def test_empty_bank_is_not_a_completed_defence_test(self):
        with self.assertRaises(ValueError):
            resolve_defences({'actions':{}})
