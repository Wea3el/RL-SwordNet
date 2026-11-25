"""Publication boundaries for timing-only diagnostic candidates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sim.a1_local_retime import publish


class PublishTimingTests(unittest.TestCase):
    def fixture(self, folder):
        q = [1., 1., 1., 1., 1., 1., 0.]
        row = {'knot_time_s': [0., 1.], 'knot_qpos': [q, q],
               'candidate_pass': False, 'admitted_to_training': False}
        source = folder/'source.json'
        source.write_text(json.dumps({'actions': {'attack': row, 'other': {'keep': 1}}}))
        candidate = folder/'candidate.json'
        new = dict(row, knot_time_s=[0., 2.], candidate_pass=True)
        candidate.write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                         'action': 'attack', 'candidate_action': new}))
        return source, candidate

    def test_publish_preserves_source_and_unrelated_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, candidate = self.fixture(folder)
            before = source.read_bytes()
            result = publish(source, candidate, folder/'result')
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result['actions']['other'], {'keep': 1})
            self.assertEqual(result['actions']['attack']['knot_time_s'], [0., 2.])
            self.assertFalse(result['admitted_to_training'])

    def test_wrong_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, candidate = self.fixture(folder)
            source.write_text(source.read_text()+' ')
            with self.assertRaises(ValueError):
                publish(source, candidate, folder/'result')

    def test_changed_joint_pose_is_not_a_timing_only_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, candidate = self.fixture(folder)
            data = json.loads(candidate.read_text())
            data['candidate_action']['knot_qpos'][1][0] += .1
            candidate.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                publish(source, candidate, folder/'result')


if __name__ == '__main__':
    unittest.main()
