"""Isolated provenance/identity regression tests; no physics or rendering."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sim.a1_triangle_parry import strict_review


class TriangleParryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="triangle_review_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True)+"\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fixture(self, *, freeze_contacts=True, actor="a1", legacy_identity=False):
        experiment = self.root/"experiment"
        experiment.mkdir()
        summaries, audits = {}, {}
        for label, dirname in (("baseline", "baseline"), ("defended", "nominal")):
            folder = experiment/dirname
            folder.mkdir()
            summary = {
                "native_execution_pass": True,
                "source_unchanged": True,
                "failure": None,
                "runtime_config": {
                    "minimum_blade_force_n": .1,
                    "minimum_blade_impulse_ns": .02,
                },
            }
            if not legacy_identity:
                summary["metadata"] = {"attacker_slot": actor}
            self.write_json(folder/"summary.json", summary)
            # strict_review hashes these files, without loading a simulator.
            (folder/"trace.npz").write_bytes(b"isolated recorded trace fixture\n")
            (folder/"scene.xml").write_bytes(b"isolated recorded scene fixture\n")
            audit = {
                "all_entries": ([{"agent": actor, "region": "upper", "time_s": 3.,
                                  "step_t0_s": 2.999}] if label == "baseline" else []),
                "intended_region_first": label == "baseline",
                "initial_overlap": {},
                "sweep_failures": {},
                "native_trace_source_sha256": {
                    name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                    for name in ("summary.json", "trace.npz", "scene.xml")
                },
            }
            if not legacy_identity:
                audit["attacker_slot"] = actor
            summaries[label], audits[label] = summary, audit
        contacts = [{
            "interval_start_s": 2., "interval_end_s": 2.1,
            "contacts": [{"category": "cross_blade_blade",
                          "geom1": "a1_shinai_blade_collision_0",
                          "geom2": "a2_shinai_blade_collision_0",
                          "normal_force_n": 1.}],
        }]
        contact_path = experiment/"nominal"/"contacts.json"
        contact_hash = self.write_json(contact_path, contacts)
        previous = {
            "status": "qualified",
            "matching_checks": {"same_attack": True, "complete_native_windows": True},
            "baseline_triangle_audit": audits["baseline"],
            "defended_triangle_audit": audits["defended"],
        }
        if freeze_contacts:
            previous["native_contact_source_sha256"] = contact_hash
        comparison = experiment/"nominal_comparison.json"
        self.write_json(comparison, previous)
        return experiment, comparison, contact_path

    def review(self, experiment, comparison, *, actor="a1"):
        original = comparison.read_bytes()
        output = self.root/"review"
        result = strict_review(experiment, output, attacker=actor)["nominal"]
        self.assertEqual(comparison.read_bytes(), original, "Historical comparison must remain unchanged")
        self.assertEqual(json.loads((output/"nominal.json").read_text()), result)
        self.assertEqual(result["input_comparison_sha256"], hashlib.sha256(original).hexdigest())
        self.assertFalse(result["admitted_to_training"])
        return result

    def test_new_frozen_evidence_is_qualified_without_rewriting_history(self):
        experiment, comparison, _ = self.fixture()
        result = self.review(experiment, comparison)
        self.assertEqual(result["status"], "qualified")
        self.assertTrue(result["matching_checks"]["recorded_contacts_unchanged"])
        self.assertEqual(result["contact_hash_provenance"], "verified_against_original_comparison")
        self.assertEqual(result["attacker_slot"], "a1")

    def test_tampered_contacts_are_unmatched_even_if_still_plausible(self):
        experiment, comparison, contact_path = self.fixture()
        contacts = json.loads(contact_path.read_text())
        contacts[0]["contacts"][0]["normal_force_n"] = 2.
        self.write_json(contact_path, contacts)
        result = self.review(experiment, comparison)
        self.assertEqual(result["status"], "unmatched")
        self.assertFalse(result["matching_checks"]["recorded_contacts_unchanged"])
        self.assertFalse(result["successful_physical_defence"])

    def test_legacy_evidence_keeps_explicit_unfrozen_contact_provenance(self):
        experiment, comparison, _ = self.fixture(freeze_contacts=False, legacy_identity=True)
        result = self.review(experiment, comparison)
        self.assertEqual(result["status"], "qualified")
        self.assertEqual(result["contact_hash_provenance"], "legacy_comparison_did_not_freeze_contact_hash")
        self.assertNotIn("recorded_contacts_unchanged", result["matching_checks"])
        self.assertIn("current_contact_snapshot_sha256", result)

    def test_wrong_attacker_slot_is_unmatched_not_another_certification(self):
        experiment, comparison, _ = self.fixture(actor="a2")
        result = self.review(experiment, comparison, actor="a1")
        self.assertEqual(result["status"], "unmatched")
        self.assertFalse(result["matching_checks"]["baseline_attacker_identity"])
        self.assertFalse(result["matching_checks"]["defended_attacker_identity"])

    def test_correct_mirror_identity_can_be_reviewed(self):
        experiment, comparison, _ = self.fixture(actor="a2")
        result = self.review(experiment, comparison, actor="a2")
        self.assertEqual(result["status"], "qualified")
        self.assertTrue(result["matching_checks"]["baseline_attacker_identity"])
        self.assertEqual(result["attacker_slot"], "a2")

    def test_changed_trace_is_unmatched_and_prior_review_is_preserved(self):
        experiment, comparison, _ = self.fixture()
        (experiment/"nominal"/"trace.npz").write_bytes(b"changed recorded trace\n")
        result = self.review(experiment, comparison)
        self.assertEqual(result["status"], "unmatched")
        self.assertFalse(result["matching_checks"]["defended_recorded_evidence_unchanged"])
        prior_review = (self.root/"review"/"nominal.json").read_bytes()
        with self.assertRaises(FileExistsError):
            strict_review(experiment, self.root/"review")
        self.assertEqual((self.root/"review"/"nominal.json").read_bytes(), prior_review)


if __name__ == "__main__":
    unittest.main()
