import json
import unittest

import numpy as np

from sim.a1_triangle_candidates import attack_waypoints, retimed_curve, travel_cost
from sim.assembly import ROOT
from sim.triangle_task import TriangleGeometry


class TriangleCandidateTests(unittest.TestCase):
    def test_geometry_has_disjoint_finite_regions_and_is_not_live(self):
        cfg = json.loads((ROOT/'config/scoring_triangle_v1_candidate.json').read_text())
        geometry = TriangleGeometry.from_mapping({k: cfg[k] for k in ('base_distance', 'regions')})
        self.assertFalse(cfg['runtime_activation_allowed'])
        self.assertFalse(cfg['admitted_to_training'])
        for r in geometry.regions:
            center = geometry.world_center(r.name, 'a2')
            points = attack_waypoints(center, 'thrust')
            self.assertLess(np.linalg.norm(points[1]-center), r.radius)
            self.assertGreater(np.linalg.norm(points[0]-center), r.radius)

    def test_guard_exit_can_differ_but_is_c2_at_rest(self):
        nodes = [np.ones(7), np.ones(7)+.2, np.ones(7)+.4]
        curve = retimed_curve(nodes, .7)
        np.testing.assert_array_equal(curve.sample(0), nodes[0])
        np.testing.assert_array_equal(curve.sample(curve.duration_s), nodes[-1])
        self.assertLessEqual(max(curve.analytic_extrema()['peak_speed_rad_s']), .7+1e-9)
        for t in (0, curve.duration_s):
            np.testing.assert_array_equal(curve.derivatives(t)[0], np.zeros(7))
            np.testing.assert_array_equal(curve.derivatives(t)[1], np.zeros(7))

    def test_no_wrapping_makes_long_wrong_way_motion_look_short(self):
        nodes = np.zeros((2, 7)); nodes[1, 0] = np.deg2rad(350)
        self.assertAlmostEqual(travel_cost(nodes, 1), np.deg2rad(350))

    def test_cut_direction_is_a_request_not_a_label_certificate(self):
        points = attack_waypoints([1.43, -.2, .93], 'lateral', [0, 1, 0])
        self.assertGreater(points[2][1], points[1][1])
        with self.assertRaises(ValueError):
            attack_waypoints([1.43, -.2, .93], 'lateral', [0, 0, 0])

    def test_negative_or_nonfinite_reference_speed_is_rejected(self):
        for speed in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                retimed_curve([np.zeros(7), np.ones(7)], speed)


if __name__ == '__main__':
    unittest.main()
