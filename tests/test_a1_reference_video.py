import unittest
import numpy as np
from sim.a1_reference_video import display_times


class ReferenceVideoTests(unittest.TestCase):
    def test_full_duration_and_exact_endpoint(self):
        times=display_times(9.5,20)
        self.assertEqual(times[0],0.)
        self.assertEqual(times[-1],9.5)
        self.assertEqual(len(times),191)
        np.testing.assert_allclose(np.diff(times),.05)

    def test_noninteger_duration_is_not_sped_up(self):
        times=display_times(1.123,20)
        self.assertEqual(times[-1],1.123)
        self.assertTrue(np.all(np.diff(times)>0))
        self.assertLessEqual(np.max(np.diff(times)),.05+1e-12)

    def test_bad_clock_rejected(self):
        for duration,fps in [(0,20),(float('nan'),20),(1,0),(1,True)]:
            with self.assertRaises(ValueError): display_times(duration,fps)


if __name__=='__main__': unittest.main()
