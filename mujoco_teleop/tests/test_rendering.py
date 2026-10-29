"""A visible viewer must not change the policy's offscreen camera images."""
from pathlib import Path
import runpy
import unittest

PROJECT = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT / 'run.py'), run_name='bootstrap')

import numpy as np
from omy_repro.runtime import OMYEnvironment


class RenderingTests(unittest.TestCase):
    def test_gui_render_preserves_camera_pixels_and_physics(self):
        environment = OMYEnvironment(seed=0, headless=False)
        try:
            state = environment.snapshot()
            before = environment.images()
            environment.show(*before)
            after = environment.images()
            for original, rendered in zip(before, after):
                # Context switches can round a few MSAA pixels by one uint8
                # level on this driver. Large corruption must still fail.
                difference = np.abs(original.astype(np.int16) - rendered.astype(np.int16))
                self.assertLessEqual(int(difference.max()), 1)
                self.assertLess(np.count_nonzero(difference) / difference.size, 0.001)
            np.testing.assert_array_equal(environment.snapshot(), state)
        finally:
            environment.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
