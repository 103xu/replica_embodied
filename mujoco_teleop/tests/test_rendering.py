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
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            np.testing.assert_array_equal(environment.snapshot(), state)
        finally:
            environment.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
