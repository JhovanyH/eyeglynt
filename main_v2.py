"""
EyeGlynt - run the app with gaze_engine_v2.py
=============================================
Same screens, dwell buttons and calibration as main.py, but the gaze
comes from gaze_engine_v2.py instead of gaze_engine.py. main.py itself
is not changed, so both versions can be run from the same folder.

RUN
---
    python main.py       the team's version (gaze_engine.py)
    python main_v2.py    jhova's version (gaze_engine_v2.py)
"""

import main
from gaze_engine_v2 import GazeEngine

# main.py creates its engine with GazeEngine() inside EyeGlyntApp.build(),
# so pointing that name at the v2 class is all it takes to swap engines.
main.GazeEngine = GazeEngine

if __name__ == "__main__":
    main.EyeGlyntApp().run()
