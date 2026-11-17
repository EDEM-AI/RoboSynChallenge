# -*- coding: utf-8 -*-
"""P001 单测入口：python3 tests/run_all.py [--official-config <gym_config.json>]"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import test_p001  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(test_p001.main())
