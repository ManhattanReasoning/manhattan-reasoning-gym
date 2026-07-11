"""Make ffn_accel imports (rtl.*, golden) resolve when pytest collects here."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
