"""Runtime API configuration for AiTM experiments.

Set ``AITM_API_KEY`` in the shell before launching an experiment. Keeping the
secret outside this repository prevents it from being committed or written to
result files.
"""

import os


API_KEY = os.environ.get("AITM_API_KEY", "")
BASE_URL = "https://www.dmxapi.cn/v1"

# Use one model condition for MAS agents, the adversarial agent, and AIRA's
# role inference so that compared methods remain directly comparable.
DEFAULT_MODEL = "DeepSeek-V3.2"
ADVERSARIAL_MODEL = "DeepSeek-V3.2"
