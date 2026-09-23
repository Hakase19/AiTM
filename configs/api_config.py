"""Runtime API configuration for AiTM experiments.

Set ``AITM_API_KEY`` in the shell before launching an experiment. Keeping the
secret outside this repository prevents it from being committed or written to
result files.
"""

import os


API_KEY = os.environ.get("AITM_API_KEY", "")
BASE_URL = "https://www.dmxapi.cn/v1"

# Use one model condition for MAS agents and the adversarial agent so that
# compared methods remain directly comparable.
DEFAULT_MODEL = "DeepSeek-V3.2"
ADVERSARIAL_MODEL = "DeepSeek-V3.2"

# The pinned DeepSeek-V3.2 tokenizer declares a 131,072-token context window.
# A 4,096-token output allowance is large enough for full agent reasoning and
# structured judges while remaining below the model family's 8K non-thinking
# output limit. Input prompts are kept intact for the current small MAS graphs.
MODEL_CONTEXT_WINDOW_TOKENS = 131_072
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
