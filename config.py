"""
Configuration and constants for the Ethical AI Assessment Tool.
"""

import os
import json
import logging
from typing import Dict
from pathlib import Path
from rich.logging import RichHandler

# --- Constants ---
CONFIG_FILE = 'config.json'
QUESTIONS_FILE = 'questions.txt'
PROMPT_FILE = 'prompt.txt'
RESULTS_DIR = 'results'  # Directory to store results
DASHBOARD_DIR = 'dashboard'  # Directory to store dashboard
ASSESSMENT_DATA_FILE = 'assessment_data.jsonl'  # File to store assessment data

# --- Provider Names ---
PROVIDER_LMSTUDIO = 'lmstudio'
PROVIDER_OPENAI = 'openai'
PROVIDER_ANTHROPIC = 'anthropic'
PROVIDER_GOOGLE = 'google'
PROVIDER_GENERIC = 'generic_openai'
SUPPORTED_PROVIDERS = [PROVIDER_LMSTUDIO, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GOOGLE, PROVIDER_GENERIC]

# --- Defaults ---
DEFAULT_MAX_TOKENS = 512
SCORE_RANGE = (0, 100)
LOG_FILE = 'assessment.log'
REQUEST_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STRIP_THINK_TAGS = True
DEFAULT_NUM_SAMPLES = 3
DEFAULT_RETRY_EDGE_CASES = True
DEFAULT_MAX_RETRIES_EDGE = 3
DEFAULT_RANDOM_TEMP_MIN = 0.1
DEFAULT_RANDOM_TEMP_MAX = 0.7
DEFAULT_RETRY_CONFIRM_THRESHOLD = 0.5
DEFAULT_REQUEST_DELAY = 0  # seconds to delay between API requests (rate limiting)
DEFAULT_CATEGORY_MAPPING = {
    "transparency": list(range(1, 21)),
    "fairness": list(range(21, 41)),
    "safety": list(range(41, 61)),
    "reliability": list(range(61, 71)),
    "ethics": list(range(71, 91)),
    "social_impact": list(range(91, 101))
}

# --- Environment Variable Names ---
ENV_LMSTUDIO_API_KEY = 'LMSTUDIO_API_KEY'
ENV_OPENAI_API_KEY = 'OPENAI_API_KEY'
ENV_ANTHROPIC_API_KEY = 'ANTHROPIC_API_KEY'
ENV_GOOGLE_API_KEY = 'GEMINI_API_KEY'
ENV_GENERIC_API_KEY = 'GENERIC_API_KEY'

# --- Setup directories ---
def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'docs'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), DASHBOARD_DIR), exist_ok=True)

# --- Logging Setup ---
def setup_logging(level=logging.INFO): # Add level argument with default
    """Configure logging for the application."""
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Create results directory if it doesn't exist

    logging.basicConfig(
        level=level, # Use the passed level
        filename=LOG_FILE,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = RichHandler(rich_tracebacks=True, show_time=True, level=level) # Use the passed level
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    root_logger = logging.getLogger()
    # Remove default StreamHandler to avoid duplicate console logs
    for handler in root_logger.handlers[:]: # Iterate over a copy
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RichHandler):
            root_logger.removeHandler(handler)
    
    # Set the level for the root logger explicitly
    root_logger.setLevel(level)
    
    return logging.getLogger(__name__)

# Initialize logger with default level initially
# The level will be potentially reconfigured when main() calls setup_logging again
log = setup_logging()

# --- File Loading Functions ---
def load_config(config_file: str = CONFIG_FILE) -> Dict:
    """Load configuration data from a JSON file."""
    log.info(f"Loading configuration from '{config_file}'")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            # --- Validate Provider Config ---
            for provider in SUPPORTED_PROVIDERS:
                if provider not in config_data:
                    raise ValueError(f"'{provider}' section missing in '{config_file}'")
                provider_config = config_data[provider]
                if 'api_endpoint' not in provider_config:
                    raise ValueError(f"'api_endpoint' missing in '{provider}' section of '{config_file}'")
                if 'model' not in provider_config:
                    raise ValueError(f"'model' missing in '{provider}' section of '{config_file}'")

                # --- Add Default Multi-Sample/Retry Config if Missing ---
                provider_config.setdefault('num_samples_per_question', DEFAULT_NUM_SAMPLES)
                provider_config.setdefault('retry_edge_cases', DEFAULT_RETRY_EDGE_CASES)
                provider_config.setdefault('max_retries_for_edge_case', DEFAULT_MAX_RETRIES_EDGE)
                provider_config.setdefault('random_temp_min', DEFAULT_RANDOM_TEMP_MIN)
                provider_config.setdefault('random_temp_max', DEFAULT_RANDOM_TEMP_MAX)
                provider_config.setdefault('retry_confirm_threshold', DEFAULT_RETRY_CONFIRM_THRESHOLD)
                provider_config.setdefault('max_tokens', DEFAULT_MAX_TOKENS)
                provider_config.setdefault('temperature', DEFAULT_TEMPERATURE)
                provider_config.setdefault('strip_think_tags', DEFAULT_STRIP_THINK_TAGS)
                provider_config.setdefault('request_delay', DEFAULT_REQUEST_DELAY)
                provider_config.setdefault('category_mapping', DEFAULT_CATEGORY_MAPPING)

            log.info("Configuration loaded and defaults applied.")
            return config_data
    except FileNotFoundError:
        log.error(f"Configuration file not found: {config_file}")
        raise
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in config file '{config_file}': {e}")
        raise
    except ValueError as e:
        log.error(f"Configuration Error: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected error loading config '{config_file}': {e}", exc_info=True)
        raise

def load_text_file(filepath: str) -> str:
    """Load text content from a file, stripping leading/trailing whitespace."""
    log.info(f"Loading text file from '{filepath}'")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            log.info(f"Text file '{filepath}' loaded successfully.")
            return content
    except FileNotFoundError:
        log.error(f"Required text file not found: {filepath}")
        raise
    except Exception as e:
        log.error(f"Unexpected error loading text file '{filepath}': {e}", exc_info=True)
        raise

# Load configuration and required files
def load_required_files():
    """Load all required configuration files."""
    try:
        config_data = load_config()
        questions_list = [q for q in load_text_file(QUESTIONS_FILE).splitlines() if q.strip()]
        if not questions_list:
            log.error(f"No questions found or loaded from '{QUESTIONS_FILE}'.")
            raise ValueError(f"No questions loaded from {QUESTIONS_FILE}.")
        prompt_template_text = load_text_file(PROMPT_FILE)
        
        return config_data, questions_list, prompt_template_text
    except Exception as e:
        log.critical(f"Initialization failed: {e}", exc_info=True)
        raise

# Setup directories when module is imported
setup_directories()
