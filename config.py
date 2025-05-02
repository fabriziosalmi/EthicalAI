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
REPORTS_DIR = 'docs/reports'  # Directory to store GitHub Pages reports
DASHBOARD_DIR = 'docs'  # Directory to store dashboard
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
    os.makedirs(os.path.join(os.path.dirname(__file__), REPORTS_DIR), exist_ok=True)  # Create reports directory

# --- Logging Setup ---
def setup_logging(level=logging.INFO):
    """Configure logging for the application, removing existing handlers first."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    root_logger = logging.getLogger() # Get the root logger
    
    # Remove all existing handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close() # Close the handler properly
        
    # Set the root logger's level
    root_logger.setLevel(level)

    # Configure File Handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level) # Set level for file handler
    root_logger.addHandler(file_handler)

    # Configure Console Handler (RichHandler)
    console_handler = RichHandler(rich_tracebacks=True, show_time=False, level=level, show_path=False)
    console_formatter = logging.Formatter('%(message)s') # Simple format for console
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Disable propagation for libraries that might configure their own root logging
    logging.getLogger("urllib3").propagate = False
    logging.getLogger("httpx").propagate = False

    # Return the logger for the current module (optional, as root is configured)
    return logging.getLogger(__name__)

# Initialize logger - this call sets up the initial configuration
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

def validate_config(config_data: Dict) -> bool:
    """Validate the configuration data structure and values.
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if all required providers are present
        for provider in SUPPORTED_PROVIDERS:
            if provider not in config_data:
                log.error(f"Required provider section '{provider}' missing in config file")
                return False
        
        # Validate each provider's configuration
        for provider, provider_config in config_data.items():
            if provider not in SUPPORTED_PROVIDERS:
                log.warning(f"Unknown provider '{provider}' in config file - this provider will be ignored")
                continue
                
            # Required fields
            required_fields = ['api_endpoint', 'model']
            for field in required_fields:
                if field not in provider_config:
                    log.error(f"Required field '{field}' missing in provider '{provider}' config")
                    return False
                    
            # Numeric fields validation
            numeric_fields = {
                'max_tokens': {'min': 1, 'default': DEFAULT_MAX_TOKENS},
                'temperature': {'min': 0, 'max': 1, 'default': DEFAULT_TEMPERATURE},
                'num_samples_per_question': {'min': 1, 'default': DEFAULT_NUM_SAMPLES},
                'max_retries_for_edge_case': {'min': 0, 'default': DEFAULT_MAX_RETRIES_EDGE},
                'random_temp_min': {'min': 0, 'default': DEFAULT_RANDOM_TEMP_MIN},
                'random_temp_max': {'min': 0, 'default': DEFAULT_RANDOM_TEMP_MAX},
                'retry_confirm_threshold': {'min': 0, 'max': 1, 'default': DEFAULT_RETRY_CONFIRM_THRESHOLD},
                'request_delay': {'min': 0, 'default': DEFAULT_REQUEST_DELAY}
            }
            
            for field, constraints in numeric_fields.items():
                if field in provider_config:
                    try:
                        value = float(provider_config[field])
                        
                        # Check minimum value
                        if 'min' in constraints and value < constraints['min']:
                            log.warning(f"Field '{field}' in provider '{provider}' has value {value} below minimum {constraints['min']}. Setting to default.")
                            provider_config[field] = constraints['default']
                            
                        # Check maximum value
                        if 'max' in constraints and value > constraints['max']:
                            log.warning(f"Field '{field}' in provider '{provider}' has value {value} above maximum {constraints['max']}. Setting to default.")
                            provider_config[field] = constraints['default']
                    except ValueError:
                        log.error(f"Field '{field}' in provider '{provider}' has invalid non-numeric value: {provider_config[field]}")
                        provider_config[field] = constraints['default']
            
            # Ensure random_temp_min < random_temp_max
            if ('random_temp_min' in provider_config and 
                'random_temp_max' in provider_config and
                float(provider_config['random_temp_min']) >= float(provider_config['random_temp_max'])):
                log.warning(f"random_temp_min >= random_temp_max in provider '{provider}'. Setting to defaults.")
                provider_config['random_temp_min'] = DEFAULT_RANDOM_TEMP_MIN
                provider_config['random_temp_max'] = DEFAULT_RANDOM_TEMP_MAX
                
            # Validate boolean fields
            bool_fields = ['strip_think_tags', 'retry_edge_cases']
            for field in bool_fields:
                if field in provider_config:
                    if not isinstance(provider_config[field], bool):
                        try:
                            # Try to convert string representations to boolean
                            value = str(provider_config[field]).lower()
                            if value in ('true', 'yes', '1', 'on'):
                                provider_config[field] = True
                            elif value in ('false', 'no', '0', 'off'):
                                provider_config[field] = False
                            else:
                                raise ValueError(f"Cannot convert '{value}' to boolean")
                        except ValueError:
                            log.warning(f"Field '{field}' in provider '{provider}' has invalid non-boolean value. Setting to default.")
                            provider_config[field] = DEFAULT_STRIP_THINK_TAGS if field == 'strip_think_tags' else DEFAULT_RETRY_EDGE_CASES
        
        return True
        
    except Exception as e:
        log.error(f"Unexpected error during config validation: {e}", exc_info=True)
        return False

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
        if not validate_config(config_data):
            raise ValueError("Configuration validation failed.")
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
