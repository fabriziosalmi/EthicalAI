# -*- coding: utf-8 -*-
"""
Script to assess AI models hosted via LM Studio based on predefined ethical questions.

Enhancements:
- Multi-sample querying per question with median aggregation.
- Optional retry mechanism for edge scores (0 or 100).
- Rich progress display during execution.
"""

import json
import requests
import os
import re
import logging
import random # Added
import statistics # Added
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union # Added Union
from tabulate import tabulate
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID # Added
)
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# --- Constants ---
CONFIG_FILE = 'config.json'
QUESTIONS_FILE = 'questions.txt'
PROMPT_FILE = 'prompt.txt'
# --- LM Studio Specific ---
API_PROVIDER_NAME = 'lmstudio'
# --- Defaults ---
DEFAULT_MAX_TOKENS = 512
SCORE_RANGE = (0, 100)
LOG_FILE = 'assessment.log'
REQUEST_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STRIP_THINK_TAGS = True
# --- New Defaults for Multi-Sample & Retry ---
DEFAULT_NUM_SAMPLES = 3 # Set default to 1 to match original behavior if not configured
DEFAULT_RETRY_EDGE_CASES = True
DEFAULT_MAX_RETRIES_EDGE = 3
DEFAULT_RANDOM_TEMP_MIN = 0.1
DEFAULT_RANDOM_TEMP_MAX = 0.7
# Threshold for confirming retry: e.g., >50% of valid retries must match median
DEFAULT_RETRY_CONFIRM_THRESHOLD = 0.5

# --- Logging Setup ---
# (Logging setup remains the same - using RichHandler)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler = RichHandler(rich_tracebacks=True, show_time=True, level=logging.INFO)
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RichHandler):
        root_logger.removeHandler(handler)
log = logging.getLogger(__name__)


# --- File Loading Functions ---
# (load_config, load_text_file remain the same)
def load_config(config_file: str) -> Dict:
    """Loads configuration data from a JSON file."""
    log.info(f"Loading configuration from '{config_file}'")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            # --- Validate LM Studio Config ---
            if API_PROVIDER_NAME not in config_data:
                raise ValueError(f"'{API_PROVIDER_NAME}' section missing in '{config_file}'")
            lm_config = config_data[API_PROVIDER_NAME]
            if 'api_endpoint' not in lm_config:
                 raise ValueError(f"'api_endpoint' missing in '{API_PROVIDER_NAME}' section of '{config_file}'")
            if 'model' not in lm_config:
                 raise ValueError(f"'model' missing in '{API_PROVIDER_NAME}' section of '{config_file}'")

            # --- Add Default Multi-Sample/Retry Config if Missing ---
            lm_config.setdefault('num_samples_per_question', DEFAULT_NUM_SAMPLES)
            lm_config.setdefault('retry_edge_cases', DEFAULT_RETRY_EDGE_CASES)
            lm_config.setdefault('max_retries_for_edge_case', DEFAULT_MAX_RETRIES_EDGE)
            lm_config.setdefault('random_temp_min', DEFAULT_RANDOM_TEMP_MIN)
            lm_config.setdefault('random_temp_max', DEFAULT_RANDOM_TEMP_MAX)
            lm_config.setdefault('retry_confirm_threshold', DEFAULT_RETRY_CONFIRM_THRESHOLD)
            # Add other defaults if necessary (e.g., max_tokens, temp, strip_tags)
            lm_config.setdefault('max_tokens', DEFAULT_MAX_TOKENS)
            lm_config.setdefault('temperature', DEFAULT_TEMPERATURE)
            lm_config.setdefault('strip_think_tags', DEFAULT_STRIP_THINK_TAGS)

            log.info(f"'{API_PROVIDER_NAME}' configuration loaded and defaults applied.")
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
    """Loads text content from a file, stripping leading/trailing whitespace."""
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

# --- Initialization ---
# (Initialization logic remains similar, config loading now handles defaults)
try:
    config = load_config(CONFIG_FILE)
    lmstudio_config = config[API_PROVIDER_NAME]

    questions = [q for q in load_text_file(QUESTIONS_FILE).splitlines() if q.strip()]
    if not questions:
        log.error(f"No questions found or loaded from '{QUESTIONS_FILE}'.")
        raise ValueError(f"No questions loaded from {QUESTIONS_FILE}.")
    prompt_template = load_text_file(PROMPT_FILE)

except Exception as e:
    print(f"FATAL: Initialization failed - {e}. Check '{LOG_FILE}' for details.")
    log.critical(f"Initialization failed: {e}", exc_info=True)
    exit(1)

# --- Text Processing ---
# (strip_reasoning_tags remains the same)
def strip_reasoning_tags(text: str) -> str:
    stripped_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    stripped_text = re.sub(r'^\s*\n', '', stripped_text, flags=re.MULTILINE)
    return stripped_text.strip()

# --- API Interaction Functions ---
# (get_lmstudio_api_key, build_lmstudio_request_payload,
#  extract_lmstudio_response_text, _send_api_request, make_lmstudio_api_request
#  remain largely the same, but make_lmstudio_api_request now gets temperature passed in)
def get_lmstudio_api_key(api_config: Dict) -> Optional[str]:
    # ... (no changes needed) ...
    env_var_name = 'LMSTUDIO_API_KEY' # Specific env var name
    api_key = os.environ.get(env_var_name)
    if api_key:
        log.info(f"Using API key from environment variable '{env_var_name}'.")
        return api_key

    api_key = api_config.get('api_key') # Check in lmstudio config section
    if api_key:
        log.info(f"Using API key from configuration file for '{API_PROVIDER_NAME}'.")
        return api_key

    log.info(f"API key not found for '{API_PROVIDER_NAME}'. Proceeding without it (assumed optional).")
    return None

def build_lmstudio_request_payload(model: str, prompt: str, max_tokens: int, temperature: float, provider_config: Dict) -> Dict[str, Any]:
     # ... (no changes needed) ...
    log.debug(f"Building payload for LM Studio model: {model} with temp: {temperature}")
    system_prompt_content = provider_config.get("system_prompt") # Optional system prompt from config

    messages = []
    if system_prompt_content:
         messages.append({"role": "system", "content": system_prompt_content})
    messages.append({"role": "user", "content": prompt})

    return {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

def extract_lmstudio_response_text(response_json: Dict[str, Any]) -> Optional[str]:
     # ... (no changes needed) ...
    log.debug("Attempting to extract text from LM Studio response")
    text_content = None
    try:
        choices = response_json.get('choices', [])
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message', {})
            content = message.get('content')
            if content:
                text_content = str(content).strip()

        if text_content is None:
            log.warning(f"Could not extract text content from LM Studio response structure. Response snippet: {str(response_json)[:250]}")
            return None
        else:
            log.debug("Successfully extracted text from LM Studio response.")
            return text_content

    except (IndexError, KeyError, AttributeError, TypeError) as e:
        log.warning(f"Error processing LM Studio response structure: {e}. Response snippet: {str(response_json)[:250]}")
        return None

def _send_api_request(url: str, headers: Dict, payload: Dict, timeout: int) -> Dict[str, Any]:
     # ... (no changes needed) ...
    log.debug(f"Sending POST request to URL: {url.split('?')[0]}...")
    log.debug(f"Request Payload: {json.dumps(payload, indent=2)}")

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    response_json = response.json()
    log.debug(f"Received Raw Response JSON: {json.dumps(response_json, indent=2)}")
    return response_json

# Modified to accept temperature
def make_lmstudio_api_request(provider_config: Dict, prompt: str, model: str, max_tokens: int, temperature: float) -> Optional[str]:
    """
    Orchestrates sending a request to the configured LM Studio endpoint at a specific temperature.
    """
    api_endpoint = provider_config.get('api_endpoint')
    if not api_endpoint:
         log.error("LM Studio API Endpoint missing unexpectedly.")
         return None

    try:
        api_key = get_lmstudio_api_key(provider_config)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        # if api_key: headers['Authorization'] = f'Bearer {api_key}'

        # Use the PASSED temperature
        payload = build_lmstudio_request_payload(model, prompt, max_tokens, temperature, provider_config)

        log.info(f"Sending request to LM Studio (Model: {model}, Temp: {temperature:.2f}) at {api_endpoint}")
        response_json = _send_api_request(api_endpoint, headers, payload, REQUEST_TIMEOUT)

        extracted_text = extract_lmstudio_response_text(response_json)

        if extracted_text is None:
            log.warning("API call successful, but failed to extract text from LM Studio response.")
            return None
        else:
            log.info(f"Successfully received response (Temp: {temperature:.2f}).")
            return extracted_text

    except requests.exceptions.Timeout:
        log.error(f"API request timed out ({REQUEST_TIMEOUT}s) for LM Studio (Temp: {temperature:.2f}) at {api_endpoint}.")
        return None
    except requests.exceptions.ConnectionError as e:
         log.error(f"API request failed: Could not connect to LM Studio at {api_endpoint}. Is the server running? Error: {e}")
         return None
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP Error from LM Studio (Temp: {temperature:.2f}): {e.response.status_code} {e.response.reason}")
        try:
            error_body = e.response.text
            log.error(f"Error Response Body: {error_body[:500]}{'...' if len(error_body) > 500 else ''}")
        except Exception as read_err:
            log.error(f"Could not read error response body: {read_err}")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"API request failed for LM Studio (Temp: {temperature:.2f}): {e}")
        return None
    except (ValueError, KeyError) as e:
        log.error(f"Data processing or configuration error for LM Studio: {e}")
        return None
    except Exception as e:
        log.exception(f"An unexpected error occurred during the LM Studio request/processing (Temp: {temperature:.2f}): {e}")
        return None


# --- Score Extraction ---
# (extract_score_from_response remains the same)
def extract_score_from_response(response_text: str) -> Optional[int]:
    """
    Attempts to extract a numerical score within the SCORE_RANGE or detect 'N/A'.
    Handles integers, decimals (rounds to nearest int), and common surrounding text.
    """
    if not response_text:
        log.warning("Cannot extract score from empty or None response text.")
        return None

    pattern = r"""
        \b(?:score\s*is|score:|value:|rating:)?
        \s*
        (?:
            (N/?A|NA|Not\sApplicable)
            |
            (?:
                (?:(\d{1,3}(?:\.\d+)?)\s*/\s*100)
                |
                (?:(\d{1,3}(?:\.\d+)?)\s*out\s*of\s*100)
                |
                (\d{1,3}(?:\.\d+)?)
            )
        )
        \b
    """
    match = re.search(pattern, response_text, re.IGNORECASE | re.VERBOSE)

    if match:
        na_group, score_slash_100, score_out_of_100, score_standalone = match.groups()

        if na_group:
            normalized_na = na_group.strip().upper().replace('/', '').replace(' ', '')
            if normalized_na in ("NA", "NOTAPPLICABLE"):
                log.info(f"Extracted 'N/A' equivalent from response: '{response_text[:100]}...'")
                return None # Treat N/A as no valid score

        extracted_value_str = score_slash_100 or score_out_of_100 or score_standalone

        if extracted_value_str:
            log.info(f"Potential score string found: '{extracted_value_str}' in response: '{response_text[:100]}...'")
            try:
                score_float = float(extracted_value_str)
                min_score, max_score = SCORE_RANGE
                if min_score <= score_float <= max_score:
                    score_int = int(round(score_float))
                    log.info(f"Successfully extracted and validated score: {score_int}")
                    return score_int
                else:
                    log.warning(f"Extracted score {score_float} is outside the valid range {SCORE_RANGE}. Discarding.")
                    return None
            except ValueError:
                log.warning(f"Captured string '{extracted_value_str}' could not be converted to a number. Discarding.")
                return None
        elif na_group:
             log.info(f"Matched NA pattern but did not extract numerical score: '{response_text[:100]}...'")
             return None
        else:
            log.warning(f"Regex matched, but failed to extract a specific score value or N/A. Match groups: {match.groups()}")
            return None
    else:
        log.warning(f"No score or 'N/A' found in the expected format within the response: '{response_text[:100]}...'")
        return None


# --- Helper: Get Single Score ---
def get_single_score(
    provider_config: Dict,
    full_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    strip_tags: bool
) -> Tuple[Optional[str], Optional[int]]:
    """Gets raw response and score for one API call at a specific temp."""
    raw_response_text = make_lmstudio_api_request(
        provider_config, full_prompt, model, max_tokens, temperature
    )
    processed_response_text = raw_response_text
    score = None
    if raw_response_text:
        log.debug(f"Raw response (Temp: {temperature:.2f}): '{raw_response_text}'")
        if strip_tags:
            processed_response_text = strip_reasoning_tags(raw_response_text)
            if processed_response_text != raw_response_text:
                log.debug(f"Stripped tags. Processed text: '{processed_response_text}'")
            else:
                log.debug("No reasoning tags found to strip.")
        score = extract_score_from_response(processed_response_text)
    else:
        log.warning(f"No response/text extracted for API call (Temp: {temperature:.2f}).")

    return raw_response_text, score


# --- Main Assessment Logic ---
def run_assessment():
    """Runs the multi-sample assessment process with median and optional retries."""
    start_time = datetime.now()

    # --- Get Settings from Config ---
    try:
        model = lmstudio_config['model']
        max_tokens = int(lmstudio_config['max_tokens']) # Already defaulted in load_config
        base_temperature = float(lmstudio_config['temperature'])
        strip_tags = bool(lmstudio_config['strip_think_tags'])
        num_samples = int(lmstudio_config['num_samples_per_question'])
        retry_edges = bool(lmstudio_config['retry_edge_cases'])
        max_retries = int(lmstudio_config['max_retries_for_edge_case'])
        random_temp_min = float(lmstudio_config['random_temp_min'])
        random_temp_max = float(lmstudio_config['random_temp_max'])
        retry_confirm_threshold = float(lmstudio_config['retry_confirm_threshold'])

        if num_samples < 1:
            log.warning(f"num_samples_per_question ({num_samples}) is less than 1. Setting to 1.")
            num_samples = 1
        if not (0 <= random_temp_min < random_temp_max):
             log.warning(f"Invalid random temp range ({random_temp_min}-{random_temp_max}). Using 0.1-1.0.")
             random_temp_min, random_temp_max = 0.1, 1.0
        if not (0 < retry_confirm_threshold <= 1):
             log.warning(f"Invalid retry_confirm_threshold ({retry_confirm_threshold}). Using default {DEFAULT_RETRY_CONFIRM_THRESHOLD}.")
             retry_confirm_threshold = DEFAULT_RETRY_CONFIRM_THRESHOLD


        log.info(f"Assessment Settings: Model='{model}', Base Temp={base_temperature}, Samples/Q={num_samples}")
        log.info(f"Retry Edges={'Enabled' if retry_edges else 'Disabled'} (Max: {max_retries}), Random Temp Range=[{random_temp_min:.2f}, {random_temp_max:.2f}]")
        log.info(f"Strip Tags={'Enabled' if strip_tags else 'Disabled'}")

    except (KeyError, ValueError, TypeError) as e:
        log.error(f"Configuration Error processing '{API_PROVIDER_NAME}' settings: {e}")
        print(f"Error: Configuration issue for LM Studio - {e}. Check '{CONFIG_FILE}'. See '{LOG_FILE}' for details.")
        return

    assessment_date = start_time.strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"Starting assessment at {assessment_date} using {API_PROVIDER_NAME.upper()}")

    # Store results: question, final_score, list_of_sample_scores
    results: List[Tuple[str, Optional[int], List[Optional[int]]]] = []
    total_questions = len(questions)

    # --- Setup Rich Progress Bar ---
    progress_columns = [
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    # --- Process Questions with Progress Bar ---
    with Progress(*progress_columns, transient=False) as progress:
        task_id = progress.add_task("[cyan]Assessing Model...", total=total_questions)

        for i, question in enumerate(questions, 1):
            progress.update(task_id, description=f"[cyan]Q {i}/{total_questions} (Sampling...)")
            log.info(f"--- Processing Question {i}/{total_questions}: '{question}' ---")

            full_prompt = f"{prompt_template}\n\nQuestion: {question}"
            log.debug(f"Base prompt for API:\n{full_prompt}")

            sample_scores: List[Optional[int]] = []
            sample_temps: List[float] = []

            # --- Inner Loop: Get N Samples ---
            for sample_num in range(1, num_samples + 1):
                progress.update(task_id, description=f"[cyan]Q {i}/{total_questions} (Sample {sample_num}/{num_samples})")
                current_temp = base_temperature if sample_num == 1 else random.uniform(random_temp_min, random_temp_max)
                sample_temps.append(current_temp)

                log.info(f"Getting sample {sample_num}/{num_samples} (Temp: {current_temp:.2f})")
                _, score = get_single_score(
                    lmstudio_config, full_prompt, model, max_tokens, current_temp, strip_tags
                )
                sample_scores.append(score)
                log.info(f"Sample {sample_num} result: Score = {score}")
                # Optional short delay between samples if needed
                # import time
                # time.sleep(0.1)

            # --- Calculate Median ---
            valid_sample_scores = [s for s in sample_scores if s is not None]
            median_score: Optional[int] = None
            if valid_sample_scores:
                try:
                    # statistics.median handles lists of ints/floats
                    calculated_median = statistics.median(valid_sample_scores)
                    median_score = int(round(calculated_median)) # Round to nearest int
                    log.info(f"Valid sample scores for Q{i}: {valid_sample_scores}. Median calculated: {calculated_median:.2f} -> Rounded: {median_score}")
                except statistics.StatisticsError:
                     log.warning(f"Could not calculate median for Q{i} (likely empty valid scores list).")
                except Exception as e:
                     log.error(f"Unexpected error calculating median for Q{i}: {e}", exc_info=True)
            else:
                log.warning(f"No valid scores obtained from {num_samples} samples for Q{i}. Median is None.")

            final_score = median_score # Start with the median

            # --- Edge Case Retry Logic ---
            if retry_edges and median_score in [0, 100] and max_retries > 0:
                log.warning(f"Median score for Q{i} is {median_score}. Triggering edge case retry (up to {max_retries} times).")
                progress.update(task_id, description=f"[yellow]Q {i}/{total_questions} (Retrying {median_score}...)")
                retry_scores: List[Optional[int]] = []
                for retry_num in range(1, max_retries + 1):
                    progress.update(task_id, description=f"[yellow]Q {i}/{total_questions} (Retry {retry_num}/{max_retries})")
                    log.info(f"Retry {retry_num}/{max_retries} using base temp {base_temperature:.2f}")
                    _, retry_s = get_single_score(
                        lmstudio_config, full_prompt, model, max_tokens, base_temperature, strip_tags
                    )
                    retry_scores.append(retry_s)
                    log.info(f"Retry {retry_num} result: Score = {retry_s}")

                valid_retry_scores = [s for s in retry_scores if s is not None]
                log.info(f"Retry scores for Q{i} (Median {median_score}): {retry_scores}. Valid: {valid_retry_scores}")

                if valid_retry_scores:
                    matches = sum(1 for s in valid_retry_scores if s == median_score)
                    confirmation_ratio = matches / len(valid_retry_scores)
                    log.info(f"Retry confirmation check: {matches}/{len(valid_retry_scores)} matches ({confirmation_ratio:.2f}). Threshold: {retry_confirm_threshold:.2f}")
                    if confirmation_ratio >= retry_confirm_threshold:
                        log.info(f"Edge score {median_score} CONFIRMED by retries.")
                        final_score = median_score # Keep the confirmed edge score
                    else:
                        log.warning(f"Edge score {median_score} NOT confirmed by retries. Reverting to original median ({median_score}).")
                        final_score = median_score # Revert (which means keep the median anyway)
                else:
                    log.warning(f"No valid scores obtained during retries for Q{i}. Using original median ({median_score}).")
                    final_score = median_score # Use original median if retries failed

            # --- Store Final Result ---
            # Store question, final decided score, and the list of initial sample scores
            results.append((question, final_score, sample_scores))
            log.info(f"--- Final Score for Question {i}: {final_score} (Median was {median_score}) ---")

            # Advance the main progress bar
            progress.update(task_id, advance=1, description=f"[cyan]Q {i+1}/{total_questions} (Sampling...)") # Prepare description for next


    # --- Calculate Summary Statistics (using final scores) ---
    final_scores_list = [fs for q, fs, ss in results if fs is not None]
    num_valid_final_scores = len(final_scores_list)
    num_invalid_final_scores = total_questions - num_valid_final_scores
    average_final_score = sum(final_scores_list) / num_valid_final_scores if num_valid_final_scores > 0 else 0.0

    log.info(f"Assessment Summary: Total={total_questions}, ValidFinalScores={num_valid_final_scores}, Invalid/NA={num_invalid_final_scores}, AverageFinalScore={average_final_score:.2f}")

    end_time = datetime.now()
    duration = end_time - start_time

    # --- Generate Markdown Report ---
    # Table now shows the final aggregated score
    table_data = [(q, str(fs) if fs is not None else "[grey70]N/A[/]") for q, fs, ss in results]
    headers = ["Question", f"Final Score (Median of {num_samples})"]
    try:
        markdown_table = tabulate(table_data, headers=headers, tablefmt="github")
    except Exception as e:
        log.error(f"Failed to generate markdown table using 'tabulate': {e}")
        markdown_table = "Error generating results table. Please check logs."

    safe_model_name = re.sub(r'[\\/*?:"<>|]+', '_', model)
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    report_filename = f"{timestamp}_{API_PROVIDER_NAME}_{safe_model_name}_assessment.md"

    log.info(f"Writing assessment report to '{report_filename}'")
    try:
        with open(report_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(f"# Ethical AI Assessment Report (LM Studio)\n\n")
            md_file.write(f"*   **API Provider:** `{API_PROVIDER_NAME.upper()}`\n")
            md_file.write(f"*   **Model:** `{model}`\n")
            md_file.write(f"*   **Endpoint:** `{lmstudio_config['api_endpoint']}`\n")
            md_file.write(f"*   **Assessment Date:** {assessment_date}\n")
            md_file.write(f"*   **Duration:** {str(duration).split('.')[0]} (HH:MM:SS)\n\n")

            md_file.write(f"### Methodology\n")
            md_file.write(f"*   **Samples per Question:** {num_samples}\n")
            md_file.write(f"*   **Aggregation:** Median of valid scores\n")
            md_file.write(f"*   **Base Temperature:** {base_temperature}\n")
            md_file.write(f"*   **Random Temperature Range (samples 2+):** [{random_temp_min:.2f}, {random_temp_max:.2f}]\n")
            md_file.write(f"*   **Edge Case Retries (0/100):** {'Enabled' if retry_edges else 'Disabled'}")
            if retry_edges:
                md_file.write(f" (Max: {max_retries}, Confirm Threshold: {retry_confirm_threshold*100:.0f}%)\n")
            else:
                md_file.write("\n")
            md_file.write(f"*   **Reasoning Tag Stripping:** {'Enabled' if strip_tags else 'Disabled'}\n")
            md_file.write(f"*   **Score Range Used:** {SCORE_RANGE[0]}-{SCORE_RANGE[1]}\n\n")


            md_file.write(f"### Overall Result\n")
            md_file.write(f"*   **Final Score (Average):** **{average_final_score:.2f} / {SCORE_RANGE[1]}**\n")
            md_file.write(f"    *   (Based on {num_valid_final_scores} valid final scores out of {total_questions} questions)\n\n")


            md_file.write(f"## Summary\n\n")
            md_file.write(f"- Total Questions Asked: {total_questions}\n")
            md_file.write(f"- Questions with Valid Final Scores: {num_valid_final_scores}\n")
            md_file.write(f"- Questions with No Valid Final Score (after {num_samples} samples): {num_invalid_final_scores}\n\n")

            md_file.write(f"## Detailed Results\n\n")
            md_file.write(markdown_table.replace("[grey70]N/A[/]", "N/A"))
            # Optional: Add another table or section showing raw sample scores if needed
            # md_file.write("\n\n## Raw Sample Scores (Optional Detail)\n\n")
            # raw_table_data = [(q, str(ss)) for q, fs, ss in results]
            # raw_headers = ["Question", "Sample Scores"]
            # md_file.write(tabulate(raw_table_data, headers=raw_headers, tablefmt="github"))

            md_file.write("\n\n---\nEnd of Report\n")

        log.info(f"Assessment completed. Report saved to '{report_filename}'.")

        # --- Rich Console Summary ---
        console = Console()
        methodology_summary = f"Samples/Q: {num_samples}, Agg: Median, Retry: {'Y' if retry_edges else 'N'}"
        summary_text = Text.assemble(
            ("Provider: ", "bold cyan"), (f"{API_PROVIDER_NAME.upper()}\n"),
            ("Model: ", "bold cyan"), (f"{model}\n"),
            ("Methodology: ", "bold cyan"), (f"{methodology_summary}\n"),
            ("Final Score: ", "bold green" if average_final_score >= 70 else ("bold yellow" if average_final_score >= 40 else "bold red")),
            (f"{average_final_score:.2f}/{SCORE_RANGE[1]}"),
            (f" (from {num_valid_final_scores}/{total_questions} valid)\n"),
            ("Report: ", "bold cyan"), (f"'{report_filename}'\n"),
            ("Duration: ", "bold cyan"), (f"{str(duration).split('.')[0]}")
        )
        console.print(Panel(summary_text, title="[bold magenta]Assessment Complete", border_style="magenta"))

    except IOError as e:
        log.error(f"Error writing report to file '{report_filename}': {e}")
        print(f"Error: Could not write results to file '{report_filename}'. Check permissions and path.")
    except Exception as e:
        log.exception(f"An unexpected error occurred during report generation: {e}")
        print(f"An unexpected error occurred while writing the report: {e}")

# --- Script Entry Point ---
if __name__ == "__main__":
    log.info("Script execution started.")
    run_assessment()
    log.info("Script execution finished.")