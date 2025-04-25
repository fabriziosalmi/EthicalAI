# -*- coding: utf-8 -*-
"""
Ethical AI Assessment Tool

A comprehensive tool to evaluate AI models across multiple providers based on ethical questions.
The tool performs multi-sample querying with robust score aggregation and supports visualization.

Supported providers:
- LM Studio (local inference)
- OpenAI (GPT models)
- Anthropic (Claude models)
- Generic OpenAI-compatible API endpoints
"""

import json
import requests
import os
import re
import logging
import random
import statistics
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from tabulate import tabulate
from pathlib import Path
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID
)
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from collections import defaultdict

# Add imports for HTML/PDF report generation
import markdown
import weasyprint
from jinja2 import Environment, FileSystemLoader, Template
import base64
from io import BytesIO
from pathlib import Path

# --- Constants ---
CONFIG_FILE = 'config.json'
QUESTIONS_FILE = 'questions.txt'
PROMPT_FILE = 'prompt.txt'
RESULTS_DIR = 'results'  # Directory to store results

# --- Provider Names ---
PROVIDER_LMSTUDIO = 'lmstudio'
PROVIDER_OPENAI = 'openai'
PROVIDER_ANTHROPIC = 'anthropic'
PROVIDER_GOOGLE = 'google'  # Add Google Gemini provider
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
ENV_GOOGLE_API_KEY = 'GEMINI_API_KEY'  # Add Google API key environment variable
ENV_GENERIC_API_KEY = 'GENERIC_API_KEY'

# --- Logging Setup ---
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create results directory if it doesn't exist

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
def load_config(config_file: str) -> Dict:
    """Loads configuration data from a JSON file."""
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
try:
    config = load_config(CONFIG_FILE)
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
def strip_reasoning_tags(text: str) -> str:
    stripped_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    stripped_text = re.sub(r'^\s*\n', '', stripped_text, flags=re.MULTILINE)
    return stripped_text.strip()

# --- API Interaction Functions ---
def get_api_key(provider: str, api_config: Dict) -> Optional[str]:
    env_var_name = {
        PROVIDER_LMSTUDIO: ENV_LMSTUDIO_API_KEY,
        PROVIDER_OPENAI: ENV_OPENAI_API_KEY,
        PROVIDER_ANTHROPIC: ENV_ANTHROPIC_API_KEY,
        PROVIDER_GOOGLE: ENV_GOOGLE_API_KEY,  # Add Google API key environment variable
        PROVIDER_GENERIC: ENV_GENERIC_API_KEY
    }.get(provider)
    if env_var_name:
        api_key = os.environ.get(env_var_name)
        if api_key:
            log.info(f"Using API key from environment variable '{env_var_name}' for provider '{provider}'.")
            return api_key

    api_key = api_config.get('api_key')
    if api_key:
        log.info(f"Using API key from configuration file for provider '{provider}'.")
        return api_key

    log.info(f"API key not found for provider '{provider}'. Proceeding without it (assumed optional).")
    return None

def build_request_payload(provider: str, model: str, prompt: str, max_tokens: int, temperature: float, provider_config: Dict) -> Dict[str, Any]:
    log.debug(f"Building payload for provider '{provider}' model: {model} with temp: {temperature}")
    system_prompt_content = provider_config.get("system_prompt")

    if provider == PROVIDER_LMSTUDIO:
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
    elif provider == PROVIDER_OPENAI:
        return {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
    elif provider == PROVIDER_ANTHROPIC:
        return {
            'model': model,
            'prompt': prompt,
            'max_tokens_to_sample': max_tokens,
            'temperature': temperature
        }
    elif provider == PROVIDER_GOOGLE:
        # Format for Google Gemini API
        return {
            'contents': [{
                'parts': [{'text': prompt}]
            }],
            'generationConfig': {
                'temperature': temperature,
                'maxOutputTokens': max_tokens,
                'topP': 0.95,
                'topK': 40
            }
        }
    elif provider == PROVIDER_GENERIC:
        return {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def extract_response_text(provider: str, response_json: Dict[str, Any]) -> Optional[str]:
    log.debug(f"Attempting to extract text from provider '{provider}' response")
    text_content = None
    try:
        if provider == PROVIDER_LMSTUDIO:
            choices = response_json.get('choices', [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                message = choices[0].get('message', {})
                content = message.get('content')
                if content:
                    text_content = str(content).strip()
        elif provider == PROVIDER_OPENAI:
            text_content = response_json.get('choices', [{}])[0].get('text', '').strip()
        elif provider == PROVIDER_ANTHROPIC:
            text_content = response_json.get('completion', '').strip()
        elif provider == PROVIDER_GOOGLE:
            # Extract text from Google Gemini API response format
            candidates = response_json.get('candidates', [])
            if candidates and isinstance(candidates, list) and len(candidates) > 0:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts and isinstance(parts, list) and len(parts) > 0:
                    text_content = parts[0].get('text', '').strip()
        elif provider == PROVIDER_GENERIC:
            text_content = response_json.get('choices', [{}])[0].get('text', '').strip()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if text_content is None:
            log.warning(f"Could not extract text content from provider '{provider}' response structure. Response snippet: {str(response_json)[:250]}")
            return None
        else:
            log.debug(f"Successfully extracted text from provider '{provider}' response.")
            return text_content
    except (IndexError, KeyError, AttributeError, TypeError) as e:
        log.warning(f"Error processing provider '{provider}' response structure: {e}. Response snippet: {str(response_json)[:250]}")
        return None

def _send_api_request(url: str, headers: Dict, payload: Dict, timeout: int) -> Dict[str, Any]:
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

def make_api_request(provider: str, provider_config: Dict, prompt: str, model: str, max_tokens: int, temperature: float) -> Optional[str]:
    api_endpoint = provider_config.get('api_endpoint')
    if not api_endpoint:
        log.error(f"API Endpoint missing unexpectedly for provider '{provider}'.")
        return None

    try:
        api_key = get_api_key(provider, provider_config)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        
        # For Google Gemini, the API key is passed as a query parameter rather than in the header
        if provider == PROVIDER_GOOGLE and api_key:
            if '?' in api_endpoint:
                api_endpoint = f"{api_endpoint}&key={api_key}"
            else:
                api_endpoint = f"{api_endpoint}?key={api_key}"
        elif api_key and api_key.lower() != 'none':
            headers['Authorization'] = f'Bearer {api_key}'

        payload = build_request_payload(provider, model, prompt, max_tokens, temperature, provider_config)

        log.info(f"Sending request to provider '{provider}' (Model: {model}, Temp: {temperature:.2f}) at {api_endpoint}")
        response_json = _send_api_request(api_endpoint, headers, payload, REQUEST_TIMEOUT)

        extracted_text = extract_response_text(provider, response_json)

        if extracted_text is None:
            log.warning(f"API call successful, but failed to extract text from provider '{provider}' response.")
            return None
        else:
            log.info(f"Successfully received response from provider '{provider}' (Temp: {temperature:.2f}).")
            return extracted_text
    except requests.exceptions.Timeout:
        log.error(f"API request timed out ({REQUEST_TIMEOUT}s) for provider '{provider}' (Temp: {temperature:.2f}) at {api_endpoint}.")
        return None
    except requests.exceptions.ConnectionError as e:
        log.error(f"API request failed: Could not connect to provider '{provider}' at {api_endpoint}. Is the server running? Error: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP Error from provider '{provider}' (Temp: {temperature:.2f}): {e.response.status_code} {e.response.reason}")
        try:
            error_body = e.response.text
            log.error(f"Error Response Body: {error_body[:500]}{'...' if len(error_body) > 500 else ''}")
        except Exception as read_err:
            log.error(f"Could not read error response body: {read_err}")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"API request failed for provider '{provider}' (Temp: {temperature:.2f}): {e}")
        return None
    except (ValueError, KeyError) as e:
        log.error(f"Data processing or configuration error for provider '{provider}': {e}")
        return None
    except Exception as e:
        log.exception(f"An unexpected error occurred during the provider '{provider}' request/processing (Temp: {temperature:.2f}): {e}")
        return None

# --- Score Extraction ---
def extract_score_from_response(response_text: str) -> Optional[int]:
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
                return None

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
    provider: str,
    provider_config: Dict,
    full_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    strip_tags: bool
) -> Tuple[Optional[str], Optional[int]]:
    raw_response_text = make_api_request(
        provider, provider_config, full_prompt, model, max_tokens, temperature
    )
    processed_response_text = raw_response_text
    score = None
    if raw_response_text:
        log.debug(f"Raw response from provider '{provider}' (Temp: {temperature:.2f}): '{raw_response_text}'")
        if strip_tags:
            processed_response_text = strip_reasoning_tags(raw_response_text)
            if processed_response_text != raw_response_text:
                log.debug(f"Stripped tags. Processed text: '{processed_response_text}'")
            else:
                log.debug("No reasoning tags found to strip.")
        score = extract_score_from_response(processed_response_text)
    else:
        log.warning(f"No response/text extracted for API call to provider '{provider}' (Temp: {temperature:.2f}).")

    return raw_response_text, score

# --- Main Assessment Logic ---
def run_assessment(provider: str, generate_reports: bool = True):
    """
    Run the assessment for a given provider.
    
    Args:
        provider: Name of the AI provider to assess
        generate_reports: Whether to automatically generate HTML and PDF reports
    """
    start_time = datetime.now()

    try:
        provider_config = config[provider]
        model = provider_config['model']
        max_tokens = int(provider_config['max_tokens'])
        base_temperature = float(provider_config['temperature'])
        strip_tags = bool(provider_config['strip_think_tags'])
        num_samples = int(provider_config['num_samples_per_question'])
        retry_edges = bool(provider_config['retry_edge_cases'])
        max_retries = int(provider_config['max_retries_for_edge_case'])
        random_temp_min = float(provider_config['random_temp_min'])
        random_temp_max = float(provider_config['random_temp_max'])
        retry_confirm_threshold = float(provider_config['retry_confirm_threshold'])
        request_delay = float(provider_config['request_delay'])
        category_mapping = provider_config['category_mapping']

        if num_samples < 1:
            log.warning(f"num_samples_per_question ({num_samples}) is less than 1. Setting to 1.")
            num_samples = 1
        if not (0 <= random_temp_min < random_temp_max):
            log.warning(f"Invalid random temp range ({random_temp_min}-{random_temp_max}). Using 0.1-1.0.")
            random_temp_min, random_temp_max = 0.1, 1.0
        if not (0 < retry_confirm_threshold <= 1):
            log.warning(f"Invalid retry_confirm_threshold ({retry_confirm_threshold}). Using default {DEFAULT_RETRY_CONFIRM_THRESHOLD}.")
            retry_confirm_threshold = DEFAULT_RETRY_CONFIRM_THRESHOLD

        log.info(f"Assessment Settings for provider '{provider}': Model='{model}', Base Temp={base_temperature}, Samples/Q={num_samples}")
        log.info(f"Retry Edges={'Enabled' if retry_edges else 'Disabled'} (Max: {max_retries}), Random Temp Range=[{random_temp_min:.2f}, {random_temp_max:.2f}]")
        log.info(f"Strip Tags={'Enabled' if strip_tags else 'Disabled'}, Request Delay={request_delay}s")

    except (KeyError, ValueError, TypeError) as e:
        log.error(f"Configuration Error processing '{provider}' settings: {e}")
        print(f"Error: Configuration issue for provider '{provider}' - {e}. Check '{CONFIG_FILE}'. See '{LOG_FILE}' for details.")
        return

    assessment_date = start_time.strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"Starting assessment at {assessment_date} using provider '{provider.upper()}'")

    results: List[Tuple[str, Optional[int], List[Optional[int]]]] = []
    total_questions = len(questions)

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

    with Progress(*progress_columns, transient=False) as progress:
        task_id = progress.add_task("[cyan]Assessing Model...", total=total_questions)

        for i, question in enumerate(questions, 1):
            progress.update(task_id, description=f"[cyan]Q {i}/{total_questions} (Sampling...)")
            log.info(f"--- Processing Question {i}/{total_questions}: '{question}' ---")

            full_prompt = f"{prompt_template}\n\nQuestion: {question}"
            log.debug(f"Base prompt for API:\n{full_prompt}")

            sample_scores: List[Optional[int]] = []
            sample_temps: List[float] = []

            for sample_num in range(1, num_samples + 1):
                progress.update(task_id, description=f"[cyan]Q {i}/{total_questions} (Sample {sample_num}/{num_samples})")
                current_temp = base_temperature if sample_num == 1 else random.uniform(random_temp_min, random_temp_max)
                sample_temps.append(current_temp)

                log.info(f"Getting sample {sample_num}/{num_samples} (Temp: {current_temp:.2f})")
                _, score = get_single_score(
                    provider, provider_config, full_prompt, model, max_tokens, current_temp, strip_tags
                )
                sample_scores.append(score)
                log.info(f"Sample {sample_num} result: Score = {score}")
                time.sleep(request_delay)

            valid_sample_scores = [s for s in sample_scores if s is not None]
            median_score: Optional[int] = None
            if valid_sample_scores:
                try:
                    calculated_median = statistics.median(valid_sample_scores)
                    median_score = int(round(calculated_median))
                    log.info(f"Valid sample scores for Q{i}: {valid_sample_scores}. Median calculated: {calculated_median:.2f} -> Rounded: {median_score}")
                except statistics.StatisticsError:
                    log.warning(f"Could not calculate median for Q{i} (likely empty valid scores list).")
                except Exception as e:
                    log.error(f"Unexpected error calculating median for Q{i}: {e}", exc_info=True)
            else:
                log.warning(f"No valid scores obtained from {num_samples} samples for Q{i}. Median is None.")

            final_score = median_score

            if retry_edges and median_score in [0, 100] and max_retries > 0:
                log.warning(f"Median score for Q{i} is {median_score}. Triggering edge case retry (up to {max_retries} times).")
                progress.update(task_id, description=f"[yellow]Q {i}/{total_questions} (Retrying {median_score}...)")
                retry_scores: List[Optional[int]] = []
                for retry_num in range(1, max_retries + 1):
                    progress.update(task_id, description=f"[yellow]Q {i}/{total_questions} (Retry {retry_num}/{max_retries})")
                    log.info(f"Retry {retry_num}/{max_retries} using base temp {base_temperature:.2f}")
                    _, retry_s = get_single_score(
                        provider, provider_config, full_prompt, model, max_tokens, base_temperature, strip_tags
                    )
                    retry_scores.append(retry_s)
                    log.info(f"Retry {retry_num} result: Score = {retry_s}")
                    time.sleep(request_delay)

                valid_retry_scores = [s for s in retry_scores if s is not None]
                log.info(f"Retry scores for Q{i} (Median {median_score}): {retry_scores}. Valid: {valid_retry_scores}")

                if valid_retry_scores:
                    matches = sum(1 for s in valid_retry_scores if s == median_score)
                    confirmation_ratio = matches / len(valid_retry_scores)
                    log.info(f"Retry confirmation check: {matches}/{len(valid_retry_scores)} matches ({confirmation_ratio:.2f}). Threshold: {retry_confirm_threshold:.2f}")
                    if confirmation_ratio >= retry_confirm_threshold:
                        log.info(f"Edge score {median_score} CONFIRMED by retries.")
                        final_score = median_score
                    else:
                        log.warning(f"Edge score {median_score} NOT confirmed by retries. Reverting to original median ({median_score}).")
                        final_score = median_score
                else:
                    log.warning(f"No valid scores obtained during retries for Q{i}. Using original median ({median_score}).")
                    final_score = median_score

            results.append((question, final_score, sample_scores))
            log.info(f"--- Final Score for Question {i}: {final_score} (Median was {median_score}) ---")

            progress.update(task_id, advance=1, description=f"[cyan]Q {i+1}/{total_questions} (Sampling...)")

    final_scores_list = [fs for q, fs, ss in results if fs is not None]
    num_valid_final_scores = len(final_scores_list)
    num_invalid_final_scores = total_questions - num_valid_final_scores
    average_final_score = sum(final_scores_list) / num_valid_final_scores if num_valid_final_scores > 0 else 0.0

    log.info(f"Assessment Summary for provider '{provider}': Total={total_questions}, ValidFinalScores={num_valid_final_scores}, Invalid/NA={num_invalid_final_scores}, AverageFinalScore={average_final_score:.2f}")

    end_time = datetime.now()
    duration = end_time - start_time

    table_data = [(q, str(fs) if fs is not None else "[grey70]N/A[/]") for q, fs, ss in results]
    headers = ["Question", f"Final Score (Median of {num_samples})"]
    try:
        markdown_table = tabulate(table_data, headers=headers, tablefmt="github")
    except Exception as e:
        log.error(f"Failed to generate markdown table using 'tabulate': {e}")
        markdown_table = "Error generating results table. Please check logs."

    safe_model_name = re.sub(r'[\\/*?:"<>|]+', '_', model)
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    report_filename = f"{RESULTS_DIR}/{timestamp}_{provider}_{safe_model_name}_assessment.md"

    log.info(f"Writing assessment report to '{report_filename}'")
    try:
        with open(report_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(f"# Ethical AI Assessment Report ({provider.upper()})\n\n")
            md_file.write(f"*   **API Provider:** `{provider.upper()}`\n")
            md_file.write(f"*   **Model:** `{model}`\n")
            md_file.write(f"*   **Endpoint:** `{provider_config['api_endpoint']}`\n")
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

            md_file.write("\n\n---\nEnd of Report\n")

        log.info(f"Assessment completed for provider '{provider}'. Report saved to '{report_filename}'.")
        
        # Generate visualization charts
        viz_dir = None
        if generate_reports:
            log.info("Generating visualizations...")
            viz_dir = generate_visualizations(provider, results, category_mapping)
        
        # Generate HTML and PDF reports if requested
        if generate_reports:
            log.info("Generating HTML and PDF reports...")
            # Generate HTML report
            html_file = generate_html_report(report_filename, include_charts=True)
            if html_file:
                log.info(f"HTML report generated: {html_file}")
                # Generate PDF report from HTML
                pdf_file = generate_pdf_report(report_filename, html_file)
                if pdf_file:
                    log.info(f"PDF report generated: {pdf_file}")
                else:
                    log.warning("Failed to generate PDF report")
            else:
                log.warning("Failed to generate HTML report")

        console = Console()
        methodology_summary = f"Samples/Q: {num_samples}, Agg: Median, Retry: {'Y' if retry_edges else 'N'}"
        summary_text = Text.assemble(
            ("Provider: ", "bold cyan"), (f"{provider.upper()}\n"),
            ("Model: ", "bold cyan"), (f"{model}\n"),
            ("Methodology: ", "bold cyan"), (f"{methodology_summary}\n"),
            ("Final Score: ", "bold green" if average_final_score >= 70 else ("bold yellow" if average_final_score >= 40 else "bold red")),
            (f"{average_final_score:.2f}/{SCORE_RANGE[1]}"),
            (f" (from {num_valid_final_scores}/{total_questions} valid)\n"),
            ("Report: ", "bold cyan"), (f"'{report_filename}'\n")
        )
        
        # Add info about additional report formats if they were generated
        if generate_reports:
            if html_file:
                summary_text.append(("HTML Report: ", "bold cyan"))
                summary_text.append(f"'{html_file}'\n")
            if pdf_file:
                summary_text.append(("PDF Report: ", "bold cyan"))
                summary_text.append(f"'{pdf_file}'\n")
                
        summary_text.append(("Duration: ", "bold cyan"))
        summary_text.append(f"{str(duration).split('.')[0]}")
                
        console.print(Panel(summary_text, title="[bold magenta]Assessment Complete", border_style="magenta"))

    except IOError as e:
        log.error(f"Error writing report to file '{report_filename}': {e}")
        print(f"Error: Could not write results to file '{report_filename}'. Check permissions and path.")
    except Exception as e:
        log.exception(f"An unexpected error occurred during report generation: {e}")
        print(f"An unexpected error occurred while writing the report: {e}")

def generate_visualizations(provider: str, results: List[Tuple[str, Optional[int], List[Optional[int]]]], category_mapping: Dict[str, List[int]]):
    """
    Generate visualizations for assessment results and save them to the results directory.
    
    Args:
        provider: Name of the AI provider
        results: List of assessment results (question, final score, sample scores)
        category_mapping: Mapping of category names to question indices
    """
    log.info(f"Generating visualizations for provider '{provider}'")
    
    # Create directory for visualizations
    viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract valid scores and their indices
    valid_scores = []
    question_indices = []
    categories = []
    
    for i, (question, score, _) in enumerate(results, 1):
        if score is not None:
            valid_scores.append(score)
            question_indices.append(i)
            
            # Determine category for this question
            category = "Unknown"
            for cat, indices in category_mapping.items():
                if i in indices:
                    category = cat.capitalize()
                    break
            categories.append(category)
    
    if not valid_scores:
        log.warning("No valid scores available for visualization")
        return
    
    # 1. Overall Score Distribution
    plt.figure(figsize=(12, 8))
    plt.hist(valid_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Score Distribution - {provider.upper()}', fontsize=16)
    plt.xlabel('Score (0-100)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    dist_file = os.path.join(viz_dir, f"{timestamp}_{provider}_score_distribution.png")
    plt.savefig(dist_file)
    plt.close()
    
    # 2. Category-wise Average Scores
    category_scores = defaultdict(list)
    for cat, score in zip(categories, valid_scores):
        category_scores[cat].append(score)
    
    avg_category_scores = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}
    
    plt.figure(figsize=(12, 8))
    cats = list(avg_category_scores.keys())
    avgs = list(avg_category_scores.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(cats)))
    bars = plt.bar(cats, avgs, color=colors)
    
    plt.title(f'Average Score by Category - {provider.upper()}', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.ylim(0, 100)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}', ha='center', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    cat_file = os.path.join(viz_dir, f"{timestamp}_{provider}_category_scores.png")
    plt.savefig(cat_file)
    plt.close()
    
    # 3. Question-wise scores (radar chart for categories with sufficient questions)
    for category, indices in category_mapping.items():
        # Get questions that belong to this category and have valid scores
        cat_scores = []
        cat_labels = []
        
        for i, (question, score, _) in enumerate(results, 1):
            if i in indices and score is not None:
                cat_scores.append(score)
                # Extract first few words for label
                words = question.split()
                short_label = ' '.join(words[:3]) + '...' if len(words) > 3 else question
                cat_labels.append(f"Q{i}: {short_label}")
        
        if len(cat_scores) < 3:  # Skip categories with too few questions
            continue
            
        # Create radar chart
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Compute angles for each question
        angles = np.linspace(0, 2*np.pi, len(cat_scores), endpoint=False).tolist()
        
        # Complete the loop
        cat_scores.append(cat_scores[0])
        angles.append(angles[0])
        
        # Plot data
        ax.plot(angles, cat_scores, 'o-', linewidth=2, label=category.capitalize())
        ax.fill(angles, cat_scores, alpha=0.25)
        
        # Set category labels
        ax.set_thetagrids(np.degrees(angles[:-1]), cat_labels, fontsize=8)
        
        # Set radial limits
        ax.set_ylim(0, 100)
        
        # Add title
        plt.title(f'{category.capitalize()} Scores - {provider.upper()}', fontsize=16)
        
        plt.tight_layout()
        radar_file = os.path.join(viz_dir, f"{timestamp}_{provider}_{category}_radar.png")
        plt.savefig(radar_file)
        plt.close()
    
    log.info(f"Visualizations saved to {viz_dir}")
    return viz_dir

def compare_providers(providers: List[str] = None):
    """
    Compare assessment results across multiple providers and generate a comparative report.
    
    Args:
        providers: List of provider names to compare. If None, compares all providers 
                  that have results in the results directory.
    """
    if not providers:
        # Find all provider results in the results directory
        result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.md') and '_assessment.md' in f]
        providers = list(set([f.split('_')[1] for f in result_files if len(f.split('_')) > 2]))
    
    log.info(f"Comparing results across providers: {providers}")
    
    # Find the most recent result file for each provider
    provider_results = {}
    for provider in providers:
        result_files = [f for f in os.listdir(RESULTS_DIR) 
                      if f.endswith('.md') and f'_{provider}_' in f]
        if not result_files:
            log.warning(f"No results found for provider '{provider}'")
            continue
            
        # Get the most recent file based on timestamp in filename
        result_files.sort(reverse=True)
        most_recent = result_files[0]
        provider_results[provider] = os.path.join(RESULTS_DIR, most_recent)
    
    if len(provider_results) < 2:
        log.error("Need at least two providers with results for comparison")
        print("Error: Need at least two providers with results for comparison")
        return
    
    # Extract scores from each provider's results
    provider_scores = {}
    for provider, result_file in provider_results.items():
        scores = extract_scores_from_report(result_file)
        if scores:
            provider_scores[provider] = scores
    
    # Generate comparison visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_dir = os.path.join(RESULTS_DIR, 'comparisons', timestamp)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate comparative bar chart for average scores
    plt.figure(figsize=(12, 8))
    providers_list = list(provider_scores.keys())
    avg_scores = [scores.get('average', 0) for scores in provider_scores.values()]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(providers_list)))
    bars = plt.bar(providers_list, avg_scores, color=colors)
    
    plt.title('Ethical AI Assessment - Provider Comparison', fontsize=16)
    plt.xlabel('Provider', fontsize=14)
    plt.ylabel('Average Score (0-100)', fontsize=14)
    plt.ylim(0, 100)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}', ha='center', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    comparison_file = os.path.join(comparison_dir, "provider_comparison_overall.png")
    plt.savefig(comparison_file)
    plt.close()
    
    # Generate radar chart comparing providers across categories
    if any('categories' in scores for scores in provider_scores.values()):
        # Get all categories
        all_categories = set()
        for scores in provider_scores.values():
            if 'categories' in scores:
                all_categories.update(scores['categories'].keys())
        
        categories_list = sorted(list(all_categories))
        if len(categories_list) >= 3:  # Need at least 3 categories for radar chart
            plt.figure(figsize=(12, 10))
            ax = plt.subplot(111, polar=True)
            
            # Compute angles for each category
            angles = np.linspace(0, 2*np.pi, len(categories_list), endpoint=False).tolist()
            
            # Complete the loop
            categories_list.append(categories_list[0])
            angles.append(angles[0])
            
            # Plot each provider
            for i, (provider, scores) in enumerate(provider_scores.items()):
                if 'categories' not in scores:
                    continue
                    
                values = [scores['categories'].get(cat, 0) for cat in categories_list[:-1]]
                values.append(values[0])  # Complete the loop
                
                color = plt.cm.viridis(i/len(provider_scores))
                ax.plot(angles, values, 'o-', linewidth=2, label=provider.upper(), color=color)
                ax.fill(angles, values, alpha=0.1, color=color)
            
            # Set category labels
            plt.xticks(angles[:-1], [c.capitalize() for c in categories_list[:-1]], fontsize=12)
            
            # Set radial limits
            plt.ylim(0, 100)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title('Ethical AI Assessment - Category Comparison', fontsize=16)
            plt.tight_layout()
            radar_file = os.path.join(comparison_dir, "provider_comparison_radar.png")
            plt.savefig(radar_file)
            plt.close()
    
    # Generate markdown comparison report
    report_filename = os.path.join(comparison_dir, "providers_comparison_report.md")
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("# Ethical AI Assessment - Provider Comparison Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Scores\n\n")
        overall_table = [
            ["Provider", "Average Score", "Valid Question Count", "Assessment Date"]
        ]
        
        for provider, scores in provider_scores.items():
            assessment_date = scores.get('assessment_date', 'Unknown')
            overall_table.append([
                provider.upper(),
                f"{scores.get('average', 0):.2f}",
                str(scores.get('valid_count', 0)),
                assessment_date
            ])
        
        f.write(tabulate(overall_table, headers="firstrow", tablefmt="pipe"))
        f.write("\n\n")
        
        # Add comparison charts
        f.write("## Visual Comparisons\n\n")
        f.write(f"![Overall Comparison](provider_comparison_overall.png)\n\n")
        
        if os.path.exists(os.path.join(comparison_dir, "provider_comparison_radar.png")):
            f.write(f"![Category Comparison](provider_comparison_radar.png)\n\n")
        
        # Add category-specific comparisons
        if any('categories' in scores for scores in provider_scores.values()):
            f.write("## Category Scores\n\n")
            
            for category in sorted(all_categories):
                f.write(f"### {category.capitalize()}\n\n")
                
                category_table = [["Provider", f"{category.capitalize()} Score"]]
                for provider, scores in provider_scores.items():
                    if 'categories' in scores:
                        score = scores['categories'].get(category, 'N/A')
                        if isinstance(score, (int, float)):
                            score = f"{score:.2f}"
                        category_table.append([provider.upper(), score])
                
                f.write(tabulate(category_table, headers="firstrow", tablefmt="pipe"))
                f.write("\n\n")
        
        f.write("## Detailed Question Comparisons\n\n")
        # For each provider, list top 5 highest and lowest scoring questions
        for provider, scores in provider_scores.items():
            f.write(f"### {provider.upper()} Highlights\n\n")
            
            if 'questions' in scores:
                # Sort questions by score
                sorted_questions = sorted(
                    [(q, s) for q, s in scores['questions'].items() if s is not None],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if sorted_questions:
                    f.write("#### Top 5 Highest Scores\n\n")
                    top_table = [["Question", "Score"]]
                    for q, s in sorted_questions[:5]:
                        top_table.append([q[:100] + "..." if len(q) > 100 else q, str(s)])
                    f.write(tabulate(top_table, headers="firstrow", tablefmt="pipe"))
                    f.write("\n\n")
                    
                    f.write("#### 5 Lowest Scores\n\n")
                    bottom_table = [["Question", "Score"]]
                    for q, s in sorted_questions[-5:]:
                        bottom_table.append([q[:100] + "..." if len(q) > 100 else q, str(s)])
                    f.write(tabulate(bottom_table, headers="firstrow", tablefmt="pipe"))
                    f.write("\n\n")
            
            else:
                f.write("No detailed question scores available.\n\n")
    
    log.info(f"Comparison report generated at {report_filename}")
    print(f"Comparison report generated at {report_filename}")
    
    return report_filename

def extract_scores_from_report(report_file: str) -> Dict:
    """
    Extract scores and metadata from an assessment report file.
    
    Args:
        report_file: Path to the assessment report markdown file
        
    Returns:
        Dictionary containing extracted scores and metadata
    """
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        result = {
            'average': 0.0,
            'valid_count': 0,
            'assessment_date': 'Unknown',
            'questions': {},
            'categories': {}
        }
        
        # Extract assessment date
        date_match = re.search(r'\*\*Assessment Date:\*\* (.*)', content)
        if date_match:
            result['assessment_date'] = date_match.group(1).strip()
            
        # Extract average score
        avg_match = re.search(r'\*\*Final Score \(Average\):\*\* \*\*([\d.]+)', content)
        if avg_match:
            try:
                result['average'] = float(avg_match.group(1))
            except ValueError:
                pass
                
        # Extract valid count
        valid_match = re.search(r'Based on (\d+) valid final scores', content)
        if valid_match:
            try:
                result['valid_count'] = int(valid_match.group(1))
            except ValueError:
                pass
                
        # Extract question scores
        # Look for the markdown table with questions and scores
        table_match = re.search(r'## Detailed Results\s+\n\s*(.*?)\n\n', content, re.DOTALL)
        if table_match:
            table_content = table_match.group(1)
            table_lines = [line.strip() for line in table_content.strip().split('\n') if '|' in line]
            
            # Skip header and separator lines
            for line in table_lines[2:]:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 3:
                    question = parts[1].strip()
                    score_text = parts[2].strip()
                    
                    # Convert score to number or None for N/A
                    if score_text.lower() == 'n/a':
                        score = None
                    else:
                        try:
                            score = int(score_text)
                        except ValueError:
                            try:
                                score = float(score_text)
                            except ValueError:
                                score = None
                    
                    if question:
                        result['questions'][question] = score
            
        # Calculate category averages if we have question scores and category mapping
        if result['questions'] and 'category_mapping' in config:
            category_mapping = config['category_mapping']
            # Invert mapping to get question number -> category
            question_to_category = {}
            for category, question_indices in category_mapping.items():
                for idx in question_indices:
                    question_to_category[idx] = category
            
            # Group scores by category
            category_scores = defaultdict(list)
            for i, (question, score) in enumerate(result['questions'].items(), 1):
                if i in question_to_category and score is not None:
                    category = question_to_category[i]
                    category_scores[category].append(score)
            
            # Calculate average for each category
            for category, scores in category_scores.items():
                if scores:
                    result['categories'][category] = sum(scores) / len(scores)
        
        return result
        
    except Exception as e:
        log.error(f"Error extracting scores from report {report_file}: {e}", exc_info=True)
        return None

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ethical AI Assessment Tool")
    parser.add_argument('--provider', type=str, choices=SUPPORTED_PROVIDERS, help="Specify the AI provider to assess")
    parser.add_argument('--model', type=str, help="Override the model specified in config.json")
    parser.add_argument('--api-endpoint', type=str, help="Override the API endpoint specified in config.json")
    parser.add_argument('--max-tokens', type=int, help="Override max_tokens in config.json")
    parser.add_argument('--temperature', type=float, help="Override base temperature in config.json")
    parser.add_argument('--samples', type=int, help="Override number of samples per question")
    parser.add_argument('--timeout', type=int, help=f"API request timeout in seconds (default: {REQUEST_TIMEOUT})")
    parser.add_argument('--no-retry-edges', action='store_true', help="Disable retry mechanism for edge scores (0 or 100)")
    parser.add_argument('--request-delay', type=float, help="Delay between API requests in seconds (default: 0)")
    parser.add_argument('--compare', action='store_true', help="Compare results across multiple providers")
    return parser.parse_args()

def main():
    """Main entry point for the assessment tool."""
    log.info("Script execution started.")
    
    args = parse_arguments()
    
    # Determine which provider to use
    active_provider = args.provider if args.provider else config.get('active_provider', PROVIDER_LMSTUDIO)
    
    if active_provider not in SUPPORTED_PROVIDERS:
        log.error(f"Unsupported provider specified: {active_provider}")
        print(f"Error: Unsupported provider '{active_provider}'. Supported providers are: {', '.join(SUPPORTED_PROVIDERS)}")
        exit(1)

    provider_config = config[active_provider]

    if args.model:
        log.info(f"Overriding model from command line: {args.model}")
        provider_config['model'] = args.model
    
    if args.api_endpoint:
        log.info(f"Overriding API endpoint from command line: {args.api_endpoint}")
        provider_config['api_endpoint'] = args.api_endpoint
    
    if args.max_tokens:
        log.info(f"Overriding max_tokens from command line: {args.max_tokens}")
        provider_config['max_tokens'] = args.max_tokens
    
    if args.temperature is not None:
        log.info(f"Overriding temperature from command line: {args.temperature}")
        provider_config['temperature'] = args.temperature
    
    if args.samples:
        log.info(f"Overriding samples per question from command line: {args.samples}")
        provider_config['num_samples_per_question'] = args.samples
    
    # Store the custom timeout if provided
    custom_timeout = REQUEST_TIMEOUT
    if args.timeout:
        log.info(f"Overriding request timeout from command line: {args.timeout}")
        custom_timeout = args.timeout
    
    if args.no_retry_edges:
        log.info("Disabling edge case retry mechanism from command line")
        provider_config['retry_edge_cases'] = False
    
    if args.request_delay is not None:
        log.info(f"Overriding request delay from command line: {args.request_delay}")
        provider_config['request_delay'] = args.request_delay
    
    # Create a custom API request function that uses our timeout
    def custom_send_api_request(url, headers, payload, _):
        """Custom API request function that uses our specific timeout."""
        return requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=custom_timeout
        ).json()
    
    # Store the original function to restore it later
    original_send_api_request = globals()['_send_api_request']
    
    try:
        # Replace the global function with our custom one
        globals()['_send_api_request'] = custom_send_api_request
        
        if args.compare:
            compare_providers()
        else:
            run_assessment(active_provider)
    finally:
        # Restore the original function
        globals()['_send_api_request'] = original_send_api_request
    
    log.info("Script execution finished.")

if __name__ == "__main__":
    main()

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# HTML template for the assessment report
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --accent-color: #e74c3c;
            --background-color: #f8f9fa;
            --card-bg-color: #ffffff;
            --text-color: #2c3e50;
            --border-color: #e0e0e0;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background-color: var(--card-bg-color);
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
        
        h2 {
            color: var(--secondary-color);
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        h3 {
            color: var (--secondary-color);
            margin-top: 1.5rem;
        }
        
        .metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .metadata-item {
            background-color: var(--card-bg-color);
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }
        
        .metadata-item strong {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--secondary-color);
        }
        
        .methodology {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .method-item {
            background-color: var(--card-bg-color);
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }
        
        .method-item strong {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--secondary-color);
        }
        
        .highlight-box {
            background-color: var(--primary-color);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            margin: 2rem 0;
        }
        
        .highlight-box h3 {
            color: white;
            margin-top: 0;
        }
        
        .score {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        
        .score-high {
            color: var(--success-color);
        }
        
        .score-medium {
            color: var(--warning-color);
        }
        
        .score-low {
            color: var(--danger-color);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
        }
        
        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background-color: var(--primary-color);
            color: white;
        }
        
        tr:nth-child(even) {
            background-color: rgba(0, 0, 0, 0.02);
        }
        
        tr:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .summary-item {
            background-color: var(--card-bg-color);
            padding: 1.5rem;
            border-radius: 6px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .summary-item strong {
            display: block;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: var(--primary-color);
        }
        
        .summary-item span {
            color: var(--secondary-color);
        }
        
        .charts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .chart {
            background-color: var(--card-bg-color);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            text-align: center;
        }
        
        .chart img {
            max-width: 100%;
            height: auto;
        }
        
        footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            color: var(--secondary-color);
            font-size: 0.9rem;
        }
        
        @media print {
            body {
                background-color: white;
            }
            
            .container {
                max-width: 100%;
                margin: 0;
                padding: 1rem;
                box-shadow: none;
            }
            
            @page {
                margin: 1.5cm;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p>Generated on {{ current_date }}</p>
        </header>
        
        <div class="metadata">
            <div class="metadata-item">
                <strong>API Provider:</strong>
                {{ provider }}
            </div>
            <div class="metadata-item">
                <strong>Model:</strong>
                {{ model }}
            </div>
            <div class="metadata-item">
                <strong>Assessment Date:</strong>
                {{ assessment_date }}
            </div>
            <div class="metadata-item">
                <strong>Duration:</strong>
                {{ duration }}
            </div>
        </div>
        
        <h2>Methodology</h2>
        <div class="methodology">
            <div class="method-item">
                <strong>Samples per Question:</strong>
                {{ samples_per_question }}
            </div>
            <div class="method-item">
                <strong>Aggregation:</strong>
                {{ aggregation }}
            </div>
            <div class="method-item">
                <strong>Base Temperature:</strong>
                {{ base_temperature }}
            </div>
            <div class="method-item">
                <strong>Temp Range (samples 2+):</strong>
                {{ temp_range }}
            </div>
            <div class="method-item">
                <strong>Edge Case Retries:</strong>
                {{ edge_case_retries }}
            </div>
            <div class="method-item">
                <strong>Reasoning Tag Stripping:</strong>
                {{ tag_stripping }}
            </div>
            <div class="method-item">
                <strong>Score Range:</strong>
                {{ score_range }}
            </div>
        </div>
        
        <div class="highlight-box">
            <h3>Overall Result</h3>
            <div class="score {{ score_class }}">{{ average_score }} / {{ max_score }}</div>
            <p>Based on {{ valid_scores }} valid final scores out of {{ total_questions }} questions</p>
        </div>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-item">
                <strong>{{ total_questions }}</strong>
                <span>Total Questions Asked</span>
            </div>
            <div class="summary-item">
                <strong>{{ valid_scores }}</strong>
                <span>Questions with Valid Scores</span>
            </div>
            <div class="summary-item">
                <strong>{{ invalid_scores }}</strong>
                <span>Questions without Valid Scores</span>
            </div>
        </div>
        
        {% if charts %}
        <h2>Visualizations</h2>
        <div class="charts">
            {% for chart in charts %}
            <div class="chart">
                <h3>{{ chart.title }}</h3>
                <img src="{{ chart.src }}" alt="{{ chart.title }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Final Score (Median of {{ samples_per_question }})</th>
                </tr>
            </thead>
            <tbody>
                {% for question, score in results %}
                <tr>
                    <td>{{ question }}</td>
                    <td>{{ score }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <footer>
            <p>Generated by Ethical AI Assessment Tool | {{ current_date }}</p>
        </footer>
    </div>
</body>
</html>
"""

def generate_html_report(markdown_file: str, include_charts: bool = True) -> str:
    """
    Generate an HTML report from a markdown assessment report.
    
    Args:
        markdown_file: Path to the markdown assessment report
        include_charts: Whether to include charts in the HTML report
        
    Returns:
        Path to the generated HTML report
    """
    try:
        log.info(f"Generating HTML report from {markdown_file}")
        
        # Extract data from markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse basic information
        provider_match = re.search(r'\*\*API Provider:\*\* `([^`]+)`', content)
        model_match = re.search(r'\*\*Model:\*\* `([^`]+)`', content)
        date_match = re.search(r'\*\*Assessment Date:\*\* (.*)', content)
        duration_match = re.search(r'\*\*Duration:\*\* (.*)', content)
        
        # Parse methodology
        samples_match = re.search(r'\*\*Samples per Question:\*\* (\d+)', content)
        temp_match = re.search(r'\*\*Base Temperature:\*\* ([0-9.]+)', content)
        temp_range_match = re.search(r'\*\*Random Temperature Range \(samples 2\+\):\*\* \[([^]]+)\]', content)
        retries_match = re.search(r'\*\*Edge Case Retries \(0/100\):\*\* ([^(]+)', content)
        strip_match = re.search(r'\*\*Reasoning Tag Stripping:\*\* (Enabled|Disabled)', content)
        range_match = re.search(r'\*\*Score Range Used:\*\* ([0-9]+-[0-9]+)', content)
        
        # Parse scores
        avg_score_match = re.search(r'\*\*Final Score \(Average\):\*\* \*\*([0-9.]+)', content)
        valid_scores_match = re.search(r'Based on (\d+) valid final scores out of (\d+) questions', content)
        
        # Parse table with questions and scores
        results = []
        table_pattern = r'\| (.*?) \| (.*?) \|'
        table_matches = re.findall(table_pattern, content)
        for q, s in table_matches:
            if q != "Question" and q != "-":  # Skip header and separator
                results.append((q, s))
        
        # Prepare data for template
        if provider_match:
            provider = provider_match.group(1)
        else:
            provider = "Unknown"
            
        if model_match:
            model = model_match.group(1)
        else:
            model = "Unknown"
        
        title = f"Ethical AI Assessment Report - {provider}"
        
        template_data = {
            "title": title,
            "current_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "provider": provider,
            "model": model,
            "assessment_date": date_match.group(1) if date_match else "Unknown",
            "duration": duration_match.group(1) if duration_match else "Unknown",
            "samples_per_question": samples_match.group(1) if samples_match else "Unknown",
            "aggregation": "Median of valid scores",
            "base_temperature": temp_match.group(1) if temp_match else "Unknown",
            "temp_range": temp_range_match.group(1) if temp_range_match else "Unknown",
            "edge_case_retries": retries_match.group(1) if retries_match else "Unknown",
            "tag_stripping": strip_match.group(1) if strip_match else "Unknown",
            "score_range": range_match.group(1) if range_match else "Unknown",
            "results": results
        }
        
        # Handle scores and set appropriate styling class
        if avg_score_match and valid_scores_match:
            avg_score = float(avg_score_match.group(1))
            valid_count = int(valid_scores_match.group(1))
            total_count = int(valid_scores_match.group(2))
            
            template_data["average_score"] = f"{avg_score:.2f}"
            template_data["valid_scores"] = valid_count
            template_data["total_questions"] = total_count
            template_data["invalid_scores"] = total_count - valid_count
            template_data["max_score"] = "100"
            
            # Set score class for styling
            if avg_score >= 70:
                template_data["score_class"] = "score-high"
            elif avg_score >= 40:
                template_data["score_class"] = "score-medium"
            else:
                template_data["score_class"] = "score-low"
        else:
            template_data["average_score"] = "0.00"
            template_data["valid_scores"] = 0
            template_data["total_questions"] = len(results)
            template_data["invalid_scores"] = len(results)
            template_data["max_score"] = "100"
            template_data["score_class"] = "score-low"
        
        # Include charts if requested
        if include_charts:
            # Look for visualization charts in the directory
            viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
            if os.path.exists(viz_dir):
                chart_files = [f for f in os.listdir(viz_dir) if f.endswith('.png') and provider.lower() in f.lower()]
                
                # If we have charts, include them in the template
                if chart_files:
                    charts = []
                    for chart_file in chart_files:
                        chart_path = os.path.join(viz_dir, chart_file)
                        # Convert image to base64 to embed in HTML
                        with open(chart_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # Determine title from filename
                        if 'distribution' in chart_file:
                            title = 'Score Distribution'
                        elif 'category_scores' in chart_file:
                            title = 'Category Scores'
                        elif 'radar' in chart_file:
                            category = chart_file.split('_')[-2] if len(chart_file.split('_')) > 3 else 'Category'
                            title = f'{category.capitalize()} Scores'
                        else:
                            title = 'Chart'
                        
                        charts.append({
                            'title': title,
                            'src': f"data:image/png;base64,{img_data}"
                        })
                    
                    template_data["charts"] = charts
                else:
                    template_data["charts"] = []
            else:
                template_data["charts"] = []
        else:
            template_data["charts"] = []
        
        # Render template
        template = Template(HTML_TEMPLATE)
        html_content = template.render(**template_data)
        
        # Create output filepath
        html_file = markdown_file.replace('.md', '.html')
        
        # Write HTML to file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        log.info(f"HTML report generated at {html_file}")
        return html_file
    
    except Exception as e:
        log.error(f"Error generating HTML report: {e}", exc_info=True)
        return None

def generate_pdf_report(markdown_file: str, html_file: str = None) -> str:
    """
    Generate a PDF report from a markdown or HTML assessment report.
    
    Args:
        markdown_file: Path to the markdown assessment report
        html_file: Path to an HTML file to use instead of generating from markdown
        
    Returns:
        Path to the generated PDF report
    """
    try:
        log.info(f"Generating PDF report from {'HTML file' if html_file else 'markdown file'}")
        
        # If no HTML file provided, generate one
        if not html_file:
            html_file = generate_html_report(markdown_file)
            if not html_file:
                log.error("Failed to generate HTML report for PDF conversion")
                return None
        
        # Create output filepath
        pdf_file = markdown_file.replace('.md', '.pdf')
        
        # Generate PDF from HTML using WeasyPrint
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        html = weasyprint.HTML(string=html_content)
        css = weasyprint.CSS(string='@page { size: A4; margin: 1cm; }')
        html.write_pdf(pdf_file, stylesheets=[css])
        
        log.info(f"PDF report generated at {pdf_file}")
        return pdf_file
    
    except Exception as e:
        log.error(f"Error generating PDF report: {e}", exc_info=True)
        return None