# -*- coding: utf-8 -*-
"""
Script to assess AI models hosted via LM Studio based on predefined ethical questions.

It loads questions from a file, queries the configured LM Studio API endpoint,
optionally strips reasoning tags (<think>...</think>), attempts to extract a
numerical score from the remaining response, and generates a markdown report
with a rich progress display during execution.
"""

import json
import requests
import os
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from tabulate import tabulate # For markdown table generation
# --- Added for Rich Progress ---
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.logging import RichHandler # Optional: For richer logging output to console
# -----------------------------

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

# --- Logging Setup ---
# Configure root logger for file output
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Added %(name)s
)

# Configure RichHandler for console output (shows INFO and above by default)
# Set level higher (e.g., ERROR) if you want less console noise from INFO/DEBUG
console_handler = RichHandler(
    rich_tracebacks=True, # Enable rich tracebacks
    show_time=True,       # Show time in console logs
    level=logging.INFO    # Or logging.ERROR to reduce verbosity
)
formatter = logging.Formatter('%(message)s') # Keep console format simple
console_handler.setFormatter(formatter)

# Add RichHandler to the root logger
logging.getLogger().addHandler(console_handler)
# Remove default StreamHandler if it exists to avoid duplicate console output
# Note: This assumes the default handler is a StreamHandler. If not, adjust.
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RichHandler):
        root_logger.removeHandler(handler)

# Create a specific logger for this script if needed (optional)
log = logging.getLogger(__name__) # Use script's module name


# --- File Loading Functions ---
def load_config(config_file: str) -> Dict:
    """Loads configuration data from a JSON file."""
    log.info(f"Loading configuration from '{config_file}'")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            # --- Validate LM Studio Config ---
            if API_PROVIDER_NAME not in config_data:
                raise ValueError(f"'{API_PROVIDER_NAME}' section missing in '{config_file}'")
            if 'api_endpoint' not in config_data[API_PROVIDER_NAME]:
                 raise ValueError(f"'api_endpoint' missing in '{API_PROVIDER_NAME}' section of '{config_file}'")
            if 'model' not in config_data[API_PROVIDER_NAME]:
                 raise ValueError(f"'model' missing in '{API_PROVIDER_NAME}' section of '{config_file}'")
            log.info(f"'{API_PROVIDER_NAME}' configuration found.")
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
    # Load config and implicitly validate LM Studio section via load_config
    config = load_config(CONFIG_FILE)
    lmstudio_config = config[API_PROVIDER_NAME] # Get LM Studio specific config

    questions = [q for q in load_text_file(QUESTIONS_FILE).splitlines() if q.strip()]
    if not questions:
        log.error(f"No questions found or loaded from '{QUESTIONS_FILE}'.")
        raise ValueError(f"No questions loaded from {QUESTIONS_FILE}.")
    prompt_template = load_text_file(PROMPT_FILE)

except Exception as e:
    # Using print here because logging might not be fully set up if init fails early
    print(f"FATAL: Initialization failed - {e}. Check '{LOG_FILE}' for details.")
    log.critical(f"Initialization failed: {e}", exc_info=True)
    exit(1)


# --- Text Processing ---
def strip_reasoning_tags(text: str) -> str:
    """Removes <think>...</think> blocks from the text."""
    # Use re.DOTALL to make '.' match newline characters
    stripped_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Also remove potential empty lines left after stripping
    stripped_text = re.sub(r'^\s*\n', '', stripped_text, flags=re.MULTILINE)
    return stripped_text.strip()


# --- API Interaction Functions (Simplified for LM Studio OpenAI-compatible API) ---

def get_lmstudio_api_key(api_config: Dict) -> Optional[str]:
    """
    Retrieves the API key for LM Studio if configured.
    Priority: Environment Variable (LMSTUDIO_API_KEY) > Config File.
    Returns None if not found, as it's often not required for local LM Studio.
    """
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
    """Constructs the Chat Completions payload for LM Studio."""
    log.debug(f"Building payload for LM Studio model: {model}")
    system_prompt_content = provider_config.get("system_prompt") # Optional system prompt from config

    messages = []
    if system_prompt_content:
         messages.append({"role": "system", "content": system_prompt_content})
    messages.append({"role": "user", "content": prompt})

    # Note: LM Studio might ignore the 'model' field if only one model is loaded,
    # but we include it for compatibility/clarity. It should match the loaded model name or be ignored.
    return {
        'model': model, # Model name expected by LM Studio (or can be arbitrary if only one loaded)
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False # Ensure streaming is off for single response
    }


def extract_lmstudio_response_text(response_json: Dict[str, Any]) -> Optional[str]:
    """Extracts text content from LM Studio's Chat Completions response."""
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
    """Internal helper to send the POST request and handle HTTP/Request exceptions."""
    log.debug(f"Sending POST request to URL: {url.split('?')[0]}...")
    log.debug(f"Request Payload: {json.dumps(payload, indent=2)}") # Indent for better log readability

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    response_json = response.json()
    log.debug(f"Received Raw Response JSON: {json.dumps(response_json, indent=2)}")
    return response_json


def make_lmstudio_api_request(provider_config: Dict, prompt: str, model: str, max_tokens: int, temperature: float) -> Optional[str]:
    """
    Orchestrates sending a request to the configured LM Studio endpoint.
    Args/Returns remain the same.
    """
    api_endpoint = provider_config.get('api_endpoint')
    if not api_endpoint:
         log.error("LM Studio API Endpoint missing unexpectedly.")
         return None

    try:
        api_key = get_lmstudio_api_key(provider_config)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        # if api_key: headers['Authorization'] = f'Bearer {api_key}' # Uncomment if needed

        payload = build_lmstudio_request_payload(model, prompt, max_tokens, temperature, provider_config)

        log.info(f"Sending request to LM Studio (Model: {model}) at {api_endpoint}")
        response_json = _send_api_request(api_endpoint, headers, payload, REQUEST_TIMEOUT)

        extracted_text = extract_lmstudio_response_text(response_json)

        if extracted_text is None:
            log.warning("API call successful, but failed to extract text from LM Studio response.")
            return None
        else:
            log.info("Successfully received and extracted response from LM Studio.")
            return extracted_text

    except requests.exceptions.Timeout:
        log.error(f"API request timed out ({REQUEST_TIMEOUT}s) for LM Studio at {api_endpoint}.")
        return None
    except requests.exceptions.ConnectionError as e:
         log.error(f"API request failed: Could not connect to LM Studio at {api_endpoint}. Is the server running? Error: {e}")
         return None
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP Error from LM Studio: {e.response.status_code} {e.response.reason}")
        try:
            error_body = e.response.text
            log.error(f"Error Response Body: {error_body[:500]}{'...' if len(error_body) > 500 else ''}")
        except Exception as read_err:
            log.error(f"Could not read error response body: {read_err}")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"API request failed for LM Studio: {e}")
        return None
    except (ValueError, KeyError) as e:
        log.error(f"Data processing or configuration error for LM Studio: {e}")
        return None
    except Exception as e:
        log.exception(f"An unexpected error occurred during the LM Studio request/processing: {e}") # Use log.exception for tracebacks
        return None


# --- Score Extraction ---
def extract_score_from_response(response_text: str) -> Optional[int]:
    """
    Attempts to extract a numerical score within the SCORE_RANGE or detect 'N/A'.
    Handles integers, decimals (rounds to nearest int), and common surrounding text.
    """
    if not response_text:
        log.warning("Cannot extract score from empty or None response text.")
        return None

    # Using the more robust regex pattern
    pattern = r"""
        \b(?:score\s*is|score:|value:|rating:)?       # Optional preceding phrases
        \s*                                          # Optional whitespace
        (?:                                          # Group for different score formats
            (N/?A|NA|Not\sApplicable)                # Capture N/A, NA, or "Not Applicable" (Group 1)
            |                                        # OR
            (?:                                      # Group for numerical scores
                (?:(\d{1,3}(?:\.\d+)?)\s*/\s*100)     # Capture score like X/100 or X.Y/100 (Group 2)
                |                                    # OR
                (?:(\d{1,3}(?:\.\d+)?)\s*out\s*of\s*100) # Capture score like X out of 100 (Group 3)
                |                                    # OR
                (\d{1,3}(?:\.\d+)?)                  # Capture standalone number X or X.Y (Group 4)
            )
        )
        \b                                           # Word boundary
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
             # This case handles if NA was matched but logic above didn't return None
             log.info(f"Matched NA pattern but did not extract numerical score: '{response_text[:100]}...'")
             return None
        else:
            # This case should theoretically not be reached if regex matches, but good to have
            log.warning(f"Regex matched, but failed to extract a specific score value or N/A. Match groups: {match.groups()}")
            return None
    else:
        log.warning(f"No score or 'N/A' found in the expected format within the response: '{response_text[:100]}...'")
        return None


# --- Main Assessment Logic ---
def run_assessment():
    """Runs the full assessment process using the configured LM Studio endpoint with rich progress."""
    start_time = datetime.now()

    # --- Get LM Studio Settings from Config ---
    try:
        model = lmstudio_config['model']
        max_tokens = int(lmstudio_config.get('max_tokens', DEFAULT_MAX_TOKENS))
        temperature = float(lmstudio_config.get('temperature', DEFAULT_TEMPERATURE))
        strip_tags_config = lmstudio_config.get('strip_think_tags', DEFAULT_STRIP_THINK_TAGS)
        strip_tags_env = os.environ.get('STRIP_THINK_TAGS')
        if strip_tags_env is not None:
            strip_tags = strip_tags_env.lower() in ['true', '1', 'yes']
        else:
            strip_tags = strip_tags_config

        log.info(f"Using LM Studio Model: '{model}', Max Tokens: {max_tokens}, Temperature: {temperature}")
        log.info(f"Reasoning tag stripping (<think>...</think>) is {'ENABLED' if strip_tags else 'DISABLED'}.")

    except (KeyError, ValueError, TypeError) as e:
        log.error(f"Configuration Error in '{API_PROVIDER_NAME}' section: {e}")
        print(f"Error: Configuration issue for LM Studio - {e}. Check '{CONFIG_FILE}'. See '{LOG_FILE}' for details.")
        return

    assessment_date = start_time.strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"Starting assessment at {assessment_date} using {API_PROVIDER_NAME.upper()}")

    results: List[Tuple[str, Optional[int]]] = []
    total_questions = len(questions)

    # --- Setup Rich Progress Bar ---
    progress_columns = [
        SpinnerColumn(spinner_name="dots"), # Choose a spinner style
        TextColumn("[progress.description]{task.description}", justify="right"),
        BarColumn(bar_width=None), # Auto-width bar
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    # --- Process Questions with Progress Bar ---
    with Progress(*progress_columns, transient=False) as progress: # transient=False keeps bar after completion
        task_id = progress.add_task("[cyan]Assessing Model...", total=total_questions)

        for i, question in enumerate(questions, 1):
            # Update progress description for the current question
            progress.update(task_id, description=f"[cyan]Q {i}/{total_questions}")
            log.info(f"Processing question {i}/{total_questions}: '{question}'") # Keep logging detailed info

            full_prompt = f"{prompt_template}\n\nQuestion: {question}"
            log.debug(f"Full prompt for API:\n{full_prompt}")

            # Get response from LM Studio
            raw_response_text = make_lmstudio_api_request(
                lmstudio_config,
                full_prompt,
                model,
                max_tokens,
                temperature
            )

            processed_response_text = raw_response_text
            score = None
            if raw_response_text:
                log.debug(f"Raw response text received for Q{i}: '{raw_response_text}'")
                if strip_tags:
                    processed_response_text = strip_reasoning_tags(raw_response_text)
                    if processed_response_text != raw_response_text:
                         log.info(f"Stripped reasoning tags for Q{i}.")
                         log.debug(f"Processed response text for Q{i}: '{processed_response_text}'")
                    else:
                         log.debug(f"No reasoning tags found to strip for Q{i}.")

                score = extract_score_from_response(processed_response_text)
            else:
                log.warning(f"No response received or text extracted for question {i}. Score will be None.")

            results.append((question, score))
            log.info(f"Result for question {i}: Score = {score}")

            # Advance the progress bar
            progress.update(task_id, advance=1)

            # Optional delay
            # import time
            # time.sleep(0.2) # Smaller delay likely fine

    # --- Calculate Summary Statistics ---
    valid_scores = [s for q, s in results if s is not None]
    num_valid_responses = len(valid_scores)
    num_invalid_responses = total_questions - num_valid_responses
    average_score = sum(valid_scores) / num_valid_responses if num_valid_responses > 0 else 0.0

    log.info(f"Assessment Summary: Total={total_questions}, ValidScores={num_valid_responses}, Invalid/NA={num_invalid_responses}, AverageScore={average_score:.2f}")

    end_time = datetime.now()
    duration = end_time - start_time

    # --- Generate Markdown Report ---
    table_data = [(q, str(s) if s is not None else "[grey70]N/A[/]") for q, s in results] # Use Rich markup for N/A
    headers = ["Question", "Score (0-100)"] # Clarify score range
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
            md_file.write(f"*   **Model:** `{model}` (as configured)\n")
            md_file.write(f"*   **Endpoint:** `{lmstudio_config['api_endpoint']}`\n")
            md_file.write(f"*   **Assessment Date:** {assessment_date}\n")
            md_file.write(f"*   **Duration:** {str(duration).split('.')[0]} (HH:MM:SS)\n")
            md_file.write(f"*   **Score Range Used:** {SCORE_RANGE[0]}-{SCORE_RANGE[1]}\n")
            md_file.write(f"*   **Temperature Used:** {temperature}\n")
            md_file.write(f"*   **Reasoning Tag Stripping:** {'Enabled' if strip_tags else 'Disabled'}\n")
            md_file.write(f"*   **Final Score (Average):** **{average_score:.2f} / {SCORE_RANGE[1]}**\n") # Bold score
            md_file.write(f"    *   (Based on {num_valid_responses} valid scores out of {total_questions} questions)\n\n")

            md_file.write(f"## Summary\n\n")
            md_file.write(f"- Total Questions Asked: {total_questions}\n")
            md_file.write(f"- Valid Numerical Scores Received: {num_valid_responses}\n")
            md_file.write(f"- Invalid / N/A / No Response: {num_invalid_responses}\n\n")

            md_file.write(f"## Detailed Results\n\n")
            md_file.write(markdown_table.replace("[grey70]N/A[/]", "N/A")) # Replace rich markup for plain markdown
            md_file.write("\n\n---\nEnd of Report\n")

        log.info(f"Assessment completed. Report saved to '{report_filename}'.")
        # --- Use Rich Console for final output ---
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        summary_text = Text.assemble(
            ("Provider: ", "bold cyan"), (f"{API_PROVIDER_NAME.upper()}\n"),
            ("Model: ", "bold cyan"), (f"{model}\n"),
            ("Endpoint: ", "bold cyan"), (f"{lmstudio_config['api_endpoint']}\n"),
            ("Final Score: ", "bold green" if average_score >= 70 else ("bold yellow" if average_score >= 40 else "bold red")),
            (f"{average_score:.2f}/{SCORE_RANGE[1]}"),
            (f" (from {num_valid_responses}/{total_questions} valid responses)\n"),
            ("Report saved to: ", "bold cyan"), (f"'{report_filename}'\n"),
            ("Duration: ", "bold cyan"), (f"{str(duration).split('.')[0]}")
        )
        console.print(Panel(summary_text, title="[bold magenta]Assessment Complete", border_style="magenta"))
        # -----------------------------------------

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