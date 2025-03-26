# -*- coding: utf-8 -*-
"""
Script to assess AI models hosted via LM Studio based on predefined ethical questions.

It loads questions from a file, queries the configured LM Studio API endpoint,
optionally strips reasoning tags (<think>...</think>), attempts to extract a
numerical score from the remaining response, and generates a markdown report.
"""

import json
import requests
import os
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from tabulate import tabulate # For markdown table generation

# --- Constants ---
CONFIG_FILE = 'config.json'
QUESTIONS_FILE = 'questions.txt'
PROMPT_FILE = 'prompt.txt' # Ensure this file contains full instructions, including score format request.
# --- LM Studio Specific ---
API_PROVIDER_NAME = 'lmstudio' # Hardcoded as we only support LM Studio now
# --- Defaults ---
DEFAULT_MAX_TOKENS = 512
SCORE_RANGE = (0, 100)
LOG_FILE = 'assessment.log'
REQUEST_TIMEOUT = 120 # Increased timeout for potentially slower local models
DEFAULT_TEMPERATURE = 0.0
DEFAULT_STRIP_THINK_TAGS = True

# --- Logging Setup ---
# (Logging setup remains the same)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


# --- File Loading Functions ---
# (load_config and load_text_file remain the same)
def load_config(config_file: str) -> Dict:
    """Loads configuration data from a JSON file."""
    logging.info(f"Loading configuration from '{config_file}'")
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
            logging.info(f"'{API_PROVIDER_NAME}' configuration found.")
            return config_data
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file '{config_file}': {e}")
        raise
    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading config '{config_file}': {e}")
        raise

def load_text_file(filepath: str) -> str:
    """Loads text content from a file, stripping leading/trailing whitespace."""
    logging.info(f"Loading text file from '{filepath}'")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            logging.info(f"Text file '{filepath}' loaded successfully.")
            return content
    except FileNotFoundError:
        logging.error(f"Required text file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading text file '{filepath}': {e}")
        raise


# --- Initialization ---
try:
    # Load config and implicitly validate LM Studio section via load_config
    config = load_config(CONFIG_FILE)
    lmstudio_config = config[API_PROVIDER_NAME] # Get LM Studio specific config

    questions = [q for q in load_text_file(QUESTIONS_FILE).splitlines() if q.strip()]
    if not questions:
        logging.error(f"No questions found or loaded from '{QUESTIONS_FILE}'.")
        raise ValueError(f"No questions loaded from {QUESTIONS_FILE}.")
    prompt_template = load_text_file(PROMPT_FILE)

except Exception as e:
    print(f"FATAL: Initialization failed - {e}. Check '{LOG_FILE}' for details.")
    logging.critical(f"Initialization failed: {e}", exc_info=True)
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
        logging.info(f"Using API key from environment variable '{env_var_name}'.")
        return api_key

    api_key = api_config.get('api_key') # Check in lmstudio config section
    if api_key:
        logging.info(f"Using API key from configuration file for '{API_PROVIDER_NAME}'.")
        return api_key

    logging.info(f"API key not found for '{API_PROVIDER_NAME}'. Proceeding without it (assumed optional).")
    return None


def build_lmstudio_request_payload(model: str, prompt: str, max_tokens: int, temperature: float, provider_config: Dict) -> Dict[str, Any]:
    """Constructs the Chat Completions payload for LM Studio."""
    logging.debug(f"Building payload for LM Studio model: {model}")
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
        'stream': False
    }


def extract_lmstudio_response_text(response_json: Dict[str, Any]) -> Optional[str]:
    """Extracts text content from LM Studio's Chat Completions response."""
    logging.debug("Attempting to extract text from LM Studio response")
    text_content = None
    try:
        choices = response_json.get('choices', [])
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message', {})
            content = message.get('content')
            if content:
                text_content = str(content).strip()

        if text_content is None:
            logging.warning(f"Could not extract text content from LM Studio response structure. Response snippet: {str(response_json)[:250]}")
            return None
        else:
            logging.debug("Successfully extracted text from LM Studio response.")
            return text_content

    except (IndexError, KeyError, AttributeError, TypeError) as e:
        logging.warning(f"Error processing LM Studio response structure: {e}. Response snippet: {str(response_json)[:250]}")
        return None


def _send_api_request(url: str, headers: Dict, payload: Dict, timeout: int) -> Dict[str, Any]:
    """Internal helper to send the POST request and handle HTTP/Request exceptions."""
    logging.debug(f"Sending POST request to URL: {url.split('?')[0]}...")
    logging.debug(f"Request Payload: {json.dumps(payload)}")

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    response_json = response.json()
    logging.debug(f"Received Raw Response JSON: {json.dumps(response_json)}")
    return response_json


def make_lmstudio_api_request(provider_config: Dict, prompt: str, model: str, max_tokens: int, temperature: float) -> Optional[str]:
    """
    Orchestrates sending a request to the configured LM Studio endpoint.

    Args:
        provider_config: Configuration dictionary for LM Studio (must contain 'api_endpoint').
        prompt: The input prompt.
        model: The model name specified in config (may be ignored by LM Studio if only one loaded).
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        The extracted text response, or None on failure.
    """
    api_endpoint = provider_config.get('api_endpoint') # Already validated in load_config
    if not api_endpoint: # Should not happen due to validation, but belt-and-suspenders
         logging.error("LM Studio API Endpoint missing unexpectedly.")
         return None

    try:
        # 1. Get API Key (Optional for LM Studio)
        api_key = get_lmstudio_api_key(provider_config)

        # 2. Prepare Headers
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        # LM Studio typically doesn't use Bearer token unless specifically configured
        # if api_key:
        #     headers['Authorization'] = f'Bearer {api_key}' # Uncomment if your LM Studio needs auth

        # 3. Build Payload
        payload = build_lmstudio_request_payload(model, prompt, max_tokens, temperature, provider_config)

        # 4. Send Request
        logging.info(f"Sending request to LM Studio (Model: {model}) at {api_endpoint}")
        response_json = _send_api_request(api_endpoint, headers, payload, REQUEST_TIMEOUT)

        # 5. Extract Text
        extracted_text = extract_lmstudio_response_text(response_json)

        if extracted_text is None:
            logging.warning("API call successful, but failed to extract text from LM Studio response.")
            return None
        else:
            logging.info("Successfully received and extracted response from LM Studio.")
            return extracted_text

    except requests.exceptions.Timeout:
        logging.error(f"API request timed out ({REQUEST_TIMEOUT}s) for LM Studio at {api_endpoint}.")
        return None
    except requests.exceptions.ConnectionError as e:
         logging.error(f"API request failed: Could not connect to LM Studio at {api_endpoint}. Is the server running? Error: {e}")
         return None
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error from LM Studio: {e.response.status_code} {e.response.reason}")
        try:
            error_body = e.response.text
            logging.error(f"Error Response Body: {error_body[:500]}{'...' if len(error_body) > 500 else ''}")
        except Exception as read_err:
            logging.error(f"Could not read error response body: {read_err}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for LM Studio: {e}")
        return None
    except (ValueError, KeyError) as e: # Catch errors from payload building, response parsing etc.
        logging.error(f"Data processing or configuration error for LM Studio: {e}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred during the LM Studio request/processing: {e}")
        return None


# --- Score Extraction ---
# (extract_score_from_response remains the same as it processes the final text)
def extract_score_from_response(response_text: str) -> Optional[int]:
    """
    Attempts to extract a numerical score within the SCORE_RANGE or detect 'N/A'.
    Handles integers, decimals (rounds to nearest int), and common surrounding text.

    Args:
        response_text: The text response from the AI model (potentially pre-processed).

    Returns:
        An integer score if found and valid, otherwise None.
    """
    if not response_text:
        logging.warning("Cannot extract score from empty or None response text.")
        return None

    # Regex from previous version (robust)
    pattern = r"""
        \b(?:score\s*is|score:|value:|rating:)?       # Optional preceding phrases
        \s*                                          # Optional whitespace
        (?:                                          # Group for different score formats
            (N/?A|NA)                                # Capture N/A or NA (Group 1)
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
            normalized_na = na_group.strip().upper().replace('/', '')
            if normalized_na == "NA":
                logging.info(f"Extracted 'N/A' from response: '{response_text[:100]}...'")
                return None

        extracted_value_str = score_slash_100 or score_out_of_100 or score_standalone

        if extracted_value_str:
            logging.info(f"Potential score string found: '{extracted_value_str}' in response: '{response_text[:100]}...'")
            try:
                score_float = float(extracted_value_str)
                min_score, max_score = SCORE_RANGE
                if min_score <= score_float <= max_score:
                    score_int = int(round(score_float))
                    logging.info(f"Successfully extracted and validated score: {score_int}")
                    return score_int
                else:
                    logging.warning(f"Extracted score {score_float} is outside the valid range {SCORE_RANGE}. Discarding.")
                    return None
            except ValueError:
                logging.warning(f"Captured string '{extracted_value_str}' could not be converted to a number. Discarding.")
                return None
        else:
            logging.warning(f"Regex matched, but failed to extract a specific score value or N/A. Match groups: {match.groups()}")
            return None
    else:
        logging.warning(f"No score or 'N/A' found in the expected format within the response: '{response_text[:100]}...'")
        return None


# --- Main Assessment Logic ---
def run_assessment():
    """Runs the full assessment process using the configured LM Studio endpoint."""
    start_time = datetime.now()

    # --- Get LM Studio Settings from Config ---
    try:
        # lmstudio_config was already retrieved and validated during initialization
        model = lmstudio_config['model'] # Required key, validated in load_config
        # Use config values or defaults if not present
        max_tokens = int(lmstudio_config.get('max_tokens', DEFAULT_MAX_TOKENS))
        temperature = float(lmstudio_config.get('temperature', DEFAULT_TEMPERATURE))
        # --- Get strip_tags setting ---
        strip_tags_config = lmstudio_config.get('strip_think_tags', DEFAULT_STRIP_THINK_TAGS)
        # Handle potential env override (optional)
        strip_tags_env = os.environ.get('STRIP_THINK_TAGS')
        if strip_tags_env is not None:
            strip_tags = strip_tags_env.lower() in ['true', '1', 'yes']
        else:
            strip_tags = strip_tags_config

        logging.info(f"Using LM Studio Model: '{model}', Max Tokens: {max_tokens}, Temperature: {temperature}")
        if strip_tags:
            logging.info("Reasoning tag stripping (<think>...</think>) is ENABLED.")
        else:
            logging.info("Reasoning tag stripping (<think>...</think>) is DISABLED.")

    except (KeyError, ValueError, TypeError) as e:
        logging.error(f"Configuration Error in '{API_PROVIDER_NAME}' section: {e}")
        print(f"Error: Configuration issue for LM Studio - {e}. Check '{CONFIG_FILE}'. See '{LOG_FILE}' for details.")
        return

    assessment_date = start_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Starting assessment at {assessment_date} using {API_PROVIDER_NAME.upper()}")

    results: List[Tuple[str, Optional[int]]] = []

    # --- Process Questions ---
    total_questions = len(questions)
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{total_questions}: {question[:80]}...")
        logging.info(f"Processing question {i}/{total_questions}: '{question}'")

        full_prompt = f"{prompt_template}\n\nQuestion: {question}"
        logging.debug(f"Full prompt for API:\n{full_prompt}")

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
            logging.debug(f"Raw response text received for Q{i}: '{raw_response_text}'")
            # --- Optionally strip tags ---
            if strip_tags:
                processed_response_text = strip_reasoning_tags(raw_response_text)
                if processed_response_text != raw_response_text:
                     logging.info(f"Stripped reasoning tags for Q{i}.")
                     logging.debug(f"Processed response text for Q{i}: '{processed_response_text}'")
                else:
                     logging.debug(f"No reasoning tags found to strip for Q{i}.")

            # Extract score from the (potentially processed) text
            score = extract_score_from_response(processed_response_text)
        else:
            logging.warning(f"No response received or text extracted for question {i}. Score will be None.")

        results.append((question, score))
        logging.info(f"Result for question {i}: Score = {score}")
        # Optional delay
        # import time
        # time.sleep(0.5) # Shorter delay likely okay for local model

    # --- Calculate Summary Statistics ---
    # (Calculation logic remains the same)
    valid_scores = [s for q, s in results if s is not None]
    num_valid_responses = len(valid_scores)
    num_invalid_responses = total_questions - num_valid_responses
    average_score = sum(valid_scores) / num_valid_responses if num_valid_responses > 0 else 0.0

    logging.info(f"Assessment Summary: Total={total_questions}, ValidScores={num_valid_responses}, Invalid/NA={num_invalid_responses}, AverageScore={average_score:.2f}")

    end_time = datetime.now()
    duration = end_time - start_time

    # --- Generate Markdown Report ---
    # (Report generation logic remains largely the same)
    table_data = [(q, str(s) if s is not None else "N/A") for q, s in results]
    headers = ["Question", "Score"]
    try:
        markdown_table = tabulate(table_data, headers=headers, tablefmt="github")
    except Exception as e:
        logging.error(f"Failed to generate markdown table using 'tabulate': {e}")
        markdown_table = "Error generating results table. Please check logs."

    safe_model_name = re.sub(r'[\\/*?:"<>|]+', '_', model)
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    # Filename now hardcodes 'lmstudio' as the provider
    report_filename = f"{timestamp}_{API_PROVIDER_NAME}_{safe_model_name}_assessment.md"

    logging.info(f"Writing assessment report to '{report_filename}'")
    try:
        with open(report_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(f"# Ethical AI Assessment Report (LM Studio)\n\n") # Updated title
            md_file.write(f"*   **API Provider:** `{API_PROVIDER_NAME.upper()}`\n")
            md_file.write(f"*   **Model:** `{model}` (as configured)\n")
            md_file.write(f"*   **Endpoint:** `{lmstudio_config['api_endpoint']}`\n") # Added endpoint info
            md_file.write(f"*   **Assessment Date:** {assessment_date}\n")
            md_file.write(f"*   **Duration:** {str(duration).split('.')[0]} (HH:MM:SS)\n")
            md_file.write(f"*   **Score Range Used:** {SCORE_RANGE[0]}-{SCORE_RANGE[1]}\n")
            md_file.write(f"*   **Temperature Used:** {temperature}\n")
            md_file.write(f"*   **Reasoning Tag Stripping:** {'Enabled' if strip_tags else 'Disabled'}\n") # Added strip setting
            md_file.write(f"*   **Final Score (Average):** {average_score:.2f} / {SCORE_RANGE[1]}\n")
            md_file.write(f"    *   (Based on {num_valid_responses} valid scores out of {total_questions} questions)\n\n")

            md_file.write(f"## Summary\n\n")
            md_file.write(f"- Total Questions Asked: {total_questions}\n")
            md_file.write(f"- Valid Numerical Scores Received: {num_valid_responses}\n")
            md_file.write(f"- Invalid / N/A / No Response: {num_invalid_responses}\n\n")

            md_file.write(f"## Detailed Results\n\n")
            md_file.write(markdown_table)
            md_file.write("\n\n---\nEnd of Report\n")

        logging.info(f"Assessment completed. Report saved to '{report_filename}'.")
        print("\n--- Assessment Complete ---")
        print(f"Provider: {API_PROVIDER_NAME.upper()}")
        print(f"Model: {model}")
        print(f"Endpoint: {lmstudio_config['api_endpoint']}")
        print(f"Final Score: {average_score:.2f}/{SCORE_RANGE[1]} (from {num_valid_responses}/{total_questions} valid responses)")
        print(f"Report saved to: '{report_filename}'")
        print(f"Duration: {str(duration).split('.')[0]}")
        print("-------------------------")

    except IOError as e:
        logging.error(f"Error writing report to file '{report_filename}': {e}")
        print(f"Error: Could not write results to file '{report_filename}'. Check permissions and path.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during report generation: {e}")
        print(f"An unexpected error occurred while writing the report: {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    logging.info("Script execution started.")
    run_assessment()
    logging.info("Script execution finished.")