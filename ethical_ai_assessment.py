# -*- coding: utf-8 -*-
"""
Script to assess AI models based on predefined ethical questions.

It loads questions from a file, queries a configured AI API (OpenAI, Groq,
AI Studio, Ollama, LMStudio), attempts to extract a numerical score from the
response, and generates a markdown report summarizing the results.

Improvements:
- More robust score extraction (handles N/A, decimals, surrounding text).
- Refactored API interaction for better separation of concerns.
- Added SCORE_RANGE to the report output.
- Minor cleanup and consistency improvements.
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
DEFAULT_API_PROVIDER = 'openai' # Default provider if not specified
DEFAULT_MAX_TOKENS = 150      # Slightly increased default token limit
SCORE_RANGE = (0, 100)         # Expected range for valid scores
LOG_FILE = 'assessment.log'    # Log file name
REQUEST_TIMEOUT = 60           # Timeout for API requests in seconds
DEFAULT_TEMPERATURE = 0.0      # Default temperature for deterministic output

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
# Console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


# --- File Loading Functions ---
def load_config(config_file: str) -> Dict:
    """Loads configuration data from a JSON file."""
    logging.info(f"Loading configuration from '{config_file}'")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            logging.info("Configuration loaded successfully.")
            return config_data
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file '{config_file}': {e}")
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
    config = load_config(CONFIG_FILE)
    questions = [q for q in load_text_file(QUESTIONS_FILE).splitlines() if q.strip()]
    if not questions:
        logging.error(f"No questions found or loaded from '{QUESTIONS_FILE}'.")
        raise ValueError(f"No questions loaded from {QUESTIONS_FILE}.")
    # Ensure prompt.txt contains instructions for the desired score format (e.g., "Respond with score X/100 or N/A")
    prompt_template = load_text_file(PROMPT_FILE)
except Exception as e:
    print(f"FATAL: Initialization failed - {e}. Check '{LOG_FILE}' for details.")
    logging.critical(f"Initialization failed: {e}", exc_info=True)
    exit(1)


# --- API Interaction Functions ---

def get_api_key(api_provider: str, api_config: Dict) -> Optional[str]:
    """
    Retrieves the API key for the given provider if required.
    Priority: Environment Variable > Config File.
    Returns None if the provider doesn't typically require a key or if not found when not strictly required.

    Args:
        api_provider: The name of the API provider (lowercase).
        api_config: The configuration dictionary specific to this provider.

    Returns:
        The API key string, or None.

    Raises:
        ValueError: If an API key *is strictly* required (e.g., OpenAI, Groq, AI Studio) but cannot be found.
    """
    providers_requiring_key = ['openai', 'groq', 'aistudio'] # Add others if needed
    key_required = api_provider in providers_requiring_key

    env_var_name = f'{api_provider.upper()}_API_KEY'
    api_key = os.environ.get(env_var_name)
    if api_key:
        logging.info(f"Using API key from environment variable '{env_var_name}'.")
        return api_key

    api_key = api_config.get('api_key')
    if api_key:
        logging.info(f"Using API key from configuration file for '{api_provider}'.")
        return api_key

    # If we reach here and a key was strictly required, raise an error
    if key_required:
        error_msg = f"API Key for '{api_provider}' is required but not found in env var '{env_var_name}' or in '{CONFIG_FILE}'."
        logging.error(error_msg)
        raise ValueError(error_msg)
    else:
        # For providers like local Ollama/LMStudio, a key might not be needed by default
        logging.info(f"API key not found for '{api_provider}', proceeding without it (may be optional).")
        return None


def build_request_payload(api_provider: str, model: str, prompt: str, max_tokens: int, temperature: float, provider_config: Dict) -> Dict[str, Any]:
    """
    Constructs the appropriate request payload JSON for the specified API provider.

    Args:
        api_provider: Name of the API provider.
        model: Specific model to use.
        prompt: Input prompt for the AI model.
        max_tokens: Maximum number of tokens for the response.
        temperature: Sampling temperature.
        provider_config: Configuration specific to the provider (used for optional system prompt).


    Returns:
        A dictionary representing the request payload.

    Raises:
        ValueError: If the api_provider is not supported.
    """
    logging.debug(f"Building payload for provider: {api_provider}, model: {model}")
    system_prompt_content = provider_config.get("system_prompt") # Optional system prompt from config

    # --- OpenAI / Groq / LMStudio (Chat Completions format) ---
    if api_provider in ['openai', 'groq', 'lmstudio']:
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
    # --- Google AI Studio (Gemini API v1beta) ---
    elif api_provider == 'aistudio':
        # Note: Gemini API structure might change. This targets v1beta models/generateContent
        # System instructions can sometimes be placed before the user content.
        contents = []
        if system_prompt_content:
             # Gemini doesn't have a dedicated 'system' role in the basic structure,
             # often prepended or handled differently based on client libraries/usage.
             # Here, we'll just prepend it as text in the user part, or you could adjust based on specific Gemini best practices.
             # A common pattern is to make the *first* turn the system instruction.
             # If prompt_template includes placeholders for system/user, adapt accordingly.
             logging.warning("System prompt handling for AI Studio might need specific formatting.")
             # Simple approach: prepend to user prompt (might not be ideal)
             prompt = f"{system_prompt_content}\n\n{prompt}"

        contents.append({
             "role": "user",
             "parts": [{"text": prompt}]
        })
        return {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "candidateCount": 1 # Ensure only one response candidate
            }
        }
    # --- Ollama (Generate API) ---
    elif api_provider == 'ollama':
        # Ollama's '/api/generate' endpoint
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        if system_prompt_content:
             payload["system"] = system_prompt_content # Add system prompt if provided

        # Note: Ollama also has a '/api/chat' endpoint which uses 'messages' format like OpenAI.
        # Consider switching to that for consistency if needed.
        return payload
    else:
        logging.error(f"Payload construction failed: Unsupported API provider '{api_provider}'.")
        raise ValueError(f"Unsupported API provider for payload construction: {api_provider}")


def extract_response_text(api_provider: str, response_json: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the relevant text content from the API's JSON response.

    Args:
        api_provider: Name of the API provider.
        response_json: Parsed JSON response from the API.

    Returns:
        The extracted text content as a string, stripped of whitespace, or None if extraction fails.

    Raises:
        ValueError: If the api_provider's response structure isn't handled.
    """
    logging.debug(f"Attempting to extract text from response for provider: {api_provider}")
    text_content = None
    try:
        # --- OpenAI / Groq / LMStudio (Chat Completions format) ---
        if api_provider in ['openai', 'groq', 'lmstudio']:
            choices = response_json.get('choices', [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                message = choices[0].get('message', {})
                content = message.get('content')
                if content:
                    text_content = str(content).strip()

        # --- Google AI Studio (Gemini API v1beta - generateContent) ---
        elif api_provider == 'aistudio':
            candidates = response_json.get('candidates', [])
            if candidates and isinstance(candidates, list) and len(candidates) > 0:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts and isinstance(parts, list) and len(parts) > 0:
                    text = parts[0].get('text')
                    if text:
                        text_content = str(text).strip()

        # --- Ollama (Generate API) ---
        elif api_provider == 'ollama':
            response_text = response_json.get('response') # The 'response' field holds the generated text
            if response_text:
                text_content = str(response_text).strip()

        # --- Provider not handled ---
        else:
            logging.error(f"Response text extraction logic not implemented for provider '{api_provider}'.")
            # Raise ValueError for unhandled providers to make it explicit
            raise ValueError(f"Unsupported API provider for text extraction: {api_provider}")

        if text_content is None:
            logging.warning(f"Could not extract text content from {api_provider} response structure. JSON keys might be missing or unexpected. Response snippet: {str(response_json)[:250]}")
            return None
        else:
            logging.debug(f"Successfully extracted text for {api_provider}.")
            return text_content

    except (IndexError, KeyError, AttributeError, TypeError) as e:
        logging.warning(f"Error processing response structure for {api_provider}: {e}. Response snippet: {str(response_json)[:250]}")
        return None # Return None on unexpected structure during extraction


def _send_api_request(url: str, headers: Dict, payload: Dict, timeout: int) -> Dict[str, Any]:
    """Internal helper to send the POST request and handle HTTP/Request exceptions."""
    logging.debug(f"Sending POST request to URL: {url.split('?')[0]}...") # Log URL without query params
    # Consider selectively logging headers if sensitive info is present
    # logging.debug(f"Request Headers: {headers}")
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


def make_api_request(api_provider: str, provider_config: Dict, prompt: str, model: str, max_tokens: int, temperature: float) -> Optional[str]:
    """
    Orchestrates fetching API key, building payload, sending request, and extracting text.

    Args:
        api_provider: Name of the API provider.
        provider_config: Configuration dictionary for the provider.
        prompt: The input prompt.
        model: The model name.
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        The extracted text response, or None on failure.
    """
    api_endpoint = provider_config.get('api_endpoint')
    if not api_endpoint:
        logging.error(f"API Endpoint missing in config for provider '{api_provider}'.")
        return None

    try:
        # 1. Get API Key (raises ValueError if required and not found)
        api_key = get_api_key(api_provider, provider_config)

        # 2. Prepare Headers and URL
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        effective_url = api_endpoint
        if api_key:
            if api_provider in ['openai', 'groq']:
                headers['Authorization'] = f'Bearer {api_key}'
            elif api_provider == 'aistudio':
                # AI Studio key is typically appended to the URL
                effective_url = f"{api_endpoint}?key={api_key}"
            # Add other auth methods here if needed (e.g., custom headers for LMStudio if configured)

        # 3. Build Payload (raises ValueError if provider not supported)
        payload = build_request_payload(api_provider, model, prompt, max_tokens, temperature, provider_config)

        # 4. Send Request using the internal helper
        logging.info(f"Sending request to {api_provider} (Model: {model})")
        response_json = _send_api_request(effective_url, headers, payload, REQUEST_TIMEOUT)

        # 5. Extract Text (raises ValueError if provider not supported for extraction)
        extracted_text = extract_response_text(api_provider, response_json)

        if extracted_text is None:
            # Log warning if API call succeeded but no text was extracted
            logging.warning(f"API call successful, but failed to extract text from {api_provider} response.")
            return None # Return None to indicate no usable text
        else:
            logging.info(f"Successfully received and extracted response from {api_provider}.")
            return extracted_text

    except requests.exceptions.Timeout:
        logging.error(f"API request timed out ({REQUEST_TIMEOUT}s) for '{api_provider}' to {api_endpoint.split('?')[0]}.")
        return None
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error during API request for '{api_provider}': {e.response.status_code} {e.response.reason}")
        try:
            error_body = e.response.text
            logging.error(f"Error Response Body: {error_body[:500]}{'...' if len(error_body) > 500 else ''}")
        except Exception as read_err:
            logging.error(f"Could not read error response body: {read_err}")
        return None
    except requests.exceptions.RequestException as e:
        # Catch other requests-related errors (DNS, connection, etc.)
        logging.error(f"API request failed for '{api_provider}': {e}")
        return None
    except (ValueError, KeyError) as e:
        # Catch errors from get_key, build_payload, extract_text, or config issues
        logging.error(f"Configuration, data processing, or provider support error for '{api_provider}': {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during the process
        logging.exception(f"An unexpected error occurred during the API request/processing for '{api_provider}': {e}")
        return None


# --- Score Extraction ---
def extract_score_from_response(response_text: str) -> Optional[int]:
    """
    Attempts to extract a numerical score within the SCORE_RANGE or detect 'N/A'.
    Handles integers, decimals (rounds to nearest int), and common surrounding text.

    Args:
        response_text: The text response from the AI model.

    Returns:
        An integer score if found and valid, otherwise None.
    """
    if not response_text:
        logging.warning("Cannot extract score from empty or None response text.")
        return None

    # Enhanced regex:
    # - Optional preceding phrases like "score is", "score:", "value:", etc. (case-insensitive via flag)
    # - Captures "N/A", "NA" (case-insensitive, slash optional) OR a number (integer or decimal)
    # - Word boundaries (\b) to avoid partial matches.
    # - Allows flexible spacing (\s*)
    # - Added common formats like X/100 or X out of 100
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

        # Check for N/A first
        if na_group:
            normalized_na = na_group.strip().upper().replace('/', '')
            if normalized_na == "NA":
                logging.info(f"Extracted 'N/A' from response: '{response_text[:100]}...'")
                return None # Treat N/A explicitly as not a score

        # Determine which score format was captured
        extracted_value_str = score_slash_100 or score_out_of_100 or score_standalone

        if extracted_value_str:
            logging.info(f"Potential score string found: '{extracted_value_str}' in response: '{response_text[:100]}...'")
            try:
                # Convert to float first to handle decimals
                score_float = float(extracted_value_str)
                min_score, max_score = SCORE_RANGE

                # Validate range
                if min_score <= score_float <= max_score:
                    # Round to nearest integer before returning
                    score_int = int(round(score_float))
                    logging.info(f"Successfully extracted and validated score: {score_int}")
                    return score_int
                else:
                    logging.warning(f"Extracted score {score_float} is outside the valid range {SCORE_RANGE}. Discarding.")
                    return None
            except ValueError:
                # Handle cases where the captured string isn't a valid number
                logging.warning(f"Captured string '{extracted_value_str}' could not be converted to a number. Discarding.")
                return None
        else:
            # This case should be rare if the regex is correct but good to have
            logging.warning(f"Regex matched, but failed to extract a specific score value or N/A. Match groups: {match.groups()}")
            return None

    else:
        logging.warning(f"No score or 'N/A' found in the expected format within the response: '{response_text[:100]}...'")
        return None


# --- Main Assessment Logic ---
def run_assessment():
    """Runs the full assessment process: setup, query, scoring, reporting."""
    start_time = datetime.now()

    # --- Determine API Provider, Model, and Settings ---
    try:
        api_provider = os.environ.get('API_PROVIDER', config.get('api_provider', DEFAULT_API_PROVIDER)).lower()
        logging.info(f"Selected API Provider: '{api_provider}'")

        provider_config = config.get(api_provider)
        if not provider_config:
            raise ValueError(f"Configuration section for provider '{api_provider}' not found in '{CONFIG_FILE}'.")

        model_env_var = f'{api_provider.upper()}_MODEL'
        model = os.environ.get(model_env_var, provider_config.get('model'))
        if not model:
            raise ValueError(f"Model name missing for provider '{api_provider}'. Checked env var '{model_env_var}' and config file.")

        # Allow overriding max_tokens and temperature via config/env
        max_tokens_env_var = f'{api_provider.upper()}_MAX_TOKENS'
        max_tokens = int(os.environ.get(max_tokens_env_var, provider_config.get('max_tokens', DEFAULT_MAX_TOKENS)))

        temperature_env_var = f'{api_provider.upper()}_TEMPERATURE'
        temperature = float(os.environ.get(temperature_env_var, provider_config.get('temperature', DEFAULT_TEMPERATURE)))

        logging.info(f"Using Model: '{model}', Max Tokens: {max_tokens}, Temperature: {temperature}")

    except (ValueError, TypeError) as e:
        logging.error(f"Configuration Error: {e}")
        print(f"Error: Configuration issue - {e}. Check '{CONFIG_FILE}' and environment variables. Check '{LOG_FILE}' for details.")
        return # Stop execution if config is invalid

    assessment_date = start_time.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Starting assessment at {assessment_date}")

    results: List[Tuple[str, Optional[int]]] = [] # Store (question, score) tuples

    # --- Process Questions ---
    total_questions = len(questions)
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{total_questions}: {question[:80]}...")
        logging.info(f"Processing question {i}/{total_questions}: '{question}'")

        # Construct the full prompt using the template and the current question
        # Assumes prompt_template might have a placeholder like {question} or similar,
        # or simply needs the question appended. Adjust if your template is different.
        # Simple appending example:
        full_prompt = f"{prompt_template}\n\nQuestion: {question}"
        # If using a placeholder: full_prompt = prompt_template.format(question=question)
        logging.debug(f"Full prompt for API:\n{full_prompt}")

        # Get response using the refactored make_api_request
        response_text = make_api_request(
            api_provider,
            provider_config,
            full_prompt,
            model,
            max_tokens,
            temperature
        )

        score = None
        if response_text:
            logging.debug(f"Raw response text received for Q{i}: '{response_text}'")
            score = extract_score_from_response(response_text)
        else:
            logging.warning(f"No response received or text extracted for question {i}. Score will be None.")

        results.append((question, score))
        logging.info(f"Result for question {i}: Score = {score}")
        # Optional: Add a small delay between requests if needed
        # import time
        # time.sleep(1)

    # --- Calculate Summary Statistics ---
    valid_scores = [s for q, s in results if s is not None]
    num_valid_responses = len(valid_scores)
    num_invalid_responses = total_questions - num_valid_responses
    if num_valid_responses > 0:
        average_score = sum(valid_scores) / num_valid_responses
    else:
        average_score = 0.0 # Avoid division by zero

    logging.info(f"Assessment Summary: Total={total_questions}, ValidScores={num_valid_responses}, Invalid/NA={num_invalid_responses}, AverageScore={average_score:.2f}")

    end_time = datetime.now()
    duration = end_time - start_time

    # --- Generate Markdown Report ---
    table_data = [(q, str(s) if s is not None else "N/A") for q, s in results]
    headers = ["Question", "Score"]
    try:
        markdown_table = tabulate(table_data, headers=headers, tablefmt="github")
    except Exception as e:
        logging.error(f"Failed to generate markdown table using 'tabulate': {e}")
        markdown_table = "Error generating results table. Please check logs."

    # Create a safe filename
    safe_model_name = re.sub(r'[\\/*?:"<>|]+', '_', model) # Remove/replace invalid filename chars
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    report_filename = f"{timestamp}_{api_provider}_{safe_model_name}_assessment.md"

    logging.info(f"Writing assessment report to '{report_filename}'")
    try:
        with open(report_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(f"# Ethical AI Assessment Report\n\n")
            md_file.write(f"*   **API Provider:** `{api_provider.upper()}`\n")
            md_file.write(f"*   **Model:** `{model}`\n")
            md_file.write(f"*   **Assessment Date:** {assessment_date}\n")
            md_file.write(f"*   **Duration:** {str(duration).split('.')[0]} (HH:MM:SS)\n") # Cleaner duration format
            md_file.write(f"*   **Score Range Used:** {SCORE_RANGE[0]}-{SCORE_RANGE[1]}\n")
            md_file.write(f"*   **Temperature Used:** {temperature}\n")
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
        print(f"Provider: {api_provider.upper()}")
        print(f"Model: {model}")
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