"""
API interaction functions for communicating with different AI providers.
"""

import re
import os
import json
import requests
import logging
import time
import random
from typing import Dict, Any, Optional, Tuple

from config import (
    PROVIDER_LMSTUDIO, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GOOGLE, PROVIDER_GENERIC,
    ENV_LMSTUDIO_API_KEY, ENV_OPENAI_API_KEY, ENV_ANTHROPIC_API_KEY, ENV_GOOGLE_API_KEY, ENV_GENERIC_API_KEY,
    REQUEST_TIMEOUT
)

log = logging.getLogger(__name__)

# --- Add Constants for Retry Mechanism ---
MAX_RETRIES = 3  # Maximum number of retry attempts
INITIAL_RETRY_DELAY = 2  # Initial delay in seconds
MAX_RETRY_DELAY = 60  # Maximum delay in seconds

# --- Text Processing ---
def strip_reasoning_tags(text: str) -> str:
    """Remove reasoning tags from the response text."""
    stripped_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    stripped_text = re.sub(r'^\s*\n', '', stripped_text, flags=re.MULTILINE)
    return stripped_text.strip()

# --- API Interaction Functions ---
def get_api_key(provider: str, api_config: Dict) -> Optional[str]:
    """Get API key from environment variables or config."""
    env_var_name = {
        PROVIDER_LMSTUDIO: ENV_LMSTUDIO_API_KEY,
        PROVIDER_OPENAI: ENV_OPENAI_API_KEY,
        PROVIDER_ANTHROPIC: ENV_ANTHROPIC_API_KEY,
        PROVIDER_GOOGLE: ENV_GOOGLE_API_KEY,
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
    """Build the API request payload based on the provider."""
    log.debug(f"Building payload for provider '{provider}' model: {model} with temp: {temperature}")

    system_prompt_content = provider_config.get("system_prompt")

    if provider == PROVIDER_LMSTUDIO:
        # Allow for model specification in LM Studio by using the provided model parameter
        # instead of the model from config if it's specifically set
        model_to_use = model if model else provider_config.get('model', '')
        
        messages = []
        if system_prompt_content:
            messages.append({"role": "system", "content": system_prompt_content})
        messages.append({"role": "user", "content": prompt})
        return {
            'model': model_to_use,
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
    """Extract the response text from the API response JSON."""
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
            # Enhanced extraction for Google Gemini API with more robust error handling
            candidates = response_json.get('candidates', [])
            if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
                log.warning(f"No candidates found in Google API response: {str(response_json)[:250]}")
                return None
                
            content = candidates[0].get('content', {})
            if not content or not isinstance(content, dict):
                log.warning(f"Invalid content structure in Google API response: {str(candidates[0])[:250]}")
                return None
                
            parts = content.get('parts', [])
            if not parts or not isinstance(parts, list) or len(parts) == 0:
                log.warning(f"No parts found in Google API response content: {str(content)[:250]}")
                return None
                
            # Try different ways of extracting text based on observed response patterns
            if 'text' in parts[0]:
                text_content = parts[0]['text'].strip()
            elif isinstance(parts[0], str):
                text_content = parts[0].strip()
            elif isinstance(parts[0], dict) and 'text' in parts[0]:
                text_content = parts[0]['text'].strip()
            else:
                log.warning(f"Unexpected part structure in Google API response: {str(parts[0])[:250]}")
                # As a fallback, try to convert the entire parts object to a string
                try:
                    text_content = str(parts[0]).strip()
                    if text_content.startswith('{') or text_content.startswith('['):
                        log.warning("Part appears to be a raw JSON object, may not be usable text content")
                except Exception as e:
                    log.error(f"Failed to extract text from Google API response parts: {e}")
                    return None
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
    """Send the API request and return the JSON response."""
    log.debug(f"Sending POST request to URL: {url.split('?')[0]}...")
    log.debug(f"Request Payload: {json.dumps(payload, indent=2)}")

    retries = 0
    delay = INITIAL_RETRY_DELAY
    
    while True:
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            # Handle rate limiting (HTTP 429)
            if response.status_code == 429:
                if retries < MAX_RETRIES:
                    # Get retry-after header if available or use exponential backoff
                    retry_after = response.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff with jitter
                        wait_time = min(delay * (1.5 + random.random() * 0.5), MAX_RETRY_DELAY)
                        delay = wait_time
                    
                    retries += 1
                    log.warning(f"Rate limit exceeded (HTTP 429). Retrying in {wait_time:.1f} seconds... (Attempt {retries}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
                else:
                    log.error(f"Rate limit exceeded (HTTP 429) and maximum retries ({MAX_RETRIES}) reached. Giving up.")
                    response.raise_for_status()  # This will raise the exception
            
            # For all other errors, just raise the exception
            response.raise_for_status()
            
            # If we got here, the request was successful
            response_json = response.json()
            log.debug(f"Received Raw Response JSON: {json.dumps(response_json, indent=2)}")
            return response_json
            
        except requests.exceptions.HTTPError as e:
            # If we've already handled rate limiting above but still got here
            # this means we have a different HTTP error or max retries exceeded
            if e.response.status_code == 429:
                raise
            
            # Special handling for invalid API key or authentication errors
            if e.response.status_code == 401 or e.response.status_code == 403:
                log.error(f"Authentication error (HTTP {e.response.status_code}). Check your API key.")
                raise
                
            # For other HTTP errors, retry with backoff if we haven't exceeded max retries
            if retries < MAX_RETRIES:
                wait_time = min(delay * (1.5 + random.random() * 0.5), MAX_RETRY_DELAY)
                delay = wait_time
                retries += 1
                log.warning(f"HTTP error {e.response.status_code}. Retrying in {wait_time:.1f} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue
            else:
                log.error(f"Maximum retries ({MAX_RETRIES}) reached for HTTP errors. Giving up.")
                raise
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Network errors can be temporary, so retry with backoff
            if retries < MAX_RETRIES:
                wait_time = min(delay * (1.5 + random.random() * 0.5), MAX_RETRY_DELAY)
                delay = wait_time
                retries += 1
                log.warning(f"Network error: {str(e)}. Retrying in {wait_time:.1f} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue
            else:
                log.error(f"Maximum retries ({MAX_RETRIES}) reached for network errors. Giving up.")
                raise

def make_api_request(provider: str, provider_config: Dict, prompt: str, model: str, max_tokens: int, temperature: float) -> Optional[str]:
    """Make an API request to the specified provider."""
    api_endpoint = provider_config.get('api_endpoint')
    if not api_endpoint:
        log.error(f"API Endpoint missing unexpectedly for provider '{provider}'.")
        return None

    # Check if request parameters are valid before attempting request
    try:
        if not model:
            log.error(f"Model name is missing or empty for provider '{provider}'.")
            return None
            
        if max_tokens <= 0:
            log.error(f"Invalid max_tokens value ({max_tokens}) for provider '{provider}'.")
            return None
            
        if not (0 <= temperature <= 1.0):
            log.warning(f"Temperature value ({temperature}) outside recommended range [0, 1] for provider '{provider}'. Proceeding anyway.")
            
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
            
    # Distinguish between different types of errors for better debugging
    except requests.exceptions.Timeout:
        log.error(f"API request timed out ({REQUEST_TIMEOUT}s) for provider '{provider}' (Temp: {temperature:.2f}) at {api_endpoint}.")
        return None
    except requests.exceptions.ConnectionError as e:
        log.error(f"API request failed: Could not connect to provider '{provider}' at {api_endpoint}. Is the server running? Error: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        # Distinguish between different HTTP error types
        status_code = e.response.status_code
        
        # Client errors
        if 400 <= status_code < 500:
            if status_code == 401:
                log.error(f"Authentication error for provider '{provider}': Invalid or missing API key.")
            elif status_code == 403:
                log.error(f"Authorization error for provider '{provider}': Insufficient permissions or usage limits exceeded.")
            elif status_code == 404:
                log.error(f"Resource not found error for provider '{provider}': Check if model '{model}' exists and API endpoint is correct.")
            elif status_code == 429:
                log.error(f"Rate limit exceeded for provider '{provider}'. Consider increasing request delays.")
            else:
                log.error(f"Client error from provider '{provider}' (HTTP {status_code}): {e.response.reason}")
        # Server errors
        else:
            log.error(f"Server error from provider '{provider}' (HTTP {status_code}): {e.response.reason}. This is likely temporary.")
        
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
    """Extract a numerical score from the model's response."""
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
                min_score, max_score = 0, 100  # SCORE_RANGE
                if min_score <= score_float <= max_score:
                    score_int = int(round(score_float))
                    log.info(f"Successfully extracted and validated score: {score_int}")
                    return score_int
                else:
                    log.warning(f"Extracted score {score_float} is outside the valid range (0-100). Discarding.")
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
    """Make a single API call and extract a score from the response."""
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
