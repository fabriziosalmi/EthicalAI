"""
API interaction functions for communicating with different AI providers.
"""

import re
import os
import json
import requests
import logging
from typing import Dict, Any, Optional, Tuple

from config import (
    PROVIDER_LMSTUDIO, PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GOOGLE, PROVIDER_GENERIC,
    ENV_LMSTUDIO_API_KEY, ENV_OPENAI_API_KEY, ENV_ANTHROPIC_API_KEY, ENV_GOOGLE_API_KEY, ENV_GENERIC_API_KEY,
    REQUEST_TIMEOUT
)

log = logging.getLogger(__name__)

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
    """Send the API request and return the JSON response."""
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
    """Make an API request to the specified provider."""
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
