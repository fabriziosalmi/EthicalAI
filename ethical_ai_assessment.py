import json
import requests
from datetime import datetime
import markdown_table
import os
import re
import logging
from typing import Optional, Dict, Any, List

# Constants
CONFIG_FILE = 'config.json'
QUESTIONS_FILE = 'questions.txt'
PROMPT_FILE = 'prompt.txt'
DEFAULT_API_PROVIDER = 'openai'
DEFAULT_MAX_TOKENS = 50
SCORE_RANGE = (0, 100)

# Logging setup
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO for less verbose output
    filename='assessment.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config(config_file: str) -> Dict:
    """Loads the configuration from the specified JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_file}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in config file: {config_file}")
        raise


def load_text_file(filepath: str) -> str:
    """Loads text content from a file and returns it as a string."""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise


# Load configurations, questions, and prompt text
try:
    config = load_config(CONFIG_FILE)
    questions = load_text_file(QUESTIONS_FILE).splitlines()
    prompt_text = load_text_file(PROMPT_FILE)
except Exception as e:
    print(f"Error during initialization: {e}")  # Print to console for immediate feedback
    exit(1)  # Exit the program as initialization failed


def get_api_key(api_provider: str, api_config: Dict) -> str:
    """Retrieves the API key from environment variables or config file."""
    env_var_name = f'{api_provider.upper()}_API_KEY'
    api_key = os.environ.get(env_var_name) or api_config.get('api_key')
    if not api_key:
        logging.error(f"API Key not found in environment variables or config file for {api_provider}")
        raise ValueError(f"Missing API key for {api_provider}")
    return api_key


def build_request_payload(api_provider: str, model: str, prompt: str) -> Dict:
    """Builds the request payload based on the API provider."""
    if api_provider == 'openai':
        return {
            'model': model,
            'prompt': prompt,
            'max_tokens': DEFAULT_MAX_TOKENS
        }
    elif api_provider == 'groq':
        return {
            'model': model,
            'messages': [{"role": "user", "content": prompt}],
            'max_tokens': DEFAULT_MAX_TOKENS
        }
    elif api_provider == 'aistudio':
        return {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
    elif api_provider == 'ollama':
        return {
            "prompt": prompt,
            "model": model,
            "stream": False,
            "options": {
                "num_predict": DEFAULT_MAX_TOKENS
            }
        }
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")


def extract_response_text(api_provider: str, response_json: Dict) -> str:
    """Extracts the response text from the API response."""
    if api_provider == 'openai':
        return response_json.get('choices', [{}])[0].get('text', '').strip()
    elif api_provider == 'groq':
        return response_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
    elif api_provider == 'aistudio':
        return response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
    elif api_provider == 'ollama':
        return response_json.get('response', '').strip()
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")


def make_api_request(api_provider: str, api_config: Dict, prompt: str, model: str) -> Optional[str]:
    """Makes a request to the specified AI API."""
    api_endpoint = api_config.get('api_endpoint')
    if not api_endpoint:
        logging.error(f"API Endpoint missing for '{api_provider}'.")
        return None

    try:
        api_key = get_api_key(api_provider, api_config)
        headers = {'Content-Type': 'application/json'}

        if api_provider == 'openai':
            headers['Authorization'] = f'Bearer {api_key}'
        elif api_provider == 'groq':
            headers['Authorization'] = f'Bearer {api_key}'
        elif api_provider == 'aistudio':
            headers['x-goog-api-key'] = api_key

        payload = build_request_payload(api_provider, model, prompt)

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)  # Added timeout

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()
        return extract_response_text(api_provider, response_json)

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for provider '{api_provider}': {e}")
        return None
    except (ValueError, KeyError) as e:
        logging.error(f"Error configuring API request for '{api_provider}': {e}")
        return None


def ask_question_and_extract_score(question: str, api_provider: str, config: Dict, model: str) -> Optional[int]:
    """Asks a question to the AI and extracts a numerical score from the response."""
    full_prompt = f"{prompt_text}\n\n{question}"
    response_text = make_api_request(api_provider, config, full_prompt, model)

    if not response_text:
        return None

    match = re.search(r'\b(N/A|NA|\d+\.?\d*)\b', response_text, re.IGNORECASE)

    if match:
        value = match.group(0).upper()
        if value in ("N/A", "NA"):
            return None
        try:
            score = float(value)
            if SCORE_RANGE[0] <= score <= SCORE_RANGE[1]:
                return int(score)
            else:
                logging.warning(f"Extracted score {score} is outside range {SCORE_RANGE} from response: {response_text}")
                return None
        except ValueError:
            logging.warning(f"Extracted value '{value}' is not a valid number from response: {response_text}")
            return None
    else:
        logging.warning(f"No valid score found from response: {response_text}")
        return None


def run_assessment():
    """Runs the assessment by asking questions to the AI and generating a markdown report."""
    scores: List[str] = []
    valid_scores: List[int] = []
    invalid_responses: int = 0

    # Load API provider and model, allowing overrides by environment variables
    api_provider = os.environ.get('API_PROVIDER', config.get('api_provider', DEFAULT_API_PROVIDER))
    api_config = config.get(api_provider, {})
    model = os.environ.get(f'{api_provider.upper()}_MODEL', api_config.get('model', None))

    if not model:
        logging.error(f"No model was configured or found for api '{api_provider}'")
        print(f"Error: No model configured for API Provider: {api_provider}")
        return

    assessment_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Starting assessment using API Provider: {api_provider}, Model: {model}, Date: {assessment_date}")

    for i, question in enumerate(questions):
        logging.info(f"Processing question {i+1}/{len(questions)}")
        score = ask_question_and_extract_score(question, api_provider, config, model)
        if score is not None:
            scores.append(str(score))
            valid_scores.append(score)
        else:
            invalid_responses += 1

    # Calculate final score (average of valid responses)
    final_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    # Generating Markdown Table for valid responses
    headers = ["Question", "Score"]
    # Use questions list as is, matching 1-to-1 with the number of scores collected
    markdown_content = markdown_table.render(headers, zip(questions[:len(scores)], scores))

    # Generating the output file name with date, API and Model used
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{api_provider}_{model.replace('/', '_')}.md"

    # Writing results to the markdown file
    try:
        with open(filename, 'w') as md_file:
            md_file.write(f"# Ethical AI Assessment Results\n\n")
            md_file.write(f"- API Evaluated: {api_provider.upper()} (Model: {model})\n")
            md_file.write(f"- Assessment Date: {assessment_date}\n")
            md_file.write(f"- Final Score: {final_score:.2f}/100 (Based on valid responses only)\n\n")
            md_file.write(f"- Total Questions: {len(questions)}\n")
            md_file.write(f"- Valid Responses: {len(valid_scores)}\n")
            md_file.write(f"- Invalid Responses: {invalid_responses}\n\n")
            md_file.write(markdown_content)
        logging.info(f"Assessment completed. Results saved to '{filename}'.")
        print(f"Assessment completed. Results saved to '{filename}'.") # Print summary to console

    except IOError as e:
        logging.error(f"Error writing results to file: {e}")
        print(f"Error writing results to file: {e}")


if __name__ == "__main__":
    run_assessment()
