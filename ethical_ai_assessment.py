import json
import requests
from datetime import datetime
import markdown_table
import os
import re
import logging
from typing import Optional, Dict, Any


# Setting up logging
logging.basicConfig(level=logging.DEBUG, filename='assessment.log',
                    format='%(asctime)s %(levelname)s:%(message)s')

# Load API key and other configurations
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Read questions from questions.txt
with open('questions.txt', 'r') as questions_file:
    questions = questions_file.read().splitlines()

# Read prompt text from prompt.txt
with open('prompt.txt', 'r') as prompt_file:
    prompt_text = prompt_file.read().strip()


def make_api_request(api_provider: str, config: Dict, prompt: str, model: str) -> Optional[str]:
    """
    Makes a request to the specified AI API.

    Args:
        api_provider: The API provider to use (e.g., "openai", "groq").
        config: The configuration dictionary loaded from config.json.
        prompt: The prompt to send to the AI.
        model: The model to use.

    Returns:
        The response text from the API, or None if there is an error.
    """
    try:
        api_config = config.get(api_provider)
        if not api_config:
            logging.error(f"API Provider '{api_provider}' not configured.")
            return None
        api_endpoint = api_config.get('api_endpoint')
        if not api_endpoint:
           logging.error(f"API Endpoint missing for '{api_provider}'.")
           return None
        api_key = os.environ.get(f'{api_provider.upper()}_API_KEY') if os.environ.get(f'{api_provider.upper()}_API_KEY') else api_config.get('api_key')
        headers = {}

        if api_provider == 'openai':
            headers = {'Authorization': f'Bearer {api_key}'}
            payload = {
                'model': model,
                'prompt': prompt,
                'max_tokens': 50
            }
            response = requests.post(api_endpoint, headers=headers, json=payload)

        elif api_provider == 'groq':
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': model,
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': 50
            }
            response = requests.post(api_endpoint, headers=headers, json=payload)

        elif api_provider == 'aistudio':
            headers = {
               'Content-Type': 'application/json',
               'x-goog-api-key': api_key
             }
            payload = {
                "contents": [
                     {
                     "parts": [{"text": prompt}]
                     }
                   ]
                }
            response = requests.post(api_endpoint, headers=headers, json=payload)
        
        elif api_provider == 'ollama':
            headers = {
                'Content-Type': 'application/json'
            }
            payload = {
               "prompt": prompt,
               "model": model,
               "stream": False,
               "options": {
                   "num_predict": 50
               }
            }
            response = requests.post(api_endpoint, headers=headers, json=payload)
        else:
             logging.error(f"API provider '{api_provider}' is not supported")
             return None

        if response.status_code != 200:
           logging.error(f"API request failed for provider '{api_provider}': {response.text}")
           return None

        if api_provider == 'openai':
           return response.json().get('choices', [{}])[0].get('text', '').strip()
        elif api_provider == 'groq':
           return response.json().get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        elif api_provider == 'aistudio':
           return response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        elif api_provider == 'ollama':
           return response.json().get('response', '').strip()
        else:
            logging.error(f"Unsupported API provider: {api_provider}")
            return None
    except Exception as e:
        logging.error(f"Error during API request for provider '{api_provider}': {e}")
        return None


def ask_question_and_extract_score(question: str, api_provider: str, config: Dict, model: str) -> Optional[int]:
    """
    Asks a question to the AI and extracts a numerical score from the response.

    Args:
        question: The question to ask the AI.
        api_provider: The API provider to use (e.g., "openai", "groq").
        config: The configuration dictionary loaded from config.json.
        model: The model to use.

    Returns:
        The extracted numerical score (int) if found, or None if not found/invalid response.
    """
    try:
        full_prompt = f"{prompt_text}\n\n{question}"
        response_text = make_api_request(api_provider, config, full_prompt, model)

        if not response_text:
            return None

        match = re.search(r'\b(N/A|NA|\d+\.?\d*)\b', response_text, re.IGNORECASE)

        if match:
            value = match.group(0).upper()
            if value == "N/A" or value == "NA":
                return None
            try:
                score = float(value)
                if 0 <= score <= 100:
                    return int(score)
                else:
                  logging.warning(f"Extracted score {score} is outside range [0-100] from response: {response_text}")
                  return None
            except ValueError:
                logging.warning(f"Extracted value '{value}' is not a valid number from response: {response_text}")
                return None
        else:
            logging.warning(f"No valid score found from response: {response_text}")
            return None
    except Exception as e:
        logging.error(f"Error processing question '{question}': {e}")
        return None



def run_assessment():
    """Runs the assessment by asking questions to the AI and generating a markdown report."""
    scores = []
    invalid_responses = 0

    # Load API provider and model, allowing overrides by environment variables
    api_provider = os.environ.get('API_PROVIDER', config.get('api_provider', 'openai'))
    api_config = config.get(api_provider, {})
    model = os.environ.get(f'{api_provider.upper()}_MODEL', api_config.get('model', None))
    if not model:
        logging.error(f"No model was configured or found for api '{api_provider}'")
        return None

    assessment_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    logging.info(f"Starting assessment using API Provider: {api_provider}, Model: {model}, Date: {assessment_date}")

    for question in questions:
        score = ask_question_and_extract_score(question, api_provider, config, model)
        if score is not None:
            scores.append(str(score))
        else:
            invalid_responses += 1

    # Calculate final score (average of valid responses)
    final_score = sum(map(int, scores)) / len(scores) if scores else 0

    # Generating Markdown Table for valid responses
    headers = ["Question", "Score"]
    markdown_content = markdown_table.render(headers, zip(questions, scores))

    # Generating the output file name with date, API and Model used
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{api_provider}_{model.replace('/', '_')}.md"

    # Writing results to the results.md file
    with open(filename, 'w') as md_file:
        md_file.write(f"# Ethical AI Assessment Results\n\n")
        md_file.write(f"- API Evaluated: {api_provider.upper()} (Model: {model})\n")
        md_file.write(f"- Assessment Date: {assessment_date}\n")
        md_file.write(f"- Final Score: {final_score:.2f}/100 (Based on valid responses only)\n\n")
        md_file.write(f"- Total Questions: {len(questions)}\n")
        md_file.write(f"- Valid Responses: {len(scores)}\n")
        md_file.write(f"- Invalid Responses: {invalid_responses}\n\n")
        md_file.write(markdown_content)
    logging.info(f"Assessment completed. Results saved to '{filename}'.")

if __name__ == "__main__":
    run_assessment()
