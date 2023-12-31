import json
import requests
from datetime import datetime
import markdown_table  # This module is used for generating Markdown tables
import os
import re
import logging

# Setting up logging
logging.basicConfig(level=logging.DEBUG, filename='assessment.log', 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Load API key and other configurations
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
api_key = os.environ['OPENAI_API_KEY']
api_endpoint = config['api_endpoint']

# Read questions from questions.txt
with open('questions.txt', 'r') as questions_file:
    questions = questions_file.read().splitlines()

# Read prompt text from prompt.txt
with open('prompt.txt', 'r') as prompt_file:
    prompt_text = prompt_file.read().strip()

# Function to ask a question to the AI and extract a numerical response
def ask_question_and_extract_score(question):
    try:
        full_prompt = f"{prompt_text}\n\n{question}"
        response = requests.post(api_endpoint, json={
            'model': 'text-davinci-003',  # Use either 'model' or 'engine', but not both
            'prompt': full_prompt,
            'max_tokens': 50
        }, headers={
            'Authorization': f'Bearer {api_key}'
        })

        if response.status_code != 200:
            logging.error(f"API request failed for question '{question}': {response.text}")
            return None

        response_text = response.json().get('choices', [{}])[0].get('text', '').strip()

        match = re.search(r'\b([1-9][0-9]?|100)\b', response_text)
        return int(match.group(0)) if match else None
    except Exception as e:
        logging.error(f"Error processing question '{question}': {e}")
        return None


# Main function to run the assessment
def run_assessment():
    scores = []
    invalid_responses = 0
    assessment_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_used = "text-davinci-003"  # Replace with the actual model used if different

    for question in questions:
        score = ask_question_and_extract_score(question)
        if score is not None:
            scores.append(str(score))
        else:
            invalid_responses += 1
    
    # Calculate final score (average of valid responses)
    final_score = sum(map(int, scores)) / len(scores) if scores else 0

    # Generating Markdown Table for valid responses
    headers = ["Question", "Score"]
    markdown_content = markdown_table.render(headers, zip(questions, scores))

    # Writing results to the results.md file
    with open('davinci003.md', 'w') as md_file:
        md_file.write(f"# Ethical AI Assessment Results\n\n")
        md_file.write(f"- API Evaluated: OpenAI (Model: {model_used})\n")
        md_file.write(f"- Assessment Date: {assessment_date}\n")
        md_file.write(f"- Final Score: {final_score:.2f}/100 (Based on valid responses only)\n\n")
        md_file.write(f"- Total Questions: {len(questions)}\n")
        md_file.write(f"- Valid Responses: {len(scores)}\n")
        md_file.write(f"- Invalid Responses: {invalid_responses}\n\n")
        md_file.write(markdown_content)

if __name__ == "__main__":
    run_assessment()
