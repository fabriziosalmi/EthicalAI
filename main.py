"""
Main entry point for the Ethical AI Assessment tool.
"""

import argparse
import logging
import sys
import os

from config import load_config, setup_logging
from assessment import run_assessment

log = logging.getLogger(__name__)

def load_text_file(filepath: str, description: str) -> str:
    """Load content from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        log.info(f"Successfully loaded {description} from '{filepath}'")
        return content
    except FileNotFoundError:
        log.error(f"{description.capitalize()} file not found at '{filepath}'.")
        print(f"Error: {description.capitalize()} file not found at '{filepath}'. Exiting.", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        log.error(f"Error reading {description} file '{filepath}': {e}")
        print(f"Error: Could not read {description} file '{filepath}': {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        log.exception(f"An unexpected error occurred loading {description} file '{filepath}': {e}")
        print(f"An unexpected error occurred loading {description} file '{filepath}': {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to parse arguments and run the assessment."""
    parser = argparse.ArgumentParser(description="Run Ethical AI Assessment.")
    parser.add_argument(
        "provider",
        help="Name of the AI provider to assess (must match a key in config.json)"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to the configuration file (default: config.json)"
    )
    parser.add_argument(
        "-q", "--questions",
        default="questions.txt",
        help="Path to the questions file (default: questions.txt)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="prompt.txt",
        help="Path to the prompt template file (default: prompt.txt)"
    )
    parser.add_argument(
        "-l", "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-reports",
        action="store_false",
        dest="generate_reports",
        help="Disable automatic generation of HTML and PDF reports"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log.upper())

    log.info("--- Ethical AI Assessment Tool Starting ---")

    # Load configuration
    config = load_config(args.config)
    if not config:
        sys.exit(1) # Error message already printed by load_config

    # Check if the specified provider exists in the config
    if args.provider not in config:
        log.error(f"Provider '{args.provider}' not found in configuration file '{args.config}'.")
        print(f"Error: Provider '{args.provider}' not found in configuration file '{args.config}'.", file=sys.stderr)
        print(f"Available providers: {list(config.keys())}", file=sys.stderr)
        sys.exit(1)

    # Load questions
    questions_content = load_text_file(args.questions, "questions")
    questions_list = [q.strip() for q in questions_content.splitlines() if q.strip()]
    if not questions_list:
        log.error(f"No questions found in '{args.questions}'.")
        print(f"Error: No questions found in '{args.questions}'. Exiting.", file=sys.stderr)
        sys.exit(1)
    log.info(f"Loaded {len(questions_list)} questions.")

    # Load prompt template
    prompt_template = load_text_file(args.prompt, "prompt template")
    if not prompt_template:
        log.error(f"Prompt template file '{args.prompt}' is empty.")
        print(f"Error: Prompt template file '{args.prompt}' is empty. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Ensure results directory exists
    results_dir = config.get("results_dir", "results") # Use default if not in config
    os.makedirs(results_dir, exist_ok=True)
    log.info(f"Ensured results directory exists: '{results_dir}'")

    # Run the assessment
    try:
        log.info(f"Running assessment for provider: {args.provider}")
        run_assessment(
            provider=args.provider,
            config=config,
            questions=questions_list,
            prompt_template=prompt_template,
            generate_reports=args.generate_reports
        )
        log.info(f"Assessment for provider '{args.provider}' completed.")
    except Exception as e:
        log.exception(f"An unexpected error occurred during the assessment run: {e}")
        print(f"An unexpected error occurred: {e}. Check logs for details.", file=sys.stderr)
        sys.exit(1)

    log.info("--- Ethical AI Assessment Tool Finished ---")

if __name__ == "__main__":
    main()
