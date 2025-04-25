"""
Main assessment logic for evaluating AI models on ethical questions.
"""

import re
import os
import json
import time
import random
import logging
import statistics
from datetime import datetime, timedelta # Import timedelta directly
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tabulate import tabulate
from collections import defaultdict

from config import (
    RESULTS_DIR, ASSESSMENT_DATA_FILE, SCORE_RANGE,
    DEFAULT_RETRY_CONFIRM_THRESHOLD
)
from api import get_single_score
from visualization import generate_visualizations
from reporting import generate_html_report, generate_pdf_report

log = logging.getLogger(__name__)

def run_assessment(provider: str, config: Dict, questions: List[str], prompt_template: str, generate_reports: bool = True):
    """
    Run the assessment for a given provider.
    
    Args:
        provider: Name of the AI provider to assess
        config: Configuration dictionary
        questions: List of questions to ask
        prompt_template: Template for the prompts
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
        
        # Get category mapping from provider config, or from root config, or use default
        if 'category_mapping' in provider_config:
            category_mapping = provider_config['category_mapping']
        elif 'category_mapping' in config:
            category_mapping = config['category_mapping']
            log.info(f"Using root-level category mapping for provider '{provider}'")
        else:
            from config import DEFAULT_CATEGORY_MAPPING
            category_mapping = DEFAULT_CATEGORY_MAPPING
            log.warning(f"No category mapping found for provider '{provider}'. Using default mapping.")

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
        print(f"Error: Configuration issue for provider '{provider}' - {e}. Check config file. See logs for details.")
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
                        log.warning(f"Edge score {median_score} NOT confirmed by retries.")
                        # Calculate a new score based on all samples (original + retries)
                        all_valid_scores = valid_sample_scores + valid_retry_scores
                        new_median = statistics.median(all_valid_scores)
                        new_median_rounded = int(round(new_median))
                        log.info(f"New median from all samples: {new_median:.2f} -> Rounded: {new_median_rounded}")
                        final_score = new_median_rounded
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

    report_filename = generate_assessment_report(
        provider, model, provider_config, results, start_time, 
        assessment_date, duration, num_samples, base_temperature,
        random_temp_min, random_temp_max, retry_edges, max_retries,
        retry_confirm_threshold, strip_tags, total_questions,
        num_valid_final_scores, num_invalid_final_scores, average_final_score
    )
    
    # Generate visualization charts
    viz_dir = None
    if generate_reports and report_filename:
        log.info("Generating visualizations...")
        viz_dir = generate_visualizations(provider, results, category_mapping)
        
        # Generate HTML and PDF reports
        log.info("Generating HTML and PDF reports...")
        html_file = generate_html_report(report_filename, include_charts=True)
        pdf_file = None
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

        # Display summary
        display_summary(provider, model, num_samples, retry_edges, 
                       average_final_score, num_valid_final_scores, 
                       total_questions, report_filename, html_file, 
                       pdf_file, duration)

        # Save assessment data for dashboard
        save_assessment_data(provider, model, average_final_score, num_valid_final_scores, 
                            total_questions, assessment_date, duration, results, category_mapping)

    return results, average_final_score

def generate_assessment_report(
    provider: str, model: str, provider_config: Dict,
    results: List[Tuple[str, Optional[int], List[Optional[int]]]],
    start_time: datetime, assessment_date: str, duration: timedelta, # Use timedelta
    num_samples: int, base_temperature: float, random_temp_min: float,
    random_temp_max: float, retry_edges: bool, max_retries: int,
    retry_confirm_threshold: float, strip_tags: bool, total_questions: int,
    num_valid_final_scores: int, num_invalid_final_scores: int, average_final_score: float
) -> str:
    """Generate the assessment report in markdown format."""
    try:
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
        return report_filename

    except IOError as e:
        log.error(f"Error writing report: {e}")
        print(f"Error: Could not write results to file. Check permissions and path.")
        return ""
    except Exception as e:
        log.exception(f"An unexpected error occurred during report generation: {e}")
        print(f"An unexpected error occurred while writing the report: {e}")
        return ""

def display_summary(
    provider: str, model: str, num_samples: int, retry_edges: bool,
    average_final_score: float, num_valid_final_scores: int, 
    total_questions: int, report_filename: str, html_file: Optional[str],
    pdf_file: Optional[str], duration: timedelta # Use timedelta
):
    """Display a summary of the assessment results in the console."""
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
    if html_file:
        summary_text.append(Text.from_markup("HTML Report: ", style="bold cyan"))
        summary_text.append(f"'{html_file}'\n")
    if pdf_file:
        summary_text.append(Text.from_markup("PDF Report: ", style="bold cyan"))
        summary_text.append(f"'{pdf_file}'\n")
            
    summary_text.append(Text.from_markup("Duration: ", style="bold cyan"))
    summary_text.append(f"{str(duration).split('.')[0]}")
            
    console.print(Panel(summary_text, title="[bold magenta]Assessment Complete", border_style="magenta"))

def save_assessment_data(
    provider: str, model: str, average_final_score: float,
    num_valid_final_scores: int, total_questions: int,
    assessment_date: str, duration: timedelta, # Use timedelta
    results: List[Tuple[str, Optional[int], List[Optional[int]]]],
    category_mapping: Dict[str, List[int]]
):
    """Save assessment data to the JSONL file for dashboard."""
    try:
        # Calculate category scores
        category_scores = defaultdict(list)
        for i, (question, score, _) in enumerate(results, 1):
            if score is not None:
                # Determine category for this question
                for cat, indices in category_mapping.items():
                    if i in indices:
                        category_scores[cat].append(score)
                        break
        
        # Create assessment data
        assessment_data = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'model': model,
            'average_score': average_final_score,
            'valid_scores': num_valid_final_scores,
            'total_questions': total_questions,
            'assessment_date': assessment_date,
            'duration_seconds': duration.total_seconds(),
            'categories': {}
        }
        
        # Add category averages to assessment data
        for category, scores in category_scores.items():
            if scores:
                assessment_data['categories'][category] = sum(scores) / len(scores)
        
        # Save to JSONL file (append mode)
        with open(ASSESSMENT_DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(assessment_data) + '\n')
        log.info(f"Assessment data saved to {ASSESSMENT_DATA_FILE}")
        
        # Import here to avoid circular import
        from dashboard import update_dashboard
        # Update dashboard with new data
        update_dashboard()
        
    except Exception as e:
        log.error(f"Failed to save assessment data: {e}")
