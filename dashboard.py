"""
Dashboard functionality for visualizing assessment results.
"""

import json
import logging
from config import ASSESSMENT_DATA_FILE

log = logging.getLogger(__name__)

def load_assessment_data():
    """Load assessment data from the JSONL file."""
    data = []
    try:
        with open(ASSESSMENT_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    log.warning(f"Skipping invalid JSON line in {ASSESSMENT_DATA_FILE}: {line.strip()}")
        log.info(f"Loaded {len(data)} assessment records from {ASSESSMENT_DATA_FILE}")
        return data
    except FileNotFoundError:
        log.warning(f"Assessment data file not found: {ASSESSMENT_DATA_FILE}. Returning empty list.")
        return []
    except Exception as e:
        log.error(f"Error loading assessment data: {e}")
        return []

def update_dashboard():
    """
    Placeholder function to update the dashboard.
    This function will be called after new assessment data is saved.
    It should load the data and update any dashboard views (e.g., HTML, console).
    """
    log.info("Updating dashboard (placeholder implementation)...")
    assessment_data = load_assessment_data()
    if assessment_data:
        # In a real implementation, you would process this data
        # and generate/update a dashboard view (e.g., an HTML file, console output)
        log.info(f"Dashboard update triggered with {len(assessment_data)} records.")
        # Example: Print latest assessment average score
        latest_assessment = assessment_data[-1]
        print(f"[Dashboard Update] Latest assessment for {latest_assessment.get('provider')} ({latest_assessment.get('model')}): Avg Score = {latest_assessment.get('average_score'):.2f}")
    else:
        log.info("No assessment data found to update dashboard.")

if __name__ == '__main__':
    # Example of how to manually trigger an update (e.g., for testing)
    logging.basicConfig(level=logging.INFO)
    print("Manually triggering dashboard update...")
    update_dashboard()
