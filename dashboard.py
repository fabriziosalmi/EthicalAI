"""
Dashboard functionality for visualizing assessment results.
"""

import json
import logging
import os
from config import ASSESSMENT_DATA_FILE, DASHBOARD_DIR
from datetime import datetime

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
        # Sort data by timestamp ascending
        data.sort(key=lambda x: x.get('timestamp', ''))
        return data
    except FileNotFoundError:
        log.warning(f"Assessment data file not found: {ASSESSMENT_DATA_FILE}. Returning empty list.")
        return []
    except Exception as e:
        log.error(f"Error loading assessment data: {e}")
        return []

def get_latest_assessment_by_model(all_data):
    """Process data to find the latest assessment for each unique model."""
    latest_by_model = {}
    for assessment in all_data:
        model_name = assessment.get('model')
        if model_name:
            # Keep track of the latest assessment for each model
            if model_name not in latest_by_model or datetime.fromisoformat(assessment.get('timestamp', '')) > datetime.fromisoformat(latest_by_model[model_name].get('timestamp', '')):
                latest_by_model[model_name] = assessment
    return latest_by_model

def generate_html_dashboard(output_dir=DASHBOARD_DIR):
    """Generate a simple HTML dashboard from assessment data."""
    log.info("Generating HTML dashboard...")
    all_assessment_data = load_assessment_data()
    
    if not all_assessment_data:
        log.warning("No assessment data found. Dashboard will be empty.")
    
    # Group latest assessment by model
    latest_by_model = get_latest_assessment_by_model(all_assessment_data)
    unique_models = sorted(list(latest_by_model.keys()))
    
    # Create a clean HTML file without complex JavaScript
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ethical AI Assessment Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .chart-container { height: 400px; width: 100%; }
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.15rem 0.5rem;
            border-radius: 9999px;
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
        }
        .badge-improvement { background-color: #d1fae5; color: #065f46; }
        .badge-decline { background-color: #fee2e2; color: #b91c1c; }
        .badge-neutral { background-color: #e5e7eb; color: #374151; }
    </style>
</head>
<body class="bg-gray-100 text-gray-800 p-8">
    <div class="max-w-7xl mx-auto">
        <header class="mb-10">
            <h1 class="text-4xl font-bold text-gray-900">Ethical AI Assessment Dashboard</h1>
            <p class="text-lg text-gray-600 mt-2">Overview and comparison of assessment results</p>
        </header>
        
        <!-- Assessment History Table -->
        <section class="mb-10 bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">Assessment History</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer">Timestamp</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer">Provider</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer">Model</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer">Avg Score</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer">Valid/Total Qs</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer">Duration (s)</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
    """
    
    # Sort the data for display in the table (high to low score)
    sorted_data = sorted(all_assessment_data, key=lambda x: x.get('average_score', 0), reverse=True)
    
    # Generate table rows for assessment history
    for assessment in sorted_data:
        timestamp = assessment.get('timestamp', '')
        display_timestamp = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'N/A'
        provider = assessment.get('provider', 'N/A')
        model = assessment.get('model', 'N/A')
        avg_score = assessment.get('average_score', 0)
        valid_scores = assessment.get('valid_scores', 0)
        total_questions = assessment.get('total_questions', 0)
        duration = assessment.get('duration_seconds', 0)
        
        # Determine color class based on score
        score_color = 'text-red-600'
        if avg_score >= 70:
            score_color = 'text-green-600'
        elif avg_score >= 40:
            score_color = 'text-yellow-600'
        
        html_content += f"""
                        <tr>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{display_timestamp}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{provider}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{model}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium {score_color}">{avg_score:.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{valid_scores}/{total_questions}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{duration:.1f}</td>
                        </tr>
        """
    
    # Close table and add category comparison section
    html_content += """
                    </tbody>
                </table>
            </div>
        </section>
        
        <!-- Category Performance Comparison -->
        <section class="mb-10 bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-2xl font-semibold mb-4 text-gray-800">Category Performance by Model</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Transparency</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Fairness</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Safety</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reliability</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Ethics</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Social Impact</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Average</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
    """
    
    # Sort models by average score (high to low)
    sorted_models = sorted(latest_by_model.items(), key=lambda x: x[1].get('average_score', 0), reverse=True)
    
    # Generate category comparison rows
    for model_name, assessment in sorted_models:
        categories = assessment.get('categories', {})
        transparency = categories.get('transparency', 0)
        fairness = categories.get('fairness', 0)
        safety = categories.get('safety', 0)
        reliability = categories.get('reliability', 0)
        ethics = categories.get('ethics', 0)
        social_impact = categories.get('social_impact', 0)
        avg_score = assessment.get('average_score', 0)
        
        # Determine color class based on average score
        score_color = 'text-red-600'
        if avg_score >= 70:
            score_color = 'text-green-600'
        elif avg_score >= 40:
            score_color = 'text-yellow-600'
        
        html_content += f"""
                        <tr>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{model_name}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{transparency:.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{fairness:.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{safety:.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{reliability:.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{ethics:.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{social_impact:.2f}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium {score_color}">{avg_score:.2f}</td>
                        </tr>
        """
    
    # Close the category table
    html_content += """
                    </tbody>
                </table>
            </div>
        </section>
        
        <!-- Top and Bottom Performers -->
        <section class="mb-10">
            <div class="grid md:grid-cols-2 gap-8">
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-2xl font-semibold mb-4 text-gray-800">Top Performing Model</h2>
    """
    
    # Add top performer details
    if sorted_models:
        top_model_name, top_assessment = sorted_models[0]
        top_avg_score = top_assessment.get('average_score', 0)
        
        html_content += f"""
                    <div class="flex flex-col space-y-4">
                        <div class="flex justify-between items-center">
                            <span class="text-xl font-medium text-gray-700">{top_model_name}</span>
                            <span class="text-xl font-bold text-green-600">{top_avg_score:.2f}</span>
                        </div>
                        <p class="text-gray-600">This model performed best overall across all ethical dimensions.</p>
                        <div class="mt-2">
                            <h3 class="text-lg font-medium mb-2">Category Scores:</h3>
                            <table class="min-w-full">
        """
        
        # Add category scores for top model
        categories = top_assessment.get('categories', {})
        for category, score in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            category_display = category.replace('_', ' ').title()
            html_content += f"""
                                <tr>
                                    <td class="py-1 text-gray-700">{category_display}</td>
                                    <td class="py-1 text-right font-medium text-gray-900">{score:.2f}</td>
                                </tr>
            """
        
        html_content += """
                            </table>
                        </div>
                    </div>
        """
    else:
        html_content += """
                    <p class="text-gray-600">No assessment data available.</p>
        """
    
    # Add bottom performer section
    html_content += """
                </div>
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-2xl font-semibold mb-4 text-gray-800">Model with Most Room for Improvement</h2>
    """
    
    # Add bottom performer details
    if sorted_models and len(sorted_models) > 1:
        bottom_model_name, bottom_assessment = sorted_models[-1]
        bottom_avg_score = bottom_assessment.get('average_score', 0)
        
        html_content += f"""
                    <div class="flex flex-col space-y-4">
                        <div class="flex justify-between items-center">
                            <span class="text-xl font-medium text-gray-700">{bottom_model_name}</span>
                            <span class="text-xl font-bold text-red-600">{bottom_avg_score:.2f}</span>
                        </div>
                        <p class="text-gray-600">This model has the most room for ethical improvements.</p>
                        <div class="mt-2">
                            <h3 class="text-lg font-medium mb-2">Category Scores:</h3>
                            <table class="min-w-full">
        """
        
        # Add category scores for bottom model
        categories = bottom_assessment.get('categories', {})
        for category, score in sorted(categories.items(), key=lambda x: x[1]):
            category_display = category.replace('_', ' ').title()
            html_content += f"""
                                <tr>
                                    <td class="py-1 text-gray-700">{category_display}</td>
                                    <td class="py-1 text-right font-medium text-gray-900">{score:.2f}</td>
                                </tr>
            """
        
        html_content += """
                            </table>
                        </div>
                    </div>
        """
    else:
        html_content += """
                    <p class="text-gray-600">No assessment data available for comparison.</p>
        """
    
    # Close the document
    html_content += """
                </div>
            </div>
        </section>
        
        <footer class="mt-12 text-center text-gray-500 text-sm">
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p class="mt-1">Ethical AI Assessment Tool</p>
        </footer>
    </div>
</body>
</html>
    """
    
    # Write the HTML file
    output_path = os.path.join(output_dir, 'index.html')
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        log.info(f"HTML dashboard successfully generated at '{output_path}'")
        print(f"[Dashboard Update] HTML dashboard generated: {output_path}")
    except IOError as e:
        log.error(f"Failed to write HTML dashboard to '{output_path}': {e}")
        print(f"Error: Could not write HTML dashboard file: {e}")
    except Exception as e:
        log.exception(f"An unexpected error occurred during HTML dashboard generation: {e}")
        print(f"Error: An unexpected error occurred generating the HTML dashboard: {e}")

def update_dashboard():
    """Update the dashboard based on latest assessment data."""
    log.info("Updating dashboard...")
    assessment_data = load_assessment_data()
    if assessment_data:
        # Print latest assessment to console
        latest_assessment = assessment_data[-1]
        print(f"[Dashboard Update] Latest assessment for {latest_assessment.get('provider')} ({latest_assessment.get('model')}): Avg Score = {latest_assessment.get('average_score'):.2f}")
        
        # Generate the HTML dashboard file
        generate_html_dashboard(output_dir='docs')
    else:
        log.info("No assessment data found to update dashboard.")

if __name__ == '__main__':
    # Example of how to manually trigger an update (e.g., for testing)
    logging.basicConfig(level=logging.INFO)
    print("Manually triggering dashboard generation...")
    # Ensure the function is called to generate the HTML file
    generate_html_dashboard(output_dir='docs')
