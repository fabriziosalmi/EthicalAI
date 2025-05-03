"""
Dashboard functionality for visualizing assessment results.
"""

import json
import logging
import os
import matplotlib.pyplot as plt
import io
import base64
from config import ASSESSMENT_DATA_FILE, DASHBOARD_DIR
from datetime import datetime
import numpy as np

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

def generate_chart_base64(fig):
    """Convert a matplotlib figure to a base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def create_radar_chart(data, categories, title):
    """Create a radar chart for the given data."""
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Adjust categories for display
    display_categories = [cat.replace('_', ' ').title() for cat in categories]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_categories)
    
    # Add data
    values = [data.get(cat, 0) for cat in categories]
    values += values[:1]  # Close the loop
    
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.1)
    
    # Add title
    plt.title(title, size=15, y=1.1)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    return fig

def create_score_comparison_chart(models_data):
    """Create a bar chart comparing average scores across models."""
    models = []
    scores = []
    
    for model_name, assessment in models_data:
        models.append(model_name)
        scores.append(assessment.get('average_score', 0))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, scores, color='skyblue')
    
    # Add score labels above bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Score')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 100)
    
    # Rotate model names if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def create_category_comparison_chart(models_data, category):
    """Create a bar chart comparing a specific category across models."""
    models = []
    scores = []
    
    for model_name, assessment in models_data:
        models.append(model_name)
        categories = assessment.get('categories', {})
        scores.append(categories.get(category, 0))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, scores, color='lightgreen')
    
    # Add score labels above bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    category_display = category.replace('_', ' ').title()
    ax.set_title(f'{category_display} Score Comparison')
    ax.set_ylim(0, 100)
    
    # Rotate model names if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def create_score_trend_chart(assessment_data, model_name=None):
    """Create a line chart showing score trends over time for a model or all models."""
    # Group data by model
    model_data = {}
    
    for assessment in assessment_data:
        curr_model = assessment.get('model')
        if model_name and curr_model != model_name:
            continue
            
        if curr_model not in model_data:
            model_data[curr_model] = {'dates': [], 'scores': []}
            
        timestamp = assessment.get('timestamp', '')
        if timestamp:
            date = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d')
            model_data[curr_model]['dates'].append(date)
            model_data[curr_model]['scores'].append(assessment.get('average_score', 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Add a line for each model
    for model, data in model_data.items():
        if data['dates'] and data['scores']:
            ax.plot(data['dates'], data['scores'], marker='o', label=model)
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Score')
    if model_name:
        ax.set_title(f'Score Trend for {model_name}')
    else:
        ax.set_title('Score Trends Across Models')
        
    ax.set_ylim(0, 100)
    
    # Add legend if multiple models
    if len(model_data) > 1:
        ax.legend()
    
    # Rotate date labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def create_category_breakdown_chart(assessment):
    """Create a horizontal bar chart showing category scores for an assessment."""
    categories = assessment.get('categories', {})
    
    # Sort categories from highest to lowest score
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    cats = []
    scores = []
    colors = []
    
    for category, score in sorted_categories:
        cat_display = category.replace('_', ' ').title()
        cats.append(cat_display)
        scores.append(score)
        
        # Determine color based on score
        if score >= 70:
            colors.append('green')
        elif score >= 40:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(5, len(cats) * 0.5)))
    bars = ax.barh(cats, scores, color=colors)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.annotate(f'{score:.2f}',
                  xy=(width, bar.get_y() + bar.get_height() / 2),
                  xytext=(5, 0),  # 5 points horizontal offset
                  textcoords="offset points",
                  ha='left', va='center')
    
    # Add labels and title
    ax.set_xlabel('Score')
    ax.set_title(f'Category Breakdown for {assessment.get("model", "Unknown Model")}')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    
    return fig

def generate_html_dashboard(output_dir=DASHBOARD_DIR):
    """Generate a comprehensive HTML dashboard from assessment data."""
    log.info("Generating HTML dashboard...")
    all_assessment_data = load_assessment_data()
    
    if not all_assessment_data:
        log.warning("No assessment data found. Dashboard will be empty.")
    
    # Group latest assessment by model
    latest_by_model = get_latest_assessment_by_model(all_assessment_data)
    
    # Sort models by average score (high to low)
    sorted_models = sorted(latest_by_model.items(), key=lambda x: x[1].get('average_score', 0), reverse=True)
    
    # Get unique categories across all assessments
    all_categories = set()
    for assessment in all_assessment_data:
        if 'categories' in assessment:
            all_categories.update(assessment['categories'].keys())
    all_categories = sorted(list(all_categories))
    
    # Generate charts
    charts = {}
    
    # Model comparison chart
    if sorted_models:
        charts['model_comparison'] = generate_chart_base64(create_score_comparison_chart(sorted_models))
    
    # Category comparison charts for each category
    for category in all_categories:
        charts[f'category_{category}'] = generate_chart_base64(create_category_comparison_chart(sorted_models, category))
    
    # Overall score trends
    charts['score_trends'] = generate_chart_base64(create_score_trend_chart(all_assessment_data))
    
    # Top model category breakdown
    if sorted_models:
        top_model_name, top_assessment = sorted_models[0]
        charts['top_model_breakdown'] = generate_chart_base64(create_category_breakdown_chart(top_assessment))
    
    # Bottom model category breakdown
    if len(sorted_models) > 1:
        bottom_model_name, bottom_assessment = sorted_models[-1]
        charts['bottom_model_breakdown'] = generate_chart_base64(create_category_breakdown_chart(bottom_assessment))
    
    # Top model radar chart
    if sorted_models and all_categories:
        top_model_name, top_assessment = sorted_models[0]
        radar_data = top_assessment.get('categories', {})
        charts['top_model_radar'] = generate_chart_base64(
            create_radar_chart(radar_data, all_categories, f'Performance Profile: {top_model_name}')
        )
    
    # Create the HTML dashboard content
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ethical AI Assessment Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .chart-container { width: 100%; overflow: hidden; }
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
        .tab-button { 
            padding: 0.5rem 1rem;
            border-radius: 0.375rem 0.375rem 0 0;
            font-weight: 500;
            cursor: pointer;
        }
        .tab-button.active {
            background-color: white;
            border-color: #e5e7eb;
            border-bottom-color: white;
            margin-bottom: -1px;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800 p-8">
    <div class="max-w-7xl mx-auto">
        <header class="mb-10">
            <h1 class="text-4xl font-bold text-gray-900">Ethical AI Assessment Dashboard</h1>
            <p class="text-lg text-gray-600 mt-2">Comprehensive analysis of AI model ethical performance</p>
            <div class="flex items-center mt-4">
                <div class="text-sm bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
                    Latest update: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
                </div>
                <div class="ml-4 text-sm text-gray-500">
                    """ + str(len(all_assessment_data)) + """ assessments | """ + str(len(latest_by_model)) + """ unique models
                </div>
            </div>
        </header>

        <!-- Dashboard Tabs -->
        <div class="mb-6">
            <div class="flex border-b border-gray-200">
                <button class="tab-button active" data-tab="overview">Overview</button>
                <button class="tab-button" data-tab="models">Model Comparison</button>
                <button class="tab-button" data-tab="categories">Category Analysis</button>
                <button class="tab-button" data-tab="details">Detailed Results</button>
            </div>
        </div>

        <!-- Overview Tab -->
        <div id="overview-tab" class="tab-content active">
            <!-- Key Stats Cards -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
    """
    
    # Add stats cards
    if sorted_models:
        # Top performer card
        top_model_name, top_assessment = sorted_models[0]
        top_score = top_assessment.get('average_score', 0)
        
        html_content += f"""
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Top Performing Model</h3>
                    <p class="text-3xl font-bold text-green-600">{top_score:.2f}</p>
                    <p class="text-gray-500">{top_model_name}</p>
                </div>
        """
        
        # Average score card
        all_scores = [a.get('average_score', 0) for a in all_assessment_data]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        html_content += f"""
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Average Model Score</h3>
                    <p class="text-3xl font-bold text-blue-600">{avg_score:.2f}</p>
                    <p class="text-gray-500">Across all assessments</p>
                </div>
        """
        
        # Lowest performer card
        if len(sorted_models) > 1:
            bottom_model_name, bottom_assessment = sorted_models[-1]
            bottom_score = bottom_assessment.get('average_score', 0)
            
            html_content += f"""
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h3 class="text-lg font-semibold text-gray-700 mb-2">Model Needing Improvement</h3>
                    <p class="text-3xl font-bold text-red-600">{bottom_score:.2f}</p>
                    <p class="text-gray-500">{bottom_model_name}</p>
                </div>
            """
    
    html_content += """
            </div>

            <!-- Charts -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
    """
    
    # Add model comparison chart
    if 'model_comparison' in charts:
        html_content += f"""
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Model Performance Comparison</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{charts['model_comparison']}" class="w-full" alt="Model Comparison">
                    </div>
                </div>
        """
    
    # Add score trends chart
    if 'score_trends' in charts:
        html_content += f"""
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Score Trends Over Time</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{charts['score_trends']}" class="w-full" alt="Score Trends">
                    </div>
                </div>
        """
    
    # Add top model radar chart
    if 'top_model_radar' in charts:
        html_content += f"""
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Top Model Performance Profile</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{charts['top_model_radar']}" class="w-full" alt="Top Model Radar">
                    </div>
                </div>
        """
    
    # Add top model breakdown chart
    if 'top_model_breakdown' in charts:
        html_content += f"""
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">Top Model Category Breakdown</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{charts['top_model_breakdown']}" class="w-full" alt="Top Model Breakdown">
                    </div>
                </div>
        """
    
    html_content += """
            </div>
        </div>

        <!-- Model Comparison Tab -->
        <div id="models-tab" class="tab-content">
            <div class="bg-white p-6 rounded-lg shadow-md mb-8">
                <h2 class="text-2xl font-semibold mb-6">Model Performance Comparison</h2>
                
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Provider</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Average Score</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Assessed</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reports</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
    """
    
    # Add model comparison rows
    for rank, (model_name, assessment) in enumerate(sorted_models, 1):
        provider = assessment.get('provider', 'N/A')
        avg_score = assessment.get('average_score', 0)
        timestamp = assessment.get('timestamp', '')
        display_date = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d') if timestamp else 'N/A'
        assessment_date = assessment.get('assessment_date', '').replace(' ', '_')
        
        # Determine score color class and badge
        score_color = 'text-red-600'
        badge_class = 'bg-red-100 text-red-800'
        if avg_score >= 70:
            score_color = 'text-green-600'
            badge_class = 'bg-green-100 text-green-800'
        elif avg_score >= 40:
            score_color = 'text-yellow-600'
            badge_class = 'bg-yellow-100 text-yellow-800'
        
        # Build report links if available
        if assessment.get('filesystem_date') or assessment.get('timestamp'):
            # First try to use the filesystem_date which is already properly formatted
            formatted_timestamp = ""
            if assessment.get('filesystem_date'):
                # Convert filesystem_date (2025-05-03 14_46_11) to timestamp format (20250503_144611)
                try:
                    fs_date = assessment.get('filesystem_date')
                    date_part = fs_date.split(' ')[0].replace('-', '')
                    time_part = fs_date.split(' ')[1].replace('_', '')
                    formatted_timestamp = f"{date_part}_{time_part}"
                except Exception as e:
                    log.warning(f"Error formatting filesystem_date: {e}")
            
            # If filesystem_date didn't work, fall back to timestamp
            if not formatted_timestamp and assessment.get('timestamp'):
                try:
                    # Format from ISO timestamp
                    timestamp_obj = datetime.fromisoformat(assessment.get('timestamp'))
                    formatted_timestamp = timestamp_obj.strftime("%Y%m%d_%H%M%S")
                except Exception as e:
                    log.warning(f"Error formatting timestamp: {e}")
            
            # Ensure model name is properly sanitized for filenames, just like in assessment.py
            safe_model_name = model_name.lower()
            if formatted_timestamp:
                report_links = f"""
                    <div class="flex space-x-2">
                        <a href="reports/{formatted_timestamp}_{provider.lower()}_{safe_model_name}_assessment.html" class="px-3 py-1 bg-blue-100 text-blue-600 rounded hover:bg-blue-200 transition-colors duration-200 inline-flex items-center text-xs font-medium" target="_blank">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            HTML
                        </a>
                        <a href="reports/{formatted_timestamp}_{provider.lower()}_{safe_model_name}_assessment.pdf" class="px-3 py-1 bg-red-100 text-red-600 rounded hover:bg-red-200 transition-colors duration-200 inline-flex items-center text-xs font-medium" target="_blank">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 01-2 2z" />
                            </svg>
                            PDF
                        </a>
                    </div>
                """
            else:
                report_links = "<span class='text-gray-400 italic text-xs'>Report links unavailable</span>"
        else:
            report_links = "<span class='text-gray-400 italic text-xs'>No reports available</span>"
        
        html_content += f"""
                            <tr class="hover:bg-gray-50 transition-colors duration-150">
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center font-semibold text-gray-600">{rank}</div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="text-sm font-medium text-gray-900">{model_name}</div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                                        {provider}
                                    </span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="w-full bg-gray-200 rounded-full h-2.5 mb-2">
                                        <div class="{score_color.replace('text', 'bg')} h-2.5 rounded-full" style="width: {min(100, avg_score)}%"></div>
                                    </div>
                                    <div class="text-sm font-medium {score_color}">{avg_score:.2f}</div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{display_date}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{report_links}</td>
                            </tr>
        """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Individual Model Charts -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
    """
    
    # Add breakdown charts for each model
    for model_name, assessment in sorted_models:
        if len(sorted_models) <= 2 and 'top_model_breakdown' in charts and model_name == sorted_models[0][0]:
            # We already showed this in the overview for the top model
            continue
            
        if len(sorted_models) <= 2 and 'bottom_model_breakdown' in charts and model_name == sorted_models[-1][0]:
            # We already showed this in the overview for the bottom model
            continue
            
        # Generate category breakdown for this model
        model_breakdown_chart = generate_chart_base64(create_category_breakdown_chart(assessment))
        
        html_content += f"""
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4">{model_name} - Category Breakdown</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{model_breakdown_chart}" class="w-full" alt="{model_name} Breakdown">
                    </div>
                </div>
        """
    
    html_content += """
            </div>
        </div>

        <!-- Category Analysis Tab -->
        <div id="categories-tab" class="tab-content">
            <div class="bg-white p-6 rounded-lg shadow-md mb-8">
                <h2 class="text-2xl font-semibold mb-6">Category Performance Analysis</h2>
                
                <div class="mb-6">
                    <p class="text-gray-600">This section compares how different models perform across each ethical dimension.</p>
                </div>
                
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
    """
    
    # Add category comparison charts
    for category in all_categories:
        category_display = category.replace('_', ' ').title()
        if f'category_{category}' in charts:
            html_content += f"""
                    <div class="bg-white border border-gray-200 rounded-lg shadow-sm">
                        <div class="border-b border-gray-200 px-4 py-3 bg-gray-50 rounded-t-lg">
                            <h3 class="text-lg font-medium text-gray-800">{category_display}</h3>
                        </div>
                        <div class="p-4">
                            <div class="chart-container">
                                <img src="data:image/png;base64,{charts[f'category_{category}']}" class="w-full" alt="{category_display} Comparison">
                            </div>
                        </div>
                    </div>
            """
    
    html_content += """
                </div>
            </div>
            
            <!-- Category Statistics -->
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-xl font-semibold mb-4">Category Statistics</h2>
                
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Score</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Best Model</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Best Score</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Worst Model</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Worst Score</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
    """
    
    # Calculate category statistics
    category_stats = {}
    for category in all_categories:
        category_stats[category] = {
            'total': 0,
            'count': 0,
            'best_model': '',
            'best_score': 0,
            'worst_model': '',
            'worst_score': 100
        }
    
    # Collect statistics
    for model_name, assessment in sorted_models:
        categories = assessment.get('categories', {})
        for category, score in categories.items():
            category_stats[category]['total'] += score
            category_stats[category]['count'] += 1
            
            if score > category_stats[category]['best_score']:
                category_stats[category]['best_score'] = score
                category_stats[category]['best_model'] = model_name
                
            if score < category_stats[category]['worst_score']:
                category_stats[category]['worst_score'] = score
                category_stats[category]['worst_model'] = model_name
    
    # Add category statistics rows
    for category in all_categories:
        category_display = category.replace('_', ' ').title()
        stats = category_stats[category]
        
        avg_score = stats['total'] / stats['count'] if stats['count'] > 0 else 0
        
        # Determine color classes
        avg_color = 'text-red-600'
        if avg_score >= 70:
            avg_color = 'text-green-600'
        elif avg_score >= 40:
            avg_color = 'text-yellow-600'
            
        best_color = 'text-green-600'
        worst_color = 'text-red-600'
        
        html_content += f"""
                            <tr>
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{category_display}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium {avg_color}">{avg_score:.2f}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{stats['best_model']}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium {best_color}">{stats['best_score']:.2f}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{stats['worst_model']}</td>
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium {worst_color}">{stats['worst_score']:.2f}</td>
                            </tr>
        """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Detailed Results Tab -->
        <div id="details-tab" class="tab-content">
            <!-- Assessment History Table -->
            <section class="mb-10 bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-2xl font-semibold mb-4 text-gray-800">Assessment History</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Provider</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Score</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Valid/Total Qs</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration (s)</th>
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reports</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
    """
    
    # Sort the data for display in the table (most recent first)
    sorted_data = sorted(all_assessment_data, key=lambda x: x.get('timestamp', ''), reverse=True)
    
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
        score_bg = 'bg-red-100'
        if avg_score >= 70:
            score_color = 'text-green-600'
            score_bg = 'bg-green-100'
        elif avg_score >= 40:
            score_color = 'text-yellow-600'
            score_bg = 'bg-yellow-100'
        
        # Build report links
        assessment_date_str = assessment.get('assessment_date', '')
        report_links = ""
        
        # Only create links if we have a proper assessment date
        if assessment.get('filesystem_date') or assessment_date_str:
            try:
                # Use filesystem_date if available, otherwise format from assessment_date
                formatted_timestamp = ""
                if assessment.get('filesystem_date'):
                    fs_date = assessment.get('filesystem_date')
                    date_part = fs_date.split(' ')[0].replace('-', '')
                    time_part = fs_date.split(' ')[1].replace('_', '')
                    formatted_timestamp = f"{date_part}_{time_part}"
                elif assessment_date_str and ' ' in assessment_date_str:
                    date_part = assessment_date_str.split(' ')[0].replace('-', '')
                    time_part = assessment_date_str.split(' ')[1].replace(':', '')
                    formatted_timestamp = f"{date_part}_{time_part}"
                
                # Use model name as-is to match actual filenames
                safe_model = model.lower()
                
                if formatted_timestamp:
                    report_links = f"""
                        <div class="flex space-x-2">
                            <a href="reports/{formatted_timestamp}_{provider.lower()}_{safe_model}_assessment.html" class="px-2 py-1 bg-blue-100 text-blue-600 rounded hover:bg-blue-200 transition-colors duration-200 inline-flex items-center text-xs font-medium" target="_blank">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                HTML
                            </a>
                            <a href="reports/{formatted_timestamp}_{provider.lower()}_{safe_model}_assessment.pdf" class="px-2 py-1 bg-red-100 text-red-600 rounded hover:bg-red-200 transition-colors duration-200 inline-flex items-center text-xs font-medium" target="_blank">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 01-2 2z" />
                                </svg>
                                PDF
                            </a>
                        </div>
                    """
                else:
                    report_links = "<span class='text-gray-400 italic text-xs'>Reports unavailable</span>"
            except (IndexError, AttributeError) as e:
                log.warning(f"Could not create report links for assessment {timestamp}: {e}")
                report_links = "<span class='text-gray-400 italic text-xs'>Reports unavailable</span>"
        else:
            report_links = "<span class='text-gray-400 italic text-xs'>No reports available</span>"
        
        html_content += f"""
                            <tr class="hover:bg-gray-50 transition-colors duration-150">
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="text-sm text-gray-900">{display_timestamp}</div>
                                    <div class="text-xs text-gray-500 mt-1">ID: {timestamp.split('T')[0]}</div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                                        {provider}
                                    </span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="text-sm font-medium text-gray-900">{model}</div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <span class="px-3 py-1 rounded-full text-xs font-medium {score_bg} {score_color}">
                                        {avg_score:.2f}
                                    </span>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="text-sm text-gray-900">{valid_scores}/{total_questions}</div>
                                    <div class="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                                        <div class="bg-blue-600 h-1.5 rounded-full" style="width: {(valid_scores/total_questions*100) if total_questions else 0}%"></div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="text-sm text-gray-500">
                                        <span class="font-medium">{duration:.1f}</span> sec
                                    </div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    {report_links}
                                </td>
                            </tr>
        """
    
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
    """
    
    # Add category headers
    for category in all_categories:
        category_display = category.replace('_', ' ').title()
        html_content += f"""
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{category_display}</th>
        """
    
    html_content += """
                                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Average</th>
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
    """
    
    # Generate category comparison rows
    for model_name, assessment in sorted_models:
        categories = assessment.get('categories', {})
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
        """
        
        # Add category scores
        for category in all_categories:
            score = categories.get(category, 0)
            cat_color = 'text-red-600'
            if score >= 70:
                cat_color = 'text-green-600'
            elif score >= 40:
                cat_color = 'text-yellow-600'
                
            html_content += f"""
                                <td class="px-6 py-4 whitespace-nowrap text-sm {cat_color}">{score:.2f}</td>
            """
        
        html_content += f"""
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium {score_color}">{avg_score:.2f}</td>
                            </tr>
        """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
        
        <footer class="mt-12 text-center text-gray-500 text-sm">
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p class="mt-1"><a href="https://github.com/fabriziosalmi/ethical-ai" target="_blank">Ethical AI Assessment Tool</a></p>
        </footer>
    </div>

    <script>
        // Tab switching functionality
        document.addEventListener('DOMContentLoaded', function() {
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    // Remove active class from all buttons and contents
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    // Add active class to clicked button and corresponding content
                    button.classList.add('active');
                    const tabId = button.getAttribute('data-tab');
                    document.getElementById(`${tabId}-tab`).classList.add('active');
                });
            });
        });
    </script>
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
