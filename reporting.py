"""
Functions for generating reports (HTML, PDF, Markdown) from assessment results.
"""

import os
import re
import logging
import base64
import markdown
import weasyprint
from datetime import datetime
from jinja2 import Template
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from config import RESULTS_DIR

log = logging.getLogger(__name__)

# HTML template for the assessment report
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --accent-color: #e74c3c;
            --background-color: #f8f9fa;
            --card-bg-color: #ffffff;
            --text-color: #2c3e50;
            --border-color: #e0e0e0;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background-color: var(--card-bg-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        h1 {
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
        
        h2 {
            color: var (--secondary-color);
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        h3 {
            color: var(--secondary-color);
            margin-top: 1.5rem;
        }
        
        .metadata {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 2rem.
        }
        
        .metadata-item {
            background-color: var(--card-bg-color);
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }
        
        .metadata-item strong {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--secondary-color);
        }
        
        .methodology {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 2rem.
        }
        
        .method-item {
            background-color: var(--card-bg-color);
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var (--border-color);
        }
        
        .method-item strong {
            display: block;
            color: var(--secondary-color);
        }
        
        .highlight-box {
            background-color: var(--primary-color);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            margin: 2rem 0;
        }
        
        .highlight-box h3 {
            color: white;
            margin-top: 0;
        }
        
        .score {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        
        .score-high {
            color: var(--success-color);
        }
        
        .score-medium {
            color: var(--warning-color);
        }
        
        .score-low {
            color: var(--danger-color);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
        }
        
        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background-color: var(--primary-color);
            color: white;
        }
        
        tr:nth-child(even) {
            background-color: rgba(0, 0, 0, 0.02);
        }
        
        tr:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .summary-item {
            background-color: var(--card-bg-color);
            padding: 1.5rem;
            border-radius: 6px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .summary-item strong {
            display: block;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: var(--primary-color);
        }
        
        .summary-item span {
            color: var(--secondary-color);
        }
        
        .charts {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .chart {
            background-color: var(--card-bg-color);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            text-align: center;
        }
        
        .chart img {
            max-width: 100%;
            height: auto;
        }
        
        footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            color: var(--secondary-color);
            font-size: 0.9rem;
        }
        
        @media print {
            body {
                background-color: white;
            }
            
            .container {
                max-width: 100%;
                margin: 0;
                padding: 1rem;
                border: none;
            }
            
            @page {
                margin: 1.5cm;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p>Generated on {{ current_date }}</p>
        </header>
        
        <div class="metadata">
            <div class="metadata-item">
                <strong>API Provider:</strong>
                {{ provider }}
            </div>
            <div class="metadata-item">
                <strong>Model:</strong>
                {{ model }}
            </div>
            <div class="metadata-item">
                <strong>Assessment Date:</strong>
                {{ assessment_date }}
            </div>
            <div class="metadata-item">
                <strong>Duration:</strong>
                {{ duration }}
            </div>
        </div>
        
        <h2>Methodology</h2>
        <div class="methodology">
            <div class="method-item">
                <strong>Samples per Question:</strong>
                {{ samples_per_question }}
            </div>
            <div class="method-item">
                <strong>Aggregation:</strong>
                {{ aggregation }}
            </div>
            <div class="method-item">
                <strong>Base Temperature:</strong>
                {{ base_temperature }}
            </div>
            <div class="method-item">
                <strong>Temp Range (samples 2+):</strong>
                {{ temp_range }}
            </div>
            <div class="method-item">
                <strong>Edge Case Retries:</strong>
                {{ edge_case_retries }}
            </div>
            <div class="method-item">
                <strong>Reasoning Tag Stripping:</strong>
                {{ tag_stripping }}
            </div>
            <div class="method-item">
                <strong>Score Range:</strong>
                {{ score_range }}
            </div>
        </div>
        
        <div class="highlight-box">
            <h3>Overall Result</h3>
            <div class="score {{ score_class }}">{{ average_score }} / {{ max_score }}</div>
            <p>Based on {{ valid_scores }} valid final scores out of {{ total_questions }} questions</p>
        </div>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-item">
                <strong>{{ total_questions }}</strong>
                <span>Total Questions Asked</span>
            </div>
            <div class="summary-item">
                <strong>{{ valid_scores }}</strong>
                <span>Questions with Valid Scores</span>
            </div>
            <div class="summary-item">
                <strong>{{ invalid_scores }}</strong>
                <span>Questions without Valid Scores</span>
            </div>
        </div>
        
        {% if charts %}
        <h2>Visualizations</h2>
        <div class="charts">
            {% for chart in charts %}
            <div class="chart">
                <h3>{{ chart.title }}</h3>
                <img src="{{ chart.src }}" alt="{{ chart.title }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Final Score (Median of {{ samples_per_question }})</th>
                </tr>
            </thead>
            <tbody>
                {% for question, score in results %}
                <tr>
                    <td>{{ question }}</td>
                    <td>{{ score }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <footer>
            <p>Generated by Ethical AI Assessment Tool | {{ current_date }}</p>
        </footer>
    </div>
</body>
</html>
"""

def generate_html_report(markdown_file: str, include_charts: bool = True) -> Optional[str]:
    """
    Generate an HTML report from a markdown assessment report.
    
    Args:
        markdown_file: Path to the markdown assessment report
        include_charts: Whether to include charts in the HTML report
        
    Returns:
        Path to the generated HTML report or None if generation failed
    """
    try:
        if not os.path.exists(markdown_file):
            log.error(f"Markdown file not found: {markdown_file}")
            return None
            
        log.info(f"Generating HTML report from {markdown_file}")
        
        # Extract data from markdown file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse basic information
        provider_match = re.search(r'\*\*API Provider:\*\* `([^`]+)`', content)
        model_match = re.search(r'\*\*Model:\*\* `([^`]+)`', content)
        date_match = re.search(r'\*\*Assessment Date:\*\* (.*)', content)
        duration_match = re.search(r'\*\*Duration:\*\* (.*)', content)
        
        # Parse methodology
        samples_match = re.search(r'\*\*Samples per Question:\*\* (\d+)', content)
        temp_match = re.search(r'\*\*Base Temperature:\*\* ([0-9.]+)', content)
        temp_range_match = re.search(r'\*\*Random Temperature Range \(samples 2\+\):\*\* \[([^]]+)\]', content)
        retries_match = re.search(r'\*\*Edge Case Retries \(0/100\):\*\* ([^(]+)', content)
        strip_match = re.search(r'\*\*Reasoning Tag Stripping:\*\* (Enabled|Disabled)', content)
        range_match = re.search(r'\*\*Score Range Used:\*\* ([0-9]+-[0-9]+)', content)
        
        # Parse scores
        avg_score_match = re.search(r'\*\*Final Score \(Average\):\*\* \*\*([0-9.]+)', content)
        valid_scores_match = re.search(r'Based on (\d+) valid final scores out of (\d+) questions', content)
        
        # Parse table with questions and scores
        results = []
        table_pattern = r'\| (.*?) \| (.*?) \|'
        table_matches = re.findall(table_pattern, content)
        for q, s in table_matches:
            if q != "Question" and q != "-":  # Skip header and separator
                results.append((q, s))
        
        # Prepare data for template
        if provider_match:
            provider = provider_match.group(1)
        else:
            provider = "Unknown"
            
        if model_match:
            model = model_match.group(1)
        else:
            model = "Unknown"
        
        title = f"Ethical AI Assessment Report - {provider}"
        
        template_data = {
            "title": title,
            "current_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "provider": provider,
            "model": model,
            "assessment_date": date_match.group(1) if date_match else "Unknown",
            "duration": duration_match.group(1) if duration_match else "Unknown",
            "samples_per_question": samples_match.group(1) if samples_match else "Unknown",
            "aggregation": "Median of valid scores",
            "base_temperature": temp_match.group(1) if temp_match else "Unknown",
            "temp_range": temp_range_match.group(1) if temp_range_match else "Unknown",
            "edge_case_retries": retries_match.group(1) if retries_match else "Unknown",
            "tag_stripping": strip_match.group(1) if strip_match else "Unknown",
            "score_range": range_match.group(1) if range_match else "Unknown",
            "results": results
        }
        
        # Handle scores and set appropriate styling class
        if avg_score_match and valid_scores_match:
            avg_score = float(avg_score_match.group(1))
            valid_count = int(valid_scores_match.group(1))
            total_count = int(valid_scores_match.group(2))
            
            template_data["average_score"] = f"{avg_score:.2f}"
            template_data["valid_scores"] = valid_count
            template_data["total_questions"] = total_count
            template_data["invalid_scores"] = total_count - valid_count
            template_data["max_score"] = "100"
            
            # Set score class for styling
            if avg_score >= 70:
                template_data["score_class"] = "score-high"
            elif avg_score >= 40:
                template_data["score_class"] = "score-medium"
            else:
                template_data["score_class"] = "score-low"
        
        else:
            template_data["average_score"] = "0.00"
            template_data["max_score"] = "100"
            template_data["score_class"] = "score-low"
            template_data["valid_scores"] = 0
            template_data["total_questions"] = 0
            template_data["invalid_scores"] = 0
        
        # Include charts if requested
        if include_charts:
            # Look for visualization charts in the directory
            viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
            if os.path.exists(viz_dir):
                timestamp_pattern = re.search(r'\/(\d{8}_\d{6})_', markdown_file)
                timestamp = timestamp_pattern.group(1) if timestamp_pattern else None
                
                # If we have a timestamp, look for charts with that timestamp
                if timestamp:
                    chart_files = [f for f in os.listdir(viz_dir) if f.startswith(timestamp) and f.endswith('.png') and provider.lower() in f.lower()]
                else:
                    # Otherwise, just look for charts with the provider name
                    chart_files = [f for f in os.listdir(viz_dir) if f.endswith('.png') and provider.lower() in f.lower()]
                
                # Sort files by creation time (newest first) so we get the most recent charts
                if not chart_files and os.listdir(viz_dir):
                    # Fallback: try to find any charts for this provider
                    chart_files = [f for f in os.listdir(viz_dir) if f.endswith('.png') and provider.lower() in f.lower()]
                    if chart_files:
                        chart_files.sort(key=lambda f: os.path.getctime(os.path.join(viz_dir, f)), reverse=True)
                        # Take only the most recent set (they should have the same timestamp)
                        most_recent_time = os.path.getctime(os.path.join(viz_dir, chart_files[0]))
                        chart_files = [f for f in chart_files if abs(os.path.getctime(os.path.join(viz_dir, f)) - most_recent_time) < 10]  # Within 10 seconds
                
                # If we have charts, include them in the template
                if chart_files:
                    charts = []
                    for chart_file in chart_files:
                        chart_path = os.path.join(viz_dir, chart_file)
                        # Convert image to base64 to embed in HTML
                        with open(chart_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # Determine title from filename
                        if 'distribution' in chart_file:
                            title = 'Score Distribution'
                        elif 'category_scores' in chart_file:
                            title = 'Category Scores'
                        elif 'radar' in chart_file:
                            category = chart_file.split('_')[-2] if len(chart_file.split('_')) > 3 else 'Category'
                            title = f'{category.capitalize()} Scores'
                        else:
                            title = 'Chart'
                        
                        charts.append({
                            'title': title,
                            'src': f"data:image/png;base64,{img_data}"
                        })
                    
                    template_data["charts"] = charts
                else:
                    template_data["charts"] = []
                    log.warning(f"No visualization charts found for provider '{provider}' in {viz_dir}")
            else:
                template_data["charts"] = []
                log.warning(f"Visualization directory not found: {viz_dir}")
        else:
            template_data["charts"] = []
        
        # Render template
        template = Template(HTML_TEMPLATE)
        html_content = template.render(**template_data)
        
        # Create output filepath
        if markdown_file.endswith('.md'):
            html_file = markdown_file.replace('.md', '.html')
        else:
            # In case the file doesn't have .md extension
            html_file = markdown_file + '.html'
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(html_file)), exist_ok=True)
        
        # Write HTML to file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        log.info(f"HTML report generated at {html_file}")
        return html_file
    
    except Exception as e:
        log.error(f"Error generating HTML report: {e}", exc_info=True)
        return None

def generate_pdf_report(markdown_file: str, html_file: Optional[str] = None) -> Optional[str]:
    """
    Generate a PDF report from a markdown or HTML assessment report.
    
    Args:
        markdown_file: Path to the markdown assessment report
        html_file: Path to an HTML file to use instead of generating from markdown
        
    Returns:
        Path to the generated PDF report or None if generation failed
    """
    try:
        if not markdown_file:
            log.error("No markdown file provided for PDF generation")
            return None
            
        if not os.path.exists(markdown_file):
            log.error(f"Markdown file not found: {markdown_file}")
            return None
            
        log.info(f"Generating PDF report from {'HTML file' if html_file else 'markdown file'}")
        
        # If no HTML file provided, generate one
        if not html_file:
            html_file = generate_html_report(markdown_file)
            if not html_file:
                log.error("Failed to generate HTML report for PDF conversion")
                return None
        elif not os.path.exists(html_file):
            log.error(f"Provided HTML file not found: {html_file}")
            html_file = generate_html_report(markdown_file)
            if not html_file:
                log.error("Failed to generate HTML report for PDF conversion")
                return None
        
        # Create output filepath
        if markdown_file.endswith('.md'):
            pdf_file = markdown_file.replace('.md', '.pdf')
        else:
            # In case the file doesn't have .md extension
            pdf_file = markdown_file + '.pdf'
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(pdf_file)), exist_ok=True)
        
        try:
            # Generate PDF from HTML using WeasyPrint
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            html = weasyprint.HTML(string=html_content, base_url=os.path.dirname(os.path.abspath(html_file)))
            css = weasyprint.CSS(string='@page { size: A4; margin: 1cm; }')
            html.write_pdf(pdf_file, stylesheets=[css])
            
            if not os.path.exists(pdf_file):
                log.error(f"PDF file was not created at the expected location: {pdf_file}")
                return None
                
            log.info(f"PDF report successfully generated at {pdf_file}")
            return pdf_file
        except Exception as e:
            log.error(f"Error during PDF generation with WeasyPrint: {e}", exc_info=True)
            return None
    
    except Exception as e:
        log.error(f"Error generating PDF report: {e}", exc_info=True)
        return None

def generate_comparison_report(
    provider_scores: Dict, 
    comparison_dir: str, 
    all_categories: set = None
) -> str:
    """
    Generate a markdown comparison report for multiple providers.
    
    Args:
        provider_scores: Dictionary of provider scores
        comparison_dir: Directory to save the report
        all_categories: Set of all categories (optional)
        
    Returns:
        Path to the generated report
    """
    try:
        from tabulate import tabulate
        from collections import defaultdict
        
        if not provider_scores:
            log.error("No provider scores provided for comparison report")
            return ""
            
        if not comparison_dir:
            log.error("No comparison directory specified")
            return ""
            
        # Ensure the comparison directory exists
        os.makedirs(comparison_dir, exist_ok=True)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = os.path.join(comparison_dir, f"{timestamp}_providers_comparison_report.md")
        
        providers_with_scores = [p for p, s in provider_scores.items() if s.get('average') is not None]
        
        if not providers_with_scores:
            log.warning("No providers with valid scores found for comparison report")
            return ""
            
        log.info(f"Generating comparison report for {len(providers_with_scores)} providers")
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("# Ethical AI Assessment - Provider Comparison Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overall Scores\n\n")
            overall_table = [
                ["Provider", "Average Score", "Valid Question Count", "Assessment Date"]
            ]
            
            for provider, scores in sorted(provider_scores.items(), 
                                          key=lambda x: x[1].get('average', 0), 
                                          reverse=True):
                assessment_date = scores.get('assessment_date', 'Unknown')
                overall_table.append([
                    provider.upper(),
                    f"{scores.get('average', 0):.2f}",
                    str(scores.get('valid_count', 0)),
                    assessment_date
                ])
            
            f.write(tabulate(overall_table, headers="firstrow", tablefmt="pipe"))
            f.write("\n\n")
            
            # Add comparison charts - verify files exist first
            f.write("## Visual Comparisons\n\n")
            overall_chart_path = os.path.join(comparison_dir, "provider_comparison_overall.png")
            radar_chart_path = os.path.join(comparison_dir, "provider_comparison_radar.png")
            
            if os.path.exists(overall_chart_path):
                f.write(f"![Overall Comparison](provider_comparison_overall.png)\n\n")
            else:
                f.write("*Overall comparison chart not available*\n\n")
                
            if os.path.exists(radar_chart_path):
                f.write(f"![Category Comparison](provider_comparison_radar.png)\n\n")
            else:
                f.write("*Category comparison radar chart not available*\n\n")
            
            # Add category-specific comparisons
            if all_categories and any('categories' in scores for scores in provider_scores.values()):
                f.write("## Category Scores\n\n")
                
                # Organize by category scores
                category_comparison = defaultdict(list)
                for category in sorted(all_categories):
                    for provider, scores in provider_scores.items():
                        if 'categories' in scores and category in scores['categories']:
                            score = scores['categories'][category]
                            category_comparison[category].append((provider, score))
                
                # Write comparison tables for each category, sorted by score
                for category, provider_scores_list in sorted(category_comparison.items()):
                    if not provider_scores_list:
                        continue
                        
                    f.write(f"### {category.capitalize()}\n\n")
                    
                    # Sort providers by category score (highest first)
                    provider_scores_list.sort(key=lambda x: x[1], reverse=True)
                    
                    category_table = [["Provider", f"{category.capitalize()} Score"]]
                    for provider, score in provider_scores_list:
                        if isinstance(score, (int, float)):
                            formatted_score = f"{score:.2f}"
                        else:
                            formatted_score = 'N/A'
                        category_table.append([provider.upper(), formatted_score])
                    
                    f.write(tabulate(category_table, headers="firstrow", tablefmt="pipe"))
                    f.write("\n\n")
            
            # Add detailed question comparisons - use a more robust approach to find questions
            f.write("## Detailed Question Comparisons\n\n")
            
            # Get all questions from all providers to enable cross-provider comparison
            all_questions = set()
            for scores in provider_scores.values():
                if 'questions' in scores:
                    all_questions.update(scores['questions'].keys())
            
            if all_questions:
                # First, list highlights for each provider
                for provider, scores in provider_scores.items():
                    f.write(f"### {provider.upper()} Highlights\n\n")
                    
                    if 'questions' in scores and any(s is not None for s in scores['questions'].values()):
                        # Sort questions by score
                        sorted_questions = sorted(
                            [(q, s) for q, s in scores['questions'].items() if s is not None],
                            key=lambda x: x[1],
                            reverse=True
                        )
                        
                        if sorted_questions:
                            # Top scores
                            f.write("#### Top 5 Highest Scores\n\n")
                            top_table = [["Question", "Score"]]
                            for q, s in sorted_questions[:min(5, len(sorted_questions))]:
                                question_text = q[:100] + "..." if len(q) > 100 else q
                                top_table.append([question_text, str(s)])
                            f.write(tabulate(top_table, headers="firstrow", tablefmt="pipe"))
                            f.write("\n\n")
                            
                            # Lowest scores
                            f.write("#### 5 Lowest Scores\n\n")
                            bottom_table = [["Question", "Score"]]
                            for q, s in sorted_questions[max(-5, -len(sorted_questions)):]:
                                question_text = q[:100] + "..." if len(q) > 100 else q
                                bottom_table.append([question_text, str(s)])
                            f.write(tabulate(bottom_table, headers="firstrow", tablefmt="pipe"))
                            f.write("\n\n")
                    else:
                        f.write("No detailed question scores available.\n\n")
                
                # Then, add a cross-provider question comparison if we have multiple providers with questions
                providers_with_questions = [p for p, s in provider_scores.items() if 'questions' in s and s['questions']]
                if len(providers_with_questions) > 1:
                    f.write("### Cross-Provider Question Comparison\n\n")
                    f.write("Questions with the largest score differences across providers:\n\n")
                    
                    # Find questions that appear in multiple providers and calculate the score variance
                    question_comparison = []
                    for question in all_questions:
                        scores = []
                        providers_for_q = []
                        for provider, provider_data in provider_scores.items():
                            if 'questions' in provider_data and question in provider_data['questions']:
                                score = provider_data['questions'][question]
                                if score is not None:
                                    scores.append(score)
                                    providers_for_q.append(provider)
                        
                        # Only include questions that have scores from multiple providers
                        if len(scores) > 1:
                            score_variance = max(scores) - min(scores)
                            avg_score = sum(scores) / len(scores)
                            question_comparison.append({
                                'question': question,
                                'variance': score_variance,
                                'avg_score': avg_score,
                                'providers': providers_for_q,
                                'scores': scores
                            })
                    
                    # Sort by variance (largest first) and take top 10
                    question_comparison.sort(key=lambda x: x['variance'], reverse=True)
                    top_variance_questions = question_comparison[:min(10, len(question_comparison))]
                    
                    if top_variance_questions:
                        variance_table = [["Question", "Variance", "Providers", "Scores"]]
                        for item in top_variance_questions:
                            # Truncate question text for display
                            question_text = item['question'][:80] + "..." if len(item['question']) > 80 else item['question']
                            providers_text = ", ".join(p.upper() for p in item['providers'])
                            scores_text = ", ".join(f"{s:.1f}" for s in item['scores'])
                            variance_table.append([
                                question_text,
                                f"{item['variance']:.1f}",
                                providers_text,
                                scores_text
                            ])
                        f.write(tabulate(variance_table, headers="firstrow", tablefmt="pipe"))
                        f.write("\n\n")
                    else:
                        f.write("No questions with significant variance found.\n\n")
            else:
                f.write("No detailed question data available for comparison.\n\n")
            
            # Add conclusion with summary of findings
            f.write("## Conclusion\n\n")
            
            # Sort providers by average score
            ranked_providers = sorted(
                [(p, s.get('average', 0)) for p, s in provider_scores.items() if s.get('average') is not None],
                key=lambda x: x[1],
                reverse=True
            )
            
            if ranked_providers:
                top_provider, top_score = ranked_providers[0]
                f.write(f"Overall, **{top_provider.upper()}** achieved the highest score ({top_score:.2f}/100) ")
                f.write(f"among the {len(ranked_providers)} providers evaluated.\n\n")
                
                # Add category-specific insights if available
                if all_categories and any('categories' in scores for scores in provider_scores.values()):
                    top_by_category = {}
                    
                    for category in sorted(all_categories):
                        providers_with_category = [(p, s['categories'][category]) 
                                               for p, s in provider_scores.items() 
                                               if 'categories' in s and category in s['categories']]
                        
                        if providers_with_category:
                            top_in_category = max(providers_with_category, key=lambda x: x[1])
                            top_by_category[category] = top_in_category
                    
                    if top_by_category:
                        f.write("### Category Leaders\n\n")
                        for category, (provider, score) in sorted(top_by_category.items()):
                            f.write(f"* **{category.capitalize()}**: {provider.upper()} ({score:.2f}/100)\n")
                        f.write("\n")
            else:
                f.write("No valid scores available for comparison.\n\n")
                
            f.write("---\n\n")
            f.write(f"Report generated by Ethical AI Assessment Tool on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}")
        
        log.info(f"Comparison report generated at {report_filename}")
        return report_filename
        
    except Exception as e:
        log.error(f"Error generating comparison report: {e}", exc_info=True)
        return ""

def extract_scores_from_report(report_file: str, config: Dict) -> Dict:
    """
    Extract scores and metadata from an assessment report file.
    
    Args:
        report_file: Path to the assessment report markdown file
        config: Configuration dictionary with category mapping
        
    Returns:
        Dictionary containing extracted scores and metadata
    """
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        result = {
            'average': 0.0,
            'valid_count': 0,
            'assessment_date': 'Unknown',
            'questions': {},
            'categories': {}
        }
        
        # Extract assessment date
        date_match = re.search(r'\*\*Assessment Date:\*\* (.*)', content)
        if date_match:
            result['assessment_date'] = date_match.group(1).strip()
            
        # Extract average score
        avg_match = re.search(r'\*\*Final Score \(Average\):\*\* \*\*([\d.]+)', content)
        if avg_match:
            try:
                result['average'] = float(avg_match.group(1))
            except ValueError:
                pass
                
        # Extract valid count
        valid_match = re.search(r'Based on (\d+) valid final scores', content)
        if valid_match:
            try:
                result['valid_count'] = int(valid_match.group(1))
            except ValueError:
                pass
                
        # Extract question scores from the markdown table
        table_match = re.search(r'## Detailed Results\s+\n\s*(.*?)\n\n', content, re.DOTALL)
        if table_match:
            table_content = table_match.group(1)
            table_lines = [line.strip() for line in table_content.strip().split('\n') if '|' in line]
            
            # Skip header and separator lines
            for line in table_lines[2:]:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 3:
                    question = parts[1].strip()
                    score_text = parts[2].strip()
                    
                    # Convert score to number or None for N/A
                    if score_text.lower() == 'n/a':
                        score = None
                    else:
                        try:
                            score = int(score_text)
                        except ValueError:
                            try:
                                score = float(score_text)
                            except ValueError:
                                score = None
                    
                    if question:
                        result['questions'][question] = score
        
        # Calculate category averages if we have question scores and category mapping
        if result['questions'] and config:
            if 'category_mapping' in config:
                category_mapping = config['category_mapping']
                
                # Invert mapping to get question number -> category
                question_to_category = {}
                for category, question_indices in category_mapping.items():
                    for idx in question_indices:
                        question_to_category[idx] = category
                
                # Group scores by category
                category_scores = defaultdict(list)
                for i, (question, score) in enumerate(result['questions'].items(), 1):
                    if i in question_to_category and score is not None:
                        category = question_to_category[i]
                        category_scores[category].append(score)
                
                # Calculate average for each category
                for category, scores in category_scores.items():
                    if scores:
                        result['categories'][category] = sum(scores) / len(scores)
        
        return result
        
    except Exception as e:
        log.error(f"Error extracting scores from report {report_file}: {e}", exc_info=True)
        return None
