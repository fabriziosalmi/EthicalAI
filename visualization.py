"""
Data visualization functions for the assessment results.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from config import RESULTS_DIR

log = logging.getLogger(__name__)

def generate_visualizations(provider: str, results: List[Tuple[str, Optional[int], List[Optional[int]]]], category_mapping: Dict[str, List[int]]):
    """
    Generate visualizations for assessment results and save them to the results directory.
    
    Args:
        provider: Name of the AI provider
        results: List of assessment results (question, final score, sample scores)
        category_mapping: Mapping of category names to question indices
    """
    log.info(f"Generating visualizations for provider '{provider}'")
    
    # Create directory for visualizations
    viz_dir = os.path.join(RESULTS_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract valid scores and their indices
    valid_scores = []
    question_indices = []
    categories = []
    
    for i, (question, score, _) in enumerate(results, 1):
        if score is not None:
            valid_scores.append(score)
            question_indices.append(i)
            
            # Determine category for this question
            category = "Unknown"
            for cat, indices in category_mapping.items():
                if i in indices:
                    category = cat.capitalize()
                    break
            categories.append(category)
    
    if not valid_scores:
        log.warning("No valid scores available for visualization")
        return
    
    # 1. Overall Score Distribution
    plt.figure(figsize=(12, 8))
    plt.hist(valid_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Score Distribution - {provider.upper()}', fontsize=16)
    plt.xlabel('Score (0-100)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    dist_file = os.path.join(viz_dir, f"{timestamp}_{provider}_score_distribution.png")
    plt.savefig(dist_file)
    plt.close()
    
    # 2. Category-wise Average Scores
    category_scores = defaultdict(list)
    for cat, score in zip(categories, valid_scores):
        category_scores[cat].append(score)
    
    avg_category_scores = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}
    
    plt.figure(figsize=(12, 8))
    cats = list(avg_category_scores.keys())
    avgs = list(avg_category_scores.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(cats)))
    bars = plt.bar(cats, avgs, color=colors)
    
    plt.title(f'Average Score by Category - {provider.upper()}', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.ylim(0, 100)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}', ha='center', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    cat_file = os.path.join(viz_dir, f"{timestamp}_{provider}_category_scores.png")
    plt.savefig(cat_file)
    plt.close()
    
    # 3. Question-wise scores (radar chart for categories with sufficient questions)
    for category, indices in category_mapping.items():
        # Get questions that belong to this category and have valid scores
        cat_scores = []
        cat_labels = []
        
        for i, (question, score, _) in enumerate(results, 1):
            if i in indices and score is not None:
                cat_scores.append(score)
                # Extract first few words for label
                words = question.split()
                short_label = ' '.join(words[:3]) + '...' if len(words) > 3 else question
                cat_labels.append(f"Q{i}: {short_label}")
        
        if len(cat_scores) < 3:  # Skip categories with too few questions
            continue
            
        # Create radar chart
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Compute angles for each question
        angles = np.linspace(0, 2*np.pi, len(cat_scores), endpoint=False).tolist()
        
        # Complete the loop
        cat_scores.append(cat_scores[0])
        angles.append(angles[0])
        
        # Plot data
        ax.plot(angles, cat_scores, 'o-', linewidth=2, label=category.capitalize())
        ax.fill(angles, cat_scores, alpha=0.25)
        
        # Set category labels
        ax.set_thetagrids(np.degrees(angles[:-1]), cat_labels, fontsize=8)
        
        # Set radial limits
        ax.set_ylim(0, 100)
        
        # Add title
        plt.title(f'{category.capitalize()} Scores - {provider.upper()}', fontsize=16)
        
        plt.tight_layout()
        radar_file = os.path.join(viz_dir, f"{timestamp}_{provider}_{category}_radar.png")
        plt.savefig(radar_file)
        plt.close()
    
    log.info(f"Visualizations saved to {viz_dir}")
    return viz_dir

def generate_comparison_visualizations(provider_scores, comparison_dir):
    """
    Generate visualizations comparing different providers.
    
    Args:
        provider_scores: Dictionary of provider scores
        comparison_dir: Directory to save the visualizations
    """
    # Generate comparative bar chart for average scores
    plt.figure(figsize=(12, 8))
    providers_list = list(provider_scores.keys())
    avg_scores = [scores.get('average', 0) for scores in provider_scores.values()]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(providers_list)))
    bars = plt.bar(providers_list, avg_scores, color=colors)
    
    plt.title('Ethical AI Assessment - Provider Comparison', fontsize=16)
    plt.xlabel('Provider', fontsize=14)
    plt.ylabel('Average Score (0-100)', fontsize=14)
    plt.ylim(0, 100)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}', ha='center', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    comparison_file = os.path.join(comparison_dir, "provider_comparison_overall.png")
    plt.savefig(comparison_file)
    plt.close()
    
    # Generate radar chart comparing providers across categories
    if any('categories' in scores for scores in provider_scores.values()):
        # Get all categories
        all_categories = set()
        for scores in provider_scores.values():
            if 'categories' in scores:
                all_categories.update(scores['categories'].keys())
        
        categories_list = sorted(list(all_categories))
        if len(categories_list) >= 3:  # Need at least 3 categories for radar chart
            plt.figure(figsize=(12, 10))
            ax = plt.subplot(111, polar=True)
            
            # Compute angles for each category
            angles = np.linspace(0, 2*np.pi, len(categories_list), endpoint=False).tolist()
            
            # Complete the loop
            categories_list.append(categories_list[0])
            angles.append(angles[0])
            
            # Plot each provider
            for i, (provider, scores) in enumerate(provider_scores.items()):
                if 'categories' not in scores:
                    continue
                    
                values = [scores['categories'].get(cat, 0) for cat in categories_list[:-1]]
                values.append(values[0])  # Complete the loop
                
                color = plt.cm.viridis(i/len(provider_scores))
                ax.plot(angles, values, 'o-', linewidth=2, label=provider.upper(), color=color)
                ax.fill(angles, values, alpha=0.1, color=color)
            
            # Set category labels
            plt.xticks(angles[:-1], [c.capitalize() for c in categories_list[:-1]], fontsize=12)
            
            # Set radial limits
            plt.ylim(0, 100)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title('Ethical AI Assessment - Category Comparison', fontsize=16)
            plt.tight_layout()
            radar_file = os.path.join(comparison_dir, "provider_comparison_radar.png")
            plt.savefig(radar_file)
            plt.close()
            
    return all_categories if 'all_categories' in locals() else set()
