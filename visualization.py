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

from config import REPORTS_DIR  # Change from RESULTS_DIR to REPORTS_DIR

log = logging.getLogger(__name__)

def generate_visualizations(provider: str, results: List[Tuple[str, Optional[int], List[Optional[int]]]], category_mapping: Dict[str, List[int]]):
    """
    Generate visualizations for assessment results and save them to the docs/reports directory.
    
    Args:
        provider: Name of the AI provider
        results: List of assessment results (question, final score, sample scores)
        category_mapping: Mapping of category names to question indices
        
    Returns:
        str: Path to the visualization directory, or None if visualization failed
    """
    try:
        if not provider:
            log.error("Provider name is empty or None")
            return None
            
        if not results:
            log.warning("No results provided for visualization")
            return None
            
        if not category_mapping:
            log.warning("No category mapping provided for visualization. Using a default mapping.")
            # Create a simple default mapping with all questions in one category
            category_mapping = {"uncategorized": list(range(1, len(results) + 1))}
            
        log.info(f"Generating visualizations for provider '{provider}'")
        
        # Create directory for visualizations - save directly to docs/reports
        os.makedirs(REPORTS_DIR, exist_ok=True)
        viz_dir = REPORTS_DIR
        
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
            return viz_dir  # Return the directory even though no visualizations were created
        
        # 1. Overall Score Distribution
        try:
            plt.figure(figsize=(12, 8))
            plt.hist(valid_scores, bins=min(20, len(set(valid_scores))), color='skyblue', edgecolor='black')
            plt.title(f'Score Distribution - {provider.upper()}', fontsize=16)
            plt.xlabel('Score (0-100)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            dist_file = os.path.join(viz_dir, f"{timestamp}_{provider.lower()}_score_distribution.png")
            plt.savefig(dist_file)
            plt.close()
            log.info(f"Generated score distribution chart: {dist_file}")
        except Exception as e:
            log.error(f"Failed to generate score distribution chart: {e}")
            # Continue with other visualizations even if this one fails
        
        # 2. Category-wise Average Scores
        try:
            category_scores = defaultdict(list)
            for cat, score in zip(categories, valid_scores):
                category_scores[cat].append(score)
            
            if not category_scores:
                log.warning("No category scores available for bar chart")
            else:
                avg_category_scores = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}
                
                plt.figure(figsize=(12, 8))
                cats = list(avg_category_scores.keys())
                avgs = list(avg_category_scores.values())
                
                # Handle the case where there's only one category
                if len(cats) == 1:
                    bars = plt.bar(cats, avgs, color='skyblue')
                else:
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
                cat_file = os.path.join(viz_dir, f"{timestamp}_{provider.lower()}_category_scores.png")
                plt.savefig(cat_file)
                plt.close()
                log.info(f"Generated category scores chart: {cat_file}")
        except Exception as e:
            log.error(f"Failed to generate category scores chart: {e}")
        
        # 3. Question-wise scores (radar chart for categories with sufficient questions)
        for category, indices in category_mapping.items():
            try:
                # Get questions that belong to this category and have valid scores
                cat_scores = []
                cat_labels = []
                
                for i, (question, score, _) in enumerate(results, 1):
                    if i in indices and score is not None:
                        cat_scores.append(score)
                        # Extract first few words for label
                        if question:  # Ensure question is not None or empty
                            words = question.split()
                            short_label = ' '.join(words[:3]) + '...' if len(words) > 3 else question
                        else:
                            short_label = f"Q{i}"
                        cat_labels.append(f"Q{i}: {short_label}")
                
                if len(cat_scores) < 3:  # Skip categories with too few questions
                    log.info(f"Skipping radar chart for category '{category}' - not enough valid questions ({len(cat_scores)})")
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
                
                # Set category labels but handle potential matplotlib issues with too many labels
                if len(cat_labels) > 20:
                    # For many labels, show a subset to avoid overlapping
                    simplified_labels = ["" if i % 3 != 0 else label for i, label in enumerate(cat_labels)]
                    ax.set_thetagrids(np.degrees(angles[:-1]), simplified_labels, fontsize=8)
                    log.info(f"Simplified radar chart labels for category '{category}' (showing {len([l for l in simplified_labels if l])}/{len(cat_labels)} labels)")
                else:
                    ax.set_thetagrids(np.degrees(angles[:-1]), cat_labels, fontsize=8)
                
                # Set radial limits
                ax.set_ylim(0, 100)
                
                # Add title
                plt.title(f'{category.capitalize()} Scores - {provider.upper()}', fontsize=16)
                
                plt.tight_layout()
                radar_file = os.path.join(viz_dir, f"{timestamp}_{provider.lower()}_{category.lower()}_radar.png")
                plt.savefig(radar_file)
                plt.close()
                log.info(f"Generated radar chart for category '{category}': {radar_file}")
            except Exception as e:
                log.error(f"Failed to generate radar chart for category '{category}': {e}")
        
        log.info(f"All visualizations saved to {viz_dir}")
        return viz_dir
        
    except Exception as e:
        log.error(f"Unexpected error during visualization generation: {e}", exc_info=True)
        return None

def generate_comparison_visualizations(provider_scores, comparison_dir):
    """
    Generate visualizations comparing different providers.
    
    Args:
        provider_scores: Dictionary of provider scores
        comparison_dir: Directory to save the visualizations
        
    Returns:
        set: Set of all category names found across all providers, or empty set if visualization failed
    """
    try:
        if not provider_scores:
            log.warning("No provider scores provided for comparison visualization")
            return set()
        
        if not comparison_dir:
            log.error("No comparison directory specified")
            return set()
        
        # Ensure the comparison directory exists
        os.makedirs(comparison_dir, exist_ok=True)
        
        log.info(f"Generating comparison visualizations for {len(provider_scores)} providers")
        
        # Generate comparative bar chart for average scores
        try:
            plt.figure(figsize=(12, 8))
            providers_list = list(provider_scores.keys())
            avg_scores = [scores.get('average', 0) for scores in provider_scores.values()]
            
            if len(providers_list) == 0:
                log.warning("No providers found for bar chart")
                return set()
            
            if len(providers_list) == 1:
                bars = plt.bar(providers_list, avg_scores, color='skyblue')
            else:
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
            log.info(f"Generated provider comparison bar chart: {comparison_file}")
        except Exception as e:
            log.error(f"Failed to generate provider comparison bar chart: {e}")
        
        # Generate radar chart comparing providers across categories
        all_categories = set()
        try:
            if any('categories' in scores for scores in provider_scores.values()):
                # Get all categories
                for scores in provider_scores.values():
                    if 'categories' in scores and isinstance(scores['categories'], dict):
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
                    legend_handles = []
                    for i, (provider, scores) in enumerate(provider_scores.items()):
                        if 'categories' not in scores or not isinstance(scores['categories'], dict):
                            log.warning(f"Provider '{provider}' has no valid categories data, skipping in radar chart")
                            continue
                            
                        values = [scores['categories'].get(cat, 0) for cat in categories_list[:-1]]
                        values.append(values[0])  # Complete the loop
                        
                        color = plt.cm.viridis(i/max(1, len(provider_scores)))
                        line, = ax.plot(angles, values, 'o-', linewidth=2, label=provider.upper(), color=color)
                        ax.fill(angles, values, alpha=0.1, color=color)
                        legend_handles.append(line)
                    
                    # Set category labels
                    plt.xticks(angles[:-1], [c.capitalize() for c in categories_list[:-1]], fontsize=12)
                    
                    # Set radial limits
                    plt.ylim(0, 100)
                    
                    # Add legend only if we have providers to show
                    if legend_handles:
                        plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    
                    plt.title('Ethical AI Assessment - Category Comparison', fontsize=16)
                    plt.tight_layout()
                    radar_file = os.path.join(comparison_dir, "provider_comparison_radar.png")
                    plt.savefig(radar_file)
                    plt.close()
                    log.info(f"Generated provider comparison radar chart: {radar_file}")
                else:
                    log.warning(f"Not enough categories ({len(categories_list)}) for radar chart - minimum 3 required")
        except Exception as e:
            log.error(f"Failed to generate provider comparison radar chart: {e}")
        
        return all_categories
        
    except Exception as e:
        log.error(f"Unexpected error during comparison visualization generation: {e}", exc_info=True)
        return set()
