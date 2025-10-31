import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from PIL import Image, ImageEnhance



class DatasetAnalyzer:
    """
    Class for analyzing image datasets and extracting key metrics

    Attributes:
        metrics (dict): Dictionary of aggregate metrics for the dataset.
        distributions (dict): Dictionary of statistical distributions for each metric.
        raw_metrics (list): List of raw metrics for KDE visualization.

    Methods:
        analyze_image(image):
            Extract key metrics from a single image with proper normalization.
                image (PIL.Image or str): Image object or path to the image file.
            Returns:
                dict: Dictionary containing extracted metrics.
        analyze_dataset(dataset_path, file_pattern=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            Analyze entire dataset and build statistical distributions.
                dataset_path (str): Path to dataset folder.
                file_pattern (tuple): Tuple of valid file extensions. Default: ('.png', '.jpg', '.jpeg', '.tif', '.tiff').
            Returns:
                dict: Dictionary containing statistical distributions of metrics.
        visualize_distributions_kde(metrics_to_plot=None, save=False):
                metrics_to_plot (list): List of metrics to plot. If None, plot all metrics. Default: None.
                save (bool): Whether to save the plot as an image file. Default: False.
        save_analysis(path):
            Save the analysis results.
                path (str): Path to save the analysis results.
        load_analysis(path):
            Load previously saved analysis.
                path (str): Path to the saved analysis file.
            Returns:
                DatasetAnalyzer: An instance of DatasetAnalyzer with loaded analysis.

    """
    def __init__(self):
        """Initialize the analyzer with empty metrics and distributions       
        """
        self.metrics = {}
        self.distributions = {}
        
    def _calculate_entropy(self, image):
        """Calculate image entropy from histogram"""
        histogram = image.histogram()
        total_pixels = sum(histogram)
        
        # Calculate probabilities and entropy
        probabilities = [h/total_pixels for h in histogram if h > 0]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        
        return entropy
    
    def _normalize_image(self, image):
        """Normalize image to 0-255 range properly"""
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif image.mode != 'L':
            image = image.convert('L')
            
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if normalization is needed
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Ensure proper range
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array)
        
    def analyze_image(self, image):
        """Extract key metrics from a single image with proper normalization"""
        # Normalize image first
        image = self._normalize_image(image)
        img_array = np.array(image)
        
        p1, p99 = np.percentile(img_array, [1, 99])
        p25, p75 = np.percentile(img_array, [25, 75])
        
        return {
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'contrast_ratio': float(p99 / (p1 + 1e-6)),
            'median': float(np.median(img_array)),
            'dynamic_range': float(p99 - p1),
            'entropy': float(self._calculate_entropy(image)),
            'iqr': float(p75 - p25),  # Inter-quartile range
            'non_empty_ratio': float(np.count_nonzero(img_array > 10) / img_array.size)
        }
    
    
    def analyze_dataset(self, dataset_path, file_pattern=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        """Analyze entire dataset and build statistical distributions
        Args:
            dataset_path (str): Path to dataset folder.
            file_pattern (tuple): Tuple of valid file extensions. Default: ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        
        """
        all_metrics = []
        
        # Find all images
        image_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(file_pattern):
                    image_files.append(os.path.join(root, file))
        
        # Process each image
        print(f"Analyzing {len(image_files)} images...")
        for img_path in tqdm(image_files):
            try:
                metrics = self.analyze_image(img_path)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Store raw metrics for KDE visualization
        self.raw_metrics = all_metrics
        
        # Calculate statistical distributions
        metrics_array = {key: [m[key] for m in all_metrics] for key in all_metrics[0].keys()}
        
        for key, values in metrics_array.items():
            values = np.array(values)
            self.distributions[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentiles': np.percentile(values, [5, 25, 50, 75, 95]).tolist(),
                'n_samples': len(values),
                'values': values.tolist()  # Store actual values
            }
        
        return self.distributions

    def visualize_distributions_kde(self, metrics_to_plot=None, save=False):
        """
        Visualize model statistics using KDE plots with real data points.
        
        Args:
            metrics_to_plot (bool): List of metrics to plot (if None, plot all metrics). Default: None.
            save (bool): Whether to save the plot as an image file. Default: False.
        """
        if not self.distributions:
            raise ValueError("Must analyze dataset first!")
        
        # If no metrics specified, plot all
        if metrics_to_plot is None:
            metrics_to_plot = list(self.distributions.keys())
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        # Color scheme
        colors = {
            'kde': '#4ECDC4',
            'mean': '#FF6B6B',
            'median': '#45B7D1',
            'sigma': '#96CEB4',
            'points': '#FFD93D'
        }
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            dist = self.distributions[metric]
            
            # Get actual values
            values = np.array(dist['values'])
            
            # Plot KDE using actual values
            sns.kdeplot(data=values, ax=ax, color=colors['kde'], fill=True, alpha=0.3, 
                    label='Distribution')
            
            # Add scatter plot of actual values
            y_offset = 0.00  # Small offset from x-axis #0.02
            ax.scatter(values, [y_offset] * len(values), color=colors['points'], alpha=0.5, 
                    s=30, label='Samples', zorder=2)
            
            # Plot mean and median lines
            ax.axvline(dist['mean'], color=colors['mean'], linestyle='--', 
                    linewidth=2, label=f"Mean: {dist['mean']:.2f}")
            ax.axvline(dist['percentiles'][2], color=colors['median'], linestyle=':', 
                    linewidth=2, label=f"Median: {dist['percentiles'][2]:.2f}")
            
            # Plot 2σ range
            sigma_min = dist['mean'] - 2*dist['std']
            sigma_max = dist['mean'] + 2*dist['std']
            ax.axvspan(sigma_min, sigma_max, alpha=0.2, color=colors['sigma'], 
                    label=f'±2σ ({sigma_min:.2f}, {sigma_max:.2f})')
            
            # Add percentile markers
            percentiles = dist['percentiles']
            y_range = ax.get_ylim()
            max_y = y_range[1]
            for p, v in zip([5, 25, 50, 75, 95], percentiles):
                ax.scatter(v, 0, marker='|', color='black', s=100)
                ax.text(v, max_y*0.05, f'{p}%', ha='center', fontsize=8)
            
            # Customize plot
            ax.set_title(f'{metric} Distribution (n={dist["n_samples"]})', pad=20)
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('model_stats_kde.png', bbox_inches='tight', dpi=300)
        
        plt.show()

    def save_analysis(self, path):
        """Save the analysis results
        
        Args:
            path (str): Path to save the analysis results.
        """
        np.save(path, {
            'distributions': self.distributions
        })
    
    @classmethod
    def load_analysis(cls, path):
        """Load previously saved analysis
        
        Args:
            path (str): Path to the saved analysis file.
        Returns:
            DatasetAnalyzer: An instance of DatasetAnalyzer with loaded analysis.
        """
        analyzer = cls()
        data = np.load(path, allow_pickle=True).item()
        analyzer.distributions = data['distributions']
        return analyzer


def process_folder_metrics(input_folder, model_stats, file_extensions=('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
    """
    Process all images in a folder.
    
    Args:
        input_folder (str): Path to input images
        model_stats (dict): Dictionary of model statistics from analyzer
        file_extensions (list): Tuple of valid file extensions. Default: ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    """
    output_folder = input_folder + "_adjusted"
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in file_extensions:
        image_files.extend([f for f in os.listdir(input_folder) if f.lower().endswith(ext)])
    
    if not image_files:
        print("❌ No images found in input folder!")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    print("=" * 50)
    
    # Process each image
    processed_count = 0
    adjusted_count = 0
    
    for img_file in image_files:
        try:
            input_path = os.path.join(input_folder, img_file)
            processed_image, needed_adjustment = process_image_metrics(input_path, model_stats)
            
            # Always save the image
            output_path = os.path.join(output_folder, img_file)
            processed_image.save(output_path)
            
            if needed_adjustment:
                adjusted_count += 1
                
            processed_count += 1
            print(f"Progress: {processed_count}/{len(image_files)}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")
            continue
    
    # Print summary
    print("\n📊 Processing Summary")
    print("=" * 50)
    print(f"Total processed:  {processed_count}")
    print(f"Images adjusted:  {adjusted_count}")
    print(f"No adjustments:   {processed_count - adjusted_count}")
    print(f"\n💾 All images saved to: {output_folder}")



def visualize_metrics_change_OLD(original_metrics, adjusted_metrics, model_stats, metrics_to_plot=None, save = False):
    """
    Show where original and adjusted metrics fall on the KDE distribution.
    
    Args:
        original_metrics (dict): Dictionary of original image metrics
        adjusted_metrics (dict): Dictionary of adjusted image metrics
        model_stats (dict): Dictionary of model statistics
        metrics_to_plot (list): List of metrics to plot (if None, plot selected metrics). Default: None
        save (bool): Whether to save the plot as an image file. Default: False
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['contrast_ratio', 'mean', 'std']
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    colors = {
        'kde': '#4ECDC4',
        'mean': '#FF6B6B',
        'median': '#45B7D1',
        'sigma': '#96CEB4',
        'original': '#FF9F1C',
        'adjusted': '#2EC4B6',
        'points': '#FFD93D'
    }
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        dist = model_stats[metric]
        
        # Get actual values and plot KDE
        values = np.array(dist['values'])
        sns.kdeplot(data=values, ax=ax, color=colors['kde'], fill=True, alpha=0.3, 
                   label='Distribution')
        
        # Add scatter plot of actual values
        y_offset = 0.00
        ax.scatter(values, [y_offset] * len(values), color=colors['points'], alpha=0.5, 
                  s=30, label='Samples', zorder=2)
        
        # Plot original and adjusted values
        original_value = original_metrics[metric]
        ax.axvline(original_value, color=colors['original'], linewidth=2, 
                  label=f'Original: {original_value:.2f}')
        
        if adjusted_metrics is not None:
            adjusted_value = adjusted_metrics[metric]
            ax.axvline(adjusted_value, color=colors['adjusted'], linewidth=2, 
                      label=f'Adjusted: {adjusted_value:.2f}')
            
            # Add arrow showing change
            if abs(adjusted_value - original_value) > 1e-6:
                y_range = ax.get_ylim()
                arrow_y = y_range[1] * 0.5
                ax.annotate('', xy=(adjusted_value, arrow_y), 
                           xytext=(original_value, arrow_y),
                           arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3',
                                         color='gray',
                                         alpha=0.7))
        
        # Plot mean and median lines
        ax.axvline(dist['mean'], color=colors['mean'], linestyle='--', 
                  linewidth=2, label=f"Mean: {dist['mean']:.2f}")
        ax.axvline(dist['percentiles'][2], color=colors['median'], linestyle=':', 
                  linewidth=2, label=f"Median: {dist['percentiles'][2]:.2f}")
        
        # Plot 2σ range
        sigma_min = dist['mean'] - 2*dist['std']
        sigma_max = dist['mean'] + 2*dist['std']
        ax.axvspan(sigma_min, sigma_max, alpha=0.2, color=colors['sigma'], 
                  label=f'±2σ ({sigma_min:.2f}, {sigma_max:.2f})')
        
        # Add percentile markers
        percentiles = dist['percentiles']
        y_range = ax.get_ylim()
        max_y = y_range[1]
        for p, v in zip([5, 25, 50, 75, 95], percentiles):
            ax.scatter(v, 0, marker='|', color='black', s=100)
            ax.text(v, max_y*0.05, f'{p}%', ha='center', fontsize=8)
        
        # Customize plot
        ax.set_title(f'{metric} Distribution (n={dist["n_samples"]})', pad=20)
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()

    if save:
        plt.savefig('model_stats_comparison.png', bbox_inches='tight', dpi=300)
        
    plt.show()


def visualize_metrics_change(original_metrics, adjusted_metrics, model_stats, metrics_to_plot=None, save=False):
    """
    Show where original and adjusted metrics fall on the KDE distribution.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['contrast_ratio', 'mean', 'std']
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    colors = {
        'kde': '#4ECDC4',
        'mean': '#FF6B6B',
        'median': '#45B7D1',
        'sigma': '#96CEB4',
        'original': '#FF9F1C',
        'adjusted': '#2EC4B6',
        'points': '#FFD93D',
        'arrow': '#666666'
    }
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        dist = model_stats[metric]
        
        # Get actual values and plot KDE
        values = np.array(dist['values'])
        sns.kdeplot(data=values, ax=ax, color=colors['kde'], fill=True, alpha=0.3, 
                   label='Distribution')
        
        # Add scatter plot of actual values
        y_offset = 0.00
        ax.scatter(values, [y_offset] * len(values), color=colors['points'], alpha=0.5, 
                  s=30, label='Samples', zorder=2)
        
        # Plot original and adjusted values
        original_value = original_metrics[metric]
        ax.axvline(original_value, color=colors['original'], linewidth=2, 
                  label=f'Original: {original_value:.2f}')
        
        if adjusted_metrics is not None:
            adjusted_value = adjusted_metrics[metric]
            ax.axvline(adjusted_value, color=colors['adjusted'], linewidth=2, 
                      label=f'Adjusted: {adjusted_value:.2f}')
            
            # Add arrow showing change with annotation
            if abs(adjusted_value - original_value) > 1e-6:
                y_range = ax.get_ylim()
                arrow_y = y_range[1] * 0.5
                ax.annotate('', xy=(adjusted_value, arrow_y), 
                           xytext=(original_value, arrow_y),
                           arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3',
                                         color=colors['arrow'],
                                         alpha=0.7))
                
                # Add adjustment scale annotation
                scale = adjusted_value / original_value
                mid_x = (original_value + adjusted_value) / 2
                ax.text(mid_x, arrow_y * 1.1, f'×{scale:.2f}', 
                       ha='center', va='bottom', color=colors['arrow'])
        
        # Plot mean and median lines
        ax.axvline(dist['mean'], color=colors['mean'], linestyle='--', 
                  linewidth=2, label=f"Mean: {dist['mean']:.2f}")
        ax.axvline(dist['percentiles'][2], color=colors['median'], linestyle=':', 
                  linewidth=2, label=f"Median: {dist['percentiles'][2]:.2f}")
        
        # Plot confidence interval
        sigma_min = dist['mean'] - 2*dist['std']
        sigma_max = dist['mean'] + 2*dist['std']
        ax.axvspan(sigma_min, sigma_max, alpha=0.2, color=colors['sigma'], 
                  label=f'95% CI ({sigma_min:.2f}, {sigma_max:.2f})')
        
        # Add confidence interval edge lines
        ax.axvline(sigma_min, color=colors['sigma'], linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(sigma_max, color=colors['sigma'], linestyle='--', alpha=0.5, linewidth=1)
        
        # Add percentile markers
        percentiles = dist['percentiles']
        y_range = ax.get_ylim()
        max_y = y_range[1]
        for p, v in zip([5, 25, 50, 75, 95], percentiles):
            ax.scatter(v, 0, marker='|', color='black', s=100)
            ax.text(v, max_y*0.05, f'{p}%', ha='center', fontsize=8)
        
        # Customize plot
        ax.set_title(f'{metric} Distribution (n={dist["n_samples"]})', pad=20)
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        
        # Add legend with better spacing
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), borderaxespad=0)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adjust y-axis to show full distribution
        ax.set_ylim(bottom=ax.get_ylim()[0])  # Add small padding at bottom -0.02
    
    plt.tight_layout()

    if save:
        plt.savefig('model_stats_comparison.png', bbox_inches='tight', dpi=300)
        
    plt.show()

###################
###################

def adjust_metrics_to_confidence_interval(image, model_stats):
    """
    Adjust image metrics to fall within confidence intervals of the model stats.
    Only adjusts if metrics fall outside the confidence interval.
    
    Args:
        image (PIL.Image): Input image
        model_stats (dict): Dictionary containing model statistics
    
    Returns:
        PIL.Image: Adjusted image
    """
    analyzer = DatasetAnalyzer()
    metrics = analyzer.analyze_image(image)
    
    # Convert to array for adjustments
    img_array = np.array(image).astype(float)
    
    # Get std confidence interval
    std_mean = model_stats['std']['mean']
    std_std = model_stats['std']['std']
    std_ci_low = std_mean - 2 * std_std  # 2-sigma confidence interval
    std_ci_high = std_mean + 2 * std_std
    
    # Only adjust if std falls outside confidence interval
    if metrics['std'] < std_ci_low or metrics['std'] > std_ci_high:
        # Calculate target std close to mean
        target_std = std_mean
        current_std = metrics['std']
        
        # Calculate scaling factor to achieve target std
        scale_factor = target_std / (current_std + 1e-6)
        
        # Adjust the image to match target std
        img_mean = np.mean(img_array)
        img_array = img_mean + (img_array - img_mean) * scale_factor
        
        # Ensure values stay in valid range
        img_array = np.clip(img_array, 0, 255)
        
    # Convert back to PIL Image
    adjusted_image = Image.fromarray(img_array.astype(np.uint8))
    return adjusted_image

# Modify apply_recommended_adjustments to use this function
def apply_recommended_adjustments(image, model_stats, verbose=True):
    """
    Automatically adjust an image based on analysis against model statistics.
    Now includes std adjustment within confidence intervals.
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Check current image metrics
    analyzer = DatasetAnalyzer()
    check_result = check_image_quality(image, model_stats)
    
    if verbose and not check_result['is_compatible']:
        print("Applying adjustments:")
    
    # Start with the original image
    adjusted_image = image
    
    # Apply standard adjustments first
    for rec in check_result['recommendations']:
        if rec['type'] == 'contrast':
            if verbose:
                print(f"- Adjusting contrast with scale factor: {rec['scale_factor']:.2f}")
            
            img_array = np.array(adjusted_image).astype(float)
            mean = np.mean(img_array)
            img_array = mean + (img_array - mean) * rec['scale_factor']
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            adjusted_image = Image.fromarray(img_array)
            
        elif rec['type'] == 'brightness':
            if verbose:
                print(f"- Adjusting brightness to match target mean")
            
            current_mean = check_result['metrics']['mean']
            target_mean = model_stats['mean']['mean']
            brightness_factor = target_mean / (current_mean + 1e-6)
            brightness_factor = np.clip(brightness_factor, 0.5, 1.5)
            
            if verbose:
                print(f"  Current mean: {current_mean:.1f}")
                print(f"  Target mean: {target_mean:.1f}")
                print(f"  Brightness factor: {brightness_factor:.2f}")
            
            enhancer = ImageEnhance.Brightness(adjusted_image)
            adjusted_image = enhancer.enhance(brightness_factor)
    
    # Now apply std adjustment if needed
    adjusted_image = adjust_metrics_to_confidence_interval(adjusted_image, model_stats)
    
    # Verify the adjustments
    if verbose:
        final_metrics = analyzer.analyze_image(adjusted_image)
        print("\nFinal metrics:")
        print(f"- Contrast ratio: {final_metrics['contrast_ratio']:.2f}")
        print(f"- Mean brightness: {final_metrics['mean']:.1f}")
        print(f"- Standard deviation: {final_metrics['std']:.2f}")
    
    return adjusted_image

def check_image_quality(image, model_stats):
    """
    Check if an image needs preprocessing before being fed to the model.
    Returns recommended adjustments.
    """
    # Analyze image
    analyzer = DatasetAnalyzer()
    metrics = analyzer.analyze_image(image)
    
    recommendations = []
    
    # Check contrast ratio
    if metrics['contrast_ratio'] > model_stats['contrast_ratio']['mean'] + 2 * model_stats['contrast_ratio']['std']:
        recommended_scale = model_stats['contrast_ratio']['mean'] / metrics['contrast_ratio']
        recommendations.append({
            'type': 'contrast',
            'message': f'Contrast too high. Recommend scaling by {recommended_scale:.2f}',
            'scale_factor': recommended_scale
        })
    
    # Check if image is too bright/dark
    mean_diff = abs(metrics['mean'] - model_stats['mean']['mean'])
    if mean_diff > 2 * model_stats['mean']['std']:
        brightness_factor = model_stats['mean']['mean'] / (metrics['mean'] + 1e-6)
        brightness_factor = np.clip(brightness_factor, 0.5, 1.5)
        recommendations.append({
            'type': 'brightness',
            'message': f'Mean brightness differs by {mean_diff:.1f}. Consider adjusting.',
            'scale_factor': brightness_factor
        })
    
    # Check std
    std_mean = model_stats['std']['mean']
    std_std = model_stats['std']['std']
    std_ci_low = std_mean - 2 * std_std
    std_ci_high = std_mean + 2 * std_std
    
    if metrics['std'] < std_ci_low or metrics['std'] > std_ci_high:
        recommendations.append({
            'type': 'std',
            'message': f'Standard deviation ({metrics["std"]:.2f}) outside confidence interval [{std_ci_low:.2f}, {std_ci_high:.2f}]'
        })
    
    return {
        'metrics': metrics,
        'recommendations': recommendations,
        'is_compatible': len(recommendations) == 0
    }

def process_image_metrics(image_path, model_stats):
    """
    Analyze image metrics and apply necessary adjustments.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
    
    # Get original metrics
    analyzer = DatasetAnalyzer()
    original_metrics = analyzer.analyze_image(image)
    
    # Check if adjustments needed
    check_result = check_image_quality(image, model_stats)
    
    if check_result['is_compatible']:
        print("✅ Image metrics within normal ranges\n")
        return image, False
    
    # Print analysis
    print("\n⚙️ Image Analysis:")
    required_adjustments = []
    for rec in check_result['recommendations']:
        if rec['type'] == 'contrast':
            scale = rec['scale_factor']
            required_adjustments.append(f"Contrast: {scale:.2f}x")
        elif rec['type'] == 'brightness':
            scale = rec['scale_factor']
            required_adjustments.append(f"Brightness: {scale:.2f}x")
        elif rec['type'] == 'std':
            std_mean = model_stats['std']['mean']
            current_std = original_metrics['std']
            required_adjustments.append(f"Std: {current_std:.2f} → {std_mean:.2f}")
    
    print("└─ " + " | ".join(required_adjustments))
    
    # Apply adjustments
    adjusted_image = apply_recommended_adjustments(image, model_stats)
    print("✨ Adjustments applied\n")
    
    # Get metrics after adjustment
    adjusted_metrics = analyzer.analyze_image(adjusted_image)
    
    print("\n📊 Metrics Summary:")
    if 'contrast' in str(required_adjustments):
        print(f"  Contrast: {original_metrics['contrast_ratio']:.2f} → {adjusted_metrics['contrast_ratio']:.2f}")
    if 'brightness' in str(required_adjustments):
        print(f"  Mean: {original_metrics['mean']:.2f} → {adjusted_metrics['mean']:.2f}")
    if 'std' in str(required_adjustments):
        print(f"  Std: {original_metrics['std']:.2f} → {adjusted_metrics['std']:.2f}")
    
    return adjusted_image, adjusted_metrics