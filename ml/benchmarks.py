import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import time
import json
import os
from PIL import Image
import torchvision.transforms as transforms
from .vjepa2 import VJEPA2Handler
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class ModelBenchmark:
    """
    Benchmarking and analysis tools for V-JEPA 2 model
    """
    
    def __init__(self, vjepa_handler: VJEPA2Handler):
        """
        Initialize benchmarking tools
        
        Args:
            vjepa_handler: Initialized V-JEPA 2 handler
        """
        self.vjepa_handler = vjepa_handler
        self.results_dir = os.path.join(os.getcwd(), 'benchmark_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def benchmark_latency(self, image_size: Tuple[int, int] = (256, 256), num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference latency
        
        Args:
            image_size: Size of test images
            num_runs: Number of inference runs to average over
            
        Returns:
            Dictionary with latency statistics
        """
        print(f"Running latency benchmark with {num_runs} iterations...")
        
        # Create random test image
        test_image = torch.randn(3, *image_size)
        
        # Warmup
        for _ in range(10):
            self.vjepa_handler.encode_image(test_image)
        
        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            self.vjepa_handler.encode_image(test_image)
            latencies.append((time.time() - start_time) * 1000)  # ms
            
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        results = {
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'device': self.vjepa_handler.device
        }
        
        print(f"Latency benchmark results: {json.dumps(results, indent=2)}")
        return results
    
    def benchmark_physical_understanding(self, test_dataset_path: str) -> Dict[str, float]:
        """
        Benchmark model's physical understanding using the V-JEPA 2 physical benchmarks
        
        Args:
            test_dataset_path: Path to physical understanding benchmark dataset
            
        Returns:
            Dictionary with accuracy metrics for different physical tasks
        """
        print(f"Running physical understanding benchmark on {test_dataset_path}...")
        
        # Load test dataset (specific implementation depends on benchmark format)
        try:
            with open(test_dataset_path, 'r') as f:
                benchmark_data = json.load(f)
        except:
            print(f"Error: Could not load benchmark data from {test_dataset_path}")
            return {}
            
        results = {}
        for task_name, task_data in benchmark_data.items():
            print(f"Evaluating task: {task_name}")
            
            correct = 0
            total = len(task_data['samples'])
            
            for sample in task_data['samples']:
                # Process sample according to task type
                if sample['type'] == 'prediction':
                    # Get initial frames
                    initial_frames = [Image.open(f) for f in sample['initial_frames']]
                    initial_embeddings = [self.vjepa_handler.encode_image(frame) for frame in initial_frames]
                    
                    # Predict future frame
                    current_state = initial_embeddings[-1]
                    if 'action' in sample:
                        action = np.array(sample['action'])
                        action_encoding = self.vjepa_handler.encode_action(action)
                        predicted_state = self.vjepa_handler.predict_future_state(current_state, action_encoding)
                    else:
                        # No action provided, use model's internal dynamics
                        predicted_state = self.vjepa_handler.predict_future_state(current_state, None)
                    
                    # Compare with ground truth
                    gt_frame = Image.open(sample['ground_truth_frame'])
                    gt_embedding = self.vjepa_handler.encode_image(gt_frame)
                    
                    # Calculate error
                    error = self.vjepa_handler.compute_embedding_distance(predicted_state, gt_embedding)
                    
                    # Check if error is below threshold
                    if error < sample.get('threshold', 0.1):
                        correct += 1
                        
            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            results[task_name] = accuracy
            print(f"{task_name} accuracy: {accuracy:.2f}")
            
        # Save results
        timestamp = int(time.time())
        results_file = os.path.join(self.results_dir, f"physical_benchmark_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Benchmark results saved to {results_file}")
        return results
    
    def analyze_failure_cases(self, test_dataset_path: str, output_dir: Optional[str] = None) -> None:
        """
        Analyze and visualize failure cases to identify model limitations
        
        Args:
            test_dataset_path: Path to test dataset
            output_dir: Directory to save visualizations (default: benchmark_results/failure_analysis)
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, 'failure_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Running failure case analysis on {test_dataset_path}...")
        
        # Load test dataset
        try:
            with open(test_dataset_path, 'r') as f:
                test_data = json.load(f)
        except:
            print(f"Error: Could not load test data from {test_dataset_path}")
            return
            
        # Track errors by category
        errors_by_category = {}
        failure_examples = []
        
        for sample_id, sample in enumerate(test_data['samples']):
            # Process sample based on type
            if 'initial_frames' in sample and 'ground_truth_frame' in sample:
                # Get initial frames
                initial_frames = [Image.open(f) for f in sample['initial_frames']]
                initial_embeddings = [self.vjepa_handler.encode_image(frame) for frame in initial_frames]
                
                # Predict future frame
                current_state = initial_embeddings[-1]
                if 'action' in sample:
                    action = np.array(sample['action'])
                    action_encoding = self.vjepa_handler.encode_action(action)
                    predicted_state = self.vjepa_handler.predict_future_state(current_state, action_encoding)
                else:
                    predicted_state = self.vjepa_handler.predict_future_state(current_state, None)
                
                # Compare with ground truth
                gt_frame = Image.open(sample['ground_truth_frame'])
                gt_embedding = self.vjepa_handler.encode_image(gt_frame)
                
                # Calculate error
                error = self.vjepa_handler.compute_embedding_distance(predicted_state, gt_embedding)
                
                # Categorize sample
                category = sample.get('category', 'unknown')
                if category not in errors_by_category:
                    errors_by_category[category] = []
                errors_by_category[category].append(error)
                
                # Record high-error cases
                if error > sample.get('threshold', 0.1):
                    failure_examples.append({
                        'sample_id': sample_id,
                        'error': error,
                        'category': category,
                        'properties': sample.get('properties', {})
                    })
        
        # Analyze errors by category
        category_summary = {}
        for category, errors in errors_by_category.items():
            category_summary[category] = {
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'max_error': np.max(errors),
                'min_error': np.min(errors),
                'sample_count': len(errors)
            }
        
        # Sort failure examples by error
        failure_examples.sort(key=lambda x: x['error'], reverse=True)
        
        # Visualize error distribution by category
        plt.figure(figsize=(10, 6))
        categories = list(errors_by_category.keys())
        means = [np.mean(errors_by_category[cat]) for cat in categories]
        
        plt.bar(categories, means)
        plt.xlabel('Category')
        plt.ylabel('Mean Error')
        plt.title('Error by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(os.path.join(output_dir, 'error_by_category.png'))
        
        # Save detailed analysis
        analysis_file = os.path.join(output_dir, 'failure_analysis.json')
        analysis = {
            'category_summary': category_summary,
            'top_failures': failure_examples[:10]  # Save top 10 worst cases
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        print(f"Failure analysis saved to {output_dir}")
            
    def visualize_latent_space(self, dataset_path: str, method: str = 'tsne') -> None:
        """
        Visualize the model's latent space to gain insights into physics representations
        
        Args:
            dataset_path: Path to dataset of physical interactions
            method: Dimensionality reduction method ('tsne' or 'pca')
        """
        print(f"Visualizing latent space using {method}...")
        
        # Load dataset images and annotations
        try:
            with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
        except:
            print(f"Error: Could not load metadata from {dataset_path}")
            return
            
        # Collect embeddings and labels
        embeddings = []
        categories = []
        properties = []
        
        for item in metadata['items']:
            # Load image
            image_path = os.path.join(dataset_path, item['image_path'])
            try:
                image = Image.open(image_path)
                # Encode image
                embedding = self.vjepa_handler.encode_image(image)
                
                # Store embedding and metadata
                embeddings.append(embedding.cpu().numpy().flatten())
                categories.append(item['category'])
                properties.append(item.get('properties', {}))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        if len(embeddings) == 0:
            print("No valid embeddings collected")
            return
            
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # pca
            reducer = PCA(n_components=2)
            
        embeddings_2d = reducer.fit_transform(embeddings_array)
        
        # Create DataFrame for visualization
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'category': categories
        })
        
        # Visualize
        plt.figure(figsize=(12, 10))
        categories_set = set(categories)
        
        for category in categories_set:
            subset = df[df['category'] == category]
            plt.scatter(subset['x'], subset['y'], label=category, alpha=0.7)
            
        plt.title(f'Latent Space Visualization ({method.upper()})')
        plt.legend()
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Save visualization
        vis_path = os.path.join(self.results_dir, f'latent_space_{method}.png')
        plt.savefig(vis_path)
        plt.close()
        
        print(f"Latent space visualization saved to {vis_path}")
