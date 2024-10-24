import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from typing import Dict, List, Any, Tuple, Union
import pandas as pd
from scipy import stats
import json
import os
from datetime import datetime

class ResultsAnalyzer:
    def __init__(self):
        """Initialize ResultsAnalyzer with empty storage for results."""
        self.experiments = {}
        self.comparisons = {}
        
    def add_experiment_result(self, 
                            experiment_name: str,
                            history: Dict[str, List[float]],
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            metadata: Dict[str, Any] = None) -> None:
        """
        Add experiment results for later analysis.
        
        Args:
            experiment_name: Unique name for the experiment
            history: Training history dictionary
            y_true: True labels
            y_pred: Predicted labels/probabilities
            metadata: Additional experiment information
        """
        self.experiments[experiment_name] = {
            'history': history,
            'y_true': y_true,
            'y_pred': y_pred,
            'metadata': metadata or {}
        }

    def visualize_digit(self, image: np.ndarray, label: np.ndarray = None, 
                       prediction: np.ndarray = None) -> None:
        """Visualize a single MNIST digit with optional label and prediction."""
        if image.shape == (784,):
            image = image.reshape(28, 28)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        title = ""
        if label is not None:
            title += f"True: {np.argmax(label)}"
        if prediction is not None:
            title += f" Pred: {np.argmax(prediction)}"
        if title:
            plt.title(title)
        plt.show()

    def visualize_multiple_digits(self, images: np.ndarray, 
                                labels: np.ndarray = None,
                                predictions: np.ndarray = None,
                                num_images: int = 25,
                                title: str = "") -> None:
        """Visualize multiple MNIST digits in a grid."""
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        plt.figure(figsize=(2*grid_size, 2*grid_size))
        plt.suptitle(title, y=1.02, size=14)
        
        for i in range(min(num_images, len(images))):
            plt.subplot(grid_size, grid_size, i + 1)
            
            img = images[i].reshape(28, 28) if images[i].shape == (784,) else images[i]
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            
            title = ""
            if labels is not None:
                title += f"T:{np.argmax(labels[i])}"
            if predictions is not None:
                title += f" P:{np.argmax(predictions[i])}"
            if title:
                plt.title(title, fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            title: str = "Confusion Matrix",
                            normalize: bool = False) -> None:
        """
        Plot confusion matrix with optional normalization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            normalize: Whether to normalize confusion matrix
        """
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def plot_training_history(self, history: Dict[str, List[float]], 
                            title: str = "Training History",
                            include_validation: bool = True) -> None:
        """Plot training and validation metrics over time."""
        metrics = [m for m in history.keys() if not m.startswith('val_')]
        n_metrics = len(metrics)
        
        plt.figure(figsize=(6*n_metrics, 5))
        plt.suptitle(title, y=1.02, size=14)
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, n_metrics, i)
            plt.plot(history[metric], label=f'Training {metric}')
            if include_validation and f'val_{metric}' in history:
                plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    def compare_experiments(self, 
                          experiment_names: List[str],
                          metric: str = 'val_accuracy') -> None:
        """
        Compare multiple experiments based on a specific metric.
        
        Args:
            experiment_names: List of experiment names to compare
            metric: Metric to compare
        """
        plt.figure(figsize=(12, 6))
        
        for name in experiment_names:
            if name in self.experiments:
                history = self.experiments[name]['history']
                if metric in history:
                    plt.plot(history[metric], label=name)
        
        plt.title(f'Comparison of {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.show()

    def plot_roc_curves(self, experiment_names: List[str]) -> None:
        """Plot ROC curves for multiple experiments."""
        plt.figure(figsize=(10, 8))
        
        for name in experiment_names:
            if name in self.experiments:
                exp = self.experiments[name]
                y_true = exp['y_true']
                y_pred = exp['y_pred']
                
                # Calculate ROC curve for each class
                n_classes = y_true.shape[1]
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{name} - Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def analyze_errors(self, experiment_name: str) -> pd.DataFrame:
        """
        Analyze misclassified examples.
        
        Returns:
            DataFrame with error analysis
        """
        exp = self.experiments[experiment_name]
        y_true = np.argmax(exp['y_true'], axis=1)
        y_pred = np.argmax(exp['y_pred'], axis=1)
        
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        error_analysis = pd.DataFrame({
            'True_Label': y_true[error_indices],
            'Predicted_Label': y_pred[error_indices],
            'Confidence': np.max(exp['y_pred'][error_indices], axis=1)
        })
        
        return error_analysis

    def compute_statistical_significance(self, 
                                      experiment1: str,
                                      experiment2: str,
                                      metric: str = 'val_accuracy') -> Dict:
        """
        Compute statistical significance between two experiments.
        
        Returns:
            Dictionary with statistical test results
        """
        exp1 = self.experiments[experiment1]['history'][metric]
        exp2 = self.experiments[experiment2]['history'][metric]
        
        t_stat, p_value = stats.ttest_ind(exp1, exp2)
        cohens_d = (np.mean(exp1) - np.mean(exp2)) / np.sqrt(
            (np.std(exp1) ** 2 + np.std(exp2) ** 2) / 2)
        
        return {
            'experiment1_mean': np.mean(exp1),
            'experiment2_mean': np.mean(exp2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d
        }

    def plot_class_distribution(self, experiment_name: str) -> None:
        """Plot distribution of predictions across classes."""
        exp = self.experiments[experiment_name]
        y_true = np.argmax(exp['y_true'], axis=1)
        y_pred = np.argmax(exp['y_pred'], axis=1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.countplot(x=y_true)
        plt.title('True Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        sns.countplot(x=y_pred)
        plt.title('Predicted Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()

    def generate_summary_report(self, experiment_name: str) -> Dict:
        """
        Generate comprehensive summary report for an experiment.
        
        Returns:
            Dictionary containing summary metrics
        """
        exp = self.experiments[experiment_name]
        y_true = exp['y_true']
        y_pred = exp['y_pred']
        
        # Calculate various metrics
        accuracy = np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
        
        # Per-class metrics
        per_class_accuracy = []
        for i in range(y_true.shape[1]):
            class_acc = np.mean(
                np.argmax(y_true, axis=1)[np.argmax(y_pred, axis=1) == i] == i
            )
            per_class_accuracy.append(class_acc)
        
        # Training history summary
        history = exp['history']
        final_metrics = {
            metric: values[-1] 
            for metric, values in history.items()
        }
        
        return {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy,
            'final_metrics': final_metrics,
            'metadata': exp['metadata']
        }

    def save_results(self, filepath: str) -> None:
        """
        Save analysis results to file.
        
        Args:
            filepath: Path to save results
        """
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            name: {
                'history': exp['history'],
                'metadata': exp['metadata'],
                'summary': self.generate_summary_report(name)
            }
            for name, exp in self.experiments.items()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)

    def load_results(self, filepath: str) -> None:
        """
        Load previously saved results.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.experiments = data

    def plot_learning_rate_analysis(self, experiment_name: str) -> None:
        """Analyze and plot learning rate effects."""
        exp = self.experiments[experiment_name]
        history = exp['history']
        
        if 'lr' not in history:
            print("Learning rate history not available")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['lr'])
        plt.title('Learning Rate Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['lr'], history['loss'])
        plt.title('Loss vs Learning Rate')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.xscale('log')
        
        plt.tight_layout()
        plt.show()

    def plot_prediction_confidence(self, experiment_name: str) -> None:
        """Plot distribution of prediction confidences."""
        exp = self.experiments[experiment_name]
        y_true = np.argmax(exp['y_true'], axis=1)
        y_pred = np.argmax(exp['y_pred'], axis=1)
        confidences = np.max(exp['y_pred'], axis=1)
        
        correct = y_true == y_pred
        
        plt.figure(figsize=(10, 6))
        plt.hist([confidences[correct], confidences[~correct]], 
                label=['Correct', 'Incorrect'],
                bins=50, alpha=0.6)
        plt.title('Distribution of Prediction Confidences')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    def export_results_to_latex(self, experiment_names: List[str],
                              metrics: List[str]) -> str:
        """
        Generate LaTeX table of results.
        
        Returns:
            String containing LaTeX table code
        """
        rows = []
        for exp_name in experiment_names:
            if exp_name in self.experiments:
                summary = self.generate_summary_report(exp_name)
                row = [exp_name]
                for metric in metrics:
                    if metric in summary['final_metrics']:
                        row.append(f"{summary['final_metrics'][metric]:.4f}")
                    else:
                        row.append("N/A")
                rows.append(" & ".join(row) + " \\\\")
        
        header = ["Experiment"] + metrics
        latex_table = (
            "\\begin{tabular}{" + "c" * len(header) + "}\n"
            "\\hline\n" +
            " & ".join(header) + " \\\\\n" +
            "\\hline\n" +
            "\