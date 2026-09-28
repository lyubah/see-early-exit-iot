#!/usr/bin/env python3
"""
TEASER Complete Analysis with Tree Sizes and Meaningful Plots
==============================================================
Extracts all metrics and creates insightful visualizations showing:
- Energy vs Accuracy trade-offs
- Tree complexity vs Performance
- Earliness vs Model size relationships
"""

import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
import pickle
try:                                   # plotting is optional (not used by the multiseed runner)
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLT = True
except Exception:
    plt = sns = None
    _HAS_PLT = False
from typing import Dict, Optional, Any, List
import logging

import baseline_data as bd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for better plots (only if matplotlib is available)
if _HAS_PLT:
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

try:
    from sktime.classification.early_classification import TEASER
    logger.info("✓ TEASER imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import TEASER: {e}")
    sys.exit(1)


class SimpleTimeSeriesClassifier(BaseEstimator, ClassifierMixin):
    """Simple classifier that works with TEASER. Supports RandomForest and GradientBoosting."""
    
    def __init__(self, n_estimators=50, max_depth=None, random_state=42, base_estimator='rf'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        # Store original value (don't modify in __init__ for sklearn clone compatibility)
        self.base_estimator = base_estimator
        self.clf = None  # Can be RF or GB (changed from rf)
        self._is_fitted = False
        self.classes_ = None  # Initialize for sklearn compatibility
    
    def get_params(self, deep=True):
        """Get parameters for sklearn cloning compatibility."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'base_estimator': self.base_estimator
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn cloning compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
        
    def _flatten_data(self, X):
        """Convert data to 2D for tree-based classifiers."""
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
            n_channels = len(X.columns)
            n_timepoints = len(X.iloc[0, 0])
            X_3d = np.zeros((n_samples, n_channels, n_timepoints))
            for i in range(n_samples):
                for j, col in enumerate(X.columns):
                    X_3d[i, j, :] = X.iloc[i, j].values
            return X_3d.reshape(n_samples, -1)
        elif isinstance(X, np.ndarray):
            if X.ndim == 3:
                return X.reshape(X.shape[0], -1)
            return X
        else:
            raise ValueError(f"Unexpected data type: {type(X)}")
    
    def fit(self, X, y):
        """Fit the classifier."""
        X_flat = self._flatten_data(X)
        
        # Normalize base_estimator to lowercase for comparison
        base_est = str(self.base_estimator).lower()
        if base_est == 'gb':
            self.clf = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                learning_rate=0.1
            )
        else:  # Default to RandomForest
            self.clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        
        self.clf.fit(X_flat, y)
        self.classes_ = self.clf.classes_
        self._is_fitted = True
        return self
    
    def predict(self, X):
        """Predict classes."""
        if self.clf is None:
            raise RuntimeError("Classifier not fitted yet")
        X_flat = self._flatten_data(X)
        return self.clf.predict(X_flat)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if self.clf is None:
            raise RuntimeError("Classifier not fitted yet")
        X_flat = self._flatten_data(X)
        return self.clf.predict_proba(X_flat)


class ComprehensiveTEASERAnalyzer:
    """Enhanced TEASER analyzer with detailed tree metrics and visualizations."""
    
    def __init__(self, n_estimators=50, random_state=42, base_estimator='rf'):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_estimator = base_estimator.lower()  # 'rf' or 'gb'
        self.teaser = None
        self.n_data = None
        self.checkpoints = None
        self.detailed_tree_metrics = {}
        
    def _to_nested_dataframe(self, X):
        """Convert 3D array to nested dataframe."""
        n_samples, n_channels, n_timepoints = X.shape
        data_dict = {}
        for c in range(n_channels):
            data_dict[f'dim_{c}'] = [pd.Series(X[i, c, :]) for i in range(n_samples)]
        return pd.DataFrame(data_dict)
    
    def extract_detailed_tree_metrics(self):
        """Extract comprehensive tree metrics from all RF models in TEASER."""
        detailed_metrics = {
            'per_checkpoint': {},
            'aggregate': {
                'total_trees': 0,
                'total_nodes': 0,
                'all_node_counts': [],
                'all_depths': [],
                'all_leaf_counts': [],
                'memory_estimate_mb': 0
            }
        }
        
        # TEASER creates multiple estimators (one per checkpoint + one for full length)
        if hasattr(self.teaser, '_estimators'):
            logger.info(f"Extracting metrics from {len(self.teaser._estimators)} TEASER estimators")
            
            for i, estimator in enumerate(self.teaser._estimators):
                checkpoint = self.checkpoints[i] if i < len(self.checkpoints) else self.n_data
                
                # Extract tree-based model from the estimator
                tree_model = None
                if hasattr(estimator, 'clf'):
                    tree_model = estimator.clf
                elif hasattr(estimator, 'rf'):
                    tree_model = estimator.rf
                elif isinstance(estimator, (RandomForestClassifier, GradientBoostingClassifier)):
                    tree_model = estimator
                elif hasattr(estimator, 'steps'):  # Pipeline
                    for name, step in estimator.steps:
                        if isinstance(step, (RandomForestClassifier, GradientBoostingClassifier)):
                            tree_model = step
                            break
                
                # Handle RF and GB models
                trees = None
                if tree_model and isinstance(tree_model, RandomForestClassifier) and hasattr(tree_model, 'estimators_'):
                    trees = tree_model.estimators_
                elif tree_model and isinstance(tree_model, GradientBoostingClassifier) and hasattr(tree_model, 'estimators_'):
                    # GB estimators_ is a list of DecisionTreeRegressor objects
                    trees = tree_model.estimators_.flatten() if tree_model.estimators_.ndim > 1 else tree_model.estimators_
                
                if trees is not None and len(trees) > 0:
                    n_trees = len(trees)
                    
                    # Detailed metrics for each tree
                    node_counts = []
                    depths = []
                    leaf_counts = []
                    split_counts = []
                    
                    for tree in trees:
                        node_count = tree.tree_.node_count
                        depth = tree.tree_.max_depth
                        n_leaves = np.sum(tree.tree_.children_left == -1)
                        n_splits = node_count - n_leaves
                        
                        node_counts.append(node_count)
                        depths.append(depth)
                        leaf_counts.append(n_leaves)
                        split_counts.append(n_splits)
                    
                    # Calculate memory estimate (rough approximation)
                    # Each node ~= 80 bytes (feature, threshold, children, value, etc.)
                    total_nodes = sum(node_counts)
                    memory_mb = (total_nodes * 80) / (1024 * 1024)
                    
                    # Store metrics for this checkpoint
                    detailed_metrics['per_checkpoint'][f'cp_{checkpoint}'] = {
                        'checkpoint_value': checkpoint,
                        'n_trees': n_trees,
                        'total_nodes': total_nodes,
                        'mean_nodes': np.mean(node_counts),
                        'std_nodes': np.std(node_counts),
                        'min_nodes': min(node_counts),
                        'max_nodes': max(node_counts),
                        'mean_depth': np.mean(depths),
                        'std_depth': np.std(depths),
                        'max_depth': max(depths),
                        'mean_leaves': np.mean(leaf_counts),
                        'total_leaves': sum(leaf_counts),
                        'mean_splits': np.mean(split_counts),
                        'memory_estimate_mb': memory_mb,
                        'node_distribution': node_counts,
                        'depth_distribution': depths
                    }
                    
                    # Update aggregate metrics
                    detailed_metrics['aggregate']['total_trees'] += n_trees
                    detailed_metrics['aggregate']['total_nodes'] += total_nodes
                    detailed_metrics['aggregate']['all_node_counts'].extend(node_counts)
                    detailed_metrics['aggregate']['all_depths'].extend(depths)
                    detailed_metrics['aggregate']['all_leaf_counts'].extend(leaf_counts)
                    detailed_metrics['aggregate']['memory_estimate_mb'] += memory_mb
                    
                    logger.info(f"Checkpoint {checkpoint}: {n_trees} trees, "
                              f"{total_nodes} total nodes, "
                              f"avg depth {np.mean(depths):.1f}, "
                              f"~{memory_mb:.2f} MB")
        
        # Calculate aggregate statistics
        if detailed_metrics['aggregate']['all_node_counts']:
            detailed_metrics['aggregate']['mean_nodes_overall'] = np.mean(detailed_metrics['aggregate']['all_node_counts'])
            detailed_metrics['aggregate']['mean_depth_overall'] = np.mean(detailed_metrics['aggregate']['all_depths'])
            detailed_metrics['aggregate']['complexity_score'] = (
                detailed_metrics['aggregate']['total_nodes'] * 
                detailed_metrics['aggregate']['mean_depth_overall'] / 1000
            )
        
        self.detailed_tree_metrics = detailed_metrics
        return detailed_metrics
    
    def calculate_advanced_energy_metrics(self, window_earliness, decision_times, tree_metrics):
        """
        Calculate advanced energy metrics considering both data processing and model complexity.
        """
        # Energy model parameters
        BASE_ENERGY = 0.1
        ENERGY_PER_SAMPLE = 0.01
        ENERGY_PER_TREE_NODE = 0.0001  # Energy for tree traversal
        
        # Calculate energy for each window
        window_energy = []
        for i, (earliness, decision_time) in enumerate(zip(window_earliness, decision_times)):
            # Data processing energy
            data_energy = BASE_ENERGY + (decision_time * ENERGY_PER_SAMPLE)
            
            # Model complexity energy (tree traversal)
            # Find which checkpoint was used
            checkpoint_key = f'cp_{decision_time}'
            if checkpoint_key in tree_metrics['per_checkpoint']:
                tree_nodes = tree_metrics['per_checkpoint'][checkpoint_key]['mean_nodes']
                n_trees = tree_metrics['per_checkpoint'][checkpoint_key]['n_trees']
                model_energy = n_trees * tree_nodes * ENERGY_PER_TREE_NODE
            else:
                # Estimate based on average
                model_energy = 50 * 50 * ENERGY_PER_TREE_NODE  # Default estimate
            
            total_energy = data_energy + model_energy
            window_energy.append(total_energy)
        
        window_energy = np.array(window_energy)
        
        # Calculate savings
        max_energy = BASE_ENERGY + (self.n_data * ENERGY_PER_SAMPLE) + (50 * 50 * ENERGY_PER_TREE_NODE)
        energy_saved = max_energy - window_energy
        energy_saved_percentage = (energy_saved / max_energy) * 100
        
        # Energy efficiency score (considers both savings and accuracy)
        efficiency_score = energy_saved_percentage.mean()
        
        return {
            'per_window_energy': window_energy,
            'per_window_saved': energy_saved,
            'per_window_saved_percentage': energy_saved_percentage,
            'mean_energy': float(window_energy.mean()),
            'std_energy': float(window_energy.std()),
            'total_energy': float(window_energy.sum()),
            'mean_saved': float(energy_saved.mean()),
            'mean_saved_percentage': float(energy_saved_percentage.mean()),
            'efficiency_score': float(efficiency_score),
            'energy_per_correct': None  # Will be calculated with accuracy
        }
    
    def fit_and_analyze(self, X_train, y_train, X_test, y_test, checkpoints, n_data):
        """Fit TEASER and perform complete analysis with detailed metrics."""
        self.checkpoints = sorted(checkpoints)
        self.n_data = n_data
        
        # Convert to nested format
        X_train_nested = self._to_nested_dataframe(X_train)
        X_test_nested = self._to_nested_dataframe(X_test)
        n_test = len(X_test_nested)
        
        # Create and fit TEASER
        base_estimator = SimpleTimeSeriesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            base_estimator=self.base_estimator
        )
        
        self.teaser = TEASER(
            estimator=base_estimator,
            classification_points=self.checkpoints,
            one_class_classifier=OneClassSVM(kernel='rbf', gamma='scale'),
            one_class_param_grid={"nu": [0.1, 0.3]},
            random_state=self.random_state,
            n_jobs=1
        )
        
        logger.info(f"Fitting TEASER with checkpoints: {self.checkpoints}")
        print(f"    [TEASER] Starting training (this may take 10-30+ minutes per dataset)...")
        import time
        start_fit = time.time()
        self.teaser.fit(X_train_nested, y_train)
        fit_time = time.time() - start_fit
        logger.info("✓ TEASER fitted")
        print(f"    [TEASER] Training completed in {fit_time/60:.1f} minutes")
        
        # Make predictions
        result = self.teaser.predict(X_test_nested)
        if isinstance(result, tuple):
            predictions = result[0]
        else:
            predictions = result
        
        # Extract per-window earliness
        state_info = self.teaser.state_info
        decision_times = state_info[:, 3].astype(int)
        decision_idx = state_info[:, 0].astype(int)   # which checkpoint committed (0-based)
        window_earliness = 1 - (decision_times / n_data)

        # Per-sample class probabilities (for the records schema). predict_proba
        # re-runs and overwrites state_info, so it is called AFTER the decision_*
        # captures above.
        try:
            _pr = self.teaser.predict_proba(X_test_nested)
            proba_matrix = _pr[0] if isinstance(_pr, tuple) else _pr
        except Exception:
            proba_matrix = None
        proba_classes = getattr(self.teaser, "classes_", None)
        
        # Calculate accuracy + macro/weighted precision/recall/F1
        overall_accuracy = accuracy_score(y_test, predictions)
        prf = {
            'precision_macro': precision_score(y_test, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, predictions, average='weighted', zero_division=0),
        }
        conf = bd.confusion_aggregates(y_test, predictions)
        per_class = [{"Method": "TEASER", "config_id": "best", **pc}
                     for pc in bd.per_class_metrics(y_test, predictions)]

        # Simple earliness calculation
        mean_earliness = window_earliness.mean()
        
        # Time measurement (inference time in seconds per sample)
        import time
        start_inference = time.time()
        _ = self.teaser.predict(X_test_nested[:10])  # Small sample for timing
        inference_time_per_sample = (time.time() - start_inference) / 10
        
        # Energy calculation (convert to mJ)
        # Energy = data_used_percentage / 100 (as per your model)
        data_used_percentage = (1 - mean_earliness) * 100
        energy_per_sample = data_used_percentage / 100  # Energy ratio
        # Convert to mJ (assuming 1 unit = 1 mJ, adjust conversion factor as needed)
        energy_mj = energy_per_sample * 1000  # Convert to mJ (adjust factor if needed)
        
        return {
            'predictions': predictions,
            'decision_times': decision_times,
            'decision_idx': decision_idx,
            'y_proba_matrix': proba_matrix,
            'proba_classes': proba_classes,
            'n_checkpoints': len(self.checkpoints),
            'window_earliness': window_earliness,
            'accuracy': float(overall_accuracy),
            'energy_savings': float(mean_earliness * 100),  # sensing-energy savings %, paper axis
            **{k: float(v) for k, v in prf.items()},
            'TP': conf['TP'], 'FP': conf['FP'], 'FN': conf['FN'], 'TN': conf['TN'],
            'specificity_macro': float(conf['specificity_macro']),
            'fpr_macro': float(conf['fpr_macro']),
            'fnr_macro': float(conf['fnr_macro']),
            'support': conf['support'],
            '_per_class': per_class,
            'time_s': float(inference_time_per_sample),
            'energy_mj': float(energy_mj)
        }


def create_meaningful_comparison_plots(all_results, save_prefix="TEASER"):
    """Create insightful plots showing relationships between metrics."""
    
    # Prepare data for plotting
    plot_data = []
    for r in all_results:
        plot_data.append({
            'Dataset': r['dataset'],
            'Accuracy': r['accuracy_metrics']['overall_accuracy'],
            'Mean_Earliness': r['earliness_stats']['mean'],
            'Energy_Saved_%': r['energy_metrics']['mean_saved_percentage'],
            'Total_Trees': r['tree_metrics']['aggregate']['total_trees'],
            'Total_Nodes': r['tree_metrics']['aggregate']['total_nodes'],
            'Mean_Tree_Depth': r['tree_metrics']['aggregate'].get('mean_depth_overall', 0),
            'Memory_MB': r['tree_metrics']['aggregate']['memory_estimate_mb'],
            'Harmonic_Mean': r['composite_scores']['harmonic_mean'],
            'EAC_Score': r['composite_scores']['eac_score'],
            'Complexity_Score': r['tree_metrics']['aggregate'].get('complexity_score', 0)
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with meaningful plots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Energy Saved vs Accuracy (with bubble size = tree complexity)
    ax1 = plt.subplot(3, 4, 1)
    scatter = ax1.scatter(df['Energy_Saved_%'], df['Accuracy'], 
                         s=df['Total_Nodes']/50, alpha=0.6, 
                         c=df['Mean_Earliness'], cmap='viridis')
    for i, txt in enumerate(df['Dataset']):
        ax1.annotate(txt, (df['Energy_Saved_%'].iloc[i], df['Accuracy'].iloc[i]), 
                    fontsize=8, ha='center')
    ax1.set_xlabel('Energy Saved (%)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Energy Efficiency vs Accuracy\n(bubble size = model complexity)')
    plt.colorbar(scatter, ax=ax1, label='Mean Earliness')
    ax1.grid(True, alpha=0.3)
    
    # 2. Tree Complexity vs Performance
    ax2 = plt.subplot(3, 4, 2)
    ax2.scatter(df['Total_Nodes'], df['Accuracy'], alpha=0.7, s=100, color='darkblue')
    for i, txt in enumerate(df['Dataset']):
        ax2.annotate(txt, (df['Total_Nodes'].iloc[i], df['Accuracy'].iloc[i]), 
                    fontsize=8, ha='center')
    ax2.set_xlabel('Total RF Nodes')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Complexity vs Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['Total_Nodes'], df['Accuracy'], 1)
    p = np.poly1d(z)
    ax2.plot(df['Total_Nodes'], p(df['Total_Nodes']), "r--", alpha=0.5)
    
    # 3. Earliness vs Tree Depth
    ax3 = plt.subplot(3, 4, 3)
    ax3.scatter(df['Mean_Earliness'], df['Mean_Tree_Depth'], 
                s=100, alpha=0.7, c=df['Accuracy'], cmap='RdYlGn')
    for i, txt in enumerate(df['Dataset']):
        ax3.annotate(txt, (df['Mean_Earliness'].iloc[i], df['Mean_Tree_Depth'].iloc[i]), 
                    fontsize=8, ha='center')
    ax3.set_xlabel('Mean Earliness')
    ax3.set_ylabel('Mean Tree Depth')
    ax3.set_title('Earliness vs Model Depth')
    plt.colorbar(ax3.collections[0], ax=ax3, label='Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # 4. Memory Usage Analysis
    ax4 = plt.subplot(3, 4, 4)
    bars = ax4.bar(range(len(df)), df['Memory_MB'], alpha=0.7, 
                   color=plt.cm.viridis(df['Accuracy']))
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Memory (MB)')
    ax4.set_title('Model Memory Requirements')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['Dataset'], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Efficiency-Accuracy Trade-off (2D density)
    ax5 = plt.subplot(3, 4, 5)
    hexbin = ax5.hexbin(df['Energy_Saved_%'], df['Accuracy'], 
                        gridsize=15, cmap='YlOrRd', mincnt=1)
    ax5.set_xlabel('Energy Saved (%)')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Efficiency-Accuracy Density')
    plt.colorbar(hexbin, ax=ax5, label='Count')
    
    # Add ideal zones
    ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Accuracy')
    ax5.axvline(x=30, color='blue', linestyle='--', alpha=0.5, label='Good Efficiency')
    ax5.legend()
    
    # 6. Composite Score Comparison
    ax6 = plt.subplot(3, 4, 6)
    x = np.arange(len(df))
    width = 0.25
    bars1 = ax6.bar(x - width, df['Harmonic_Mean'], width, label='Harmonic Mean', alpha=0.7)
    bars2 = ax6.bar(x, df['EAC_Score'], width, label='EAC Score', alpha=0.7)
    bars3 = ax6.bar(x + width, df['Accuracy'], width, label='Accuracy', alpha=0.7)
    ax6.set_xlabel('Dataset')
    ax6.set_ylabel('Score')
    ax6.set_title('Performance Metrics Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(df['Dataset'], rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Tree Size Distribution (violin plot)
    ax7 = plt.subplot(3, 4, 7)
    tree_sizes_by_dataset = []
    labels = []
    for r in all_results:
        if 'per_checkpoint' in r['tree_metrics']:
            for cp_data in r['tree_metrics']['per_checkpoint'].values():
                if 'node_distribution' in cp_data:
                    tree_sizes_by_dataset.extend(cp_data['node_distribution'])
                    labels.extend([r['dataset']] * len(cp_data['node_distribution']))
    
    if tree_sizes_by_dataset:
        df_trees = pd.DataFrame({'Dataset': labels, 'Tree_Size': tree_sizes_by_dataset})
        unique_datasets = df_trees['Dataset'].unique()
        positions = range(len(unique_datasets))
        
        for i, dataset in enumerate(unique_datasets):
            data = df_trees[df_trees['Dataset'] == dataset]['Tree_Size']
            parts = ax7.violinplot([data], positions=[i], showmeans=True)
        
        ax7.set_xlabel('Dataset')
        ax7.set_ylabel('Tree Size (nodes)')
        ax7.set_title('Tree Size Distribution')
        ax7.set_xticks(positions)
        ax7.set_xticklabels(unique_datasets, rotation=45)
    ax7.grid(True, alpha=0.3)
    
    # 8. Earliness Distribution Comparison
    ax8 = plt.subplot(3, 4, 8)
    for i, r in enumerate(all_results):
        ax8.hist(r['window_earliness'], bins=20, alpha=0.5, 
                label=r['dataset'], density=True)
    ax8.set_xlabel('Earliness')
    ax8.set_ylabel('Density')
    ax8.set_title('Earliness Distribution Across Datasets')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Pareto Front: Accuracy vs Energy
    ax9 = plt.subplot(3, 4, 9)
    ax9.scatter(df['Accuracy'], df['Energy_Saved_%'], s=100, alpha=0.7)
    
    # Find Pareto optimal points
    pareto_points = []
    for i in range(len(df)):
        is_pareto = True
        for j in range(len(df)):
            if i != j:
                if df['Accuracy'].iloc[j] > df['Accuracy'].iloc[i] and \
                   df['Energy_Saved_%'].iloc[j] > df['Energy_Saved_%'].iloc[i]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append(i)
    
    # Highlight Pareto optimal points
    if pareto_points:
        ax9.scatter(df['Accuracy'].iloc[pareto_points], 
                   df['Energy_Saved_%'].iloc[pareto_points], 
                   s=200, alpha=1.0, edgecolors='red', linewidths=2, 
                   facecolors='none', label='Pareto Optimal')
    
    for i, txt in enumerate(df['Dataset']):
        ax9.annotate(txt, (df['Accuracy'].iloc[i], df['Energy_Saved_%'].iloc[i]), 
                    fontsize=8, ha='center')
    ax9.set_xlabel('Accuracy')
    ax9.set_ylabel('Energy Saved (%)')
    ax9.set_title('Pareto Front: Accuracy vs Efficiency')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Correlation Heatmap
    ax10 = plt.subplot(3, 4, 10)
    corr_columns = ['Accuracy', 'Mean_Earliness', 'Energy_Saved_%', 
                    'Total_Nodes', 'Mean_Tree_Depth', 'Memory_MB']
    corr_data = df[corr_columns].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax10, cbar_kws={'label': 'Correlation'})
    ax10.set_title('Metric Correlations')
    
    # 11. 3D Plot: Accuracy vs Earliness vs Energy
    ax11 = plt.subplot(3, 4, 11, projection='3d')
    scatter_3d = ax11.scatter(df['Accuracy'], df['Mean_Earliness'], 
                              df['Energy_Saved_%'], 
                              s=100, c=df['Total_Nodes'], cmap='plasma', alpha=0.6)
    ax11.set_xlabel('Accuracy')
    ax11.set_ylabel('Mean Earliness')
    ax11.set_zlabel('Energy Saved (%)')
    ax11.set_title('3D Performance Space')
    plt.colorbar(scatter_3d, ax=ax11, label='Total Nodes', pad=0.1)
    
    # 12. Summary Table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Create summary statistics
    summary_text = f"""
TEASER PERFORMANCE SUMMARY
{'='*40}

Best Accuracy:      {df['Accuracy'].max():.3f} ({df.loc[df['Accuracy'].idxmax(), 'Dataset']})
Best Earliness:     {df['Mean_Earliness'].max():.3f} ({df.loc[df['Mean_Earliness'].idxmax(), 'Dataset']})
Best Energy Saving: {df['Energy_Saved_%'].max():.1f}% ({df.loc[df['Energy_Saved_%'].idxmax(), 'Dataset']})
Most Efficient:     {df.loc[df['EAC_Score'].idxmax(), 'Dataset']} (EAC: {df['EAC_Score'].max():.3f})

Average Performance:
  Accuracy:      {df['Accuracy'].mean():.3f} ± {df['Accuracy'].std():.3f}
  Earliness:     {df['Mean_Earliness'].mean():.3f} ± {df['Mean_Earliness'].std():.3f}
  Energy Saved:  {df['Energy_Saved_%'].mean():.1f}% ± {df['Energy_Saved_%'].std():.1f}%
  
Model Complexity:
  Total Trees:   {df['Total_Trees'].sum()}
  Total Nodes:   {df['Total_Nodes'].sum():,}
  Memory Usage:  {df['Memory_MB'].sum():.1f} MB total

Optimal Trade-offs (Pareto):
  {', '.join(df.iloc[pareto_points]['Dataset'].tolist()) if pareto_points else 'None identified'}
"""
    
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('TEASER Comprehensive Performance Analysis (RF)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot (save_prefix already includes full path from main function)
    plt.savefig(f'{save_prefix}_meaningful_analysis.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved meaningful analysis plot to {save_prefix}_meaningful_analysis.png")
    
    return fig


def main():
    """Main function processing all datasets with comprehensive analysis."""
    # Get today's date for filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Configure datasets to run - comment/uncomment as needed
    # datasets = ["Epilepsy"]  # Local testing with Epilepsy
    datasets = ["Epilepsy", "PAMAP2", "Shoaib", "WESADchest", "EMGPhysical", "SelfRegulationSCP1"]  # All datasets
    
    # Create results directory (single labeled folder for all baselines)
    results_dir = str(bd.ensure_results_dir())
    logger.info(f"Results will be saved to: {results_dir}/")
    
    all_results = []
    
    for dataset in datasets:
        print("\n" + "="*80)
        print(f"PROCESSING DATASET: {dataset}")
        print("="*80)
        
        try:
            # Fair preprocessing via baseline_data.py: no PAMAP2 aug,
            # 60/20/20 split (rs=42), train on 60% / test on 20%. Feed TEASER the
            # channel-first canonical view (N, F, T_real) so "time" is the real
            # time axis (fixes the packed 3*T axis for triaxial Shoaib/PAMAP2).
            X, labels = bd.load_raw(dataset)
            train_idx, _, test_idx = bd.split_60_20_20(len(labels), labels)
            V = np.swapaxes(bd.temporal_view(X, dataset), 1, 2)   # (N, F, T_real)
            n_samples, n_channels, n_timepoints = V.shape

            X_train = V[train_idx]
            X_test = V[test_idx]
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            print(f"Canonical view (N, F, T): {V.shape}")
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")

            # Setup checkpoints - proportional to REAL data length
            checkpoints = [n_timepoints // 4, n_timepoints // 2, 3 * n_timepoints // 4]
            print(f"Checkpoints: {checkpoints}")

            # RF base estimator (the comparative RF baseline)
            base_est = 'rf'
            # Create analyzer and run analysis
            print(f"  Creating analyzer with {base_est.upper()} (50 estimators)...")
            analyzer = ComprehensiveTEASERAnalyzer(n_estimators=50, random_state=42, base_estimator=base_est)
            print(f"  Starting fit_and_analyze (this may take a while)...")
            print(f"  Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
            results = analyzer.fit_and_analyze(X_train, y_train, X_test, y_test, checkpoints, n_timepoints)
            print(f"  ✓ Analysis complete for {dataset}!")
            
            # Store results with base estimator info
            results['dataset'] = dataset
            results['base_estimator'] = base_est
            for pc in results.get('_per_class', []):
                pc['Dataset'] = dataset

            # Unified per-sample records: TIME axis (sensing_used = decision time).
            # Compute = MEASURED trees executed across the checkpoint forests run up
            # to the committing checkpoint (TEASER is a cascade): (decision_idx+1)
            # checkpoints x n_estimators per checkpoint.
            n_ckpt = results.get('n_checkpoints', len(checkpoints))
            dec_idx = results.get('decision_idx')
            if dec_idx is not None:
                comp_used = (np.asarray(dec_idx).astype(int) + 1) * analyzer.n_estimators
            else:
                comp_used = np.nan
            results['_records'] = bd.build_sample_records(
                dataset=dataset, method="TEASER", config_id="best", random_state=42,
                sample_ids=test_idx, y_true=y_test, y_pred=results['predictions'],
                y_proba=results.get('y_proba_matrix'), classes=results.get('proba_classes'),
                sensing_axis="time",
                sensing_used=np.asarray(results['decision_times']).astype(int),
                sensing_total=int(n_timepoints),
                compute_units_used=comp_used,
                compute_units_total=n_ckpt * analyzer.n_estimators, compute_kind="trees")
            all_results.append(results)
            
            # Print summary
            print(f"\n✓ {dataset} (RF) COMPLETE!")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Time: {results['time_s']:.4f} s")
            print(f"  Energy: {results['energy_mj']:.2f} mJ")
            
            # Save individual dataset results with GB suffix
            df = pd.DataFrame({
                'window_id': range(len(results['window_earliness'])),
                'decision_time': results['decision_times'],
                'earliness': results['window_earliness'],
                'predicted_class': results['predictions']
            })
            
            csv_file = os.path.join(results_dir, f"{date_str}_{dataset}_TEASER_RF_detailed.csv")
            df.to_csv(csv_file, index=False)
            print(f"  Saved to: {csv_file}")
            
            # IMMEDIATELY save partial summary after each dataset
            if all_results:
                summary_data = []
                for r in all_results:
                    summary_data.append({
                        'Dataset': r['dataset'],
                        'Accuracy': r['accuracy'],
                        'Precision_macro': r.get('precision_macro'),
                        'Recall_macro': r.get('recall_macro'),
                        'F1_macro': r.get('f1_macro'),
                        'Precision_weighted': r.get('precision_weighted'),
                        'Recall_weighted': r.get('recall_weighted'),
                        'F1_weighted': r.get('f1_weighted'),
                        'FP': r.get('FP'), 'FN': r.get('FN'), 'TN': r.get('TN'),
                        'Specificity_macro': r.get('specificity_macro'),
                        'Support': r.get('support'),
                        'Time_s': r['time_s'],
                        'Energy_mJ': r['energy_mj']
                    })
                df_summary = pd.DataFrame(summary_data)
                summary_file = os.path.join(results_dir, f'{date_str}_TEASER_RF_summary.csv')
                df_summary.to_csv(summary_file, index=False)
                print(f"  ✓ Updated summary: {summary_file}")
            
        except Exception as e:
            print(f"\n✗ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final summary
    if all_results:
        summary_data = []
        for r in all_results:
            summary_data.append({
                'Dataset': r['dataset'],
                'Accuracy': r['accuracy'],
                'Precision_macro': r.get('precision_macro'),
                'Recall_macro': r.get('recall_macro'),
                'F1_macro': r.get('f1_macro'),
                'Precision_weighted': r.get('precision_weighted'),
                'Recall_weighted': r.get('recall_weighted'),
                'F1_weighted': r.get('f1_weighted'),
                'FP': r.get('FP'), 'FN': r.get('FN'), 'TN': r.get('TN'),
                'Specificity_macro': r.get('specificity_macro'),
                'Support': r.get('support'),
                'Time_s': r['time_s'],
                'Energy_mJ': r['energy_mj']
            })

        df_summary = pd.DataFrame(summary_data)
        summary_file = os.path.join(results_dir, f'{date_str}_TEASER_RF_summary.csv')
        df_summary.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to: {summary_file}")

        # Unified plot-ready rows: one headline point per dataset (sensing axis).
        prf_keys = ['precision_macro', 'recall_macro', 'f1_macro',
                    'precision_weighted', 'recall_weighted', 'f1_weighted']
        plot_rows = [
            bd.plot_row(r['dataset'], 'TEASER', r['accuracy'], r.get('energy_savings', 0.0),
                        energy_type='sensing', n_estimators=50, num_exits=1,
                        config_id='best', is_headline=True,
                        window_fraction_used=1.0 - r.get('energy_savings', 0.0) / 100.0,
                        compute_fraction_used=1.0 - r.get('energy_savings', 0.0) / 100.0,
                        compute_fraction_source='proxy', savings_kind='projected',
                        prf_dict={k: r.get(k) for k in prf_keys})
            for r in all_results
        ]
        bd.write_plot_csv(plot_rows, os.path.join(results_dir, f'{date_str}_TEASER_RF_plot.csv'))
        print(f"✓ Plot-ready saved to: {os.path.join(results_dir, f'{date_str}_TEASER_RF_plot.csv')}")

        per_class_rows = [pc for r in all_results for pc in r.get('_per_class', [])]
        if per_class_rows:
            pc_file = os.path.join(results_dir, f'{date_str}_TEASER_RF_per_class.csv')
            pd.DataFrame(per_class_rows).to_csv(pc_file, index=False)
            print(f"✓ Per-class confusion saved to: {pc_file}")

        record_rows = [rec for r in all_results for rec in r.get('_records', [])]
        if record_rows:
            rec_file = os.path.join(results_dir, f'{date_str}_TEASER_RF_records.csv')
            bd.write_sample_records(record_rows, rec_file)
            print(f"✓ Per-sample records saved to: {rec_file}")

        print("\nFinal Summary:")
        print(df_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
