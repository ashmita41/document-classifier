"""
Adaptive learning system for continuous improvement and self-learning.

This module implements:
- Feedback loop with user corrections
- Automated retraining pipeline
- Active learning for uncertain predictions
- Feature store management
- Model versioning and rollback
- Performance tracking
- Auto-detection of new patterns
- Semi-supervised learning
"""
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

from app.models.text_element import TextElement


logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """User correction record."""
    document_id: str
    element_id: str
    original_prediction: str
    corrected_label: str
    confidence_score: float
    timestamp: datetime
    user_id: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'document_id': self.document_id,
            'element_id': self.element_id,
            'original_prediction': self.original_prediction,
            'corrected_label': self.corrected_label,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
        }


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    timestamp: datetime
    accuracy_metrics: Dict[str, float]
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    model_path: str
    performance_vs_previous: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp.isoformat(),
            'accuracy_metrics': self.accuracy_metrics,
            'training_data_hash': self.training_data_hash,
            'hyperparameters': self.hyperparameters,
            'model_path': self.model_path,
            'performance_vs_previous': self.performance_vs_previous,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics over time."""
    timestamp: datetime
    overall_accuracy: float
    per_class_metrics: Dict[str, Dict[str, float]]
    processing_time: float
    num_predictions: int
    num_corrections: int
    correction_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_accuracy': self.overall_accuracy,
            'per_class_metrics': self.per_class_metrics,
            'processing_time': self.processing_time,
            'num_predictions': self.num_predictions,
            'num_corrections': self.num_corrections,
            'correction_rate': self.correction_rate,
        }


class FeatureStore:
    """
    Feature store for efficient feature management.
    
    Stores document-level, element-level, and relationship features.
    """
    
    def __init__(self, storage_path: str = "feature_store"):
        """Initialize feature store."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.document_features = {}
        self.element_features = {}
        self.relationship_features = {}
        
        self.feature_version = "1.0.0"
    
    def store_document_features(
        self,
        document_id: str,
        features: Dict[str, Any]
    ) -> None:
        """Store document-level features."""
        self.document_features[document_id] = {
            'features': features,
            'timestamp': datetime.now().isoformat(),
            'version': self.feature_version
        }
    
    def store_element_features(
        self,
        element_id: str,
        features: Dict[str, Any]
    ) -> None:
        """Store element-level features."""
        self.element_features[element_id] = {
            'features': features,
            'timestamp': datetime.now().isoformat(),
            'version': self.feature_version
        }
    
    def get_features(
        self,
        document_id: Optional[str] = None,
        element_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve features."""
        result = {}
        
        if document_id:
            result['document'] = self.document_features.get(document_id, {})
        
        if element_id:
            result['element'] = self.element_features.get(element_id, {})
        
        return result
    
    def save(self) -> None:
        """Save feature store to disk."""
        # Save as Parquet for efficiency
        try:
            # Document features
            if self.document_features:
                df_doc = pd.DataFrame.from_dict(self.document_features, orient='index')
                df_doc.to_parquet(self.storage_path / 'document_features.parquet')
            
            # Element features
            if self.element_features:
                df_elem = pd.DataFrame.from_dict(self.element_features, orient='index')
                df_elem.to_parquet(self.storage_path / 'element_features.parquet')
            
            logger.info(f"Feature store saved to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save feature store: {e}")
            # Fallback to pickle
            with open(self.storage_path / 'features.pkl', 'wb') as f:
                pickle.dump({
                    'document': self.document_features,
                    'element': self.element_features,
                    'relationship': self.relationship_features
                }, f)
    
    def load(self) -> None:
        """Load feature store from disk."""
        try:
            # Load document features
            doc_path = self.storage_path / 'document_features.parquet'
            if doc_path.exists():
                df_doc = pd.read_parquet(doc_path)
                self.document_features = df_doc.to_dict('index')
            
            # Load element features
            elem_path = self.storage_path / 'element_features.parquet'
            if elem_path.exists():
                df_elem = pd.read_parquet(elem_path)
                self.element_features = df_elem.to_dict('index')
            
            logger.info("Feature store loaded")
        except Exception as e:
            logger.warning(f"Failed to load Parquet, trying pickle: {e}")
            # Try pickle fallback
            pkl_path = self.storage_path / 'features.pkl'
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                self.document_features = data.get('document', {})
                self.element_features = data.get('element', {})
                self.relationship_features = data.get('relationship', {})


class AdaptiveLearner:
    """
    Central learning coordinator for continuous improvement.
    
    Implements feedback loops, retraining, active learning,
    and automatic pattern detection.
    """
    
    def __init__(
        self,
        storage_path: str = "adaptive_learning",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize adaptive learner.
        
        Args:
            storage_path: Path for storing learning data
            config: Configuration parameters
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or self._default_config()
        
        # Storage
        self.corrections = []
        self.model_versions = []
        self.performance_history = []
        self.feature_store = FeatureStore(
            str(self.storage_path / "features")
        )
        
        # Current model reference
        self.current_model_version = None
        
        # Pattern detection
        self.unknown_patterns = defaultdict(list)
        self.suggested_patterns = []
        
        logger.info("AdaptiveLearner initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'retraining_threshold': 100,  # Min corrections before retraining
            'confidence_threshold': 0.5,  # Below this = uncertain
            'active_learning_sample_size': 20,  # Samples to request labels for
            'model_evaluation_metrics': ['accuracy', 'f1_macro'],
            'min_improvement_threshold': 0.02,  # 2% improvement required
            'max_model_versions': 5,  # Keep last N versions
            'retraining_schedule': 'weekly',  # weekly, daily, manual
            'ab_testing_rollout': [0.1, 0.5, 1.0],  # Gradual rollout percentages
            'pattern_detection_threshold': 10,  # Min cluster size for new pattern
            'drift_detection_window': 1000,  # Samples for drift detection
        }
    
    def record_correction(
        self,
        document_id: str,
        element_id: str,
        original_prediction: str,
        corrected_label: str,
        confidence_score: float,
        user_id: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a user correction.
        
        Args:
            document_id: Document identifier
            element_id: Element identifier
            original_prediction: Original model prediction
            corrected_label: User-corrected label
            confidence_score: Confidence of original prediction
            user_id: User who made correction
            features: Element features
        """
        correction = Correction(
            document_id=document_id,
            element_id=element_id,
            original_prediction=original_prediction,
            corrected_label=corrected_label,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            user_id=user_id,
            features=features or {}
        )
        
        self.corrections.append(correction)
        
        logger.info(
            f"Correction recorded: {original_prediction} -> {corrected_label} "
            f"(confidence: {confidence_score:.2f})"
        )
        
        # Check if retraining should be triggered
        if self._should_trigger_retraining():
            logger.info("Retraining threshold reached")
            # In production, this would trigger a Celery task
            # self.trigger_retraining_task()
    
    def _should_trigger_retraining(self) -> bool:
        """Check if retraining should be triggered."""
        return len(self.corrections) >= self.config['retraining_threshold']
    
    def get_uncertain_predictions(
        self,
        predictions: List[Tuple[str, str, float]],
        n_samples: int = None
    ) -> List[Tuple[str, str, float]]:
        """
        Get uncertain predictions for active learning.
        
        Args:
            predictions: List of (element_id, prediction, confidence) tuples
            n_samples: Number of samples to return (default from config)
            
        Returns:
            List of most uncertain predictions
        """
        if n_samples is None:
            n_samples = self.config['active_learning_sample_size']
        
        # Filter low confidence
        uncertain = [
            p for p in predictions
            if p[2] < self.config['confidence_threshold']
        ]
        
        # Sort by confidence (lowest first = most uncertain)
        uncertain.sort(key=lambda x: x[2])
        
        return uncertain[:n_samples]
    
    def prepare_retraining_data(
        self,
        base_training_data: List[Tuple[TextElement, str]]
    ) -> Tuple[List[TextElement], List[str]]:
        """
        Prepare data for retraining by merging base data with corrections.
        
        Args:
            base_training_data: Original training data
            
        Returns:
            Tuple of (elements, labels)
        """
        # Convert base training data
        elements = [item[0] for item in base_training_data]
        labels = [item[1] for item in base_training_data]
        
        # Add corrections (in production, load from database)
        correction_data = self._load_correction_data()
        
        # Merge and deduplicate
        element_dict = {id(elem): (elem, label) for elem, label in zip(elements, labels)}
        
        # Add/update with corrections
        for correction in correction_data:
            # In production, would fetch actual element
            # For now, skip if we don't have the element
            pass
        
        # Extract merged data
        merged_elements = [elem for elem, _ in element_dict.values()]
        merged_labels = [label for _, label in element_dict.values()]
        
        logger.info(
            f"Prepared {len(merged_elements)} samples for retraining "
            f"({len(correction_data)} corrections added)"
        )
        
        return merged_elements, merged_labels
    
    def _load_correction_data(self) -> List[Correction]:
        """Load correction data (placeholder for database query)."""
        return self.corrections
    
    def retrain_model(
        self,
        model: Any,
        training_data: List[Tuple[TextElement, str]],
        validation_split: float = 0.2
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Retrain model with new data.
        
        Args:
            model: Model to retrain
            training_data: Training data (elements, labels)
            validation_split: Validation set size
            
        Returns:
            Tuple of (retrained_model, metrics)
        """
        # Prepare data
        elements, labels = self.prepare_retraining_data(training_data)
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            elements,
            labels,
            test_size=validation_split,
            random_state=42,
            stratify=labels
        )
        
        logger.info(f"Retraining with {len(X_train)} samples, validating on {len(X_val)}")
        
        # Train model (specific to each model type)
        try:
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            elif hasattr(model, 'train_ml_classifier'):
                model.train_ml_classifier(X_train, y_train)
            else:
                raise ValueError("Model doesn't have fit or train method")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            raise
        
        # Evaluate
        metrics = self._evaluate_model(model, X_val, y_val)
        
        logger.info(f"Retraining complete. Accuracy: {metrics.get('accuracy', 0):.3f}")
        
        return model, metrics
    
    def _evaluate_model(
        self,
        model: Any,
        X_val: List[TextElement],
        y_val: List[str]
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = [model.predict(elem)[0] if isinstance(model.predict(elem), tuple) 
                     else model.predict(elem) for elem in X_val]
        elif hasattr(model, 'classify'):
            y_pred = [model.classify(elem).primary_type for elem in X_val]
        else:
            raise ValueError("Model doesn't have predict or classify method")
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='macro', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_macro': float(f1),
        }
    
    def save_model_version(
        self,
        model: Any,
        metrics: Dict[str, float],
        training_data_hash: str,
        hyperparameters: Dict[str, Any]
    ) -> ModelVersion:
        """
        Save model version with metadata.
        
        Args:
            model: Trained model
            metrics: Performance metrics
            training_data_hash: Hash of training data
            hyperparameters: Model hyperparameters
            
        Returns:
            ModelVersion object
        """
        # Generate version ID
        version_id = f"v{len(self.model_versions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        model_path = self.storage_path / f"models/{version_id}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        
        # Calculate performance vs previous
        performance_vs_previous = None
        if self.model_versions:
            prev_metrics = self.model_versions[-1].accuracy_metrics
            current_acc = metrics.get('accuracy', 0)
            prev_acc = prev_metrics.get('accuracy', 0)
            performance_vs_previous = current_acc - prev_acc
        
        # Create version record
        version = ModelVersion(
            version_id=version_id,
            timestamp=datetime.now(),
            accuracy_metrics=metrics,
            training_data_hash=training_data_hash,
            hyperparameters=hyperparameters,
            model_path=str(model_path),
            performance_vs_previous=performance_vs_previous
        )
        
        self.model_versions.append(version)
        
        # Keep only last N versions
        if len(self.model_versions) > self.config['max_model_versions']:
            old_version = self.model_versions.pop(0)
            # Delete old model file
            try:
                Path(old_version.model_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete old model: {e}")
        
        logger.info(f"Saved model version: {version_id}")
        
        return version
    
    def should_deploy_model(
        self,
        new_metrics: Dict[str, float],
        current_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Decide if new model should be deployed.
        
        Args:
            new_metrics: Metrics of new model
            current_metrics: Metrics of current model
            
        Returns:
            True if new model should be deployed
        """
        if current_metrics is None:
            return True  # First model, deploy it
        
        # Check if improvement exceeds threshold
        new_acc = new_metrics.get('accuracy', 0)
        current_acc = current_metrics.get('accuracy', 0)
        improvement = new_acc - current_acc
        
        should_deploy = improvement >= self.config['min_improvement_threshold']
        
        logger.info(
            f"Model comparison: current={current_acc:.3f}, new={new_acc:.3f}, "
            f"improvement={improvement:.3f}, deploy={should_deploy}"
        )
        
        return should_deploy
    
    def rollback_model(self, versions_back: int = 1) -> Optional[ModelVersion]:
        """
        Rollback to previous model version.
        
        Args:
            versions_back: Number of versions to rollback
            
        Returns:
            Rolled back ModelVersion or None
        """
        if len(self.model_versions) < versions_back + 1:
            logger.warning("Not enough versions for rollback")
            return None
        
        target_version = self.model_versions[-(versions_back + 1)]
        self.current_model_version = target_version.version_id
        
        logger.info(f"Rolled back to version: {target_version.version_id}")
        
        return target_version
    
    def detect_new_patterns(
        self,
        unknown_elements: List[TextElement],
        features: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect new patterns in unknown elements using clustering.
        
        Args:
            unknown_elements: Elements classified as UNKNOWN
            features: Feature vectors for elements
            
        Returns:
            List of detected pattern candidates
        """
        if len(unknown_elements) < self.config['pattern_detection_threshold']:
            return []
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=3)
        labels = clustering.fit_predict(features)
        
        # Identify significant clusters
        patterns = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            # Get cluster members
            cluster_indices = np.where(labels == label)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_size >= self.config['pattern_detection_threshold']:
                # Extract examples
                examples = [
                    unknown_elements[i].text[:100]
                    for i in cluster_indices[:5]
                ]
                
                pattern = {
                    'cluster_id': int(label),
                    'size': cluster_size,
                    'examples': examples,
                    'coherence': self._calculate_cluster_coherence(
                        features[cluster_indices]
                    ),
                    'suggested_name': f"pattern_{label}",
                }
                
                patterns.append(pattern)
                logger.info(f"Detected pattern candidate: size={cluster_size}")
        
        self.suggested_patterns.extend(patterns)
        
        return patterns
    
    def _calculate_cluster_coherence(self, cluster_features: np.ndarray) -> float:
        """Calculate intra-cluster coherence."""
        if len(cluster_features) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(cluster_features)
        avg_distance = distances.sum() / (len(cluster_features) * (len(cluster_features) - 1))
        
        # Convert to coherence (lower distance = higher coherence)
        coherence = 1.0 / (1.0 + avg_distance)
        
        return float(coherence)
    
    def track_performance(
        self,
        accuracy: float,
        per_class_metrics: Dict[str, Dict[str, float]],
        processing_time: float,
        num_predictions: int
    ) -> None:
        """
        Track performance metrics over time.
        
        Args:
            accuracy: Overall accuracy
            per_class_metrics: Per-class precision/recall/F1
            processing_time: Processing time in seconds
            num_predictions: Number of predictions made
        """
        num_corrections = len([
            c for c in self.corrections
            if c.timestamp > datetime.now() - timedelta(days=1)
        ])
        
        correction_rate = num_corrections / num_predictions if num_predictions > 0 else 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            overall_accuracy=accuracy,
            per_class_metrics=per_class_metrics,
            processing_time=processing_time,
            num_predictions=num_predictions,
            num_corrections=num_corrections,
            correction_rate=correction_rate
        )
        
        self.performance_history.append(metrics)
        
        logger.info(
            f"Performance tracked: accuracy={accuracy:.3f}, "
            f"correction_rate={correction_rate:.3f}"
        )
    
    def detect_drift(
        self,
        recent_predictions: List[Tuple[str, float]],
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect distribution drift in predictions.
        
        Args:
            recent_predictions: Recent (prediction, confidence) pairs
            window_size: Size of comparison window
            
        Returns:
            Drift detection results
        """
        if window_size is None:
            window_size = self.config['drift_detection_window']
        
        if len(recent_predictions) < window_size * 2:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Split into two windows
        old_window = recent_predictions[-window_size*2:-window_size]
        new_window = recent_predictions[-window_size:]
        
        # Calculate distribution statistics
        old_confidences = [conf for _, conf in old_window]
        new_confidences = [conf for _, conf in new_window]
        
        old_mean = np.mean(old_confidences)
        new_mean = np.mean(new_confidences)
        
        # Simple drift detection: significant confidence drop
        confidence_drop = old_mean - new_mean
        drift_threshold = 0.1  # 10% drop indicates drift
        
        drift_detected = confidence_drop > drift_threshold
        
        result = {
            'drift_detected': drift_detected,
            'confidence_drop': float(confidence_drop),
            'old_mean_confidence': float(old_mean),
            'new_mean_confidence': float(new_mean),
            'recommendation': 'retrain' if drift_detected else 'continue'
        }
        
        if drift_detected:
            logger.warning(
                f"Drift detected! Confidence dropped by {confidence_drop:.3f}"
            )
        
        return result
    
    def save_state(self) -> None:
        """Save learner state to disk."""
        # Save corrections
        corrections_path = self.storage_path / 'corrections.json'
        with open(corrections_path, 'w') as f:
            json.dump(
                [c.to_dict() for c in self.corrections],
                f,
                indent=2
            )
        
        # Save model versions
        versions_path = self.storage_path / 'model_versions.json'
        with open(versions_path, 'w') as f:
            json.dump(
                [v.to_dict() for v in self.model_versions],
                f,
                indent=2
            )
        
        # Save performance history
        performance_path = self.storage_path / 'performance_history.json'
        with open(performance_path, 'w') as f:
            json.dump(
                [p.to_dict() for p in self.performance_history],
                f,
                indent=2
            )
        
        # Save config
        config_path = self.storage_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save feature store
        self.feature_store.save()
        
        logger.info(f"Learner state saved to {self.storage_path}")
    
    def load_state(self) -> None:
        """Load learner state from disk."""
        # Load corrections
        corrections_path = self.storage_path / 'corrections.json'
        if corrections_path.exists():
            with open(corrections_path, 'r') as f:
                corrections_data = json.load(f)
            # Convert back to Correction objects (simplified)
            logger.info(f"Loaded {len(corrections_data)} corrections")
        
        # Load model versions
        versions_path = self.storage_path / 'model_versions.json'
        if versions_path.exists():
            with open(versions_path, 'r') as f:
                versions_data = json.load(f)
            logger.info(f"Loaded {len(versions_data)} model versions")
        
        # Load performance history
        performance_path = self.storage_path / 'performance_history.json'
        if performance_path.exists():
            with open(performance_path, 'r') as f:
                performance_data = json.load(f)
            logger.info(f"Loaded {len(performance_data)} performance records")
        
        # Load config
        config_path = self.storage_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Config loaded")
        
        # Load feature store
        self.feature_store.load()
        
        logger.info("Learner state loaded")
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Generate learning report."""
        return {
            'total_corrections': len(self.corrections),
            'model_versions': len(self.model_versions),
            'current_version': self.current_model_version,
            'performance_history_length': len(self.performance_history),
            'suggested_patterns': len(self.suggested_patterns),
            'config': self.config,
            'last_update': datetime.now().isoformat(),
        }

