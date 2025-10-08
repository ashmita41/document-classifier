"""
Self-learning pattern classification system for document elements.

This module implements a dynamic pattern learning system that can:
- Extract comprehensive features from document elements
- Discover patterns using unsupervised clustering
- Build supervised classifiers for element classification
- Learn incrementally from new examples
- Persist and load learned patterns
"""
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available, will use Random Forest only")

from app.models.text_element import TextElement, ElementType


logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract numerical features from TextElement objects."""
    
    # Regular expressions for semantic features
    DATE_PATTERN = re.compile(
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|'
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|'
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
        re.IGNORECASE
    )
    
    CURRENCY_PATTERN = re.compile(
        r'[$£€¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|'
        r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|dollars?|pounds?|euros?)',
        re.IGNORECASE
    )
    
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return [
            # Statistical features
            'text_length',
            'word_count',
            'char_count',
            'avg_word_length',
            
            # Formatting features
            'is_bold',
            'is_italic',
            'is_uppercase',
            'is_title_case',
            'is_lowercase',
            
            # Numeric features
            'font_size',
            'font_size_ratio',
            
            # Position features
            'relative_y_position',
            'relative_x_position',
            'indentation_level',
            
            # Content features
            'has_colon',
            'has_question_mark',
            'has_exclamation',
            'starts_with_number',
            'ends_with_period',
            'is_single_word',
            
            # Semantic features
            'contains_date',
            'contains_currency',
            'contains_email',
            'contains_url',
            'digit_ratio',
            'punctuation_ratio',
            
            # Whitespace features
            'lines_before',
            'lines_after',
            'is_isolated_line',
            'vertical_spacing',
            
            # Layout features
            'is_first_on_page',
            'is_last_on_page',
            'paragraph_id',
        ]
    
    def extract_features(
        self, 
        element: TextElement, 
        page_context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract feature vector from text element.
        
        Args:
            element: TextElement to extract features from
            page_context: Optional context about the page (avg font size, etc.)
            
        Returns:
            Feature vector as numpy array
        """
        features = {}
        text = element.text
        
        # Statistical features
        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['char_count'] = len(text.strip())
        features['avg_word_length'] = (
            sum(len(w) for w in words) / len(words) if words else 0
        )
        
        # Formatting features
        features['is_bold'] = float(element.is_bold)
        features['is_italic'] = float(element.is_italic)
        features['is_uppercase'] = float(text.isupper())
        features['is_title_case'] = float(text.istitle())
        features['is_lowercase'] = float(text.islower())
        
        # Numeric features
        features['font_size'] = element.font_size or 0.0
        
        # Calculate font size ratio relative to page average
        if page_context and 'avg_font_size' in page_context:
            avg_font = page_context['avg_font_size']
            features['font_size_ratio'] = (
                features['font_size'] / avg_font if avg_font > 0 else 1.0
            )
        else:
            features['font_size_ratio'] = 1.0
        
        # Position features
        if page_context and 'page_height' in page_context:
            features['relative_y_position'] = (
                element.bbox.y0 / page_context['page_height']
            )
        else:
            features['relative_y_position'] = 0.0
        
        if page_context and 'page_width' in page_context:
            features['relative_x_position'] = (
                element.bbox.x0 / page_context['page_width']
            )
        else:
            features['relative_x_position'] = 0.0
        
        features['indentation_level'] = element.indentation_level or 0
        
        # Content features
        features['has_colon'] = float(':' in text)
        features['has_question_mark'] = float('?' in text)
        features['has_exclamation'] = float('!' in text)
        features['starts_with_number'] = float(
            len(text) > 0 and text[0].isdigit()
        )
        features['ends_with_period'] = float(text.endswith('.'))
        features['is_single_word'] = float(len(words) == 1)
        
        # Semantic features
        features['contains_date'] = float(bool(self.DATE_PATTERN.search(text)))
        features['contains_currency'] = float(bool(self.CURRENCY_PATTERN.search(text)))
        features['contains_email'] = float(bool(self.EMAIL_PATTERN.search(text)))
        features['contains_url'] = float('http' in text.lower() or 'www.' in text.lower())
        
        # Character ratio features
        digits = sum(c.isdigit() for c in text)
        features['digit_ratio'] = digits / len(text) if text else 0.0
        
        punctuation = sum(c in '.,;:!?-()[]{}' for c in text)
        features['punctuation_ratio'] = punctuation / len(text) if text else 0.0
        
        # Whitespace features
        features['lines_before'] = element.vertical_spacing or 0.0
        features['lines_after'] = 0.0  # Will be filled by context processor
        features['is_isolated_line'] = float(
            (element.vertical_spacing or 0.0) > 15.0
        )
        features['vertical_spacing'] = element.vertical_spacing or 0.0
        
        # Layout features
        features['is_first_on_page'] = float(element.line_number == 0)
        features['is_last_on_page'] = 0.0  # Will be filled by context processor
        features['paragraph_id'] = element.paragraph_id or 0
        
        # Convert to ordered array
        feature_vector = np.array([
            features[name] for name in self.feature_names
        ], dtype=np.float32)
        
        return feature_vector
    
    def extract_batch_features(
        self, 
        elements: List[TextElement],
        compute_context: bool = True
    ) -> np.ndarray:
        """
        Extract features from multiple elements with context.
        
        Args:
            elements: List of TextElements
            compute_context: Whether to compute page context
            
        Returns:
            Feature matrix (n_elements × n_features)
        """
        if not elements:
            return np.array([])
        
        # Compute page context if requested
        page_contexts = {}
        if compute_context:
            for element in elements:
                page_num = element.page_number
                if page_num not in page_contexts:
                    page_elements = [e for e in elements if e.page_number == page_num]
                    page_contexts[page_num] = self._compute_page_context(page_elements)
        
        # Extract features for each element
        features_list = []
        for i, element in enumerate(elements):
            context = page_contexts.get(element.page_number) if compute_context else None
            features = self.extract_features(element, context)
            features_list.append(features)
        
        # Post-process to fill in relative features
        if compute_context:
            features_matrix = np.array(features_list)
            self._fill_relative_features(features_matrix, elements)
            return features_matrix
        
        return np.array(features_list)
    
    def _compute_page_context(self, page_elements: List[TextElement]) -> Dict[str, Any]:
        """Compute context statistics for a page."""
        context = {}
        
        # Average font size
        font_sizes = [e.font_size for e in page_elements if e.font_size]
        context['avg_font_size'] = (
            sum(font_sizes) / len(font_sizes) if font_sizes else 12.0
        )
        
        # Page dimensions (approximate from bounding boxes)
        if page_elements:
            context['page_height'] = max(e.bbox.y1 for e in page_elements)
            context['page_width'] = max(e.bbox.x1 for e in page_elements)
        else:
            context['page_height'] = 792.0  # US Letter default
            context['page_width'] = 612.0
        
        return context
    
    def _fill_relative_features(
        self, 
        features_matrix: np.ndarray, 
        elements: List[TextElement]
    ) -> None:
        """Fill in features that depend on neighboring elements."""
        lines_after_idx = self.feature_names.index('lines_after')
        is_last_idx = self.feature_names.index('is_last_on_page')
        
        for i, element in enumerate(elements):
            # Find next element on same page
            next_element = None
            for j in range(i + 1, len(elements)):
                if elements[j].page_number == element.page_number:
                    next_element = elements[j]
                    break
            
            if next_element:
                spacing = next_element.bbox.y0 - element.bbox.y1
                features_matrix[i, lines_after_idx] = spacing
            else:
                # Last element on page
                features_matrix[i, is_last_idx] = 1.0


class PatternLearner:
    """
    Self-learning classification system for document patterns.
    
    Combines unsupervised clustering for pattern discovery with
    supervised learning for classification.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        random_state: int = 42,
        clustering_method: str = 'dbscan',
        use_xgboost: bool = True
    ):
        """
        Initialize pattern learner.
        
        Args:
            n_clusters: Number of clusters for hierarchical clustering
            random_state: Random state for reproducibility
            clustering_method: 'dbscan' or 'hierarchical'
            use_xgboost: Whether to use XGBoost (if available)
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clustering_method = clustering_method
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        
        # Components
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        
        # Models
        self.clusterer = None
        self.classifier = None
        self.rf_classifier = None
        self.xgb_classifier = None
        
        # Pattern database
        self.patterns = {}
        self.pattern_examples = defaultdict(list)
        self.label_encoder = {}
        self.inverse_label_encoder = {}
        
        # Training data for incremental learning
        self.X_train = None
        self.y_train = None
        
        # Metadata
        self.version = "1.0.0"
        self.last_trained = None
        self.n_samples_trained = 0
        
        logger.info(
            f"Initialized PatternLearner with {clustering_method} clustering, "
            f"XGBoost: {self.use_xgboost}"
        )
    
    def discover_patterns(
        self, 
        elements: List[TextElement],
        eps: float = 0.5,
        min_samples: int = 3
    ) -> np.ndarray:
        """
        Discover patterns using unsupervised clustering.
        
        Args:
            elements: List of TextElements
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Cluster labels for each element
        """
        logger.info(f"Discovering patterns from {len(elements)} elements")
        
        # Extract features
        X = self.feature_extractor.extract_batch_features(elements)
        
        if len(X) == 0:
            logger.warning("No features extracted")
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        if self.clustering_method == 'dbscan':
            self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = self.clusterer.fit_predict(X_scaled)
        else:  # hierarchical
            self.clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            labels = self.clusterer.fit_predict(X_scaled)
        
        # Calculate cluster quality
        unique_labels = set(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            score = silhouette_score(X_scaled, labels)
            logger.info(
                f"Discovered {len(unique_labels)} patterns, "
                f"silhouette score: {score:.3f}"
            )
        else:
            logger.info(f"Discovered {len(unique_labels)} patterns")
        
        # Store pattern examples
        for i, label in enumerate(labels):
            if label != -1:  # Skip noise points in DBSCAN
                self.pattern_examples[int(label)].append({
                    'text': elements[i].text[:100],
                    'features': X[i].tolist(),
                    'element_type': elements[i].element_type
                })
        
        return labels
    
    def fit(
        self, 
        elements: List[TextElement], 
        labels: Optional[List[str]] = None
    ) -> 'PatternLearner':
        """
        Train the classifier on labeled data.
        
        Args:
            elements: List of TextElements
            labels: List of labels (strings). If None, uses element.element_type
            
        Returns:
            Self for chaining
        """
        logger.info(f"Training on {len(elements)} elements")
        
        # Extract features
        X = self.feature_extractor.extract_batch_features(elements)
        
        if len(X) == 0:
            raise ValueError("No features extracted from elements")
        
        # Get labels
        if labels is None:
            labels = [e.element_type for e in elements]
        
        # Encode labels
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.inverse_label_encoder = {i: label for label, i in self.label_encoder.items()}
        
        y = np.array([self.label_encoder[label] for label in labels])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store for incremental learning
        self.X_train = X_scaled
        self.y_train = y
        
        # Train Random Forest
        logger.info("Training Random Forest classifier")
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf_classifier.fit(X_scaled, y)
        
        # Train XGBoost if available
        if self.use_xgboost:
            logger.info("Training XGBoost classifier")
            self.xgb_classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.xgb_classifier.fit(X_scaled, y)
            
            # Create voting ensemble
            self.classifier = VotingClassifier(
                estimators=[
                    ('rf', self.rf_classifier),
                    ('xgb', self.xgb_classifier)
                ],
                voting='soft'
            )
            self.classifier.fit(X_scaled, y)
        else:
            self.classifier = self.rf_classifier
        
        # Update metadata
        self.last_trained = datetime.now().isoformat()
        self.n_samples_trained = len(elements)
        
        # Build pattern database
        self._build_pattern_database(X_scaled, y, elements)
        
        logger.info(
            f"Training complete. Learned {len(self.label_encoder)} pattern types"
        )
        
        return self
    
    def predict(
        self, 
        element: TextElement,
        return_confidence: bool = True
    ) -> Tuple[str, float]:
        """
        Predict the pattern type of an element.
        
        Args:
            element: TextElement to classify
            return_confidence: Whether to return confidence score
            
        Returns:
            Tuple of (predicted_label, confidence) if return_confidence=True,
            otherwise just predicted_label
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract features
        X = self.feature_extractor.extract_features(element).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.classifier.predict(X_scaled)[0]
        predicted_label = self.inverse_label_encoder[y_pred]
        
        if return_confidence:
            # Get probability estimates
            if hasattr(self.classifier, 'predict_proba'):
                probas = self.classifier.predict_proba(X_scaled)[0]
                confidence = float(probas[y_pred])
            else:
                confidence = 1.0
            
            return predicted_label, confidence
        
        return predicted_label
    
    def predict_batch(
        self, 
        elements: List[TextElement]
    ) -> List[Tuple[str, float]]:
        """
        Predict pattern types for multiple elements.
        
        Args:
            elements: List of TextElements
            
        Returns:
            List of (predicted_label, confidence) tuples
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract features
        X = self.feature_extractor.extract_batch_features(elements)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.classifier.predict(X_scaled)
        
        # Get probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probas = self.classifier.predict_proba(X_scaled)
            confidences = [float(probas[i, y_pred[i]]) for i in range(len(y_pred))]
        else:
            confidences = [1.0] * len(y_pred)
        
        # Convert to labels
        results = [
            (self.inverse_label_encoder[y_pred[i]], confidences[i])
            for i in range(len(y_pred))
        ]
        
        return results
    
    def update(
        self, 
        element: TextElement, 
        true_label: str
    ) -> 'PatternLearner':
        """
        Update model with a new labeled example (incremental learning).
        
        Args:
            element: TextElement with correct label
            true_label: The true label for this element
            
        Returns:
            Self for chaining
        """
        logger.debug(f"Updating model with new example: {true_label}")
        
        # Extract features
        X = self.feature_extractor.extract_features(element).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Encode label
        if true_label not in self.label_encoder:
            # New label discovered
            new_id = len(self.label_encoder)
            self.label_encoder[true_label] = new_id
            self.inverse_label_encoder[new_id] = true_label
            logger.info(f"New pattern type discovered: {true_label}")
        
        y = np.array([self.label_encoder[true_label]])
        
        # Append to training data
        if self.X_train is not None:
            self.X_train = np.vstack([self.X_train, X_scaled])
            self.y_train = np.append(self.y_train, y)
        else:
            self.X_train = X_scaled
            self.y_train = y
        
        self.n_samples_trained += 1
        
        # Periodically retrain (every 100 samples)
        if self.n_samples_trained % 100 == 0:
            logger.info("Periodic retraining triggered")
            self._retrain()
        
        return self
    
    def match_pattern(
        self, 
        element: TextElement, 
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find similar patterns using cosine similarity.
        
        Args:
            element: TextElement to match
            top_k: Number of top matches to return
            
        Returns:
            List of (pattern_label, similarity_score) tuples
        """
        if not self.patterns:
            return []
        
        # Extract features
        X = self.feature_extractor.extract_features(element).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Calculate similarities to each pattern
        similarities = []
        for label, pattern_data in self.patterns.items():
            centroid = np.array(pattern_data['centroid']).reshape(1, -1)
            sim = cosine_similarity(X_scaled, centroid)[0, 0]
            similarities.append((label, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_patterns(self, filepath: str) -> None:
        """
        Save learned patterns to disk.
        
        Args:
            filepath: Path to save patterns (JSON for metadata, joblib for models)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata and patterns
        metadata = {
            'version': self.version,
            'last_trained': self.last_trained,
            'n_samples_trained': self.n_samples_trained,
            'n_patterns': len(self.label_encoder),
            'label_encoder': self.label_encoder,
            'patterns': self.patterns,
            'feature_names': self.feature_extractor.feature_names,
            'clustering_method': self.clustering_method,
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save models
        model_path = filepath.with_suffix('.joblib')
        joblib.dump({
            'classifier': self.classifier,
            'rf_classifier': self.rf_classifier,
            'xgb_classifier': self.xgb_classifier,
            'scaler': self.scaler,
            'X_train': self.X_train,
            'y_train': self.y_train,
        }, model_path)
        
        logger.info(f"Patterns saved to {filepath} and {model_path}")
    
    def load_patterns(self, filepath: str) -> 'PatternLearner':
        """
        Load learned patterns from disk.
        
        Args:
            filepath: Path to load patterns from
            
        Returns:
            Self for chaining
        """
        filepath = Path(filepath)
        
        # Load metadata
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        self.version = metadata['version']
        self.last_trained = metadata['last_trained']
        self.n_samples_trained = metadata['n_samples_trained']
        self.label_encoder = {k: int(v) for k, v in metadata['label_encoder'].items()}
        self.inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
        self.patterns = metadata['patterns']
        self.clustering_method = metadata.get('clustering_method', 'dbscan')
        
        # Load models
        model_path = filepath.with_suffix('.joblib')
        models = joblib.load(model_path)
        
        self.classifier = models['classifier']
        self.rf_classifier = models['rf_classifier']
        self.xgb_classifier = models.get('xgb_classifier')
        self.scaler = models['scaler']
        self.X_train = models.get('X_train')
        self.y_train = models.get('y_train')
        
        logger.info(
            f"Patterns loaded from {filepath}. "
            f"{len(self.label_encoder)} pattern types, "
            f"{self.n_samples_trained} samples"
        )
        
        return self
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importances
        """
        if self.rf_classifier is None:
            raise ValueError("Model not trained")
        
        importances = self.rf_classifier.feature_importances_
        feature_names = self.feature_extractor.feature_names
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)
    
    def _build_pattern_database(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        elements: List[TextElement]
    ) -> None:
        """Build pattern database with statistics and examples."""
        self.patterns = {}
        
        for label_id, label_name in self.inverse_label_encoder.items():
            # Get indices for this pattern
            indices = np.where(y == label_id)[0]
            
            if len(indices) == 0:
                continue
            
            # Calculate centroid
            centroid = X[indices].mean(axis=0)
            
            # Get feature statistics
            feature_stats = {}
            for i, feature_name in enumerate(self.feature_extractor.feature_names):
                values = X[indices, i]
                feature_stats[feature_name] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                }
            
            # Get example texts
            examples = [
                elements[idx].text[:100] 
                for idx in indices[:5]
            ]
            
            self.patterns[label_name] = {
                'centroid': centroid.tolist(),
                'n_samples': len(indices),
                'feature_stats': feature_stats,
                'examples': examples,
            }
    
    def _retrain(self) -> None:
        """Retrain model with accumulated data."""
        if self.X_train is None or len(self.X_train) == 0:
            return
        
        logger.info(f"Retraining with {len(self.X_train)} samples")
        
        # Retrain Random Forest
        self.rf_classifier.fit(self.X_train, self.y_train)
        
        # Retrain XGBoost if available
        if self.use_xgboost and self.xgb_classifier is not None:
            self.xgb_classifier.fit(self.X_train, self.y_train)
            
            # Update voting classifier
            self.classifier = VotingClassifier(
                estimators=[
                    ('rf', self.rf_classifier),
                    ('xgb', self.xgb_classifier)
                ],
                voting='soft'
            )
            self.classifier.fit(self.X_train, self.y_train)
        
        self.last_trained = datetime.now().isoformat()
    
    def visualize_patterns(self) -> Dict[str, Any]:
        """
        Generate pattern visualization data for debugging.
        
        Returns:
            Dictionary with pattern statistics and examples
        """
        viz_data = {
            'n_patterns': len(self.patterns),
            'n_samples': self.n_samples_trained,
            'patterns': {}
        }
        
        for label, pattern_data in self.patterns.items():
            viz_data['patterns'][label] = {
                'count': pattern_data['n_samples'],
                'examples': pattern_data['examples'],
                'top_features': self._get_top_discriminative_features(label, top_n=5)
            }
        
        return viz_data
    
    def _get_top_discriminative_features(
        self, 
        label: str, 
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get most discriminative features for a pattern."""
        if label not in self.patterns:
            return []
        
        pattern_stats = self.patterns[label]['feature_stats']
        
        # Calculate feature distinctiveness (variance from mean)
        distinctiveness = []
        for feature_name, stats in pattern_stats.items():
            # Higher std = more distinctive
            distinctiveness.append((feature_name, stats['std']))
        
        distinctiveness.sort(key=lambda x: x[1], reverse=True)
        return distinctiveness[:top_n]

