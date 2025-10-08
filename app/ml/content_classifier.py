"""
Comprehensive multi-label content classifier with ensemble approach.

This module classifies document content using:
- Rule-based classifier (high precision, fast)
- ML classifier (Random Forest on structured features)
- Deep learning classifier (transformer-based semantic understanding)
- Ensemble voting for final classification
"""
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import joblib

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, deep learning classifier disabled")

from app.models.text_element import TextElement
from app.ml.pattern_learner import FeatureExtractor


logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Content type taxonomy."""
    SECTION_TITLE = "section_title"
    DESCRIPTION = "description"
    METADATA = "metadata"
    QUESTION = "question"
    ANSWER = "answer"
    TABLE = "table"
    LIST_ITEM = "list_item"
    INSTRUCTION = "instruction"
    REQUIREMENT = "requirement"
    TERM_CONDITION = "term_condition"
    CODE_SNIPPET = "code_snippet"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    UNKNOWN = "unknown"


@dataclass
class ContentClassification:
    """Result of content classification."""
    content_types: List[ContentType]
    confidences: Dict[ContentType, float]
    primary_type: ContentType
    primary_confidence: float
    classifier_votes: Dict[str, ContentType] = field(default_factory=dict)
    explanation: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content_types': [str(ct) for ct in self.content_types],
            'confidences': {str(k): v for k, v in self.confidences.items()},
            'primary_type': str(self.primary_type),
            'primary_confidence': self.primary_confidence,
            'classifier_votes': {k: str(v) for k, v in self.classifier_votes.items()},
            'explanation': self.explanation,
        }


class RuleBasedClassifier:
    """Fast rule-based classifier with high precision."""
    
    # Instruction keywords
    INSTRUCTION_KEYWORDS = [
        'must', 'shall', 'should', 'required', 'mandatory',
        'obligatory', 'need to', 'have to', 'ensure that'
    ]
    
    # Requirement indicators
    REQUIREMENT_INDICATORS = [
        'requirement', 'specification', 'criteria', 'standard',
        'minimum', 'maximum', 'at least', 'no less than'
    ]
    
    # Legal/T&C keywords
    LEGAL_KEYWORDS = [
        'agreement', 'contract', 'clause', 'terms', 'conditions',
        'liability', 'warranty', 'guarantee', 'indemnify',
        'jurisdiction', 'hereby', 'whereas', 'notwithstanding'
    ]
    
    # Code indicators
    CODE_PATTERNS = [
        re.compile(r'(?:public|private|protected)\s+(?:static\s+)?(?:void|int|string|bool)'),
        re.compile(r'(?:def|function|class|import|from)\s+\w+'),
        re.compile(r'(?:if|for|while|return)\s*\('),
        re.compile(r'[{}\[\]();].*[{}\[\]();]'),  # Multiple brackets/braces
    ]
    
    def classify(
        self,
        element: TextElement,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[ContentType, float]:
        """
        Classify element using rules.
        
        Args:
            element: TextElement to classify
            context: Optional context (avg font size, etc.)
            
        Returns:
            Tuple of (ContentType, confidence)
        """
        text = element.text.strip()
        text_lower = text.lower()
        
        # Rule 1: Section titles (large, bold)
        if self._is_section_title(element, context):
            return ContentType.SECTION_TITLE, 0.95
        
        # Rule 2: Questions (explicit markers)
        if self._is_question(text):
            return ContentType.QUESTION, 0.90
        
        # Rule 3: List items (bullets, numbering)
        if self._is_list_item(text):
            return ContentType.LIST_ITEM, 0.90
        
        # Rule 4: Metadata (key-value pattern)
        if self._is_metadata(text):
            return ContentType.METADATA, 0.85
        
        # Rule 5: Code snippet
        if self._is_code(text):
            return ContentType.CODE_SNIPPET, 0.90
        
        # Rule 6: Quote (quotation marks)
        if self._is_quote(text):
            return ContentType.QUOTE, 0.85
        
        # Rule 7: Footnote (references, superscript numbers)
        if self._is_footnote(text, element):
            return ContentType.FOOTNOTE, 0.85
        
        # Rule 8: Instructions (must, shall, should)
        if self._is_instruction(text_lower):
            return ContentType.INSTRUCTION, 0.80
        
        # Rule 9: Requirements (specification language)
        if self._is_requirement(text_lower):
            return ContentType.REQUIREMENT, 0.75
        
        # Rule 10: Terms & Conditions (legal language)
        if self._is_term_condition(text_lower):
            return ContentType.TERM_CONDITION, 0.75
        
        # Rule 11: Answer (follows question, indented)
        if context and context.get('follows_question'):
            return ContentType.ANSWER, 0.70
        
        # Default: Description
        return ContentType.DESCRIPTION, 0.50
    
    def _is_section_title(
        self,
        element: TextElement,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if element is a section title."""
        # Large font
        if context and element.font_size:
            avg_font = context.get('avg_font_size', 12.0)
            if element.font_size > avg_font * 1.3:
                if element.is_bold or element.text.isupper():
                    return True
        
        # Bold + short + uppercase
        if element.is_bold and len(element.text) < 100 and element.text.isupper():
            return True
        
        return False
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question."""
        # Has question mark
        if '?' in text:
            return True
        
        # Starts with interrogative
        interrogatives = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
        first_word = text.split()[0].lower().strip('.,!?;:') if text.split() else ''
        if first_word in interrogatives:
            return True
        
        # Question numbering
        if re.match(r'^(?:Q|Question)\s*\d+', text, re.IGNORECASE):
            return True
        
        return False
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item."""
        # Bullets
        if text.startswith(('•', '◦', '▪', '■', '□', '○', '●')):
            return True
        
        # Dash/asterisk at start
        if re.match(r'^[-*]\s+', text):
            return True
        
        # Numbered list
        if re.match(r'^\d+[\.)]\s+', text):
            return True
        
        # Lettered list
        if re.match(r'^[a-z][\.)]\s+', text, re.IGNORECASE):
            return True
        
        return False
    
    def _is_metadata(self, text: str) -> bool:
        """Check if text is metadata."""
        # Key: Value pattern
        if re.match(r'^[A-Za-z\s]+:\s*.+', text):
            words_before_colon = text.split(':')[0].split()
            if len(words_before_colon) <= 4:  # Short key
                return True
        
        return False
    
    def _is_code(self, text: str) -> bool:
        """Check if text is code."""
        for pattern in self.CODE_PATTERNS:
            if pattern.search(text):
                return True
        
        # High density of special characters
        special_chars = sum(c in '{}[]();' for c in text)
        if len(text) > 20 and special_chars / len(text) > 0.1:
            return True
        
        return False
    
    def _is_quote(self, text: str) -> bool:
        """Check if text is a quote."""
        # Surrounded by quotes
        if (text.startswith(('"', '"', '«')) and text.endswith(('"', '"', '»'))):
            return True
        
        # Contains quote attribution
        if re.search(r'-\s*[A-Z][a-z]+\s+[A-Z][a-z]+$', text):
            return True
        
        return False
    
    def _is_footnote(self, text: str, element: TextElement) -> bool:
        """Check if text is a footnote."""
        # Starts with number/symbol
        if re.match(r'^[\d†‡§¶]+\s+', text):
            return True
        
        # Small font size
        if element.font_size and element.font_size < 9:
            return True
        
        # At bottom of page
        if element.bbox.y0 > 700:  # Near bottom
            if re.match(r'^\d+\s+', text):
                return True
        
        return False
    
    def _is_instruction(self, text: str) -> bool:
        """Check if text is an instruction."""
        return any(keyword in text for keyword in self.INSTRUCTION_KEYWORDS)
    
    def _is_requirement(self, text: str) -> bool:
        """Check if text is a requirement."""
        return any(indicator in text for indicator in self.REQUIREMENT_INDICATORS)
    
    def _is_term_condition(self, text: str) -> bool:
        """Check if text is a term/condition."""
        keyword_count = sum(1 for keyword in self.LEGAL_KEYWORDS if keyword in text)
        return keyword_count >= 2


class MLClassifier:
    """Machine learning classifier using Random Forest."""
    
    def __init__(self):
        """Initialize ML classifier."""
        self.feature_extractor = FeatureExtractor()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.scaler = None
        self.label_encoder = {}
        self.inverse_label_encoder = {}
        self.is_trained = False
    
    def train(
        self,
        elements: List[TextElement],
        labels: List[ContentType]
    ) -> None:
        """Train the classifier."""
        from sklearn.preprocessing import StandardScaler
        
        # Extract features
        X = self.feature_extractor.extract_batch_features(elements)
        
        # Encode labels
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.inverse_label_encoder = {i: label for label, i in self.label_encoder.items()}
        
        y = np.array([self.label_encoder[label] for label in labels])
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        logger.info("Training ML classifier")
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"ML classifier trained on {len(elements)} samples")
    
    def predict(
        self,
        element: TextElement
    ) -> Tuple[ContentType, float, Dict[ContentType, float]]:
        """
        Predict content type.
        
        Returns:
            Tuple of (predicted_type, confidence, all_probabilities)
        """
        if not self.is_trained:
            return ContentType.UNKNOWN, 0.0, {}
        
        # Extract features
        X = self.feature_extractor.extract_features(element).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        pred = self.classifier.predict(X_scaled)[0]
        probas = self.classifier.predict_proba(X_scaled)[0]
        
        # Get probabilities for all classes
        all_probas = {
            self.inverse_label_encoder[i]: float(probas[i])
            for i in range(len(probas))
        }
        
        predicted_type = self.inverse_label_encoder[pred]
        confidence = float(probas[pred])
        
        return predicted_type, confidence, all_probas


class DLClassifier:
    """Deep learning classifier using transformers."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """Initialize DL classifier."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, DL classifier disabled")
    
    def load_model(self) -> None:
        """Load pre-trained model."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(ContentType)
            )
            self.is_trained = True
            logger.info(f"Loaded DL model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load DL model: {e}")
    
    def predict(
        self,
        element: TextElement
    ) -> Tuple[ContentType, float, Dict[ContentType, float]]:
        """
        Predict content type using deep learning.
        
        Returns:
            Tuple of (predicted_type, confidence, all_probabilities)
        """
        if not self.is_trained or not TRANSFORMERS_AVAILABLE:
            return ContentType.UNKNOWN, 0.0, {}
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                element.text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probas = torch.softmax(logits, dim=1)[0].numpy()
            
            # Get top prediction
            pred_idx = int(np.argmax(probas))
            confidence = float(probas[pred_idx])
            
            # Map to content types (simplified - would need proper training)
            content_types = list(ContentType)
            predicted_type = content_types[pred_idx] if pred_idx < len(content_types) else ContentType.UNKNOWN
            
            # All probabilities
            all_probas = {
                content_types[i]: float(probas[i])
                for i in range(min(len(probas), len(content_types)))
            }
            
            return predicted_type, confidence, all_probas
            
        except Exception as e:
            logger.error(f"DL prediction failed: {e}")
            return ContentType.UNKNOWN, 0.0, {}


class ContentClassifier:
    """
    Comprehensive multi-label content classifier with ensemble approach.
    
    Combines rule-based, ML, and DL classifiers for robust classification.
    """
    
    def __init__(
        self,
        enable_ml: bool = True,
        enable_dl: bool = False,  # Disabled by default (requires training)
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize content classifier.
        
        Args:
            enable_ml: Enable ML classifier
            enable_dl: Enable DL classifier
            weights: Custom weights for ensemble (rule, ml, dl)
        """
        # Initialize classifiers
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = MLClassifier() if enable_ml else None
        self.dl_classifier = DLClassifier() if enable_dl else None
        
        # Ensemble weights
        self.weights = weights or {
            'rule': 0.40,
            'ml': 0.35,
            'dl': 0.25
        }
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.5
        
        # Custom content types
        self.custom_types = {}
        
        logger.info("ContentClassifier initialized")
    
    def classify(
        self,
        element: TextElement,
        context: Optional[Dict[str, Any]] = None,
        multi_label: bool = False
    ) -> ContentClassification:
        """
        Classify element content type.
        
        Args:
            element: TextElement to classify
            context: Optional context information
            multi_label: Whether to return multiple labels
            
        Returns:
            ContentClassification result
        """
        # Collect predictions from all classifiers
        predictions = {}
        all_probabilities = defaultdict(list)
        
        # 1. Rule-based classifier (always run)
        rule_type, rule_conf = self.rule_classifier.classify(element, context)
        predictions['rule'] = (rule_type, rule_conf)
        all_probabilities[rule_type].append(rule_conf * self.weights['rule'])
        
        # Fast path: if rule-based has high confidence, skip others
        if rule_conf > 0.9:
            return self._create_classification(
                predictions,
                all_probabilities,
                "Rule-based (high confidence)"
            )
        
        # 2. ML classifier
        if self.ml_classifier and self.ml_classifier.is_trained:
            try:
                ml_type, ml_conf, ml_probas = self.ml_classifier.predict(element)
                predictions['ml'] = (ml_type, ml_conf)
                
                # Add weighted probabilities
                for content_type, prob in ml_probas.items():
                    all_probabilities[content_type].append(prob * self.weights['ml'])
                    
            except Exception as e:
                logger.warning(f"ML classifier failed: {e}")
        
        # 3. DL classifier
        if self.dl_classifier and self.dl_classifier.is_trained:
            try:
                dl_type, dl_conf, dl_probas = self.dl_classifier.predict(element)
                predictions['dl'] = (dl_type, dl_conf)
                
                # Add weighted probabilities
                for content_type, prob in dl_probas.items():
                    all_probabilities[content_type].append(prob * self.weights['dl'])
                    
            except Exception as e:
                logger.warning(f"DL classifier failed: {e}")
        
        # Combine predictions
        return self._create_classification(
            predictions,
            all_probabilities,
            "Ensemble voting"
        )
    
    def _create_classification(
        self,
        predictions: Dict[str, Tuple[ContentType, float]],
        all_probabilities: Dict[ContentType, List[float]],
        explanation: str
    ) -> ContentClassification:
        """Create classification result from predictions."""
        # Aggregate probabilities
        aggregated_probs = {
            content_type: sum(probs)
            for content_type, probs in all_probabilities.items()
        }
        
        # Get primary prediction
        if aggregated_probs:
            primary_type = max(aggregated_probs, key=aggregated_probs.get)
            primary_confidence = aggregated_probs[primary_type]
        else:
            primary_type = ContentType.UNKNOWN
            primary_confidence = 0.0
        
        # Multi-label: get all types above threshold
        content_types = [
            ct for ct, conf in aggregated_probs.items()
            if conf >= self.medium_confidence_threshold
        ]
        
        if not content_types:
            content_types = [primary_type]
        
        # Extract classifier votes
        classifier_votes = {
            name: pred_type
            for name, (pred_type, _) in predictions.items()
        }
        
        return ContentClassification(
            content_types=content_types,
            confidences=aggregated_probs,
            primary_type=primary_type,
            primary_confidence=primary_confidence,
            classifier_votes=classifier_votes,
            explanation=explanation
        )
    
    def train_ml_classifier(
        self,
        elements: List[TextElement],
        labels: List[ContentType]
    ) -> None:
        """Train the ML classifier."""
        if self.ml_classifier:
            self.ml_classifier.train(elements, labels)
    
    def evaluate(
        self,
        elements: List[TextElement],
        true_labels: List[ContentType],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate classifier performance.
        
        Args:
            elements: Test elements
            true_labels: True labels
            cv_folds: Cross-validation folds
            
        Returns:
            Evaluation metrics
        """
        if not self.ml_classifier or not self.ml_classifier.is_trained:
            logger.warning("ML classifier not trained, cannot evaluate")
            return {}
        
        # Extract features
        X = self.ml_classifier.feature_extractor.extract_batch_features(elements)
        X_scaled = self.ml_classifier.scaler.transform(X)
        
        # Encode labels
        y = np.array([
            self.ml_classifier.label_encoder[label]
            for label in true_labels
        ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.ml_classifier.classifier,
            X_scaled,
            y,
            cv=cv,
            scoring='accuracy'
        )
        
        # Predictions
        y_pred = self.ml_classifier.classifier.predict(X_scaled)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Classification report
        report = classification_report(
            y,
            y_pred,
            target_names=list(self.ml_classifier.label_encoder.keys()),
            output_dict=True
        )
        
        return {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': float(cv_scores.mean()),
            'std_cv_score': float(cv_scores.std()),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
        }
    
    def save(self, filepath: str) -> None:
        """Save classifier to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ML classifier
        if self.ml_classifier:
            model_data = {
                'classifier': self.ml_classifier.classifier,
                'scaler': self.ml_classifier.scaler,
                'label_encoder': self.ml_classifier.label_encoder,
                'is_trained': self.ml_classifier.is_trained,
            }
            joblib.dump(model_data, filepath.with_suffix('.joblib'))
        
        # Save config
        config = {
            'weights': self.weights,
            'custom_types': self.custom_types,
            'thresholds': {
                'high': self.high_confidence_threshold,
                'medium': self.medium_confidence_threshold,
            }
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Classifier saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load classifier from disk."""
        filepath = Path(filepath)
        
        # Load ML classifier
        if self.ml_classifier:
            try:
                model_data = joblib.load(filepath.with_suffix('.joblib'))
                self.ml_classifier.classifier = model_data['classifier']
                self.ml_classifier.scaler = model_data['scaler']
                self.ml_classifier.label_encoder = model_data['label_encoder']
                self.ml_classifier.inverse_label_encoder = {
                    v: k for k, v in model_data['label_encoder'].items()
                }
                self.ml_classifier.is_trained = model_data['is_trained']
                logger.info("ML classifier loaded")
            except Exception as e:
                logger.error(f"Failed to load ML classifier: {e}")
        
        # Load config
        try:
            with open(filepath.with_suffix('.json'), 'r') as f:
                config = json.load(f)
            
            self.weights = config.get('weights', self.weights)
            self.custom_types = config.get('custom_types', {})
            thresholds = config.get('thresholds', {})
            self.high_confidence_threshold = thresholds.get('high', 0.8)
            self.medium_confidence_threshold = thresholds.get('medium', 0.5)
            
            logger.info("Classifier config loaded")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def add_custom_type(
        self,
        type_name: str,
        rules: Dict[str, Any]
    ) -> None:
        """Add custom content type."""
        self.custom_types[type_name] = rules
        logger.info(f"Added custom content type: {type_name}")
    
    def classify_batch(
        self,
        elements: List[TextElement],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ContentClassification]:
        """Classify multiple elements efficiently."""
        return [self.classify(elem, context) for elem in elements]

