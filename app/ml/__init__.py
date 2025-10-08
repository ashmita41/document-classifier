"""
Machine learning components for document classification.
"""
from app.ml.pattern_learner import PatternLearner, FeatureExtractor
from app.ml.section_detector import SectionDetector, Section
from app.ml.metadata_extractor import MetadataExtractor, MetadataValue
from app.ml.qa_detector import QADetector, QAPair, QuestionType
from app.ml.content_classifier import ContentClassifier, ContentType, ContentClassification
from app.ml.adaptive_learner import AdaptiveLearner, FeatureStore, Correction, ModelVersion, PerformanceMetrics

__all__ = [
    "PatternLearner",
    "FeatureExtractor",
    "SectionDetector",
    "Section",
    "MetadataExtractor",
    "MetadataValue",
    "QADetector",
    "QAPair",
    "QuestionType",
    "ContentClassifier",
    "ContentType",
    "ContentClassification",
    "AdaptiveLearner",
    "FeatureStore",
    "Correction",
    "ModelVersion",
    "PerformanceMetrics",
]

