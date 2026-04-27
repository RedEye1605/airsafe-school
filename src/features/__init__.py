"""
Feature engineering sub-package.

Modules for school context features (elevation, road proximity,
land use, building density) and spatial aggregations.
"""

from src.features.school_features import compute_all_features, get_feature_columns
