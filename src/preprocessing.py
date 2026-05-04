"""
Preprocessing pipeline.

We use a single ColumnTransformer so the same preprocessor can be:
  - fit on training data,
  - applied identically to the test set,
  - pickled and loaded by the Streamlit app for inference,
  - re-applied to a single-row DataFrame from the app's input form.

handle_unknown='ignore' on OneHotEncoder is important: the app might
receive a category not seen at training time (rare but possible).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES

def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer that turns raw features into a model-ready matrix."""
    numerical_cols = NUMERICAL_FEATURES + ENGINEERED_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the post-transform column names. Useful for SHAP plots etc."""
    return list(preprocessor.get_feature_names_out())
