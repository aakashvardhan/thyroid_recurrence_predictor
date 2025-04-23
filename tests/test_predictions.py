import pytest
from sklearn.metrics import f1_score

from recurrence_model.predict import make_prediction


def test_prediction_quality(test_data):
    """Test prediction quality with F1 score."""
    # Given
    X_test, y_test = test_data

    # When
    result = make_prediction(input_data=X_test)

    # Then
    predictions = result.get("predictions")

    # Check correct number of predictions
    assert len(predictions) == len(
        X_test
    ), f"Expected {len(X_test)} predictions, got {len(predictions)}"

    # Extract predicted labels for f1 score calculation
    pred_labels = [pred["label"] for pred in predictions]

    # Calculate f1 score
    f1 = f1_score(y_test, pred_labels)

    # Check f1 score meets threshold
    min_f1_threshold = 0.75
    assert f1 >= min_f1_threshold, f"F1 score {f1} below threshold {min_f1_threshold}"

    print(f"F1 score: {f1}")
