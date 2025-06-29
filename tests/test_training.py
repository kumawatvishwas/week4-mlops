from src.train import load_data, train_model, evaluate_model
from sklearn.model_selection import train_test_split

def test_model_training():
    X, y = load_data("data/iris.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict"), "Model has no predict method!"

def test_model_accuracy():
    X, y = load_data("data/iris.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    assert acc > 0.5, f"Model accuracy too low: {acc}"

