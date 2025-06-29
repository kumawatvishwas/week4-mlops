import pandas as pd

def test_data_loading():
    df = pd.read_csv("data/iris.csv")
    # Check basic shape
    assert df.shape[0] > 0, "Dataset is empty!"
    assert all(col in df.columns for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']), "Missing columns!"

