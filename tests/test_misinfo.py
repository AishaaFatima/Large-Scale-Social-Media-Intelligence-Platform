import pandas as pd
from src.misinfo import make_pipe

def test_pipeline_fits_and_predicts():
    X = pd.Series(["the economy collapsed totally", "unemployment fell last year per the BLS"] * 10)
    y = pd.Series(["fake", "real"] * 10)
    p = make_pipe(1).fit(X, y)
    assert set(p.predict(X)) <= {"fake", "real"}
