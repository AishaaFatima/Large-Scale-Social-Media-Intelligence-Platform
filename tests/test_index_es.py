import numpy as np
import pandas as pd
from src.index_es import df_to_actions

def test_actions_convert_nan_dates_and_ids():
    df = pd.DataFrame({"id": [1, 2], "created": pd.to_datetime(["2009-04-06 22:19:45", "2010-01-01 00:00:00"]),
                       "pos_rate": [np.nan, 0.5]})
    a = list(df_to_actions(df, "idx", "id"))
    assert a[0]["_id"] == "1" and a[0]["_index"] == "idx"
    assert a[0]["_source"]["pos_rate"] is None and a[1]["_source"]["pos_rate"] == 0.5
    assert a[0]["_source"]["created"] == "2009-04-06T22:19:45"
