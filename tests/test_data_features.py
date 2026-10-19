"""Population filter, API cross-check, HMDA code handling and the feature encoder."""

import numpy as np
import pandas as pd

from mortfair import data as D
from mortfair.features import FEATURES, Encoder, application_table, parse_dti

PROTECTED = {"race", "ethnicity", "sex", "age", "age_62_plus", "tract_minority_pct"}


def test_dti_codes_and_na() -> None:
    s = pd.Series(["<20%", "20%-<30%", "30%-<36%", "36", "49", "50%-60%", ">60%", "NA", "Exempt"])
    out = parse_dti(s)
    assert out.iloc[:7].tolist() == [15.0, 25.0, 33.0, 36.0, 49.0, 55.0, 65.0]
    assert out.iloc[7:].isna().all()


def test_filter_removes_out_of_scope_and_exempt_rows(raw: pd.DataFrame) -> None:
    bad = raw.copy()
    bad.loc[0, "loan_type"] = "2"          # FHA
    bad.loc[1, "action_taken"] = "4"       # withdrawn
    bad.loc[2, "aus-1"] = "1111"           # partially exempt reporter
    bad.loc[3, "loan_to_value_ratio"] = "Exempt"
    bad.loc[4, "reverse_mortgage"] = "1"
    pop, steps = D.filter_population(bad)
    assert len(pop) == len(raw) - 5
    assert steps.rows.is_monotonic_decreasing and steps.rows.iloc[0] == len(raw)
    assert not D.exempt_mask(pop).any()


def test_crosscheck_counts_and_sums(raw: pd.DataFrame) -> None:
    agg = {}
    for param, col, _ in D.AGG_DIMENSIONS:
        g = raw.assign(a=raw.loan_amount.astype(float)).groupby(col).a.agg(["size", "sum"])
        agg[param] = {"aggregations": [{param: k, "count": int(r["size"]), "sum": float(r["sum"])} for k, r in g.iterrows()]}
    agg["races"]["aggregations"][0]["count"] += 1  # one deliberate mismatch
    chk = D.crosscheck(raw, agg)
    assert (~chk.count_match).sum() == 1 and chk.sum_rel_diff.max() == 0


def test_encoder_has_no_protected_inputs_and_no_nan(raw: pd.DataFrame) -> None:
    pop, _ = D.filter_population(raw)
    t = application_table(pop)
    assert not PROTECTED & set(FEATURES)
    enc = Encoder(top_lenders=2).fit(t.iloc[:300])
    X = enc.transform(t)
    assert list(X.columns) == FEATURES and not X.isna().any().any()
    na = t.dti.isna().to_numpy()
    assert na.any() and (X.dti_missing.to_numpy()[na] == 1).all() and (X.dti_missing.to_numpy()[~na] == 0).all()
    assert np.allclose(X.dti.to_numpy()[na], enc.medians["dti"])  # imputed with the training median
    assert X.lender.max() == 2  # third lender falls into 'other'
    # derived protected attributes are kept for the audit only
    assert set(t.race) <= {"White", "Black", "Asian", "Not available"} and set(t.age) >= {"Not available"}
