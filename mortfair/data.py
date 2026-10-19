"""HMDA download from the FFIEC Data Browser API, the cross-check against the API's own aggregations, and the
population filter.

The Data Browser accepts at most two filter criteria per query, so the pipeline downloads every 2025 Arizona
home-purchase record (all actions, all loan types) and applies the rest of the population filter locally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

API = "https://ffiec.cfpb.gov/v2/data-browser-api/view"
STATE, YEAR, LOAN_PURPOSE = "AZ", 2025, "1"  # 1 = home purchase
# sha256 of the file served on first download (2026-10-10; Last-Modified 21 Jul 2026). A different
# hash means the CFPB re-published the snapshot; the pipeline records the hash it used in run_metadata.json.
EXPECTED_SHA256 = "59e9de6843d9d90dc51d1d05484610fcb5928315aaaad0d79c795974e80274dd"
EXEMPT_TOKENS = ("Exempt", "1111")

# One aggregation query per dimension: (API parameter, CSV column, values). Each is combined with the download's own
# loan_purposes=1 filter, so every count and loan-amount sum can be recomputed from the downloaded file.
AGG_DIMENSIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("actions_taken", "action_taken", tuple(str(i) for i in range(1, 9))),
    ("loan_types", "loan_type", ("1", "2", "3", "4")),
    ("lien_statuses", "lien_status", ("1", "2")),
    ("construction_methods", "construction_method", ("1", "2")),
    ("races", "derived_race", ("White", "Asian", "Black or African American", "American Indian or Alaska Native",
                               "Native Hawaiian or Other Pacific Islander", "2 or more minority races", "Joint",
                               "Free Form Text Only", "Race Not Available")),
    ("ethnicities", "derived_ethnicity", ("Hispanic or Latino", "Not Hispanic or Latino", "Joint",
                                          "Ethnicity Not Available", "Free Form Text Only")),
    ("sexes", "derived_sex", ("Male", "Female", "Joint", "Sex Not Available")),
)


@dataclass(frozen=True)
class Download:
    """Where the raw file came from and what was received."""

    path: Path
    url: str
    final_url: str
    sha256: str
    n_bytes: int
    last_modified: str


def csv_url(state: str = STATE, year: int = YEAR, loan_purpose: str = LOAN_PURPOSE) -> str:
    """Data Browser CSV query for one state, one year and one loan purpose."""
    return f"{API}/csv?states={state}&years={year}&loan_purposes={loan_purpose}"


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(raw_dir: Path, state: str = STATE, year: int = YEAR, loan_purpose: str = LOAN_PURPOSE,
             timeout: float = 300.0) -> Download:
    """Download the CSV once (the API redirects to a static snapshot file); later calls reuse the local copy."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"hmda_{year}_{state}_purpose{loan_purpose}.csv"
    meta_path = path.with_suffix(".json")
    url = csv_url(state, year, loan_purpose)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return Download(path=path, **{k: meta[k] for k in ("url", "final_url", "sha256", "n_bytes", "last_modified")})
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        final_url, last_modified = r.url, r.headers.get("Last-Modified", "")
    with open(tmp, encoding="utf-8") as f:
        if not f.readline().startswith("activity_year,"):
            raise RuntimeError(f"{url} did not return a HMDA CSV (see {tmp})")
    tmp.replace(path)
    d = Download(path=path, url=url, final_url=final_url, sha256=sha256_of(path), n_bytes=path.stat().st_size,
                 last_modified=last_modified)
    meta_path.write_text(json.dumps({k: str(v) for k, v in d.__dict__.items() if k != "path"}, indent=1))
    return d


def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV with every column as text, so codes such as 'NA', 'Exempt' and '1111' stay explicit."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], low_memory=False)


# ------------------------------------------------------------------------------------------- official cross-check
def aggregation_url(param: str, values: tuple[str, ...], state: str = STATE, year: int = YEAR,
                    loan_purpose: str = LOAN_PURPOSE) -> str:
    """Data Browser aggregation query: counts and loan-amount sums by one dimension within the download's filter."""
    vals = ",".join(quote(v, safe="") for v in values)
    return f"{API}/aggregations?states={state}&years={year}&loan_purposes={loan_purpose}&{param}={vals}"


def fetch_aggregations(cache_file: Path, timeout: float = 60.0) -> dict:
    """Fetch one aggregation response per dimension (cached as JSON so the check can be rerun offline)."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    out = {}
    for param, _, values in AGG_DIMENSIONS:
        r = requests.get(aggregation_url(param, values), timeout=timeout)
        r.raise_for_status()
        out[param] = r.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out, indent=1))
    return out


def crosscheck(raw: pd.DataFrame, aggregations: dict) -> pd.DataFrame:
    """Compare record counts and loan-amount sums per category with the API's aggregations."""
    amount = pd.to_numeric(raw["loan_amount"], errors="coerce")
    rows = []
    for param, column, _ in AGG_DIMENSIONS:
        local = raw.assign(_amt=amount).groupby(column)["_amt"].agg(["size", "sum"])
        for a in aggregations[param]["aggregations"]:
            value = a[param]
            n_loc = int(local["size"].get(value, 0))
            s_loc = float(local["sum"].get(value, 0.0))
            rows.append({"dimension": column, "value": value, "api_count": int(a["count"]), "local_count": n_loc,
                         "api_loan_amount_sum": float(a["sum"]), "local_loan_amount_sum": s_loc,
                         "count_match": int(a["count"]) == n_loc,
                         # the API returns sums as single-precision-like floats (e.g. 4.337235E9)
                         "sum_rel_diff": abs(float(a["sum"]) - s_loc) / max(s_loc, 1.0)})
    return pd.DataFrame(rows).sort_values(["dimension", "local_count"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- population filter
FILTER_STEPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Originated (1) or denied (3)", "action_taken", ("1", "3")),
    ("Conventional loan", "loan_type", ("1",)),
    ("First lien", "lien_status", ("1",)),
    ("Principal residence", "occupancy_type", ("1",)),
    ("Site-built", "construction_method", ("1",)),
    ("Single unit", "total_units", ("1",)),
)
NOT_FLAGS = ("business_or_commercial_purpose", "reverse_mortgage", "open-end_line_of_credit")


def exempt_mask(df: pd.DataFrame) -> pd.Series:
    """Rows from reporters using the partial exemption (any field 'Exempt' or exemption code 1111)."""
    return df.isin(EXEMPT_TOKENS).any(axis=1)


def filter_population(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the population filter step by step; return the population and a table of rows left after each step."""
    df = raw
    log = [{"step": "All 2025 Arizona home-purchase records", "rows": len(df), "denied": int((df.action_taken == "3").sum())}]
    for label, col, keep in FILTER_STEPS:
        df = df[df[col].isin(keep)]
        log.append({"step": label, "rows": len(df), "denied": int((df.action_taken == "3").sum())})
    ex = exempt_mask(df)
    df = df[~ex]
    log.append({"step": "Not partially exempt (no 'Exempt' / 1111 codes)", "rows": len(df),
                "denied": int((df.action_taken == "3").sum())})
    for col in NOT_FLAGS:
        df = df[df[col] == "2"]
    log.append({"step": "Not business purpose, reverse mortgage or open-end line", "rows": len(df),
                "denied": int((df.action_taken == "3").sum())})
    out = pd.DataFrame(log)
    out["denial_rate"] = out.denied / out.rows
    return df.reset_index(drop=True), out
