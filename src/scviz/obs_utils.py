import re
import datetime
from pathlib import Path
from typing import List, Literal, Dict
import pandas as pd
from collections import defaultdict

def classify_subtokens(token, used_labels=None, keyword_map=None):
    """
    Classify a token into one or more metadata categories based on keyword matching and pattern rules.

    This function parses a token into subtokens (e.g., by splitting on digit/letter boundaries),
    and attempts to classify each subtoken using:
    - Regex patterns (e.g., for dates and well positions)
    - Fuzzy substring matching against a user-defined keyword map

    Parameters
    ----------
    token : str
        The input string token to classify (e.g., "Aur60minDIA").
    used_labels : set, optional
        A set of already-assigned labels to avoid duplicating suggestions. Currently unused but reserved for future logic.
    keyword_map : dict, optional
        A dictionary where keys are metadata categories (e.g. 'gradient') and values are lists of substrings to match.
        If None, a default keyword map will be used.

    Returns
    -------
    labels : list of str
        A list of matched metadata labels for the token (e.g., ['gradient', 'acquisition']).
        If no match is found, returns ['unknown??'].
    """

    default_map = {
        "gradient": ["min", "hr", "gradient", "short", "long", "fast", "slow"],
        "amount": ["cell", "cells", "sc", "bulk", "ng", "ug", "pg", "fmol"],
        "enzyme": ["trypsin", "lysC", "chymotrypsin", "gluc", "tryp", "lys-c", "glu-c"],
        "condition": ["ctrl", "stim", "wt", "ko", "kd", "scramble", "si", "drug"],
        "sample_type": ["embryo", "brain", "liver", "cellline", "mix", "qc"],
        "instrument": ["tims", "tof", "fusion", "exploris","astral","stellar","eclipse","OA","OE480","OE","QE","qexecutive","OTE"],
        "acquisition": ["dia", "prm", "dda", "srm"],
        "column": ['TS25','TS15','TS8','Aur']
    }

    keyword_map = keyword_map or default_map
    labels = set()

    # Split into subtokens (case preserved), in case one token has multiple labels
    subtokens = re.findall(r'[A-Za-z]+|\d+min|\d+(?:ng|ug|pg|fmol)|\d{6,8}', token)

    for sub in subtokens:
        # Check unmodified for regex-based rules
        if is_date_like(sub):
            labels.add("date")
        elif re.match(r"[A-Ha-h]\d{1,2}$", sub):
            labels.add("well_position")
        else:
            # Lowercase for keyword matches
            sub_lower = sub.lower()
            for label, keywords in keyword_map.items():
                if any(kw in sub_lower for kw in keywords):
                    labels.add(label)

    if not labels:
        labels.add("unknown??")
    return list(labels)

def is_date_like(sub):
    patterns = [
        ("%Y%m%d", r"20\d{6}$"),           # 20240913
        ("%y%m%d", r"\d{6}$"),             # 250913
        ("%d%b", r"\d{1,2}[A-Za-z]{3}$"),  # 13Aug
        ("%b%d", r"[A-Za-z]{3}\d{1,2}$"),  # Aug13
    ]
    for fmt, pat in patterns:
        if re.fullmatch(pat, sub):
            try:
                datetime.datetime.strptime(sub, fmt)
                return True
            except ValueError:
                continue
    return False

def get_filenames(source: str, source_type: Literal["diann", "pd"]) -> List[str]:
    """
    Extract the list of sample filenames from a DIA-NN or Proteome Discoverer report file.

    Parameters
    ----------
    source : str
        Path to the input file.
    source_type : {'diann', 'pd'}
        Source tool type.

    Returns
    -------
    filenames : list of str
        List of sample names (Run names for DIA-NN or column-based for PD).
    """
    source = Path(source)
    ext = source.suffix.lower()

    # --- DIA-NN ---
    if source_type == "diann":
        if ext in [".csv", ".tsv"]:
            df = pd.read_csv(source, sep="\t" if ext == ".tsv" else ",", usecols=["Run"], low_memory=False)
        elif ext == ".parquet":
            df = pd.read_parquet(source, columns=["Run"], engine="pyarrow")
        else:
            raise ValueError(f"Unsupported file type for DIA-NN: {ext}")

        filenames = df["Run"].dropna().unique().tolist()

    # --- Proteome Discoverer ---
    elif source_type == "pd":
        if ext in [".txt", ".tsv"]:
            df = pd.read_csv(source, sep="\t", nrows=0)
        elif ext == ".xlsx":
            df = pd.read_excel(source, nrows=0)
        else:
            raise ValueError(f"Unsupported file type for PD: {ext}")

        abundance_cols = [col for col in df.columns if re.search(r"Abundance: F\d+: ", col)]
        if not abundance_cols:
            raise ValueError("No 'Abundance: F#:' columns found in PD file.")

        filenames = []
        for col in abundance_cols:
            match = re.match(r"Abundance: F\d+: (.+)", col)
            if match:
                filenames.append(match.group(1).strip())

    else:
        raise ValueError("source_type must be 'pd' or 'diann'")

    return filenames

def analyze_filename_formats(filenames, delimiter="_", group_labels=None):
    """
    Analyze filename structures to detect format consistency.

    Parameters
    ----------
    filenames : list of str
        List of sample or file names.
    delimiter : str
        Delimiter used to split tokens.
    group_labels : list of str, optional
        If provided, maps group indices to labels like ['A', 'B'].

    Returns
    -------
    format_info : dict
        {
            'uniform': bool,
            'n_tokens': list of int,
            'group_map': dict of filename → group label
        }
    """
    group_counts = defaultdict(list)
    for fname in filenames:
        tokens = fname.split(delimiter)
        group_counts[len(tokens)].append(fname)

    token_lengths = list(group_counts.keys())
    uniform = len(token_lengths) == 1

    if group_labels is None:
        group_labels = [f"{n}-tokens" for n in token_lengths]

    group_map = {}
    for label, n_tok in zip(group_labels, token_lengths):
        for fname in group_counts[n_tok]:
            group_map[fname] = label

    return {
        "uniform": uniform,
        "n_tokens": token_lengths,
        "group_map": group_map
    }