from pathlib import Path
import datetime
import re
import warnings
from typing import List, Optional, Tuple, Dict, Any, Union
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

from scviz.utils import format_log_prefix

"""
Data import utilities for building `pAnnData` objects from supported proteomics tools.

This module provides functions to parse outputs from common tools such as Proteome Discoverer and DIA-NN,
automatically extracting protein and peptide quantification matrices, sample metadata, and relational mappings
between peptides and proteins.

Supported tools:
    - Proteome Discoverer (PD 1.3, PD 2.4, etc.)
    - DIA-NN

Functions:
    import_data: Main entry point that dispatches to the appropriate import function based on source_type.
    import_proteomeDiscoverer: Parses PD output files and initializes a pAnnData object.
    import_diann: Parses DIA-NN report file and initializes a pAnnData object.
    resolve_obs_columns: Extracts `.obs` column structure from filenames or metadata.
    suggest_obs_from_file: Suggests sample-level metadata based on consistent filename tokens.
    analyze_filename_formats: Analyzes filename structures to identify possible grouping patterns.
"""

def import_data(source_type: str, **kwargs):
    """
    Unified wrapper for importing data into a `pAnnData` object.

    This function routes to a specific import handler based on the `source_type`,
    such as Proteome Discoverer or DIA-NN. It parses protein/peptide expression data
    and associated sample metadata, returning a fully initialized `pAnnData` object.

    Args:
        source_type (str): The input tool or data source. Supported values:

            - `'pd'`, `'proteomeDiscoverer'`, `'pd13'`, `'pd24'`:  
              → Uses `import_proteomeDiscoverer()`.  
              Required kwargs:
                - `prot_file` (str): Path to protein-level report file
                - `obs_columns` (list of str): Columns to extract for `.obs`
              Optional kwargs:
                - `pep_file` (str): Path to peptide-level report file

            - `'diann'`, `'dia-nn'`:  
              → Uses `import_diann()`.  
              Required kwargs:
                - `report_file` (str): Path to DIA-NN report file
                - `obs_columns` (list of str): Columns to extract for `.obs`

            - `'fragpipe'`, `'fp'`: Not yet implemented  
            - `'spectronaut'`, `'sn'`: Not yet implemented

        **kwargs: Additional keyword arguments forwarded to the relevant import function.

    Returns:
        pAnnData: A populated pAnnData object with `.prot`, `.pep`, `.summary`, and identifier mappings.

    Example:
        Importing Proteome Discoverer output for single-cell data:

            >>> obs_columns = ['Sample', 'method', 'duration', 'cell_line']
            >>> pdata_untreated_sc = import_data(
            ...     source_type='pd',
            ...     prot_file='data/202312_untreated/Marion_20231218_OTE_Aur60min_CBR_prot_Proteins.txt',
            ...     pep_file='data/202312_untreated/Marion_20231218_OTE_Aur60min_CBR_pep_PeptideGroups.txt',
            ...     obs_columns=obs_columns
            ... )

        Importing PD output for bulk data from an Excel file:

            >>> obs_columns = ['Sample', 'cell_line']
            >>> pdata_bulk = import_data(
            ...     source_type='pd',
            ...     prot_file='HCT116 resistance_20230601_pdoutput.xlsx',
            ...     obs_columns=obs_columns
            ... )

    Note:
        If `obs_columns` is not provided and filename formats are inconsistent,
        fallback parsing is applied with generic columns (`"File"`, `"parsingType"`).
    """

    print(f"{format_log_prefix('user')} Importing data of type [{source_type}]")

    source_type = source_type.lower()
    obs_columns = kwargs.get('obs_columns', None)
    if obs_columns is None:
        source = kwargs.get('report_file') if 'report_file' in kwargs else kwargs.get('prot_file')
        format_info, fallback_columns, fallback_obs = resolve_obs_columns(source, source_type)

        if format_info["uniform"]:
            # Prompt user to rerun with obs_columns
            return None
        else:
            # non-uniform format, use fallback obs
            kwargs["obs_columns"] = fallback_columns
            kwargs["obs"] = fallback_obs

    if source_type in ['diann', 'dia-nn']:
        return _import_diann(**kwargs)

    elif source_type in ['pd', 'proteomediscoverer', 'proteome_discoverer', 'pd2.5', 'pd24']:
        return _import_proteomeDiscoverer(**kwargs)

    elif source_type in ['fragpipe', 'fp']:
        raise NotImplementedError("FragPipe import is not yet implemented. Stay tuned!")

    elif source_type in ['spectronaut', 'sn']:
        raise NotImplementedError("Spectronaut import is not yet implemented. Stay tuned!")

    else:
        raise ValueError(f"{format_log_prefix('error')} Unsupported import source: '{source_type}'. "
                         "Valid options: 'diann', 'proteomeDiscoverer', 'fragpipe', 'spectronaut'.")

def import_proteomeDiscoverer(prot_file: Optional[str] = None, pep_file: Optional[str] = None, obs_columns: Optional[List[str]] = ['sample'], **kwargs):
    """
    Import Proteome Discoverer (PD) output into a `pAnnData` object.

    This is a convenience wrapper for `import_data(source_type='pd')`. It loads protein- and optionally peptide-level
    expression data from PD report files and parses sample metadata columns.

    Args:
        prot_file (str): Path to the protein-level report file (required).
        pep_file (str, optional): Path to the peptide-level report file (optional but recommended).
        obs_columns (list of str): List of columns to extract for `.obs`. These should match metadata tokens
            embedded in the filenames (e.g. sample, condition, replicate).
        **kwargs: Additional keyword arguments passed to `import_data()`.

    Returns:
        pAnnData: A populated object with `.prot`, `.pep` (if provided), `.summary`, and identifier mappings.

    Example:
        To import data from Proteome Discoverer:

            >>> obs_columns = ['Sample', 'condition', 'cell_line']
            >>> pdata = import_proteomeDiscoverer(
            ...     prot_file='my_project/proteins.txt',
            ...     pep_file='my_project/peptides.txt',
            ...     obs_columns=obs_columns
            ... )

    Note:
        - If `pep_file` is omitted, the resulting `pAnnData` will not include `.pep` or an RS matrix.
        - If filename structure is inconsistent and `obs_columns` cannot be inferred, fallback columns are used.
    """
    return import_data(source_type='pd', prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)

def _import_proteomeDiscoverer(prot_file: Optional[str] = None, pep_file: Optional[str] = None, obs_columns: Optional[List[str]] = ['sample'], **kwargs):
    if not prot_file and not pep_file:
        raise ValueError(f"{format_log_prefix('error')} At least one of prot_file or pep_file must be provided to function. Try prot_file='proteome_discoverer_prot.txt' or pep_file='proteome_discoverer_pep.txt'.")
    print("--------------------------\nStarting import [Proteome Discoverer]\n--------------------------")

    if prot_file:
        # -----------------------------
        print(f"Source file: {prot_file} / {pep_file}")
        # PROTEIN DATA
        # check file format, if '.txt' then use read_csv, if '.xlsx' then use read_excel
        if prot_file.endswith('.txt') or prot_file.endswith('.tsv'):
            prot_all = pd.read_csv(prot_file, sep='\t')
        elif prot_file.endswith('.xlsx'):
            print("💡 Tip: The read_excel function is slower compared to reading .tsv or .txt files. For improved performance, consider converting your data to .tsv or .txt format.")
            prot_all = pd.read_excel(prot_file)
        # prot_X: sparse data matrix
        prot_X = sparse.csr_matrix(prot_all.filter(regex='Abundance: F', axis=1).values).transpose()
        # prot_layers['mbr']: protein MBR identification
        prot_layers_mbr = prot_all.filter(regex='Found in Sample', axis=1).values.transpose()
        # prot_var_names: protein names
        prot_var_names = prot_all['Accession'].values
        # prot_var: protein metadata
        prot_var = prot_all.loc[:, 'Protein FDR Confidence: Combined':'# Razor Peptides']
        prot_var.rename(columns={'Gene Symbol': 'Genes'}, inplace=True)
        # prot_obs_names: file names
        prot_obs_names = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract(r'Abundance: (F\d+):')[0].values
        # prot_obs: sample typing from the column name, drop column if all 'n/a'
        prot_obs = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract(r'Abundance: F\d+: (.+)$')[0].values
        prot_obs = pd.DataFrame(prot_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).applymap(str.strip).astype('category')
        if (prot_obs == "n/a").all().any():
            print(f"{format_log_prefix('warn')} Found columns with all 'n/a'. Dropping these columns.")
            prot_obs = prot_obs.loc[:, ~(prot_obs == "n/a").all()]

        print(f"Number of files: {len(prot_obs_names)}")
        print(f"Proteins: {len(prot_var)}")
    else:
        prot_X = prot_layers_mbr = prot_var_names = prot_var = prot_obs_names = prot_obs = None

    if pep_file:
        # -----------------------------
        # PEPTIDE DATA
        if pep_file.endswith('.txt') or pep_file.endswith('.tsv'):
            pep_all = pd.read_csv(pep_file, sep='\t')
        elif pep_file.endswith('.xlsx'):
            print("💡 Tip: The read_excel function is slower compared to reading .tsv or .txt files. For improved performance, consider converting your data to .tsv or .txt format.")
            pep_all = pd.read_excel(pep_file)
        # pep_X: sparse data matrix
        pep_X = sparse.csr_matrix(pep_all.filter(regex='Abundance: F', axis=1).values).transpose()
        # pep_layers['mbr']: peptide MBR identification
        pep_layers_mbr = pep_all.filter(regex='Found in Sample', axis=1).values.transpose()
        # pep_var_names: peptide sequence with modifications
        pep_var_names = (pep_all['Annotated Sequence'] + np.where(pep_all['Modifications'].isna(), '', ' MOD:' + pep_all['Modifications'])).values
        # pep_obs_names: file names
        pep_obs_names = pep_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: (F\d+):')[0].values
        # pep_var: peptide metadata
        pep_var = pep_all.loc[:, 'Modifications':'Theo. MH+ [Da]']
        # prot_obs: sample typing from the column name, drop column if all 'n/a'
        pep_obs = pep_all.filter(regex='Abundance: F', axis=1).columns.str.extract(r'Abundance: F\d+: (.+)$')[0].values
        pep_obs = pd.DataFrame(pep_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).applymap(str.strip).astype('category')
        if (pep_obs == "n/a").all().any():
            print(f"{format_log_prefix('warn')} Found columns with all 'n/a'. Dropping these columns.")
            pep_obs = pep_obs.loc[:, ~(pep_obs == "n/a").all()]

        print(f"Peptides: {len(pep_var)}")
    else:
        pep_X = pep_layers_mbr = pep_var_names = pep_var = pep_obs_names = pep_obs = None

    if prot_file and pep_file:
        # -----------------------------
        # RS DATA
        # rs is in the form of a binary matrix, protein x peptide
        pep_prot_list = pep_all['Master Protein Accessions'].str.split('; ')
        mlb = MultiLabelBinarizer()
        rs = mlb.fit_transform(pep_prot_list)
        if prot_var_names is not None:
            index_dict = {protein: index for index, protein in enumerate(mlb.classes_)}           
            reorder_indices = [index_dict[protein] for protein in prot_var_names]
            rs = rs[:, reorder_indices]
        # print("RS matrix successfully computed")
    else:
        rs = None

    # ASSERTIONS
    # -----------------------------
    # check if mlb.classes_ has overlap with prot_var
    if prot_file and pep_file:
        mlb_classes_set = set(mlb.classes_)
        prot_var_set = set(prot_var_names)

        if mlb_classes_set != prot_var_set:
            print(f"{format_log_prefix('warn')} Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
            print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
            print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
            print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")

    pdata = _create_pAnnData_from_parts(
        prot_X, pep_X, rs,
        prot_obs, prot_var, prot_obs_names, prot_var_names,
        pep_obs, pep_var, pep_obs_names, pep_var_names,
        obs_columns=obs_columns,
        X_mbr_prot=prot_layers_mbr,
        X_mbr_pep=pep_layers_mbr,
        metadata={
            "source": "proteomeDiscoverer",
            "prot_file": prot_file,
            "pep_file": pep_file
        },
        history_msg=f"Imported Proteome Discoverer data using source file(s): {prot_file}, {pep_file}."
    )

    return pdata

def import_diann(report_file: Optional[str] = None, obs_columns: Optional[List[str]] = None, prot_value: str = 'PG.MaxLFQ', pep_value: str = 'Precursor.Normalised', prot_var_columns: List[str] = ['Genes', 'Master.Protein'], pep_var_columns: List[str] = ['Genes', 'Protein.Group', 'Precursor.Charge', 'Modified.Sequence', 'Stripped.Sequence', 'Precursor.Id', 'All Mapped Proteins', 'All Mapped Genes'], **kwargs):
    """
    Import DIA-NN output into a `pAnnData` object.

    This function parses a DIA-NN report file and separates protein- and peptide-level expression matrices
    using the specified abundance and metadata columns.

    Args:
        report_file (str): Path to the DIA-NN report file (required).
        obs_columns (list of str): List of metadata columns to extract from the filename for `.obs`.
        prot_value (str): Column name in DIA-NN output to use for protein quantification.
            Default: `'PG.MaxLFQ'`.
        pep_value (str): Column name in DIA-NN output to use for peptide quantification.
            Default: `'Precursor.Normalised'`.
        prot_var_columns (list of str): Columns from the protein group table to store in `.prot.var`.
            Default includes gene and master protein annotations.
        pep_var_columns (list of str): Columns from the precursor table to store in `.pep.var`.
            Default includes peptide sequence, precursor ID, and mapping annotations.
        **kwargs: Additional keyword arguments passed to `import_data()`.

    Returns:
        pAnnData: A populated object with `.prot`, `.pep`, `.summary`, and identifier mappings.

    Example:
        To import data from a DIA-NN report file:

            >>> obs_columns = ['Sample', 'treatment', 'replicate']
            >>> pdata = import_diann(
            ...     report_file='data/project_diaNN_output.tsv',
            ...     obs_columns=obs_columns,
            ...     prot_value='PG.MaxLFQ',
            ...     pep_value='Precursor.Normalised'
            ... )

    Note:
        - DIA-NN report should contain both protein group and precursor-level information.
        - Metadata columns in filenames must be consistently formatted to extract `.obs`.
    """
    return import_data(source_type='diann', report_file=report_file, obs_columns=obs_columns, prot_value=prot_value, pep_value=pep_value, prot_var_columns=prot_var_columns, pep_var_columns=pep_var_columns, **kwargs)

def _import_diann(report_file: Optional[str] = None, obs_columns: Optional[List[str]] = None, obs: Optional[pd.DataFrame] = None, prot_value = 'PG.MaxLFQ', pep_value = 'Precursor.Normalised', prot_var_columns = ['Genes', 'Master.Protein'], pep_var_columns = ['Genes', 'Protein.Group', 'Precursor.Charge','Modified.Sequence', 'Stripped.Sequence', 'Precursor.Id', 'All Mapped Proteins', 'All Mapped Genes'], **kwargs):
    if not report_file:
        raise ValueError(f"{format_log_prefix('error')} Importing from DIA-NN: report.tsv or report.parquet must be provided to function. Try report_file='report.tsv' or report_file='report.parquet'")
    print("--------------------------\nStarting import [DIA-NN]\n--------------------------")

    print(f"Source file: {report_file}")
    # if csv, then use pd.read_csv, if parquet then use pd.read_parquet('example_pa.parquet', engine='pyarrow')
    if report_file.endswith('.tsv'):
        report_all = pd.read_csv(report_file, sep='\t')
    elif report_file.endswith('.parquet'):
        report_all = pd.read_parquet(report_file, engine='pyarrow')
    report_all['Master.Protein'] = report_all['Protein.Group'].str.split(';')
    report_all = report_all.explode('Master.Protein')
    # -----------------------------
    # PROTEIN DATA
    # prot_X: sparse data matrix
    if prot_value != 'PG.MaxLFQ':
        if report_file.endswith('.tsv') and prot_value == 'PG.Quantity':
            # check if 'PG.Quantity' is in the columns, if yes then pass, if not then throw an error that DIA-NN version >2.0 does not have PG.quantity
            if 'PG.Quantity' not in report_all.columns:
                raise ValueError("Reports generated with DIA-NN version >2.0 do not contain PG.Quantity values, please use PG.MaxLFQ .")
        else:
            print(f"{format_log_prefix('info')} Protein value specified is not PG.MaxLFQ, please check if correct.")
    prot_X_pivot = report_all.pivot_table(index='Master.Protein', columns='Run', values=prot_value, aggfunc='first', sort=False)
    prot_X = sparse.csr_matrix(prot_X_pivot.values).T
    # prot_var_names: protein names
    prot_var_names = prot_X_pivot.index.values
    # prot_obs: file names
    prot_obs_names = prot_X_pivot.columns.values

    # prot_var: protein metadata (default: Genes, Master.Protein)
    if 'First.Protein.Description' in report_all.columns:
        prot_var_columns.insert(0, 'First.Protein.Description')

    existing_prot_var_columns = [col for col in prot_var_columns if col in report_all.columns]
    missing_columns = set(prot_var_columns) - set(existing_prot_var_columns)

    if missing_columns:
        warnings.warn(
            f"{format_log_prefix('warn')} The following columns are missing: {', '.join(missing_columns)}. "
        )

    prot_var = report_all.loc[:, existing_prot_var_columns].drop_duplicates(subset='Master.Protein').drop(columns='Master.Protein')
    # prot_obs: sample typing from the column name
    if obs is not None:
        prot_obs = obs
        # obs_columns = obs_columns
    else:
        prot_obs = pd.DataFrame(prot_X_pivot.columns.values, columns=['Run'])['Run'].str.split('_', expand=True).rename(columns=dict(enumerate(obs_columns)))
    
    print(f"Number of files: {len(prot_obs_names)}")
    print(f"Proteins: {len(prot_var)}")

    # -----------------------------
    # PEPTIDE DATA
    # pep_X: sparse data matrix
    pep_X_pivot = report_all.pivot_table(index='Precursor.Id', columns='Run', values=pep_value, aggfunc='first', sort=False)
    pep_X = sparse.csr_matrix(pep_X_pivot.values).T
    # pep_var_names: peptide sequence
    pep_var_names = pep_X_pivot.index.values
    # pep_obs_names: file names
    pep_obs_names = pep_X_pivot.columns.values
    # pep_var: peptide sequence with modifications (default: Genes, Protein.Group, Precursor.Charge, Modified.Sequence, Stripped.Sequence, Precursor.Id, All Mapped Proteins, All Mapped Genes)
    existing_pep_var_columns = [col for col in pep_var_columns if col in report_all.columns]
    missing_columns = set(pep_var_columns) - set(existing_pep_var_columns)

    if missing_columns:
        warnings.warn(
            f"{format_log_prefix('warn')} The following columns are missing: {', '.join(missing_columns)}. "
            "Consider running analysis in the newer version of DIA-NN (1.8.1). "
            "Peptide-protein mapping may differ."
        )
    
    pep_var = report_all.loc[:, existing_pep_var_columns].drop_duplicates(subset='Precursor.Id').drop(columns='Precursor.Id')
    # pep_obs: sample typing from the column name, same as prot_obs
    pep_obs = prot_obs

    print(f"Peptides: {len(pep_var)}")

    # -----------------------------
    # RS DATA
    # rs: protein x peptide relational data
    pep_prot_list = report_all.drop_duplicates(subset=['Precursor.Id'])['Protein.Group'].str.split(';')
    mlb = MultiLabelBinarizer()
    rs = mlb.fit_transform(pep_prot_list)
    index_dict = {protein: index for index, protein in enumerate(mlb.classes_)}
    reorder_indices = [index_dict[protein] for protein in prot_var_names]
    rs = rs[:, reorder_indices]

    # -----------------------------
    # ASSERTIONS
    # -----------------------------
    # check if mlb.classes_ has overlap with prot_var
    mlb_classes_set = set(mlb.classes_)
    prot_var_set = set(prot_var_names)

    if mlb_classes_set != prot_var_set:
        print(f"{format_log_prefix('warn')} Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
        print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
        print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
        print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")
    
    pdata = _create_pAnnData_from_parts(
        prot_X, pep_X, rs,
        prot_obs, prot_var, prot_obs_names, prot_var_names,
        pep_obs, pep_var, pep_obs_names, pep_var_names,
        obs_columns=obs_columns,
        metadata={
            "source": "diann",
            "file": report_file,
            "protein_metric": prot_value,
            "peptide_metric": pep_value
        },
        history_msg=f"Imported DIA-NN report from {report_file} using {prot_value} (protein) and {pep_value} (peptide)."
    )

    return pdata

def _create_pAnnData_from_parts(
    prot_X, pep_X, rs,
    prot_obs, prot_var, prot_obs_names, prot_var_names,
    pep_obs=None, pep_var=None, pep_obs_names=None, pep_var_names=None,
    obs_columns=None,
    X_mbr_prot=None,
    X_mbr_pep=None,
    metadata=None,
    history_msg=""
):
    """
    Assemble a `pAnnData` object from processed matrices and metadata.

    This function is typically called internally by import functions. It constructs
    `.prot` and `.pep` AnnData objects, assigns optional metadata and MBR layers,
    adds identifier mappings and sample-level summary metrics, and returns a
    validated `pAnnData` object.

    Args:
        prot_X (csr_matrix): Protein-level expression matrix (samples × proteins).
        pep_X (csr_matrix or None): Peptide-level expression matrix (samples × peptides).
        rs (csr_matrix or None): Binary matrix linking proteins (rows) to peptides (columns).
        prot_obs (pd.DataFrame): Sample-level metadata for protein data.
        prot_var (pd.DataFrame): Feature-level metadata for proteins.
        prot_obs_names (list-like): Sample identifiers for `.prot`.
        prot_var_names (list-like): Protein accession identifiers for `.prot`.
        pep_obs (pd.DataFrame, optional): Sample metadata for `.pep`. If not provided, `.prot.obs` is reused.
        pep_var (pd.DataFrame, optional): Feature metadata for peptides.
        pep_obs_names (list-like, optional): Sample identifiers for `.pep`.
        pep_var_names (list-like, optional): Peptide identifiers.
        obs_columns (list of str, optional): Columns from filenames to include in `.summary` and `.obs`.
        X_mbr_prot (np.ndarray or DataFrame, optional): Optional protein-level MBR identification info.
        X_mbr_pep (np.ndarray or DataFrame, optional): Optional peptide-level MBR identification info.
        metadata (dict, optional): Optional dictionary of import metadata (e.g. `{'source': 'diann'}`).
        history_msg (str): Operation description to append to the history log.

    Returns:
        pAnnData: Initialized object with `.prot`, `.pep`, `.summary`, `.rs`, and other metadata filled in.

    Note:
        This is a low-level function. In most cases, users should call `import_data()`, `import_proteomeDiscoverer()`, or `import_diann()` instead.
    """
    from .pAnnData import pAnnData

    print("")
    pdata = pAnnData(prot_X, pep_X, rs)

    # --- PROTEIN ---
    if prot_X is not None:
        pdata.prot.obs = pd.DataFrame(prot_obs) # type: ignore[attr-defined]
        pdata.prot.var = pd.DataFrame(prot_var) # type: ignore[attr-defined]
        pdata.prot.obs_names = list(prot_obs_names) # type: ignore[attr-defined]
        pdata.prot.var_names = list(prot_var_names) # type: ignore[attr-defined]
        pdata.prot.obs.columns = obs_columns if obs_columns else list(range(pdata.prot.obs.shape[1])) # type: ignore[attr-defined]
        pdata.prot.layers['X_raw'] = prot_X # type: ignore[attr-defined]
        if X_mbr_prot is not None:
            pdata.prot.layers['X_mbr'] = X_mbr_prot # type: ignore[attr-defined]

    if "Genes" in pdata.prot.var.columns and pdata.prot.var["Genes"].isna().any(): # type: ignore[attr-defined]
        pdata.update_missing_genes(gene_col="Genes", verbose=True)

    # --- PEPTIDE ---
    if pep_X is not None:
        pdata.pep.obs = pd.DataFrame(pep_obs) # type: ignore[attr-defined]
        pdata.pep.var = pd.DataFrame(pep_var) # type: ignore[attr-defined]
        pdata.pep.obs_names = list(pep_obs_names) # type: ignore[attr-defined]
        pdata.pep.var_names = list(pep_var_names) # type: ignore[attr-defined]
        pdata.pep.obs.columns = obs_columns if obs_columns else list(range(pdata.pep.obs.shape[1])) # type: ignore[attr-defined]
        pdata.pep.layers['X_raw'] = pep_X # type: ignore[attr-defined]
        if X_mbr_pep is not None:
            pdata.pep.layers['X_mbr'] = X_mbr_pep # type: ignore[attr-defined]

    # --- Metadata ---
    metadata = metadata or {}
    metadata.setdefault("imported_at", datetime.datetime.now().isoformat())

    if pdata.prot is not None:
        pdata.prot.uns['metadata'] = metadata
    if pdata.pep is not None:
        pdata.pep.uns['metadata'] = metadata

    # --- Summary + Validation ---
    pdata.update_summary(recompute=True)
    pdata._annotate_found_samples(threshold=0.0)

    print("")
    if not pdata.validate():
        print(f"{format_log_prefix('warn')} Validation issues found. Use `pdata.validate()` to inspect.")

    if history_msg:
        pdata._append_history(history_msg)

    print("--------------------------")
    print(f"{format_log_prefix('result')} Import complete. Use `print(pdata)` to view the object.")

    return pdata

def suggest_obs_columns(source=None, source_type=None, filenames=None, delimiter=None):
    """
    Suggest `.obs` column names based on parsed sample names.

    This function analyzes filenames or run names extracted from Proteome Discoverer
    or DIA-NN reports and attempts to identify consistent metadata fields. These fields
    may include `gradient`, `amount`, `cell_line`, or `well_position`, depending on
    naming conventions and regular expression matches.

    Args:
        source (str or Path, optional): Path to a DIA-NN or PD output file.
        source_type (str, optional): Type of the input file. Supports `'diann'` or `'pd'`.
            If not provided, inferred from filename or fallback heuristics.
        filenames (list of str, optional): List of sample file names or run labels to parse.
            If provided, bypasses file loading.
        delimiter (str, optional): Delimiter to use for tokenizing filenames (e.g., `','`, `'_'`).
            If not specified, will be inferred automatically.

    Returns:
        list of str: Suggested list of metadata column names to assign to `.obs`.

    Example:
        To suggest observation columns from a file:

            >>> suggest_obs_columns("my_experiment_PD.txt", source_type="pd")

        Suggested columns: ['Sample', 'gradient', 'cell_line', 'duration']

            >>> ['Sample', 'gradient', 'cell_line', 'duration']

    Note:
        This function is typically used as part of the `.import_data()` flow
        when filenames embed experimental metadata.
    """
    from pathlib import Path
    import csv
    from collections import Counter

    if filenames is None:
        if source is None or source_type is None:
            raise ValueError("If `filenames` is not provided, both `source` and `source_type` must be specified.")
        source = Path(source)
        filenames = get_filenames(source, source_type=source_type)
    
    if not filenames:
        raise ValueError("No sample filenames could be extracted from the provided source.")

    # Pick the first filename for token analysis
    fname = filenames[0]

    # Infer delimiter if not provided
    if delimiter is None:
        all_delims = re.findall(r'[^A-Za-z0-9]', fname)
        delimiter = Counter(all_delims).most_common(1)[0][0] if all_delims else '_'
        print(f"Auto-detecting '{delimiter}' as delimiter.")

    if source_type == 'pd':
        # Custom comma-based parsing for PD
        match = re.match(r'Abundance: (F\d+): (.+)', f"Abundance: F1: {fname}")
        if match:
            _, meta = match.groups()
            raw_tokens = [t.strip() for t in meta.split(',') if t.strip().lower() != 'n/a']
            fname = meta
            tokens = raw_tokens
            delimiter = ','
        else:
            raise ValueError(f"Could not parse metadata from PD filename: {fname}")

    # --- Classify tokens ---
    tokens = fname.split(delimiter)
    suggestion = {}
    obs_columns = []
    token_label_map = []
    multi_matched_tokens = []
    unrecognized_tokens = []

    for tok in tokens:
        labels = classify_subtokens(tok)
        label = labels[0]
        if label == "unknown??":
            obs_columns.append(f"<{tok}?>")
        else:
            obs_columns.append(label)
        token_label_map.append((tok, labels))
        if label != "unknown??" and label not in suggestion:
            suggestion[label] = tok
        if "unknown??" in labels:
            unrecognized_tokens.append(tok)
        elif len(labels) > 1:
            multi_matched_tokens.append((labels, tok))

    # --- Print suggestions ---
    print(f"\nFrom filename: {fname}")
    print("Suggested .obs columns:")
    for tok, labels in token_label_map:
        print(f"  {' OR '.join(labels):<26}: {tok}")
    if multi_matched_tokens:
        print(f"\nMultiple matched token(s): {[t for _, t in multi_matched_tokens]}")
    if unrecognized_tokens:
        print(f"Unrecognized token(s): {unrecognized_tokens}")
    if multi_matched_tokens or unrecognized_tokens:
        print("Please manually label these.")

    print(f"\n{format_log_prefix('info_only')} Suggested obs:\nobs_columns = {obs_columns}")

    return obs_columns

def resolve_obs_columns(source: str, source_type: str, delimiter: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[List[str]], Optional[pd.DataFrame]]:
    """
    Resolve observation columns from sample filenames or metadata fields.

    This function attempts to infer sample-level metadata (`.obs`) from filenames
    or a report file (DIA-NN or Proteome Discoverer). It classifies tokens using 
    regex patterns and known metadata heuristics.

    Args:
        source (str): Path to the report file (DIA-NN or PD).
        source_type (str): Source type — one of {'diann', 'pd'}.
        delimiter (str, optional): Delimiter used to split filename tokens. If None, auto-inferred.

    Returns:
        Tuple[dict, list[str] or None, pd.DataFrame or None]: A tuple of:
        
        - **metadata** (dict): Metadata extracted during parsing, including fallback flags.
        - **suggested_obs** (list of str or None): Suggested observation column names, or None if inconsistent format.
        - **obs_df** (pd.DataFrame or None): Parsed observation DataFrame.

    Note:
        If filename formats are inconsistent across samples, the fallback `.obs` will include:
        - A generic 'File' column with raw filenames
        - A 'parsingType' column indicating parsing structure

    Example:
        Inferring observation columns from a PD file:

            >>> resolve_obs_columns('filepaths/pd_report.xlsx', source_type='pd')

        Inferring from a DIA-NN report with custom delimiter:

            >>> resolve_obs_columns('filepaths/diann.tsv', source_type='diann', delimiter='_')
    """
    
    filenames = get_filenames(source, source_type=source_type)
    if not filenames:
        raise ValueError(f"{format_log_prefix('error')} No sample filenames could be extracted from the provided source: {source}.")

    if delimiter is None:
        first_fname = filenames[0]
        all_delims = re.findall(r'[^A-Za-z0-9]', first_fname)
        delimiter = Counter(all_delims).most_common(1)[0][0] if all_delims else '_'
        print(f"      Auto-detecting '{delimiter}' as delimiter from first filename.")

    format_info = analyze_filename_formats(filenames, delimiter=delimiter)

    if format_info["uniform"]:
        # Uniform format — suggest obs_columns using classification
        print(f"{format_log_prefix('info_only')} Filenames are uniform. Using `suggest_obs_columns()` to recommend obs_columns...")
        obs_columns = suggest_obs_columns(filenames=filenames, source_type=source_type, delimiter=delimiter)
        print(f"{format_log_prefix('warn')} Please review the suggested `obs_columns` above.")
        print("   → If acceptable, rerun `import_data(..., obs_columns=...)` with this list.\n")
        return format_info, obs_columns, None
    else:
        # Non-uniform format — return fallback DataFrame
        print(f"{format_log_prefix('warn',indent=2)} {len(format_info['n_tokens'])} different filename formats detected. Proceeding with fallback `.obs` structure... (File Number, Parsing Type)")
        
        obs = pd.DataFrame({
            "File": list(range(1, len(filenames) + 1)),
            "parsingType": [format_info['group_map'][fname] for fname in filenames]
        })
        obs_columns = ["File", "parsingType"]
        return format_info, obs_columns, obs

def classify_subtokens(token, used_labels=None, keyword_map=None):
    """
    Classify a token into one or more metadata categories based on keyword or pattern matching.

    This function splits a token (e.g. from a filename) into subtokens using character-type transitions 
    (e.g., "Aur60minDIA" → "Aur", "60", "min", "DIA"), then attempts to classify each subtoken using:

    - Regex patterns (e.g., dates, well positions like A01)
    - Fuzzy substring matching via a user-defined or default keyword map

    Args:
        token (str): The input string to classify (e.g., "Aur60minDIA").
        used_labels (set, optional): Reserved for future logic to avoid assigning the same label twice.
        keyword_map (dict, optional): A dictionary of metadata categories (e.g., 'gradient') to example substrings.

    Returns:
        list of str: A list of predicted metadata labels for the token (e.g., ['gradient', 'acquisition']).
                     If no match is found, returns ['unknown??'].

    Example:
        Classify a gradient+time token:

            >>> classify_subtokens("Aur60minDIA")
            ['gradient', 'acquisition']

        Classify a well position:

            >>> classify_subtokens("B07")
            ['well_position']
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

def get_filenames(source: Union[str, Path], source_type: str) -> List[str]:
    """
    Extract sample filenames from a DIA-NN or Proteome Discoverer report.

    For DIA-NN reports, this extracts the 'Run' column from the table.
    For Proteome Discoverer (PD) output, it collects unique sample identifiers 
    based on column headers (e.g. abundance columns like "Abundances (SampleX)").

    Args:
        source (str or Path): Path to the input report file.
        source_type (str): Tool used to generate the report. Must be one of {'diann', 'pd'}.

    Returns:
        list of str: Extracted list of sample names or run filenames.

    Example:
        Extract DIA-NN run names:

            >>> get_filenames("diann_output.tsv", source_type="diann")
            ['Sample1.raw', 'Sample2.raw', 'Sample3.raw']

        Extract PD sample names from abundance columns:

            >>> get_filenames("pd_output.xlsx", source_type="pd")
            ['SampleA', 'SampleB', 'SampleC']
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

def analyze_filename_formats(filenames, delimiter: str = "_", group_labels=None):
    """
    Analyze filename structures to detect format consistency.

    This function checks if all filenames can be split into the same number of tokens
    using the provided delimiter. It can optionally group files by token count and assign
    custom group labels.

    Args:
        filenames (list of str): List of sample or file names.
        delimiter (str): Delimiter used to split each filename (default: "_").
        group_labels (list of str, optional): Optional group labels to assign to each unique token length group.

    Returns:
        dict: Format information containing:
            - 'uniform': True if all filenames split into the same number of tokens.
            - 'n_tokens': List of token counts for each filename.
            - 'group_map': Mapping of filename to group label (if labels are provided).

    Example:
        Check if filenames have a uniform structure:

            >>> filenames = ["A_60min_KD", "B_60min_SC", "C_120min_KD"]
            >>> analyze_filename_formats(filenames)
            {
                'uniform': True,
                'n_tokens': [3, 3, 3],
                'group_map': {}
            }

        With group labels:

            >>> analyze_filename_formats(filenames, group_labels=["Group1"])
            {
                'uniform': True,
                'n_tokens': [3, 3, 3],
                'group_map': {
                    'A_60min_KD': 'Group1',
                    'B_60min_SC': 'Group1',
                    'C_120min_KD': 'Group1'
                }
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