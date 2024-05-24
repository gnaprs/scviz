import pandas as pd
import anndata as ad
import numpy as np

from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer, KNNImputer

from typing import (  # Meta  # Generic ABCs  # Generic
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    List
)

# TODO!: methods to write
# easy way to assign obs, names, var to prot and pep
# def set_prot_obs(self, obs):

class pAnnData:
    """
    Class for storing protein and peptide data, with additional relational data between protein and peptide.
    """

    def __init__(self, 
                 prot = None, # np.ndarray | sparse.spmatrix 
                 pep = None, # np.ndarray | sparse.spmatrix
                 rs = None): # np.ndarray | sparse.spmatrix, protein x peptide relational data

        if prot is not None:
            self._prot = ad.AnnData(prot)
        if pep is not None:
            self._pep = ad.AnnData(pep)
        if rs is not None:
            self._set_RS(rs)

        self.history=[]
    
    # GETTERS
    @property
    def prot(self):
        return self._prot

    @property
    def pep(self):
        return self._pep
    
    @property
    def rs(self):
        return self._rs

    # SETTERS
    @prot.setter
    def prot(self, value):
        self._prot = value

    @pep.setter
    def pep(self, value):
        self._pep = value

    @rs.setter
    def rs(self, value):
        self._set_RS(value)

    def _set_RS(self, rs):
        # print rs shape, as well as protein and peptide shape if available
        print(f"Setting rs matrix with dimensions {rs.shape}")
        # assert that the dimensions of rs match either protein x peptide or peptide x protein
        assert ((self.prot is None or rs.shape[0] == self.prot.shape[1]) and (self.pep is None or rs.shape[1] == self.pep.shape[1])) or \
            ((self.prot is None or rs.shape[1] == self.prot.shape[1]) and (self.pep is None or rs.shape[0] == self.pep.shape[1])), \
            f"The dimensions of rs ({rs.shape}) must match either protein x peptide ({self.prot.shape[1] if self.prot is not None else None} x {self.pep.shape[1] if self.pep is not None else None}) or peptide x protein ({self.pep.shape[1] if self.pep is not None else None} x {self.prot.shape[1] if self.prot is not None else None})"

        # check dimensions of rs, make sure protein (row) x peptide (column) format
        if self.prot is not None and rs.shape[0] != self.prot.shape[1]:
            print("Transposing rs matrix to protein x peptide format")
            rs = rs.T
        self._rs = sparse.csr_matrix(rs)

    # UTILITY FUNCTIONS 
    def __repr__(self):
        if self.prot is not None:
            prot_shape = f"{self.prot.shape[0]} files by {self.prot.shape[1]} proteins"
            prot_obs = ', '.join(self.prot.obs.columns[:5]) + ('...' if len(self.prot.obs.columns) > 5 else '')
            prot_var = ', '.join(self.prot.var.columns[:5]) + ('...' if len(self.prot.var.columns) > 5 else '')
            prot_obsm = ', '.join(self.prot.obsm.keys()) + ('...' if len(self.prot.obsm.keys()) > 5 else '')
            prot_info = f"Protein (shape: {prot_shape})\nobs: {prot_obs}\nvar: {prot_var}\nobsm: {prot_obsm}"
        else:
            prot_info = "Protein: None"

        if self.pep is not None:
            pep_shape = f"{self.pep.shape[0]} files by {self.pep.shape[1]} peptides"
            pep_obs = ', '.join(self.pep.obs.columns[:5]) + ('...' if len(self.pep.obs.columns) > 5 else '')
            pep_var = ', '.join(self.pep.var.columns[:5]) + ('...' if len(self.pep.var.columns) > 5 else '')
            pep_info = f"Peptide (shape: {pep_shape})\nobs: {pep_obs}\nvar: {pep_var}"
        else:
            pep_info = "Peptide: None"

        if self.rs is not None:
            rs_shape = f"{self.rs.shape[0]} proteins by {self.rs.shape[1]} peptides"
            rs_info = f"RS (shape: {rs_shape})\n"
        else:
            rs_info = "RS: None"

        return (f"pAnnData object\n"
                f"{prot_info}\n\n"
                f"{pep_info}\n\n"
                f"{rs_info}\n")
    
    def has_data(self):
        return self.prot is not None or self.pep is not None
    
    def print_history(self):
        for i, action in enumerate(self.history, 1):
            print(f"{i}: {action}")

    # PROCESSING FUNCTIONS
    def impute(self,  method = 'median', on = 'protein'):
        if not(self.has_data()):
            raise ValueError("No protein or peptide data found in AnnData object.")

        if on not in ['protein', 'peptide']:
            raise ValueError("on must be either 'protein' or 'peptide'.")

        imputers = {
            'median': SimpleImputer(missing_values=np.nan, strategy='median', keep_empty_features=True),
            'mean': SimpleImputer(missing_values=np.nan, strategy='mean', keep_empty_features=True),
            'knn': KNNImputer(n_neighbors=2, weights="uniform", keep_empty_features=True)
        }

        if method in imputers:
            imputer = imputers[method]

            if on == 'protein':
                self.prot.obsm['X_raw'] = self.prot.X
                if method == 'knn':
                    self.prot.X = sparse.csr_matrix(imputer.fit_transform(self.prot.X.toarray()))
                else:
                    self.prot.X = imputer.fit_transform(self.prot.X)
                print(f"Imputed {on} data using {method}. New data stored in `X` with shape {self.prot.X.shape}.")
            else:
                self.pep.obsm['X_raw'] = self.pep.X
                if method == 'knn':
                    self.pep.X = sparse.csr_matrix(imputer.fit_transform(self.pep.X.toarray()))
                else:
                    self.pep.X = imputer.fit_transform(self.pep.X)
                print(f"Imputed {on} data using {method}. New data stored in `X` with shape {self.pep.X.shape}.")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.history.append(f"Imputed data using {method} on {on}")

    
def import_proteomeDiscoverer(prot_file: Optional[str] = None, pep_file: Optional[str] = None, obs_columns: Optional[List[str]] = None):
    if not prot_file and not pep_file:
        raise ValueError("At least one of prot_file or pep_file must be provided")

    if prot_file:
        # -----------------------------
        # PROTEIN DATA
        prot_all = pd.read_csv(prot_file, sep='\t')
        # prot_X: sparse data matrix
        prot_X = sparse.csr_matrix(prot_all.filter(regex='Abundance: F', axis=1).values).transpose()
        # prot_obsm['mbr']: protein MBR identification
        prot_obsm_mbr = prot_all.filter(regex='Found in Sample', axis=1).values.transpose()
        # prot_var_names: protein names
        prot_var_names = prot_all['Accession'].values
        # prot_var: protein metadata
        prot_var = prot_all.loc[:, 'Protein FDR Confidence: Combined':'# Razor Peptides']
        # prot_obs_names: file names
        prot_obs_names = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: (F\d+):')[0].values
        # prot_obs: sample typing from the column name
        prot_obs = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: F\d+: (.+)$')[0].values
        prot_obs = pd.DataFrame(prot_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).map(str.strip).astype('category')
    else:
        prot_X = prot_obsm_mbr = prot_var_names = prot_var = prot_obs_names = prot_obs = None

    if pep_file:
        # -----------------------------
        # PEPTIDE DATA
        pep_all = pd.read_csv(pep_file, sep='\t')
        # pep_X: sparse data matrix
        pep_X = sparse.csr_matrix(pep_all.filter(regex='Abundance: F', axis=1).values).transpose()
        # pep_var_names: peptide sequence with modifications
        pep_var_names = (pep_all['Annotated Sequence'] + np.where(pep_all['Modifications'].isna(), '', ' MOD:' + pep_all['Modifications'])).values
        # pep_obs_names: file names
        pep_obs_names = pep_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: (F\d+):')[0].values
        # pep_var: peptide metadata
        pep_var = pep_all.loc[:, 'Modifications':'Theo. MH+ [Da]']
        # prot_obs: sample typing from the column name
        pep_obs = pep_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: F\d+: (.+)$')[0].values
        pep_obs = pd.DataFrame(pep_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).map(str.strip).astype('category')
    else:
        pep_X = pep_var_names = pep_obs_names = pep_var = None

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
    else:
        rs = None

    # ASSERTIONS
    # -----------------------------
    # check that all files overlap, and that the order is the same
    if prot_obs_names is not None and pep_obs_names is not None:
        assert set(pep_obs_names) == set(prot_obs_names), "The files in peptide and protein data must be the same"
    # -----------------------------
    # check if mlb.classes_ has overlap with prot_var
    if prot_file and pep_file:
        mlb_classes_set = set(mlb.classes_)
        prot_var_set = set(prot_var_names)

        if mlb_classes_set != prot_var_set:
            print("WARNING: Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
            print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
            print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
            print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")

    # pAnnData OBJECT - should be the same for all imports
    # -----------------------------

    pdata = pAnnData(prot_X, pep_X, rs)

    if prot_file:
        pdata.prot.obs = pd.DataFrame(prot_obs)
        pdata.prot.obsm['X_mbr'] = prot_obsm_mbr
        pdata.prot.var = pd.DataFrame(prot_var)
        pdata.prot.obs_names = list(prot_obs_names)
        pdata.prot.var_names = list(prot_var_names)
        pdata.prot.obs.columns = obs_columns if obs_columns else list(range(len(prot_obs.columns)))

    if pep_file:
        pdata.pep.obs = pd.DataFrame(pep_obs)
        pdata.pep.var = pd.DataFrame(pep_var)
        pdata.pep.obs_names = list(pep_obs_names)
        pdata.pep.var_names = list(pep_var_names)
        pdata.pep.obs.columns = obs_columns if obs_columns else list(range(len(pep_obs.columns)))

    return pdata

# TODO!: Need to fix
def import_diann(report_file: Optional[str] = None, obs_columns: Optional[List[str]] = None):
    if not report_file:
        raise ValueError("DIA-NN report.tsv must be provided")

    report_all = pd.read_csv(report_file, sep='\t')
    # -----------------------------
    # PROTEIN DATA
    # prot_X: sparse data matrix
    prot_X_pivot = report_all.pivot_table(index='Protein.Group', columns='Run', values='PG.MaxLFQ', aggfunc='first')
    prot_X_pivot.fillna(0, inplace=True)
    prot_X = sparse.csr_matrix(prot_X_pivot.values)
    # prot_var_names: protein names
    prot_var_names = prot_X_pivot.index.values
    # prot_obs: file names
    prot_obs_names = prot_X_pivot.columns.values

    # MISSING
    # prot_var: protein metadata
    prot_var = report_all.loc[:, 'Protein FDR Confidence: Combined':'# Razor Peptides']
    # prot_obs: sample typing from the column name
    prot_obs = report_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: F\d+: (.+)$')[0].values
    prot_obs = pd.DataFrame(prot_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).map(str.strip).astype('category')

    # -----------------------------
    # PEPTIDE DATA
    # pep_X: sparse data matrix
    pep_X_pivot = report_all.pivot_table(index='Precursor.Id', columns='Run', values='Precursor.Translated', aggfunc='first')
    pep_X_pivot.fillna(0, inplace=True)
    pep_X = sparse.csr_matrix(pep_X_pivot.values)
    # pep_var: peptide sequence
    pep_var = pep_X_pivot.index.values
    # pep_obs: file names
    pep_obs = pep_X_pivot.columns.values

    # MISSING
    # pep_var_names: peptide sequence with modifications
    pep_var_names = (report_all['Annotated Sequence'] + np.where(pep_all['Modifications'].isna(), '', ' MOD:' + pep_all['Modifications'])).values
    # pep_obs_names: file names
    pep_obs_names = report_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: (F\d+):')[0].values

    # -----------------------------
    # RS DATA
    # rs: protein x peptide relational data
    pep_prot_list = report_all.drop_duplicates(subset=['Precursor.Id'])['Protein.Group'].str.split('; ')
    mlb = MultiLabelBinarizer()
    rs = mlb.fit_transform(pep_prot_list)
    index_dict = {protein: index for index, protein in enumerate(mlb.classes_)}
    reorder_indices = [index_dict[protein] for protein in prot_var]
    rs = rs[:, reorder_indices]

    # ASSERTIONS
    # -----------------------------
    # check that all files overlap, and that the order is the same
    if prot_obs_names is not None and pep_obs_names is not None:
        assert set(pep_obs_names) == set(prot_obs_names), "The files in peptide and protein data must be the same"
    # -----------------------------
    # check if mlb.classes_ has overlap with prot_var

    mlb_classes_set = set(mlb.classes_)
    prot_var_set = set(prot_var_names)

    if mlb_classes_set != prot_var_set:
        print("WARNING: Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
        print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
        print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
        print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")

    # pAnnData OBJECT - should be the same for all imports
    # -----------------------------

    pdata = pAnnData(prot_X, pep_X, rs)

    pdata.prot.obs = pd.DataFrame(prot_obs)
    pdata.prot.var = pd.DataFrame(prot_var)
    pdata.prot.obs_names = list(prot_obs_names)
    pdata.prot.var_names = list(prot_var_names)
    pdata.prot.obs.columns = obs_columns if obs_columns else list(range(len(prot_obs.columns)))

    pdata.pep.obs = pd.DataFrame(pep_obs)
    pdata.pep.var = pd.DataFrame(pep_var)
    pdata.pep.obs_names = list(pep_obs_names)
    pdata.pep.var_names = list(pep_var_names)
    pdata.pep.obs.columns = obs_columns if obs_columns else list(range(len(pep_obs.columns)))

    return pdata