"""
PTB-XL classification dataset for MedTsLLM.

PTB-XL (Wagner et al., 2020) is a large 12-lead ECG dataset. Following the
journal paper, records are mapped to the five diagnostic SUPERCLASSES
(NORM, MI, STTC, CD, HYP) and the task is whole-sequence classification.

Data layout expected (download from PhysioNet, place under `data/ptbxl/`):
    data/ptbxl/
        ptbxl_database.csv
        scp_statements.csv
        records100/.../*.dat,*.hea     (100 Hz waveforms)
        records500/.../*.dat,*.hea     (500 Hz waveforms, optional)

Requires: wfdb  (pip install wfdb)

This module defines a generic ClassificationDataset (whole-record windows) and a
PTB-XL implementation. Register it in `datasets/__init__.py` (see README patch).
"""

import ast
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


# Fixed label order matching the paper's five superclasses.
SUPERCLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_TO_IDX = {c: i for i, c in enumerate(SUPERCLASS_ORDER)}


class ClassificationDataset(BaseDataset, ABC):
    """Whole-record classification: each item is one full window + one label.

    Unlike the sliding-window tasks, here `__getitem__` returns a single
    `x_enc` of shape [T, F] and a scalar integer `labels`. `get_data` must
    return {"data": [N, T, F], "labels": [N], (optional) "descriptions": [N]}.
    """

    supported_tasks = ["classification"]

    def __init__(self, config, split):
        super().__init__(config, split)
        assert self.task == "classification"

    # --- overridden loading (records are not a single contiguous matrix) ---
    def load_data(self):
        data = self.get_data()
        X = np.asarray(data["data"], dtype=np.float32)          # [N, T, F]
        y = np.asarray(data["labels"]).astype(np.int64)         # [N]

        X = self.normalize_records(X)

        self.records = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)
        self.record_descriptions = data.get("descriptions", None)

    def normalize_records(self, X):
        if not self.config.data.normalize:
            return X
        n, t, f = X.shape
        if self.normalizer is None:
            if self.split == "train":
                train_X = X
            else:
                train_X = np.asarray(self.get_data("train")["data"], dtype=np.float32)
            self.normalizer = StandardScaler().fit(train_X.reshape(-1, train_X.shape[-1]))
        return self.normalizer.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, idx):
        out = {"x_enc": self.records[idx], "labels": self.labels[idx]}
        if self.record_descriptions is not None:
            out["descriptions"] = self.record_descriptions[idx]
        return out

    def inverse_index(self, idx):
        return idx

    @property
    def n_points(self):
        return self.records.shape[0]

    @property
    def n_features(self):
        return self.records.shape[2]


class PTBXLClassificationDataset(ClassificationDataset):

    description = (
        "PTB-XL is a large dataset of 12-lead ECGs, each a 10-second recording. "
        "Recordings are labeled with one of five diagnostic superclasses: "
        "Normal ECG (NORM), Myocardial Infarction (MI), ST/T Changes (STTC), "
        "Conduction Disturbance (CD), and Hypertrophy (HYP)."
    )
    task_description = (
        "Classify the following 12-lead ECG recording into one of five diagnostic "
        "categories: Normal, Myocardial Infarction, ST/T Change, Conduction "
        "Disturbance, or Hypertrophy."
    )

    sampling_rate = 100  # use records100/ (1000 samples / 10 s); set 500 for records500/

    @property
    def n_classes(self):
        return len(SUPERCLASS_ORDER)

    def _fold_split(self, split):
        # Predefined stratified, patient-disjoint folds (1..10).
        if split == "train":
            return list(range(1, 9))     # folds 1-8
        elif split == "val":
            return [9]                   # fold 9
        else:
            return [10]                  # fold 10 (held-out test)

    def get_data(self, split=None):
        import wfdb  # local import so the rest of the repo works without wfdb

        split = split or self.split
        basepath = Path(__file__).parent / "../data/ptbxl/"

        db = pd.read_csv(basepath / "ptbxl_database.csv", index_col="ecg_id")
        db.scp_codes = db.scp_codes.apply(ast.literal_eval)

        agg = pd.read_csv(basepath / "scp_statements.csv", index_col=0)
        agg = agg[agg.diagnostic == 1]

        def to_superclasses(scp_dict):
            classes = set()
            for code in scp_dict.keys():
                if code in agg.index:
                    classes.add(agg.loc[code, "diagnostic_class"])
            return classes

        db["superclasses"] = db.scp_codes.apply(to_superclasses)

        # Keep single-superclass records (standard single-label PTB-XL setup).
        db = db[db.superclasses.apply(lambda s: len(s) == 1)].copy()
        db["label"] = db.superclasses.apply(
            lambda s: SUPERCLASS_TO_IDX[next(iter(s))]
        )

        folds = self._fold_split(split)
        db = db[db.strat_fold.isin(folds)]

        fn_col = "filename_lr" if self.sampling_rate == 100 else "filename_hr"
        crop = int(self.history_len)

        signals, labels, descriptions = [], [], []
        for ecg_id, row in db.iterrows():
            sig, _ = wfdb.rdsamp(str(basepath / row[fn_col]))   # [T, 12]
            sig = np.asarray(sig, dtype=np.float32)

            # Center-crop / pad to history_len (paper uses a 512-step window).
            t = sig.shape[0]
            if t >= crop:
                start = (t - crop) // 2
                sig = sig[start:start + crop]
            else:
                pad = np.zeros((crop - t, sig.shape[1]), dtype=np.float32)
                sig = np.concatenate([sig, pad], axis=0)

            signals.append(sig)
            labels.append(int(row["label"]))

            sex = "male" if int(row.get("sex", 0)) == 0 else "female"
            age = row.get("age", None)
            age_str = "unknown" if pd.isna(age) else int(age)
            # JSON-formatted demographics (the paper found JSON improves comprehension).
            descriptions.append(
                f'Patient information: {{"age": {age_str}, "sex": "{sex}"}}'
            )

        data = np.stack(signals, axis=0)     # [N, crop, 12]
        labels = np.asarray(labels, dtype=np.int64)

        return {"data": data, "labels": labels, "descriptions": descriptions}


ptbxl_datasets = {
    "classification": PTBXLClassificationDataset,
}
