"""
Sequence-level classification task for MedTsLLM.

Implements the classification head/training described in the journal paper
(MedTsLLM, IEEE JBHI 2025): the contextualized time-series representation from
the (frozen) LLM is projected to the class space, passed through a softmax to
obtain class probabilities, trained with cross-entropy, and decoded with argmax
at inference. Metrics follow Table VIII: accuracy, F1, precision, recall.

Drop this file in `tasks/` and register it in `tasks/__init__.py`
(see the patch in the accompanying README).
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from .base import BaseTask


class ClassificationTask(BaseTask):

    def __init__(self, run_id, config, newrun=True):
        self.task = "classification"
        super(ClassificationTask, self).__init__(run_id, config, newrun)

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch + 1}/{self.config.training.epochs}")
            self.model.train()
            for inputs in tqdm(self.train_dataloader):
                inputs = self.prepare_batch(inputs)

                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.mixed):
                    logits = self.model(inputs)            # [bs, K] (raw logits in train mode)
                    labels = inputs["labels"].long()       # [bs]

                    if logits.ndim == 1:                   # binary, single-logit head
                        loss = self.loss_fn(logits, labels.to(logits.dtype))
                    else:                                  # multiclass
                        loss = self.loss_fn(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_step(loss.item())

            val_scores = self.val()
            self.log_epoch(val_scores)
            self.scheduler.step()

        self.model.eval()

    def val(self):
        preds, targets = self.predict(self.val_dataloader)
        scores = self.score(preds, targets)
        scores = {f"val/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def test(self):
        preds, targets = self.predict(self.test_dataloader)
        scores = self.score(preds, targets)
        scores = {f"test/{metric}": value for metric, value in scores.items()}
        self.log_scores(scores)
        return scores

    def predict(self, dataloader):
        self.model.eval()

        all_probs, all_targets = [], []
        with torch.no_grad():
            for inputs in tqdm(dataloader, total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                probs = self.model(inputs)                 # eval mode -> softmax/sigmoid applied in model.forward

                if probs.ndim == 1:                        # binary -> prob of positive class
                    probs = torch.stack([1.0 - probs, probs], dim=-1)

                all_probs.append(probs.float().cpu())
                all_targets.append(inputs["labels"].cpu())

        preds = torch.cat(all_probs, dim=0)                # [N, K]
        targets = torch.cat(all_targets, dim=0)            # [N]
        return preds, targets

    def score(self, pred_scores, target):
        avg_mode = "binary" if pred_scores.size(1) == 2 else "macro"
        pred = pred_scores.argmax(dim=1).int().numpy()
        target = target.int().numpy()
        return {
            "accuracy": accuracy_score(target, pred),
            "f1": f1_score(target, pred, average=avg_mode, zero_division=0),
            "precision": precision_score(target, pred, average=avg_mode, zero_division=0),
            "recall": recall_score(target, pred, average=avg_mode, zero_division=0),
        }

    def build_loss(self):
        is_binary = (self.train_dataset.n_classes == 2)
        loss_name = self.config.training.loss

        if loss_name in ("bce",) or is_binary:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif loss_name in ("ce", "cross_entropy", "auto"):
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function selection: {loss_name}")
        return self.loss_fn
