"""
Sequence-level classification task for MedTsLLM-family models.

Cross-entropy training with accuracy / F1 / precision / recall (macro) scoring.
If the model exposes an `aux_loss` attribute (e.g. BiomedCoOpTS, which returns
SCCM + KDSP regularizers), it is added to the cross-entropy loss during training.
This is backward compatible: models without `aux_loss` contribute nothing.
"""

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
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
                    logits = self.model(inputs)
                    labels = inputs["labels"].long()
                    if logits.ndim == 1:
                        loss = self.loss_fn(logits, labels.to(logits.dtype))
                    else:
                        loss = self.loss_fn(logits, labels)

                    # add SCCM + KDSP (or any auxiliary regularizer) if the model exposes it
                    aux = getattr(self.model, "aux_loss", None)
                    if aux is not None:
                        loss = loss + aux

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
        scores = {f"val/{k}": v for k, v in self.score(preds, targets).items()}
        self.log_scores(scores)
        return scores

    def test(self):
        preds, targets = self.predict(self.test_dataloader)
        scores = {f"test/{k}": v for k, v in self.score(preds, targets).items()}
        self.log_scores(scores)
        return scores

    def predict(self, dataloader):
        self.model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for inputs in tqdm(dataloader, total=len(dataloader)):
                inputs = self.prepare_batch(inputs)
                probs = self.model(inputs)
                if probs.ndim == 1:
                    probs = torch.stack([1.0 - probs, probs], dim=-1)
                all_probs.append(probs.float().cpu())
                all_targets.append(inputs["labels"].cpu())
        return torch.cat(all_probs, 0), torch.cat(all_targets, 0)

    def score(self, pred_scores, target):
        avg = "binary" if pred_scores.size(1) == 2 else "macro"
        pred = pred_scores.argmax(dim=1).int().numpy()
        target = target.int().numpy()
        return {
            "accuracy": accuracy_score(target, pred),
            "f1": f1_score(target, pred, average=avg, zero_division=0),
            "precision": precision_score(target, pred, average=avg, zero_division=0),
            "recall": recall_score(target, pred, average=avg, zero_division=0),
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
