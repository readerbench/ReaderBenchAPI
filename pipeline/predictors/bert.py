import datetime
import json
import os
from typing import Dict, List

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import compute_class_weight
from ray import tune
from ray.air import Result
import ray.train
from rb import Lang
from sklearn.metrics import cohen_kappa_score, f1_score, r2_score, accuracy_score
from transformers import AutoTokenizer, AutoModel

from pipeline.predictors.predictor import Predictor
from pipeline.task import TargetType, Task


class BertModel(nn.Module):
    """PyTorch nn.Module wrapping a BERT encoder with per-task output heads."""

    def __init__(self, config, tasks: List[Task]):
        super().__init__()
        self.config = config
        self.tasks = tasks

        self.bert = AutoModel.from_pretrained(config["model"])
        bert_hidden_size = self.bert.config.hidden_size

        # Per-task feature projection / batch-norm layers
        self.feature_bns: nn.ModuleList = nn.ModuleList()
        self.feature_projs: nn.ModuleList = nn.ModuleList()
        for task in tasks:
            if config["use_features"]:
                self.feature_bns.append(nn.BatchNorm1d(len(task.features)))
                if config["features_proj"] > 0:
                    self.feature_projs.append(
                        nn.Linear(len(task.features), config["features_proj"])
                    )
                else:
                    self.feature_projs.append(None)
            else:
                self.feature_bns.append(None)
                self.feature_projs.append(None)

        # Per-task hidden + output heads
        self.hidden_layers: nn.ModuleList = nn.ModuleList()
        self.output_layers: nn.ModuleList = nn.ModuleList()
        for i, task in enumerate(tasks):
            if config["use_features"]:
                feat_dim = config["features_proj"] if config["features_proj"] > 0 else len(task.features)
                in_dim = bert_hidden_size + feat_dim
            else:
                in_dim = bert_hidden_size
            self.hidden_layers.append(nn.Linear(in_dim, config["hidden"]))
            if task.binary:
                self.output_layers.append(nn.Linear(config["hidden"], 1))
            elif task.type is TargetType.STR:
                self.output_layers.append(nn.Linear(config["hidden"], len(task.classes)))
            else:
                self.output_layers.append(nn.Linear(config["hidden"], 1))

        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, task_features=None):
        """
        Args:
            input_ids:      (B, seq_len) long tensor
            attention_mask: (B, seq_len) long tensor
            task_features:  list of (B, feat_dim) float tensors, one per task,
                            or None when use_features is False
        Returns:
            list of output tensors, one per task
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.config["pooler"]:
            # pooler_output: (B, hidden)
            emb = bert_out.pooler_output
        else:
            # masked mean over sequence dimension
            # last_hidden_state: (B, seq_len, hidden)
            hidden_states = bert_out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
            sum_hidden = (hidden_states * mask).sum(dim=1)  # (B, hidden)
            count = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            emb = sum_hidden / count  # (B, hidden)

        outputs = []
        for i, task in enumerate(self.tasks):
            task_emb = emb
            if self.config["use_features"] and task_features is not None:
                feat = self.feature_bns[i](task_features[i])
                if self.feature_projs[i] is not None:
                    feat = torch.tanh(self.feature_projs[i](feat))
                task_emb = torch.cat([task_emb, feat], dim=-1)
            hidden = self.relu(self.hidden_layers[i](task_emb))
            out = self.output_layers[i](hidden)
            if task.binary:
                out = torch.sigmoid(out)  # (B, 1)
            elif task.type is TargetType.STR:
                # raw logits – softmax applied at loss/prediction time
                pass
            # regression: raw scalar, no activation
            outputs.append(out)
        return outputs


class BertPredictor(Predictor):
    def __init__(self, lang: Lang, tasks: List[Task]):
        super().__init__(lang, tasks)
        self.time_budget = datetime.timedelta(hours=4)
        self.workers = 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def load_data(self, texts, features, targets):
        self.texts = ray.put(texts)
        self.features = ray.put(features)
        self.targets = ray.put(targets)

    def get_config(self):
        if self.lang is Lang.EN:
            models = ["roberta-base", "bert-base-uncased"]
        elif self.lang is Lang.FR:
            models = ["camembert-base"]
        elif self.lang is Lang.RO:
            models = ["readerbench/RoBERT-base"]
        elif self.lang is Lang.PT:
            models = ["neuralmind/bert-base-portuguese-cased"]
        config = {
            "model": tune.choice(models),
            "use_features": tune.choice([True, False]),
            "features_proj": tune.choice([0, 32, 64, 128]),
            "pooler": tune.choice([True, False]),
            "hidden": tune.choice([32, 64, 128, 256]),
            "lr": tune.qloguniform(5e-6, 1e-4, 5e-6),
            "finetune_epochs": 10,
        }
        return config

    def create_model(self, config):
        return BertModel(config, self.tasks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tensor_dataset(self, input_ids, attention_mask, task_features, task_targets):
        """Pack everything into a TensorDataset for use with DataLoader."""
        tensors = [
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        ]
        for feat in task_features:
            tensors.append(torch.tensor(feat, dtype=torch.float32))
        for tgt in task_targets:
            tensors.append(torch.tensor(tgt, dtype=torch.float32))
        return TensorDataset(*tensors)

    def _unpack_batch(self, batch, num_tasks, use_features):
        """Unpack a batch produced by _build_tensor_dataset."""
        input_ids = batch[0].to(self.device)
        attention_mask = batch[1].to(self.device)
        offset = 2
        task_features = []
        if use_features:
            for _ in range(num_tasks):
                task_features.append(batch[offset].to(self.device))
                offset += 1
        task_targets = []
        for _ in range(num_tasks):
            task_targets.append(batch[offset].to(self.device))
            offset += 1
        return input_ids, attention_mask, task_features, task_targets

    def _compute_losses(self, model, criterion_list, input_ids, attention_mask, task_features, task_targets, use_features):
        outputs = model(
            input_ids,
            attention_mask,
            task_features if use_features else None,
        )
        total_loss = torch.tensor(0.0, device=self.device)
        for i, (task, criterion, out, tgt) in enumerate(
            zip(self.tasks, criterion_list, outputs, task_targets)
        ):
            if task.binary:
                loss = criterion(out.squeeze(-1), tgt)
            elif task.type is TargetType.STR:
                loss = criterion(out, tgt.long())
            else:
                loss = criterion(out.squeeze(-1), tgt)
            total_loss = total_loss + loss
        return total_loss, outputs

    def _run_epoch(self, model, loader, criterion_list, optimizer, use_features, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        n_batches = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                input_ids, attention_mask, task_features, task_targets = self._unpack_batch(
                    batch, len(self.tasks), use_features
                )
                if train:
                    optimizer.zero_grad()
                loss, _ = self._compute_losses(
                    model, criterion_list, input_ids, attention_mask, task_features, task_targets, use_features
                )
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _predict_all(self, model, loader, use_features):
        model.eval()
        all_outputs = [[] for _ in self.tasks]
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, task_features, _ = self._unpack_batch(
                    batch, len(self.tasks), use_features
                )
                outputs = model(
                    input_ids,
                    attention_mask,
                    task_features if use_features else None,
                )
                for i, out in enumerate(outputs):
                    all_outputs[i].append(out.cpu().numpy())
        return [np.concatenate(parts, axis=0) for parts in all_outputs]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, config, validation=True):
        model = self.create_model(config).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        texts = ray.get(self.texts)
        features = ray.get(self.features)
        targets = ray.get(self.targets)

        # Tokenise training texts
        if validation:
            processed = tokenizer(
                texts["train"], padding="max_length", truncation=True,
                max_length=512, return_tensors="np",
            )
        else:
            processed = tokenizer(
                texts["train"] + texts["val"], padding="max_length",
                truncation=True, max_length=512, return_tensors="np",
            )
        train_input_ids = processed["input_ids"]
        train_attention_mask = processed["attention_mask"]

        train_task_features = []
        train_task_targets = []
        for i, task in enumerate(self.tasks):
            if validation:
                if config["use_features"]:
                    train_task_features.append(
                        np.array([features["train"][key] for key in task.features], dtype=np.float32).T
                    )
                train_task_targets.append(np.array(targets["train"][i], dtype=np.float32))
            else:
                if config["use_features"]:
                    train_task_features.append(
                        np.array(
                            [features["train"][key] + features["val"][key] for key in task.features],
                            dtype=np.float32,
                        ).T
                    )
                train_task_targets.append(
                    np.array(targets["train"][i] + targets["val"][i], dtype=np.float32)
                )

        # Tokenise evaluation texts
        test_partition = "val" if validation else "test"
        processed = tokenizer(
            texts[test_partition], padding="max_length", truncation=True,
            max_length=512, return_tensors="np",
        )
        test_input_ids = processed["input_ids"]
        test_attention_mask = processed["attention_mask"]

        test_task_features = []
        test_task_targets = []
        for i, task in enumerate(self.tasks):
            if config["use_features"]:
                test_task_features.append(
                    np.array([features[test_partition][key] for key in task.features], dtype=np.float32).T
                )
            test_task_targets.append(np.array(targets[test_partition][i], dtype=np.float32))

        # Build DataLoaders
        train_dataset = self._build_tensor_dataset(
            train_input_ids, train_attention_mask, train_task_features, train_task_targets
        )
        test_dataset = self._build_tensor_dataset(
            test_input_ids, test_attention_mask, test_task_features, test_task_targets
        )
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Build per-task loss functions
        criterion_list = []
        for i, task in enumerate(self.tasks):
            if task.binary:
                class_weights = compute_class_weight(
                    "balanced", classes=[0, 1], y=train_task_targets[i].astype(int)
                )
                # pos_weight for BCELoss: weight of the positive class relative to negative
                pos_weight = torch.tensor(
                    class_weights[1] / class_weights[0], dtype=torch.float32, device=self.device
                )
                criterion_list.append(nn.BCELoss(reduction="mean"))
                # We manually scale: store pos_weight alongside for use in the loop
                # Use BCEWithLogitsLoss equivalent via weight trick instead
                criterion_list[-1] = _WeightedBCELoss(class_weights, device=self.device)
            elif task.type is TargetType.STR:
                class_weights = compute_class_weight(
                    "balanced",
                    classes=np.arange(len(task.classes)),
                    y=train_task_targets[i].astype(int),
                )
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
                criterion_list.append(nn.CrossEntropyLoss(weight=weight_tensor))
            else:
                criterion_list.append(nn.MSELoss())

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # Training loop
        for epoch in range(config["finetune_epochs"]):
            train_loss = self._run_epoch(
                model, train_loader, criterion_list, optimizer,
                config["use_features"], train=True,
            )
            val_loss = self._run_epoch(
                model, test_loader, criterion_list, optimizer,
                config["use_features"], train=False,
            )
            if validation:
                ray.train.report(metrics={
                    "loss": train_loss,
                    "val_loss": val_loss,
                    "training_iteration": epoch + 1,
                })

        if not validation:
            predictions = self._predict_all(model, test_loader, config["use_features"])
            metrics = {}
            for task, output, pred in zip(self.tasks, test_task_targets, predictions):
                if task.type is TargetType.STR:
                    pred_labels = np.argmax(pred, axis=-1)
                    average = "binary" if len(task.classes) == 2 else "macro"
                    metrics[f"f1_score_{task.name}"] = f1_score(
                        output.astype(int), pred_labels, average=average
                    )
                    metrics[f"accuracy_{task.name}"] = accuracy_score(
                        output.astype(int), pred_labels
                    )
                else:
                    metrics[f"r2_score_{task.name}"] = r2_score(output, pred[:, 0])
                    pred_converted = [
                        int(round(task.convert_prediction(float(p)), 0)) for p in pred[:, 0]
                    ]
                    target_converted = [
                        int(round(task.convert_prediction(float(p)), 0)) for p in output
                    ]
                    metrics[f"qwk_{task.name}"] = cohen_kappa_score(
                        target_converted, pred_converted, weights="quadratic"
                    )
            self.model = model
            return metrics

    def process_result(self, result: Result) -> Dict:
        config = result.config
        config["finetune_epochs"] = result.metrics["training_iteration"]
        return config

    def save(self, model_obj: "Model"):
        path = f"data/models/{model_obj.id}/model.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, model_obj: "Model"):
        config = json.loads(model_obj.params)
        self.model = self.create_model(config).to(self.device)
        path = f"data/models/{model_obj.id}/model.pt"
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, model_obj: "Model", texts: List[str], features: List[Dict]) -> List[np.ndarray]:
        config = json.loads(model_obj.params)
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self.load(model_obj)
        processed = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=512, return_tensors="np",
        )
        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]

        task_features = []
        for task in self.tasks:
            if config["use_features"]:
                feat_array = np.array(
                    [
                        [
                            features_dict[key] if features_dict[key] is not None else 0.0
                            for key in task.features
                        ]
                        for features_dict in features
                    ],
                    dtype=np.float32,
                )
                task_features.append(feat_array)

        # Dummy targets (zeros) so we can reuse _build_tensor_dataset / DataLoader
        dummy_targets = [
            np.zeros(len(texts), dtype=np.float32) for _ in self.tasks
        ]
        dataset = self._build_tensor_dataset(
            input_ids, attention_mask, task_features, dummy_targets
        )
        loader = DataLoader(dataset, batch_size=12, shuffle=False)
        return self._predict_all(self.model, loader, config["use_features"])


# ---------------------------------------------------------------------------
# Helper loss class
# ---------------------------------------------------------------------------

class _WeightedBCELoss(nn.Module):
    """BCE loss with per-sample weighting based on class membership."""

    def __init__(self, class_weights, device):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: (B,) probabilities, y_true: (B,) binary float
        bce = nn.functional.binary_cross_entropy(y_pred, y_true, reduction="none")
        # weight each sample by the class weight of its true label
        weights = self.class_weights[y_true.long()]
        return (bce * weights).mean()
