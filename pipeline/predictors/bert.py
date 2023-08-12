import datetime
import json
from typing import Dict, List

import numpy as np
import ray
from sklearn.utils import compute_class_weight
import tensorflow as tf
from ray import tune
from ray.air import Result
from ray.tune.integration.keras import TuneReportCallback
from rb import Lang
from sklearn.metrics import cohen_kappa_score, f1_score, r2_score, accuracy_score
from transformers import AutoTokenizer, TFAutoModel

from pipeline.predictors.predictor import Predictor
from pipeline.task import TargetType, Task

class WeightedBinaryCrossentropyLoss(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = tf.convert_to_tensor(class_weights,dtype=tf.float32)
        
    def call(self, y_true, y_pred):
        loss = super().call(y_true, y_pred)
        weight_mask = tf.gather(self.class_weights, y_true)
        return loss * weight_mask
    
class WeightedSparseCrossentropyLoss(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = tf.convert_to_tensor(class_weights,dtype=tf.float32)
        
    def call(self, y_true, y_pred):
        loss = super().call(y_true, y_pred)
        weight_mask = tf.gather(self.class_weights, y_true)
        return loss * weight_mask

class BertPredictor(Predictor):
    def __init__(self, lang: Lang, tasks: List[Task]):
        super().__init__(lang, tasks)
        self.time_budget = datetime.timedelta(hours=4)
        self.workers = 1
        
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
        bert = TFAutoModel.from_pretrained(config["model"], from_pt=False)
        token_ids = tf.keras.layers.Input((None,), dtype=np.int32)
        attention_mask = tf.keras.layers.Input((None,), dtype=np.int32)
        inputs = [token_ids, attention_mask]
        if config["use_features"]:
            for task in self.tasks:
                inputs.append(tf.keras.layers.Input((len(task.features),), dtype=np.float32)) 
        emb = bert(input_ids=token_ids, attention_mask=attention_mask)
        if config["pooler"]:
            emb = emb.pooler_output
        else:
            emb = tf.keras.layers.GlobalAveragePooling1D()(emb.last_hidden_state, mask=attention_mask)
        
        outputs = []
        for i, task in enumerate(self.tasks):
            if config["use_features"]:
                features = tf.keras.layers.BatchNormalization()(inputs[2+i])
                if config["features_proj"] > 0:
                    features = tf.keras.layers.Dense(config["features_proj"], activation="tanh")(features)
                emb = tf.keras.layers.Concatenate(axis=-1)([emb, features])
            hidden = tf.keras.layers.Dense(config["hidden"], activation="relu")(emb)
            if task.binary:
                output = tf.keras.layers.Dense(1, activation="sigmoid", name=task.name)(hidden)
            elif task.type is TargetType.STR:
                output = tf.keras.layers.Dense(len(task.classes), activation="softmax", name=task.name)(hidden)
            else:
                output = tf.keras.layers.Dense(1, activation=None, name=task.name)(hidden)
            outputs.append(output)
        model = tf.keras.Model(
            inputs=inputs, 
            outputs=outputs)
        return model

    def train(self, config, validation=True):
        # gpus = tf.config.list_physical_devices('GPU')
        # tf.config.set_visible_devices(gpus[1], 'GPU')
        model = self.create_model(config)
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        texts = ray.get(self.texts)
        features = ray.get(self.features)
        targets = ray.get(self.targets)
        if validation:
            processed = tokenizer(texts["train"], padding="max_length", truncation=True, max_length=512, return_tensors="np") 
        else:
            processed = tokenizer(texts["train"] + texts["val"], padding="max_length", truncation=True, max_length=512, return_tensors="np") 
        train_inputs = [
            processed["input_ids"], 
            processed["attention_mask"], 
        ]
        train_outputs = []
        for i, task in enumerate(self.tasks):
            if validation:
                if config["use_features"]:
                    train_inputs.append(np.array([features["train"][key] for key in task.features]).transpose())
                train_outputs.append(np.array(targets["train"][i])) 
            else:
                if config["use_features"]:
                    train_inputs.append(np.array([features["train"][key] + features["val"][key] for key in task.features]).transpose())
                train_outputs.append(np.array(targets["train"][i] + targets["val"][i])) 
        callbacks = []
        if validation:
            test_partition = "val"
            ray_callback = TuneReportCallback()
            callbacks.append(ray_callback)
        else:
            test_partition = "test"
        processed = tokenizer(texts[test_partition], padding="max_length", truncation=True, max_length=512, return_tensors="np") 
        test_inputs = [
            processed["input_ids"], 
            processed["attention_mask"], 
        ]
        test_outputs = []
        for i, task in enumerate(self.tasks):
            if config["use_features"]:
                test_inputs.append(np.array([features[test_partition][key] for key in task.features]).transpose())
            test_outputs.append(np.array(targets[test_partition][i])) 
        validation_data=(test_inputs, test_outputs)
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config["lr"])
        losses = []
        metrics = []
        for i, task in enumerate(self.tasks):
            if task.binary:
                class_weights = compute_class_weight("balanced", classes=[0, 1], y=train_outputs[i])
                losses.append(WeightedBinaryCrossentropyLoss(class_weights))
                metrics.append(tf.keras.metrics.BinaryAccuracy(name=f"Accuracy_{task.name}"))
            elif task.type is TargetType.STR:
                class_weights = compute_class_weight("balanced", classes=range(len(task.classes)), y=train_outputs[i])
                losses.append(WeightedSparseCrossentropyLoss(class_weights))
                metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name=f"Accuracy_{task.name}"))
            else:
                losses.append("mse")
                metrics.append(tf.keras.metrics.MeanAbsoluteError(name=f"MAE_{task.name}"))
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        model.fit(train_inputs, train_outputs, batch_size=8, epochs=config["finetune_epochs"], validation_data=validation_data, 
            verbose=2, callbacks=callbacks)
        if not validation:
            metrics = {}
            predictions = model.predict(test_inputs, batch_size=16)
            if len(self.tasks) == 1:
                predictions = [predictions]
            for task, output, pred in zip(self.tasks, test_outputs, predictions):
                if task.type is TargetType.STR:
                    pred = np.argmax(pred, axis=-1)
                    if len(task.classes) == 2:
                        average = "binary"
                    else:
                        average = "macro"
                    metrics[f"f1_score_{task.name}"] = f1_score(output, pred, average=average)
                    metrics[f"accuracy_{task.name}"] = accuracy_score(output, pred)
                else:
                    metrics[f"r2_score_{task.name}"] = r2_score(output, pred[:, 0])
                    pred = [int(round(task.convert_prediction(float(p)), 0)) for p in pred]
                    target = [int(round(task.convert_prediction(float(p)), 0)) for p in output]
                    metrics[f"qwk_{task.name}"] = cohen_kappa_score(target, pred, weights="quadratic")
                    
                    
            self.model = model
            return metrics    
        
    def process_result(self, result: Result) -> Dict:
        config = result.config
        config["finetune_epochs"] = result.metrics["training_iteration"]
        return config
    
    def save(self, model_obj: "Model"):
        self.model.save_weights(f"data/models/{model_obj.id}/model")

    def load(self, model_obj: "Model"):
        self.model = self.create_model(json.loads(model_obj.params))
        self.model.load_weights(f"data/models/{model_obj.id}/model").expect_partial()

    def predict(self, model_obj: "Model", texts: List[str], features: List[Dict]) -> List[np.ndarray]:
        config = json.loads(model_obj.params)
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self.load(model_obj)
        processed = tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="np")
        inputs = [
            processed["input_ids"], 
            processed["attention_mask"], 
        ]
        for i, task in enumerate(self.tasks):
            if config["use_features"]:
                filtered_features = np.array([
                    [features_dict[key] if features_dict[key] is not None else 0. for key in task.features]
                    for features_dict in features
                ], dtype=np.float32)
                inputs.append(filtered_features)
        return self.model.predict(inputs, batch_size=12)   
        