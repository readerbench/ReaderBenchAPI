import datetime
import json
import os
from typing import Dict, List

import numpy as np
from ray import tune
import ray
from ray.air import session, Result
from ray.tune.integration.xgboost import TuneReportCallback
from rb import Lang
from sklearn.utils import compute_class_weight
import xgboost as xgb
from pipeline.predictors.predictor import Predictor

from pipeline.task import TargetType, Task
from sklearn.metrics import accuracy_score, cohen_kappa_score, r2_score, mean_absolute_error, f1_score

class XGBoostPredictor(Predictor):
    def __init__(self, lang: Lang, task: Task):
        super().__init__(lang, [task])
        self.task = task
        self.model: xgb.XGBModel = None
        self.time_budget = datetime.timedelta(minutes=10)
        self.workers = 8
        if self.task.type is TargetType.STR or self.task.binary:
            self.type = xgb.XGBClassifier
        else:
            self.type = xgb.XGBRegressor
        

    def load_data(self, features, targets):
        self.features = ray.put(features)
        self.targets = ray.put(targets)
        self.train_x = ray.put(np.array([features["train"][key] for key in self.task.features]).transpose())
        self.train_y = ray.put(targets["train"])
        self.val_x = ray.put(np.array([features["val"][key] for key in self.task.features]).transpose())
        self.val_y = ray.put(targets["val"])
        self.test_x = ray.put(np.array([features["test"][key] for key in self.task.features]).transpose())
        self.test_y = ray.put(targets["test"])
        if self.task.binary:
            class_weights = compute_class_weight("balanced", classes=[0, 1], y=targets["train"])
            self.train_weight = [class_weights[y] for y in targets["train"]]   
            class_weights = compute_class_weight("balanced", classes=[0, 1], y=targets["val"])
            self.val_weight = [class_weights[y] for y in targets["val"]]  
        elif self.task.type is TargetType.STR:
            class_weights = compute_class_weight("balanced", classes=range(len(self.task.classes)), y=targets["train"])
            self.train_weight = [class_weights[y] for y in targets["train"]]
            class_weights = compute_class_weight("balanced", classes=range(len(self.task.classes)), y=targets["val"])
            self.val_weight = [class_weights[y] for y in targets["val"]]
        else:
            self.train_weight = [1. for y in targets["train"]]
            self.val_weight = [1. for y in targets["val"]]
            
    def get_config(self):
        config = {
            "n_estimators": tune.choice([8, 16, 32, 64]),
            "max_depth": tune.randint(1, 9),
            "min_child_weight": tune.choice([1, 2, 3]),
            "subsample": tune.uniform(0.5, 1.0),
            "learning_rate": tune.loguniform(1e-4, 1e-1)
        }
        # config = {
        #     "n_estimators": 8,
        #     "max_depth": 4,
        #     "min_child_weight": 2,
        #     "subsample": 0.5,
        #     "learning_rate": 1e-3
        # }
        if self.task.type is TargetType.STR or self.task.binary:
            config["objective"] = "multi:softmax"
            config["eval_metric"] = ["mlogloss"]
            config["num_class"] = len(self.task.classes)
        else:
            config["objective"] = "reg:squarederror"
            config["eval_metric"] = ["rmse", "mae"]
        return config
    

    def train(self, config, validation=True):
        callbacks = []
        if validation:
            train_x = ray.get(self.train_x)
            train_y = ray.get(self.train_y)
            test_x = ray.get(self.val_x)
            test_y = ray.get(self.val_y)
            ray_callback = TuneReportCallback()
            callbacks.append(ray_callback)
            evals = [(test_x, test_y)]
            sample_weight = self.train_weight
        else:
            train_x = np.concatenate([ray.get(self.train_x), ray.get(self.val_x)], axis=0)
            train_y = np.concatenate([ray.get(self.train_y), ray.get(self.val_y)], axis=0)
            evals = None
            sample_weight = self.train_weight + self.val_weight
        model = self.type(**config, callbacks=callbacks)
        model.fit(train_x, train_y, eval_set=evals, sample_weight=sample_weight)
        
        if not validation:
            predicted = model.predict(ray.get(self.test_x))
            test_y = ray.get(self.test_y)
            if self.task.type is TargetType.STR or self.task.binary:
                if len(self.task.classes) == 2:
                    average = "binary"
                else:
                    average = "macro"
                metrics = {
                    "accuracy": accuracy_score(test_y, predicted),
                    "f1_score": f1_score(test_y, predicted, average=average)
                }
            else:
                metrics = {
                    "mae": mean_absolute_error(test_y, predicted),
                    "r2_score": r2_score(test_y, predicted),
                }
                pred = [int(round(self.task.convert_prediction(float(p)), 0)) for p in predicted]
                target = [int(round(self.task.convert_prediction(float(p)), 0)) for p in test_y]
                metrics[f"qwk"] = cohen_kappa_score(target, pred, weights="quadratic")
                
            self.model = model
            return metrics    
        
    def get_metric(self):
        return "validation_0-" + self.get_config()["eval_metric"][0]

    
class XGBoostMultiPredictor(Predictor):

    def __init__(self, lang: Lang, tasks: List[Task]):
        self.lang = lang
        self.tasks = tasks
        self.predictors = [XGBoostPredictor(lang, task) for task in tasks]
        self.ensemble = True

    def load_data(self, texts, features, targets):
        for i, task in enumerate(self.tasks):
            self.predictors[i].load_data(features, {key: values[i] for key, values in targets.items()})

    def search_config(self) -> Result:
        for predictor in self.predictors:
            best = predictor.search_config()
            predictor.train(best.config, validation=False)

    def save(self, model_obj: "Model"):
        os.makedirs(f"data/models/{model_obj.id}", exist_ok=True)
        for i, predictor in enumerate(self.predictors):
            predictor.model.save_model(f"data/models/{model_obj.id}/model-{i}.json")
        
    def load(self, model_obj: "Model"):
        for i, predictor in enumerate(self.predictors):
            predictor.model = predictor.type()
            predictor.model.load_model(f"data/models/{model_obj.id}/model-{i}.json")
    
    def train(self, configs, validation=True):
        result = []
        for predictor, config in zip(self.predictors, configs):
            metrics = predictor.train(config, validation)
            result.append(metrics)
        return result
    
    def search_config(self) -> List[Result]:
        result = []
        for predictor in self.predictors:
            result.append(predictor.search_config())
        return result
    
    def process_result(self, results: List[Result]) -> List[Dict]:
        return [result.config for result in results]
    
    def predict(self, model_obj: "Model", texts: List[str], features: List[Dict]) -> List[np.ndarray]:
        self.load(model_obj)
        result = []
        for task, predictor in zip(self.tasks, self.predictors):
            filtered_features = np.array([
                [features_dict[key] for key in task.features]
                for features_dict in features
            ])
            if task.type is TargetType.STR or self.task.binary:
                result.append(predictor.model.predict_proba(filtered_features, validate_features=False))
            else:
                result.append(predictor.model.predict(filtered_features, validate_features=False))
        return result