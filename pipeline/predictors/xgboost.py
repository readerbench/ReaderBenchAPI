import json
import os
from typing import Dict, List

import numpy as np
from ray import tune
from ray.air import session, Result
from ray.tune.integration.xgboost import TuneReportCallback
from rb import Lang
import xgboost as xgb
from pipeline.predictors.predictor import Predictor

from pipeline.task import TargetType, Task
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, f1_score

class XGBoostPredictor(Predictor):
    def __init__(self, lang: Lang, task: Task):
        super().__init__(lang, [task])
        self.task = task
        self.model: xgb.XGBModel = None
        
    def load_data(self, features, targets):
        self.features = features
        self.targets = targets
        self.train_x = np.array([features["train"][key] for key in self.task.features]).transpose()
        self.train_y = targets["train"]
        self.val_x = np.array([features["val"][key] for key in self.task.features]).transpose()
        self.val_y = targets["val"]
        self.test_x = np.array([features["test"][key] for key in self.task.features]).transpose()
        self.test_y = targets["test"]
        
    def get_config(self):
        config = {
            "max_depth": tune.randint(1, 9),
            "min_child_weight": tune.choice([1, 2, 3]),
            "subsample": tune.uniform(0.5, 1.0),
            "eta": tune.loguniform(1e-4, 1e-1)
        }
        if self.task.type is TargetType.STR:
            config["objective"] = "multi:softmax"
            config["eval_metric"] = ["mlogloss"]
            config["num_class"] = len(self.task.classes)
        else:
            config["objective"] = "reg:squarederror"
            config["eval_metric"] = ["rmse", "mae"]
        return config
    

    def train(self, config, validation=True):
        callbacks = []
        evals = []
        if validation:
            train_set = xgb.DMatrix(self.train_x, label=self.train_y)
            test_set = xgb.DMatrix(self.val_x, label=self.val_y)
            ray_callback = TuneReportCallback()
            callbacks.append(ray_callback)
            evals.append((test_set, "eval"))
        else:
            train_set = xgb.DMatrix(np.concatenate([self.train_x, self.val_x], axis=0), label=np.concatenate([self.train_y, self.val_y], axis=0)) 
            test_set = xgb.DMatrix(self.test_x, label=self.test_y)
        model = xgb.train(
            config,
            train_set,
            evals=evals,
            verbose_eval=False,
            callbacks=callbacks,
        )
        if not validation:
            predicted = model.predict(test_set, validate_features=False)
            if self.task.type is TargetType.STR:
                if len(self.task.classes) == 2:
                    average = "binary"
                else:
                    average = "macro"
                metrics = {
                    "accuracy": accuracy_score(self.test_y, predicted),
                    "f1_score": f1_score(self.test_y, predicted, average=average)
                }
            else:
                metrics = {
                    "mae": mean_absolute_error(self.test_y, predicted),
                    "r2_score": r2_score(self.test_y, predicted),
                }
            self.model = model
            return metrics    
        # session.report({"mean_accuracy": accuracy, "done": True})

    def get_metric(self):
        return "eval-" + self.get_config()["eval_metric"][0]
    
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
            predictor.model = xgb.Booster()
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
            inputs = xgb.DMatrix(filtered_features)
            result.append(predictor.model.predict(inputs, validate_features=False))
        return result