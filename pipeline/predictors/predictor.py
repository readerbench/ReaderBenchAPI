from abc import abstractmethod
import datetime
from typing import Dict, List

from ray import tune
from ray.air import Result
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from rb import Lang

from pipeline.task import TargetType, Task


class Predictor():
    
    def __init__(self, lang: Lang, tasks: List[Task]):
        self.lang = lang
        self.tasks = tasks
        self.time_budget = datetime.timedelta(minutes=10)
        self.workers = 1

    @abstractmethod
    def load_data(self, features, targets):
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        pass
    
    @abstractmethod
    def train(self, config, validation=True):
        pass
    
    def process_result(self, result: Result) -> Dict:
        return result.config

    @abstractmethod
    def save(self, model_obj: "Model"):
        pass

    @abstractmethod
    def load(self, model_obj: "Model"):
        pass
    
    def get_metric(self):
        return "val_loss"

    def search_config(self) -> Result:
        metric = self.get_metric()
        reporter = tune.CLIReporter(max_progress_rows=10, max_report_frequency=60, metric=metric, metric_columns=[metric])
        optuna_search = OptunaSearch(
            metric=metric,
            mode="min")
        asha_scheduler = ASHAScheduler(metric=metric, mode="min")
        tuner = tune.Tuner(
            tune.with_resources(self.train, {"cpu": 4, "gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                scheduler=asha_scheduler,
                num_samples=-1,
                time_budget_s=self.time_budget,
                reuse_actors=False,
                max_concurrent_trials=self.workers,
            ),
            param_space = self.get_config(),
            run_config=RunConfig(progress_reporter=reporter, verbose=1)
        )
        results = tuner.fit()
        return results.get_best_result(metric=metric, mode="min")