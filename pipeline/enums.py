from enum import Enum
from typing import Type

from pipeline.predictors.bert import BertPredictor
from pipeline.predictors.predictor import Predictor
from pipeline.predictors.xgboost import XGBoostMultiPredictor, XGBoostPredictor

class ModelTypeEnum(Enum):
    TRANSFORMER = 1
    XGBOOST = 2

    def predictor(self) -> Type[Predictor]:
        if self is ModelTypeEnum.TRANSFORMER:
            return BertPredictor
        else:
            return XGBoostMultiPredictor

