from typing import Generator, Optional, Protocol, Union
import logging

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn import model_selection


log = logging.getLogger(__file__)


Split = tuple[np.ndarray, np.ndarray]


class CV_Strategy(Protocol):
    def split(self, X, y=None) -> Union[Split, Generator[Split, None, None]]:
        ...


class Run:
    def __init__(
        self,
        model: Pipeline,
        X: np.ndarray,
        y: np.ndarray,
        cv_split: Split,
        scoring=None,
        groups: Optional[np.ndarray] = None,
        n_jobs: int = 1,
    ) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.cv_split = cv_split
        self.scoring = scoring
        self.groups = groups
        self.n_jobs = n_jobs

    def __call__(self) -> pd.DataFrame:
        try:
            results = model_selection.cross_validate(
                self.model,
                self.X,
                self.y,
                cv=self.cv_split,
                groups=self.groups,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            return pd.DataFrame(results)
        except ValueError as e:
            log.exception(e)
            raise e
