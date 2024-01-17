from typing import Protocol, Tuple, Union, Optional, TypeVar, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from imblearn.under_sampling import ClusterCentroids
from imblearn.base import FunctionSampler
from imblearn.pipeline import Pipeline

from data_utils import FEATURE_LIST


@runtime_checkable
class Sampler(Protocol):
    def fit_resample(self, X, y):
        ...


@runtime_checkable
class Transformer(Protocol):
    def fit(self, X, y):
        ...

    def transform(self, X, y):
        ...


Step = Union[Sampler, Transformer]
NamedStep = Tuple[str, Step]

NamedTransformer = Tuple[str, Transformer]

T = TypeVar("T", bound=Step)


class PipelineBuilder:
    def __init__(
        self,
        pipe_name: str,
        estimator: Union[Transformer, NamedTransformer],
        *steps: Union[Step, NamedStep],
        shortNames: bool = True,
    ) -> None:
        self.pipe_name: str
        self.estimator: NamedTransformer
        self.shortNames: bool = shortNames
        self.namedSteps: list[NamedStep] = list()

        self.change_pipeline_name(pipe_name)
        self.replace_estimator(estimator)

        for step in steps:
            self._add(step)

    def _get_step_index(self, name):
        for i, (n, _) in enumerate(self.namedSteps):
            if n == name:
                return i

        return -1

    def _contains(self, name):
        return self._get_step_index(name) > -1

    def _get_name(self, name, step):
        if name:
            return name

        if self.shortNames:
            return step.__class__.__name__

        return repr(name)

    def _is_named_step(self, step):
        return isinstance(step, tuple)

    def _clone_step(self, step: T) -> T:
        cln = clone(step, safe=False)
        assert isinstance(cln, step.__class__)
        return cln

    def _add(self, step):
        if self._is_named_step(step):
            self.add(*step)
        else:
            self.add("", step)

    def add(self, name: str, step: Step):
        name = self._get_name(name, step)

        if self._contains(name):
            raise ValueError(f"Step '{name}' was already added.")

        self.namedSteps.append((name, self._clone_step(step)))

    def insert(self, idx: int, name: str, step: Step):
        if self._contains(name):
            raise ValueError(f"Step '{name}' was already added.")

        self.namedSteps.insert(idx, (name, step))

    def replace(self, name: Union[int, str], step: Step):
        index: int = -1
        ident: str = ""
        if isinstance(name, str):
            index = self._get_step_index(name)
            ident = name

        if isinstance(name, int):
            index = name
            ident = self.namedSteps[index][0]

        if index < 0:
            raise ValueError(f"Step '{ident}' not contained in the pipeline.")

        self.namedSteps[index] = (ident, step)

    def replace_estimator(self, estimator: Union[Transformer, NamedTransformer]):
        name: str = ""
        estim: Transformer
        if isinstance(estimator, tuple):
            name, estim = estimator
        else:
            estim = estimator

        self.estimator = (self._get_name(name, estim), self._clone_step(estim))

    def change_pipeline_name(self, pipe_name: str):
        self.pipe_name = pipe_name

    def remove(self, name: Union[int, str]):
        if isinstance(name, str):
            index = self._get_step_index(name)
        else:
            index = name

        if index < 0:
            raise ValueError(f"Step '{name}' not contained in the pipeline.")

        self.namedSteps.remove(self.namedSteps[index])

    def __add__(self, step: Union[Step, NamedStep]):
        self._add(step)

    def __repr__(self):
        cls_name = self.__class__.__name__
        pipe_name = self.pipe_name
        estimator = repr(self.estimator[1])
        steps = {name: repr(step) for name, step in self.namedSteps}

        return f"{cls_name}({pipe_name=}, {estimator=}, {steps=})"

    def __str__(self):
        cls_name = self.__class__.__name__
        pipe_name = self.pipe_name
        estimator = self.estimator[0]
        steps = [step for step, _ in self.namedSteps]

        return f"{cls_name}({pipe_name=}, {estimator=}, {steps=})"

    def clone(self, name: Optional[str] = None) -> "PipelineBuilder":
        if not name:
            name = self.pipe_name

        return PipelineBuilder(
            name, self.estimator[1], *self.namedSteps, shortNames=self.shortNames
        )

    def build(self, verbose: bool = False) -> Pipeline:
        all_steps = [
            (name, clone(step)) for name, step in self.namedSteps + [self.estimator]
        ]

        return Pipeline(all_steps, verbose=verbose)


# Legacy stuff
def _build_estimator_step(estimator=None):
    if estimator is None:
        estimator = KNeighborsRegressor(n_neighbors=1)

    return "estimator", estimator


def build_pipeline(estimator=None, random_state=None):
    return Pipeline(
        [_build_formating_step(), _build_estimator_step(estimator)], verbose=False
    )


def resampling(X, y, sampler=lambda *_: _, **_):
    enc = OrdinalEncoder()

    y_cls = enc.fit_transform(y.reshape(-1, 1))
    x_re, y_recls = sampler.fit_resample(X, y_cls.astype(np.int32))
    y_re = enc.inverse_transform(y_recls.reshape(-1, 1))

    return x_re, y_re.squeeze()


def _build_print_shape_step(prefix: str):
    def _print_shape(X, y):
        print(f"{prefix}: {X.shape}, {y.shape}")
        return X, y

    return "debug", FunctionSampler(func=_print_shape, validate=False)


def _build_sampling_step(sampler=None, random_state=None):
    if sampler is None:
        sampler = ClusterCentroids(
            estimator=KMeans(n_init="auto", random_state=random_state),
            sampling_strategy="not minority",
            random_state=random_state,
        )

    return "sampler", FunctionSampler(
        func=resampling, kw_args=dict(sampler=sampler), validate=False
    )


def build_sampled_pipeline(sampler=None, estimator=None, random_state=None):
    return Pipeline(
        [
            _build_formating_step(),
            _build_sampling_step(sampler, random_state),
            # _build_print_shape_step(f"Shape after {sampler.__class__.__name__}"),
            _build_estimator_step(estimator),
        ],
        verbose=False,
    )


def _to_numpy(X, y=None):
    if isinstance(X, pd.DataFrame):
        return np.stack(list(X["X"]), axis=0)

    return X


def _to_pandas(X, y=None):
    if isinstance(X, pd.DataFrame):
        return X

    return pd.DataFrame(zip(X), columns=FEATURE_LIST + ["X"])


def _build_formating_step(formater=_to_numpy):
    return "formater", FunctionTransformer(func=formater)


def _filtering(X, y, filter_tag="y_radius", min_value=0.0, max_value=40.0, **_):
    assert isinstance(X, pd.DataFrame)
    filter_idx = (X[filter_tag] >= min_value) & (X[filter_tag] <= max_value)

    X_filtered = X[filter_idx]
    y_filtered = y[filter_idx]

    return X_filtered, y_filtered


def _build_filtering_steps(fltr_params, **_):
    steps = []
    for i, params in enumerate(fltr_params):
        steps.append(
            (
                f"filter{i}",
                FunctionSampler(
                    func=_filtering,
                    kw_args=params,
                    validate=False,
                ),
            )
        )

    return steps


def build_filtered_pipeline(fltr_params=[], estimator=None, random_state=None):
    return Pipeline(
        _build_filtering_steps(fltr_params)
        + [
            _build_formating_step(),
            _build_estimator_step(estimator),
        ],
        verbose=False,
    )
