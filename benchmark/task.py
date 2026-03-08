"""Task loading: enumerate (dataset, sl_id) pairs with per-group data."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyarrow.parquet as pq

from benchmark.dataset.registry import DATASET_REGISTRY, DatasetInfo

_DATA_ROOT = Path(__file__).parent / "dataset"


class _ModelFn:
    def __init__(self, dataset_name: str, sl_id: str):
        self.dataset_name = dataset_name
        self.sl_id = sl_id
        self._fn = None

    def _resolve(self):
        if self._fn is None:
            mod = _get_law_module(self.dataset_name)
            self._fn = mod.LAW_REGISTRY[self.sl_id]

    def __call__(self, theta, X):
        self._resolve()
        return self._fn(theta, X, backend="numpy")

    def __getstate__(self):
        return {"dataset_name": self.dataset_name, "sl_id": self.sl_id}

    def __setstate__(self, state):
        self.dataset_name = state["dataset_name"]
        self.sl_id = state["sl_id"]
        self._fn = None


@dataclass
class GroupData:
    """Data for a single group within a task."""
    group: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    cost_train: np.ndarray


@dataclass
class ScalingLawTask:
    task_id: str           # "dataset/sl_id"
    dataset_name: str
    sl_id: str
    groups: List[GroupData]
    model_fn: Callable
    n_params: int
    param_bounds: Optional[List] = None


def _load_parquet(path: Path) -> dict:
    table = pq.read_table(str(path))
    return {col: table.column(col).to_pylist() for col in table.column_names}


def _get_law_module(dataset_name: str):
    mod_path = f"benchmark.dataset.{dataset_name}.laws"
    return import_module(mod_path)


def load_tasks_for_dataset(dataset_name: str) -> List[ScalingLawTask]:
    info = DATASET_REGISTRY[dataset_name]
    law_mod = _get_law_module(dataset_name)
    law_registry = law_mod.LAW_REGISTRY
    param_counts = law_mod.PARAM_COUNTS
    param_bounds = getattr(law_mod, "PARAM_BOUNDS", {})

    ds_dir = _DATA_ROOT / dataset_name
    train_data = _load_parquet(ds_dir / "train-00000-of-00001.parquet")
    test_data = _load_parquet(ds_dir / "test-00000-of-00001.parquet")

    # Determine groups
    if info.group_col in train_data:
        groups = sorted(set(train_data[info.group_col]))
    else:
        groups = ["all_data"]

    # Build per-group data
    group_data_list = []
    for group in groups:
        if info.group_col in train_data:
            train_mask = [i for i, g in enumerate(train_data[info.group_col]) if g == group]
            test_mask = [i for i, g in enumerate(test_data[info.group_col]) if g == group]
        else:
            train_mask = list(range(len(next(iter(train_data.values())))))
            test_mask = list(range(len(next(iter(test_data.values())))))

        if len(train_mask) == 0 or len(test_mask) == 0:
            continue

        X_train = np.array(
            [[train_data[c][i] for c in info.feature_cols] for i in train_mask],
            dtype=np.float64,
        )
        X_test = np.array(
            [[test_data[c][i] for c in info.feature_cols] for i in test_mask],
            dtype=np.float64,
        )
        y_train = np.array(
            [[train_data[c][i] for c in info.target_cols] for i in train_mask],
            dtype=np.float64,
        )
        y_test = np.array(
            [[test_data[c][i] for c in info.target_cols] for i in test_mask],
            dtype=np.float64,
        )
        if y_train.shape[1] == 1:
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]

        all_cols = info.feature_cols + info.target_cols + info.cost_extra_cols
        if info.group_col in train_data:
            all_cols = list(dict.fromkeys([info.group_col] + all_cols))
        cost_train = np.array(
            [info.cost_fn({c: train_data[c][i] for c in all_cols}) for i in train_mask],
            dtype=np.float64,
        )

        group_data_list.append(GroupData(
            group=str(group),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cost_train=cost_train,
        ))

    # Create one task per sl_id, containing all groups
    tasks = []
    for sl_id in law_registry:
        n_p = param_counts[sl_id]
        bounds = param_bounds.get(sl_id, None)
        task = ScalingLawTask(
            task_id=f"{dataset_name}/{sl_id}",
            dataset_name=dataset_name,
            sl_id=sl_id,
            groups=group_data_list,
            model_fn=_ModelFn(dataset_name, sl_id),
            n_params=n_p,
            param_bounds=bounds,
        )
        tasks.append(task)

    return tasks


def load_all_tasks(dataset_names: Optional[List[str]] = None) -> List[ScalingLawTask]:
    if dataset_names is None:
        dataset_names = list(DATASET_REGISTRY.keys())
    tasks = []
    for name in dataset_names:
        tasks.extend(load_tasks_for_dataset(name))
    return tasks
