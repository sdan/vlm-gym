""" From https://github.com/dibyaghosh/jaxrl_m
Taken from https://github.com/kvfrans/lmpo/tree/main/utils

"""
import wandb

from typing import Any, Dict, Iterable, List

import tempfile
import absl.flags as flags
import ml_collections
from  ml_collections.config_dict import FieldReference
import datetime
import time
import numpy as np
import os

wandb_config = ml_collections.ConfigDict({
    'project': "lmpo",
    'name': 'lmpo-run',
    'entity': FieldReference(None, field_type=str),
})

_ENV_METRIC_SUMMARIES = {
    "env/stage_acc": "mean",
    "env/city_correct": "mean",
    "env/distance_km": "mean",
    "env/within_25km": "mean",
    "env/within_100km": "mean",
    "env/within_500km": "mean",
    "env/distance_p50": "mean",
    "env/distance_p90": "mean",
    "env/format_ok": "mean",
    "env/parsed_field_count": "mean",
    "env/coord_quality_score": "mean",
    "env/final_reward": "mean",
    "env/stage_idx": "last",
}

_TRACKED_MEAN_KEYS = {
    "stage_acc",
    "city_correct",
    "distance_km",
    "format_ok",
    "parsed_field_count",
    "final_reward",
}

_COORD_COMPONENT_KEYS = ["has_coords", "coords_in_range", "coords_within_tolerance"]
_ALLOWED_ENV_KEYS = set(_TRACKED_MEAN_KEYS) | {"stage_idx"} | set(_COORD_COMPONENT_KEYS)
_DISTANCE_THRESHOLDS = [25, 100, 500]
_DISTANCE_PERCENTILES = [50, 90]


def get_flag_dict():
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict

def setup_wandb(hyperparam_dict, entity=None, project="jaxtransformer", group=None, name=None,
    unique_identifier="", offline=False, random_delay=0, run_id='None', **additional_init_kwargs):
    if "exp_descriptor" in additional_init_kwargs:
        # Remove deprecated exp_descriptor
        additional_init_kwargs.pop("exp_descriptor")
        additional_init_kwargs.pop("exp_prefix")

    if not unique_identifier:
        if random_delay:
            time.sleep(np.random.uniform(0, random_delay))
        unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_identifier += f"_{np.random.randint(0, 1000000):06d}"
        flag_dict = get_flag_dict()
        if 'seed' in flag_dict:
            unique_identifier += f"_{flag_dict['seed']:02d}"

    if name is not None:
        name = name.format(**{**get_flag_dict(), **hyperparam_dict})

    name = name.replace("/", "_")

    if group is not None and name is not None:
        experiment_id = f"{name}_{unique_identifier}"
    elif name is not None:
        experiment_id = f"{name}_{unique_identifier}"
    else:
        experiment_id = None

    # check if dir exists.
    if os.path.exists("/nfs/wandb"):
        wandb_output_dir = "/nfs/wandb"
    else:
        wandb_output_dir = tempfile.mkdtemp()
    print(wandb_output_dir)
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=hyperparam_dict, project=project, entity=entity, tags=tags, group=group, dir=wandb_output_dir,
        id=experiment_id, name=name, settings=wandb.Settings(
            start_method="thread",
            _disable_stats=False,
        ), mode="offline" if offline else "online", save_code=True,
    )
    init_kwargs.update(additional_init_kwargs)

    if run_id != 'None': # Resume a run
        init_kwargs.update({
            "id": run_id,
            "resume": "must",
        })

    run = wandb.init(**init_kwargs)

    wandb.config.update(get_flag_dict(), allow_val_change=True)

    wandb_config = dict(
        exp_prefix=group,
        exp_descriptor=name,
        experiment_id=experiment_id,
    )
    wandb.config.update(wandb_config, allow_val_change=True)
    return run


def define_env_metrics():
    """Register env-related metrics once the W&B run is active."""
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        for metric, summary in _ENV_METRIC_SUMMARIES.items():
            if summary:
                wandb.define_metric(metric, summary=summary)
            else:
                wandb.define_metric(metric)
    except Exception:
        # Best-effort registration; continue if W&B is unavailable.
        pass


def _flatten_numeric(values: Any) -> List[float]:
    """Collect numeric scalars from arbitrarily nested env info."""
    result: List[float] = []
    if isinstance(values, dict):
        for v in values.values():
            result.extend(_flatten_numeric(v))
        return result
    if isinstance(values, (list, tuple, set)):
        for v in values:
            result.extend(_flatten_numeric(v))
        return result
    if hasattr(values, "shape"):
        arr = np.asarray(values)
        if arr.size == 0:
            return result
        return arr.reshape(-1).astype(np.float32).tolist()
    if np.isscalar(values) and not isinstance(values, str):
        result.append(float(values))
    return result


def summarize_env_metrics(env_infos: Dict[str, Iterable[Any]]) -> Dict[str, float]:
    """Aggregate env info values into scalar metrics for logging."""
    metrics: Dict[str, float] = {}
    numeric_arrays: Dict[str, np.ndarray] = {}

    for key, raw_values in env_infos.items():
        if key not in _ALLOWED_ENV_KEYS:
            continue
        flattened = _flatten_numeric(raw_values)
        if not flattened:
            continue
        arr = np.asarray(flattened, dtype=np.float32)
        numeric_arrays[key] = arr
        if key in _TRACKED_MEAN_KEYS:
            metrics[f"env/{key}"] = float(arr.mean())
        elif key == "stage_idx":
            metrics["env/stage_idx"] = float(arr[-1])

    distances = numeric_arrays.get("distance_km")
    if distances is not None and distances.size > 0:
        for thresh in _DISTANCE_THRESHOLDS:
            metrics[f"env/within_{thresh}km"] = float(np.mean(distances <= thresh))
        for percentile in _DISTANCE_PERCENTILES:
            metrics[f"env/distance_p{percentile}"] = float(np.percentile(distances, percentile))

    coord_arrays = [numeric_arrays[k] for k in _COORD_COMPONENT_KEYS if k in numeric_arrays]
    if coord_arrays:
        weights = np.arange(1, len(coord_arrays) + 1, dtype=np.float32)
        coord_means = np.array([arr.mean() for arr in coord_arrays], dtype=np.float32)
        coord_score = float(np.dot(coord_means, weights) / weights.sum())
        metrics["env/coord_quality_score"] = coord_score

    return metrics
