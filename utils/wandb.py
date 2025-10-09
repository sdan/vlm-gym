""" From https://github.com/dibyaghosh/jaxrl_m
Taken from https://github.com/kvfrans/lmpo/tree/main/utils

"""
import wandb

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
