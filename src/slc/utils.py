import json
import os
import tempfile
from pathlib import Path

from slc.constants import DATASET_ROOTS, DatasetNames


def set_tempdir(path) -> None:
    tempfile.tempdir = path


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_deliminted_data(fp, delim=" "):
    lines = []
    with open(fp, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(delim)
            lines.append(parts)
    return lines


def set_env_var(key: str, value: str) -> None:
    os.environ[key] = value


def setup_notebook(
    device: int | list[int] = 0,
    tmp_dir: str | Path = "/home/carlyn.1/tmp",
    hf_cache_dir: str | Path = "/local/scratch/carlyn.1/hf_cache",
):
    """_summary_

    Args:
        device (Union[int, list[int]], optional): list of cuda devices to use. Defaults to 0.
        tmp_dir (str, optional): where to set the temporary directory. Defaults to "/home/carlyn.1/tmp".
    """

    # SPECIFY GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(device, list):
        device = ",".join(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"

    # Set HF cache_dir
    os.environ["HF_HOME"] = str(Path(hf_cache_dir))

    # set tempdir
    set_tempdir("/home/carlyn.1/tmp")


def get_hostname() -> str:
    hostname = os.uname()[1]
    return hostname


def get_dataset_root(dset_name: DatasetNames) -> Path:
    hostname = get_hostname()
    return DATASET_ROOTS[hostname][dset_name]
