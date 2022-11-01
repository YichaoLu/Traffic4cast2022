import json
import pickle
from pathlib import Path
from typing import Any, Union, Optional


def to_json(obj: Any, path: Union[str, Path], indent: Optional[int] = None):
    with open(path, 'w+') as fp:
        json.dump(obj, fp=fp, indent=indent)


def read_json(path: Union[str, Path]):
    with open(path, 'r') as fp:
        return json.load(fp=fp)


def to_pickle(obj: Any, path: Union[str, Path]):
    with open(path, 'wb+') as file:
        pickle.dump(obj, file=file)


def read_pickle(path: str):
    with open(path, 'rb') as file:
        return pickle.load(file=file)
