import copy
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import pathlib

from torch.utils.data import Dataset
from agents.cnn import KartCNN

MODEL_CLASS = dict(
    cnn=KartCNN,
)
MODEL_CLASS_KEY = 'model_class'
FOLDER_PATH_KEY = 'path_name'


class SuperTuxDataset(Dataset):
    pass
# todo dataset


# ----------------------------------------------------------------------------------


def save_model(
        model: torch.nn.Module,
        folder: Union[pathlib.Path, str],
        model_name: str,
        param_dicts: Dict = None,
        save_model: bool = True
) -> None:
    """
    Saves the model so it can be loaded after

    :param model_name: name of the model to be saved (non including extension)
    :param folder: path of the folder where to save the model
    :param param_dicts: dictionary of the model parameters that can later be used to load it
    :param model: model to be saved
    :param save_model: If true the model and dictionary will be saved, otherwise only the dictionary will be saved
    """
    # create folder if it does not exist
    folder_path = f"{folder}/{model_name}"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    # save model
    if save_model:
        torch.save(model.state_dict(), f"{folder_path}/{model_name}.th")

    # save dict
    if param_dicts is None:
        param_dicts = {}

    # get class of the model
    model_class = None
    for k, v in MODEL_CLASS.items():
        if isinstance(model, v):
            model_class = k
            break
    if model_class is None:
        raise Exception("Model class unknown")
    param_dicts[MODEL_CLASS_KEY] = model_class

    # save the dictionary as plain text and pickle
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict", as_str=True)
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict.pickle", as_str=False)


def load_model(folder_path: pathlib.Path, model_class: Optional[str] = None) -> Tuple[torch.nn.Module, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)

    :param folder_path: folder path of the model to be loaded
    :param model_class: one of the model classes in `MODEL_CLASS` dict. If none, it is obtained from the dictionary
    :return: the loaded model and the dictionary of parameters
    """
    path = f"{folder_path.absolute()}/{folder_path.name}"
    # use pickle dictionary
    dict_model = load_dict(f"{path}.dict.pickle")

    # get model class
    if model_class is None:
        model_class = dict_model.get(MODEL_CLASS_KEY)

    # set folder path
    dict_model[FOLDER_PATH_KEY] = str(folder_path)

    return load_model_data(MODEL_CLASS[model_class](**dict_model), f"{path}.th"), dict_model


def load_model_data(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Loads a model than has been previously saved

    :param model_path: path from where to load model
    :param model: model into which to load the saved model
    :return: the loaded model
    """
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def save_pickle(obj, path: Union[str, Path]):
    """
    Saves an object with pickle

    :param obj: object to be saved
    :param save_path: path to the file where it will be saved
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]):
    """
    Loads an object with pickle from a file

    :param path: path to the file where the object is stored
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_dict(d: Dict, path: str, as_str: bool = False) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    :param as_str: If true, it will save as a string. If false, it will use pickle
    """
    if as_str:
        with open(path, 'w', encoding="utf-8") as file:
            file.write(str(d))
    else:
        save_pickle(d, path)


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file (plain text or pickle)

    :param path: path where the dictionary was saved
    :return: the loaded dictionary
    """
    try:
        return load_pickle(path)
    except pickle.UnpicklingError as e:
        # print(e)
        pass

    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        s = file.read()
        return dict(literal_eval(s))


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    :param seed: seed for the random generators
    """
    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # # make deterministic
        # torch.backends.cudnn.determinstic = True
        # torch.backends.cudnn.benchmark = False

        # # for deterministic behavior on cuda >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self): 
        return self.__dict__
    def __setstate__(self, d): 
        self.__dict__.update(d)

    def copy(self):
        return dotdict(super().copy())
