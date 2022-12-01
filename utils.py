import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader


class LazySuperTuxDataset(Dataset):
    def __init__(
        self,
        path:str = 'data', 
        transform=None,
    ):
        super().__init__()

        # list with paths to folders with episodes
        self.episodes = list(set([k.parent for k in pathlib.Path(path).rglob('*.pt')]))

        # transformation for images
        if transform is None:
            self.transform = transforms.ToTensor() 
        else:
            self.transform = transforms.Compose([
                transform,
                transforms.ToTensor,
            ])

        print(f"Number of episodes: {len(self.episodes)}")
        
    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        """
        returns timestep, 
                image
                velocity
                rotation
                actions 
                reward 
                reward-to-go 
        """
        path_idx = self.episodes[idx]

        imgs = []
        vel = []
        rot = []
        actions = []
        rewards = []
        timesteps = []

        for p in pathlib.Path(path_idx).rglob('*.pt'):
            # only load numeric files which hold state information
            if not (p.name.split('.')[0]).isnumeric():
                continue

            # load dictionary
            d = load_dict(p)

            # save states
            s = d['state']
            imgs.append(self.transform(s['img'])[None])
            vel.append(s['vel'])
            rot.append(s['rot'])
            # save actions
            a = d['action']
            a = [a['steer'], a['acceleration'], a['drift'], a['brake']]
            actions.append(a)
            # save rewards
            r = d['reward']
            rewards.append(r)
            # save timestep
            timesteps.append(int(p.name.split('.')[0]))

        # sort all the timesteps
        t = torch.as_tensor(timesteps, dtype=torch.float32)
        a = torch.as_tensor(actions, dtype=torch.float32)
        r = torch.as_tensor(rewards, dtype=torch.float32)
        img = torch.cat(imgs)
        v = torch.as_tensor(np.array(vel), dtype=torch.float32)
        ro = torch.as_tensor(np.array(rot), dtype=torch.float32)
        # calculate rewards to go
        r_cum = r.cumsum(0)
        rg = r - r_cum + r_cum[-1]

        ord = torch.argsort(t)
        
        return tuple(k[ord] for k in [t,img,v,ro,a,r,rg])


class SuperTuxDataset(Dataset):
    def __init__(
        self,
        path:str = 'data', 
        transform=None,
    ):
        super().__init__()

        # list of states, action, reward-to-go
        imgs = defaultdict(lambda: [])
        vel = defaultdict(lambda: [])
        rot = defaultdict(lambda: [])
        actions = defaultdict(lambda: [])
        rewards = defaultdict(lambda: [])
        timesteps = defaultdict(lambda: [])

        # transformation for images
        if transform is None:
            transform = transforms.ToTensor() 
        else:
            transform = transforms.Compose([
                transform,
                transforms.ToTensor,
            ])
        
        # get paths to all images
        for p in pathlib.Path(path).rglob('*.pt'):
            # only load numeric files which hold state information
            if not (p.name.split('.')[0]).isnumeric():
                continue

            # load dictionary
            d = load_dict(p)

            # save states
            s = d['state']
            imgs[p.parent].append(transform(s['img'])[None])
            vel[p.parent].append(s['vel'])
            rot[p.parent].append(s['rot'])
            # save actions
            a = d['action']
            a = [a['steer'], a['acceleration'], a['drift'], a['brake']]
            actions[p.parent].append(a)
            # save rewards
            r = d['reward']
            rewards[p.parent].append(r)
            # save timestep
            timesteps[p.parent].append(int(p.name.split('.')[0]))

        # sort all the timesteps
        self.data = []
        for k in timesteps.keys():
            t = torch.as_tensor(timesteps[k], dtype=torch.float32)
            a = torch.as_tensor(actions[k], dtype=torch.float32)
            r = torch.as_tensor(rewards[k], dtype=torch.float32)
            img = torch.cat(imgs[k])
            v = torch.as_tensor(vel[k], dtype=torch.float32)
            ro = torch.as_tensor(rot[k], dtype=torch.float32)
            # calculate rewards to go
            r_cum = r.cumsum(0)
            rg = r - r_cum + r_cum[-1]

            ord = torch.argsort(t)
            self.data.append(tuple(k[ord] for k in [t,img,v,ro,a,r,rg]))
        
        print(f"Number of episodes: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        returns timestep, 
                image
                velocity
                rotation
                actions 
                reward 
                reward-to-go 
        """
        return self.data[idx]


class SuperTuxImages(Dataset):
    def __init__(
        self,
        path:str = 'data', 
        train_transform=None, 
        test_transform=None, 
        test=False,
    ):
        super().__init__()

        # list of images
        self.data = []
        
        # transforms to use
        self.test = test
        self.train_transform = train_transform
        self.test_transform = test_transform
        if train_transform is None:
            self.train_transform = lambda x: x 
        if test_transform is None:
            self.test_transform = lambda x: x 
        self.to_tensor = transforms.ToTensor() 
        
        # get paths to all images
        for p in pathlib.Path(path).rglob('*.pt'):
            # only load numeric files which hold state information
            if not (p.name.split('.')[0]).isnumeric():
                continue

            # load dictionary
            img = load_dict(p)['state']['img']

            self.data.append(img)
        
        print(f"Number of images: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.to_tensor(img)
        if self.test:
            img = self.test_transform(img)
        else:
            img = self.train_transform(img)
        return img, img

def split_dataset(
    dataset,
    lengths,
):
    # Get datasets
    lengths = np.floor(np.asarray(lengths) * len(dataset))
    lengths[-1] = len(dataset) - np.sum(lengths[:-1])
    subsets = random_split(
        dataset, 
        lengths.astype(int).tolist(),
        torch.Generator().manual_seed(1234)
    )
    return subsets
        
def load_data(
    dataset, 
    batch_size: int, 
    num_workers: int,
    lengths = [0.7, 0.15,0.15],
):
    subsets = split_dataset(dataset, lengths)
    train = subsets[0]
    return (DataLoader(
        train, 
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    ),) + tuple(DataLoader(
        k, 
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    ) for k in subsets[1:])


# ----------------------------------------------------------------------------------

# add class to save model metrics

# ----------------------------------------------------------------------------------


MODEL_CLASS_KEY = 'model_class'
FOLDER_PATH_KEY = 'path_name'


def save_checkpoint(
    path: str, 
    name: str, 
    epoch: int, 
    optimizer: torch.optim.Optimizer, 
    **kwargs
):
    '''Save a checkpoint of the training parameters
    
    Parameters
    ----------
    path : str
        The path to save the checkpoint to.
    name : str
        The name of the model.
    epoch : int
        The current epoch number
    optimizer : torch.optim.Optimizer
        torch.optim.Optimizer
    loss : float
    '''
    folder_path = f"{path}/{name}"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    d = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    d.update(kwargs)
    torch.save(d, f"{folder_path}/{name}_checkpoint.pt")


def load_checkpoint(path: str, optimizer: torch.optim.Optimizer):
    '''It loads a checkpoint, and then loads the optimizer state
    
    Parameters
    ----------
    path : str
        the path to the folder where the checkpoint is stored
    name : str
        the name of the model
    optimizer : torch.optim.Optimizer
        torch.optim.Optimizer
    
    Returns
    -------
        The dictionary d is being returned.
    
    '''
    folder_path = pathlib.Path(path)
    path = f"{folder_path.absolute()}/{folder_path.name}"
    d = torch.load(f"{path}_checkpoint.pt", map_location='cpu')
    
    # load optimizer state
    optimizer.load_state_dict(d['optimizer'])
    d['optimizer'] = optimizer
    return d


def save_model(
        model: torch.nn.Module,
        folder: Union[pathlib.Path, str],
        model_name: str,
        models_dict:Dict,
        param_dicts: Dict = None,
        save_model: bool = True,
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
    for k, v in models_dict.items():
        if isinstance(model, v):
            model_class = k
            break
    if model_class is None:
        raise Exception("Model class unknown")
    param_dicts[MODEL_CLASS_KEY] = model_class

    # save the dictionary as plain text and pickle
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict", as_str=True)
    save_dict(param_dicts, f"{folder_path}/{model_name}.dict.pickle", as_str=False)


def load_model(
    folder_path: str, 
    models_dict:Dict,
    model_class: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)

    :param folder_path: folder path of the model to be loaded
    :param model_class: one of the model classes in `models_dict` dict. If none, it is obtained from the dictionary
    :return: the loaded model and the dictionary of parameters
    """
    folder_path = pathlib.Path(folder_path)
    path = f"{folder_path.absolute()}/{folder_path.name}"
    # use pickle dictionary
    dict_model = load_dict(f"{path}.dict.pickle")

    # get model class
    if model_class is None:
        model_class = dict_model.get(MODEL_CLASS_KEY)

    # set folder path
    dict_model[FOLDER_PATH_KEY] = str(folder_path)

    return load_model_data(models_dict[model_class](**dict_model), f"{path}.th"), dict_model


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


def dict_to_str(d: Dict):
    str = '{\n'
    for k,v in d.items():
        if isinstance(v, dict):
            v = dict_to_str(v)
        str += f"'{k}':{v},\n"
    return str + '}'


def save_dict(d: Dict, path: str, as_str: bool = False) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    :param as_str: If true, it will save as a string. If false, it will use pickle
    """
    if as_str:
        with open(path, 'w', encoding="utf-8") as file:
            # file.write(str(d))
            file.write(dict_to_str(d))
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


# ----------------------------------------------------------------------------------


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
    torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # make deterministic
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        # for deterministic behavior on cuda >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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


if __name__ == '__main__':
    dataset = SuperTuxDataset()
