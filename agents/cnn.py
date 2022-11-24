import pathlib
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tb
from torch.nn import MSELoss
from utils import SuperTuxImages, load_model, save_model, set_seed, load_data, save_checkpoint, load_checkpoint
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale
from torch.utils.data import DataLoader, random_split
import time
import numpy as np
from os import path
import torchmetrics

PREFIX_TRAINING_PARAMS = "tr_par_"
# SCHEDULER_MODES = dict(
#     min_loss=('train_loss', min),
# )

class KartCNN(torch.nn.Module):
    class BlockConv(torch.nn.Module):
        def __init__(
            self, 
            n_input, 
            n_output, 
            kernel_size=3, 
            stride=1, 
            residual: bool = True
        ):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=stride,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=1,
                                bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
                # torch.nn.MaxPool2d(2, stride=2)
            )
            self.residual = residual
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.downsample is None else self.downsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    class BlockUpConv(torch.nn.Module):
        def __init__(
            self, 
            n_input, 
            n_output, 
            stride=1, 
            residual: bool = True
        ):
            super().__init__()
            # if kernel == 2:
            #     temp = torch.nn.ConvTranspose2d(n_input, 
            # n_output, kernel_size=2, stride=1, bias=False)
            # elif kernel == 3:
            #     # temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=stride,
            #     #                                 output_padding=1, bias=False)
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False)
            # elif kernel == 4:
            #     temp = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=4, padding=1, stride=1, bias=False)
            # else:
            #     raise Exception()

            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=stride, output_padding=1,
                                         bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1, stride=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.residual = residual
            self.upsample = None
            if stride != 1 or n_input != n_output:
                self.upsample = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=1, stride=stride, output_padding=1,
                                             bias=False),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            if self.residual:
                identity = x if self.upsample is None else self.upsample(x)
                return self.net(x) + identity
            else:
                return self.net(x)

    def __init__(
        self, 
        dim_layers=[16, 32, 64, 128], 
        n_input=3, 
        input_normalization: bool = True,
        residual: bool = True,
        input_dim=(96,128),
        hidden_dim:int = 128,
        **kwargs,
    ):
        super().__init__()
        self.dict_model = locals().copy()
        del self.dict_model['self']

        n_output = n_input

        if input_normalization:
            self.norm = torch.nn.BatchNorm2d(n_input)
        else:
            self.norm = None

        # self.min_size = 2 ** (len(dim_layers) + 1)
        self.min_size = 2 ** (len(dim_layers)) * 4

        c = dim_layers[0]
        list_encoder = [torch.nn.Sequential(
            # torch.nn.Conv2d(n_input, c, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.Conv2d(n_input, c, kernel_size=7, padding=3, stride=4, bias=False),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU()
        )]
        list_decoder = [torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c, n_output, kernel_size=7,
                                     padding=3, stride=4, output_padding=3),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(5, 5, kernel_size=1)
        )]
        for l in dim_layers[1:]:
            list_encoder.append(self.BlockConv(c, l, stride=2, residual=residual))
            list_decoder.insert(0, self.BlockUpConv(l, c, stride=2, residual=residual))
            c = l
        
        # add linear layers after encoder and beginning decoder
        # in_linear_dim = np.asarray(input_dim) / (2 ** len(dim_layers))
        in_linear_dim = (np.asarray(input_dim) / (2 ** (len(dim_layers) - 1)) / 4).astype(int).tolist()
        in_linear_size = int((np.multiply(*in_linear_dim) * c).item())

        # self.encoder_linear = torch.nn.Sequential(
        list_encoder.append(torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_linear_size, hidden_dim),
            torch.nn.ReLU(),
        ))
        list_decoder.insert(0, torch.nn.Sequential(
            torch.nn.Linear(hidden_dim,in_linear_size),
            torch.nn.ReLU(),
            torch.nn.Unflatten(dim=1, unflattened_size=[c] + in_linear_dim),
        ))

        self.encoder = torch.nn.Sequential(*list_encoder)
        self.decoder = torch.nn.Sequential(*list_decoder)

    def forward(self, x):
        # Input Normalization
        if self.norm is not None:
            x = self.norm(x)

        h = x.size(2)
        w = x.size(3)

        if h < self.min_size or w < self.min_size:
            resize = torch.zeros([
                x.size(0),
                x.size(1),
                self.min_size if h < self.min_size else h,
                self.min_size if w < self.min_size else w
            ])
            # h_start = int((self.min_size - h) / 2 if h < self.min_size else 0)
            # w_start = int((self.min_size - w) / 2 if w < self.min_size else 0)
            # resize[:, :, h_start:h_start + h, w_start:w_start + w] = x
            resize[:, :, :h, :w] = x
            x = resize

        # Calculate
        x = self.encoder(x)
        # x = self.encoder_linear(x)
        x = self.decoder(x)

        return x[:, :, :h, :w]

    def train_model(
        self,
        save_name:str = None,
        dataset_path: str = 'data',
        checkpoint_path:str = None,
        seed:int = 1234,
        log_dir: str = './logs',
        save_path: str = './saved/ae',
        lr: float = 1e-3,
        optimizer_name: str = "adamw",
        n_epochs: int = 60,
        batch_size: int = 32,
        num_workers: int = 4,
        scheduler_mode: str = 'min_val_loss',
        debug_mode: bool = False,
        steps_save: int = 1,
        use_cpu: bool = False,
        device=None,
        scheduler_patience: int = 10,
        train_transform=Compose([
            Grayscale(),
            RandomHorizontalFlip(0.5),
        ]),
        test_transform=Grayscale(),
    ):
        model = self
        dict_model = self.dict_model

        # cpu or gpu used for training if available (gpu much faster)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
        print(device)

        # Set seed
        set_seed(seed)

        # Tensorboard
        # dictionary of training parameters
        dict_param = {f"{PREFIX_TRAINING_PARAMS}{k}": v for k, v in locals().items() if k in [
            "seed",
            "lr",
            "optimizer_name",
            "n_epochs",
            "batch_size",
            "num_workers",
            "scheduler_mode",
            "steps_save",
            "scheduler_patience",
            "train_transform",
            "test_transform",
        ]}

        # dictionary to set model name
        # name_dict = dict_model.copy()
        # name_dict.update(dict_param)
        # model name
        # name_model = '/'.join([
        #     str(name_dict)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
        # ])

        valid_logger = tb.SummaryWriter(path.join(log_dir, str(type(model).__name__), f"valid_{save_name}"), flush_secs=1)
        train_logger = tb.SummaryWriter(path.join(log_dir, str(type(model).__name__), f"train_{save_name}"), flush_secs=1)
        # valid_logger = train_logger
        # global_step = 0

        # Model
        dict_model.update(dict_param)
        model = model.to(device)

        # Loss
        loss = torch.nn.MSELoss().to(device)

        # load data
        dataset = SuperTuxImages(path=dataset_path, train_transform=train_transform, test_transform=test_transform)
        train_loader, val_loader, _ = load_data(dataset, batch_size, num_workers)

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise Exception("Optimizer not configured")

        if scheduler_mode in ["min_loss", 'min_val_loss']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)
        elif scheduler_mode in []:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience)
        else:
            raise Exception("Optimizer not configured")

        # Load checkpoint if given
        checkpoint = {}
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(
                path=checkpoint_path,
                optimizer=optimizer
            )
        old_epoch = checkpoint.get('epoch',-1) + 1
        global_step = old_epoch
        n_epochs += old_epoch

        # Initialize epoch timer
        tic = time.time()
        epoch_time_metric = torchmetrics.MeanMetric()

        for epoch in range(old_epoch, n_epochs):
            # for epoch in (p_bar := trange(n_epochs, leave = True)):
            # p_bar.set_description(f"{name_model} -> best in {dict_model['epoch']}: {dict_model['val_acc']}")

            # train_loss = []
            train_mse_metric = torchmetrics.MeanSquaredError(compute_on_cpu=True)
            train_mse_metric.to(device)

            # Start training: train mode
            model.train()
            dataset.test = False
            
            for d in train_loader:
                x,y = [k.to(device) for k in d]

                # Compute loss on training and update parameters
                pred = model(x)
                loss_train = loss(pred, y)

                # Do back propagation
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                # Add train loss and accuracy
                # train_loss.append(loss_train.cpu().detach().numpy())
                train_mse_metric.update(pred,y)

            # Evaluate the model
            # val_loss = []
            val_mse_metric = torchmetrics.MeanSquaredError(compute_on_cpu=True)
            val_mse_metric.to(device)

            # Set evaluation mode
            model.eval()
            dataset.test = True

            with torch.no_grad():
                for d in val_loader:
                    x,y = [k.to(device) for k in d]

                    pred = model(x)

                    # Add loss and accuracy
                    # val_loss.append(loss(pred, y).cpu().detach().numpy())
                    val_mse_metric.update(pred, y)

            # calculate mean metrics
            # train_loss = np.mean(train_loss)
            train_loss = train_mse_metric.compute()
            train_mse_metric.reset()
            # val_loss = np.mean(val_loss)
            val_loss = val_mse_metric.compute()
            val_mse_metric.reset()
            
            # calculate time/epoch
            toc = time.time()
            epoch_time_metric.update(toc - tic, weight=epoch)   # give more weight to recent values
            epoch_time = epoch_time_metric.compute()
            tic = toc

            # Step the scheduler to change the learning rate
            # is_better = False
            # scheduler_info = SCHEDULER_MODES[scheduler_mode]
            # met =
            is_better = False
            if scheduler_mode == "min_loss":
                met = train_loss
                if (best_met := dict_model.get('train_loss', None)) is not None:
                    is_better = met < best_met
                else:
                    dict_model['train_loss'] = met
                    is_better = True
            elif scheduler_mode == "min_val_loss":
                met = val_loss
                if (best_met := dict_model.get('val_loss', None)) is not None:
                    is_better = met < best_met
                else:
                    dict_model['val_loss'] = met
                    is_better = True
            elif scheduler_mode is None:
                met = None
            else:
                raise Exception("Unknown scheduler mode")

            if met is not None:
                scheduler.step(met)

            # log metrics
            global_step += 1
            if train_logger is not None:
                # train log
                suffix = 'train'
                train_logger.add_scalar(f'loss_{suffix}', train_loss, global_step=global_step)
                # log_confussion_matrix(train_logger, train_cm, global_step, suffix=suffix)

                # validation log
                suffix = 'val'
                valid_logger.add_scalar(f'loss_{suffix}', val_loss, global_step=global_step)
                # log_confussion_matrix(valid_logger, val_cm, global_step, suffix=suffix)

                # learning rate log
                train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            # Save the model
            if (is_periodic := epoch % steps_save == steps_save - 1) or is_better:
                # d = dict_model if is_better else dict_model.copy()
                d = dict_model.copy()

                # training info
                d["epoch"] = epoch
                d['last_epoch_sec'] = epoch_time
                # metrics
                d["train_loss"] = train_loss
                d["val_loss"] = val_loss

                # name_path = str(list(name_dict.values()))[1:-1].replace(',', '_').replace("'", '').replace(' ', '')
                if save_name is None:
                    name_path = int((np.random.rand() + train_loss) * 10000)
                else:
                    name_path = save_name    

                if is_better:
                    save_model(model, save_path, f"{name_path}_best", MODEL_CLASS, param_dicts=d)
                    save_checkpoint(
                        path=save_path, 
                        name=f"{name_path}_best", 
                        epoch=epoch, 
                        optimizer=optimizer,
                    )
                    print(f"New best {epoch}/{n_epochs} in {epoch_time:0.1f}s: loss {train_loss:0.5f}, val loss {val_loss:0.5f}")

                # if periodic save, then include epoch
                if is_periodic:
                    save_model(model, save_path, f"{name_path}_{epoch + 1}", MODEL_CLASS, param_dicts=d)
                    save_checkpoint(
                        path=save_path, 
                        name=f"{name_path}_{epoch + 1}", 
                        epoch=epoch, 
                        optimizer=optimizer,
                    )
                    print(f"{epoch}/{n_epochs} in {epoch_time:0.1f}s: loss {train_loss:0.5f}, val loss {val_loss:0.5f}")


# def log_confussion_matrix(logger, confussion_matrix: ClassConfusionMatrix, global_step: int, suffix=''):
#     """
#     Logs the data in the confussion matrix to a logger
#     :param logger: tensorboard logger to use for logging
#     :param confussion_matrix: confussion matrix from where the metrics are obtained
#     :param global_step: global step for the logger
#     """
#     logger.add_scalar(f'acc_global_{suffix}', confussion_matrix.global_accuracy, global_step=global_step)
#     logger.add_scalar(f'acc_avg_{suffix}', confussion_matrix.average_accuracy, global_step=global_step)
#     logger.add_scalar(f'mcc_{suffix}', confussion_matrix.matthews_corrcoef, global_step=global_step)
#     logger.add_scalar(f'rmse_{suffix}', confussion_matrix.rmse, global_step=global_step)
#     for idx, k in enumerate(confussion_matrix.class_accuracy):
#         logger.add_scalar(f'acc_class_{idx}_{suffix}', k, global_step=global_step)


MODEL_CLASS = dict(
    cnn=KartCNN,
)


if __name__ == '__main__':
    # cnn = KartCNN(n_input=1)
    # cnn.train_model(
    #     dataset_path='data',
    #     save_name="grayscale1", 
    #     use_cpu=False, 
    #     num_workers=0,
    #     scheduler_patience=10,
    #     train_transform=Compose([
    #         Grayscale(),
    #         RandomHorizontalFlip(0.5),
    #     ]),
    #     test_transform=Grayscale(),
    #     steps_save=np.inf,
    #     batch_size=32,
    #     lr=1e-3,
    #     n_epochs=400,
    #     # checkpoint_path='./saved/ae/grayscale3_best',
    # )

    cnn = KartCNN(n_input=3)
    cnn.train_model(
        dataset_path='data',
        save_name="color1", 
        use_cpu=False, 
        num_workers=0,
        scheduler_patience=10,
        train_transform=Compose([
            RandomHorizontalFlip(0.5),
        ]),
        test_transform=None,
        steps_save=np.inf,
        batch_size=32,
        lr=1e-3,
        n_epochs=400,
        # checkpoint_path='./saved/ae/color2_best',
    )
