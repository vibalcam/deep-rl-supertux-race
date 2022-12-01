from typing import Dict, List
from agents.abstractAgent import AbstractAgent
from agents import cnn as CNN
from agents.cnn import KartCNN
from baseline.aimPointController import AimPointController
from utils import SuperTuxDataset, load_model, save_model, set_seed, load_data, save_checkpoint, load_checkpoint, split_dataset, LazySuperTuxDataset
from transformers import DecisionTransformerModel, DecisionTransformerConfig
import torch
from typing import Dict
import torch.utils.tensorboard as tb
import time
import numpy as np
from os import path
import torchmetrics
from torchvision.transforms import ToTensor, Resize
from environments.pytux import PyTux
from environments.pytux import tracks as pytux_tracks
from transformers import get_linear_schedule_with_warmup


PREFIX_TRAINING_PARAMS = "tr_par_"
cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')


class AgentTransformerModel(torch.nn.Module):
    def __init__(
        self, 
        max_ep_len: int = 2000,
        cnn_model_path:str = 'saved/ae/grayscale1_best',
        **kwargs,
    ):
        super().__init__()
        self.dict_model = locals().copy()
        del self.dict_model['self']
        
        # Some parameters
        self.hidden_size = 128
        max_length = 2048
        
        # Load decision transformer model
        config = DecisionTransformerConfig(
            state_dim=self.hidden_size*3, # img, vel, rot
            act_dim=4,
            max_ep_len=max_ep_len,
            action_tanh=False,
            n_positions=max_length*3,
        )
        self.model = DecisionTransformerModel(config)
        self.model.config.max_length = max_length
        
        # embeddings for state
        self.embed_vel = torch.nn.Sequential(
            torch.nn.Linear(3, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
        )
        self.embed_rot = torch.nn.Sequential(
            torch.nn.Linear(4, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(self.hidden_size),
        )
        cnn, d = load_model(cnn_model_path, CNN.MODEL_CLASS)
        if d['tr_par_test_transform'] is not None:
            self.preprocess_img = torch.nn.Sequential(
                Resize((96,128)),
                d['tr_par_test_transform'],
            )
        else:
            self.preprocess_img = Resize((96,128))
        self.embed_img = torch.nn.Sequential(
            cnn.encoder,
            torch.nn.BatchNorm1d(self.hidden_size),
        )

    # Workaround to get device
    def _get_device(self):
        return next(self.parameters()).device
    
    def forward(
        self, 
        timestep, 
        image,
        velocity,
        rotation,
        actions,
        reward,
        reward_to_go,
    ):
        # reshape input
        batch_size = image.shape[0]
        image,velocity,rotation = [k.reshape(-1,*k.shape[2:]) for k in [image,velocity,rotation]]

        # preprocess image
        image = self.preprocess_img(image)  # B*t, C...
        # get embeddings for state
        image = self.embed_img(image)       # B*t, hidden
        velocity = self.embed_vel(velocity) # B*t, hidden
        rotation = self.embed_rot(rotation) # B*t, hidden
        # combine state embeddings
        state = torch.cat([image,velocity,rotation], dim=1)  # B, 3*hidden
        state = state.reshape(batch_size, -1, 3*self.hidden_size)

        # only last k previous timesteps
        # state = state[:, -self.model.config.max_length :]
        # actions = actions[:, -self.model.config.max_length :]
        # returns_to_go = returns_to_go[:, -self.model.config.max_length :]
        # timesteps = timesteps[:, -self.model.config.max_length :]

        # pad to sequence length and get attention mask
        attention_mask = torch.ones(*state.shape[:2], device=self._get_device())
        # padding = self.max_length - state.shape[1]
        # attention_mask = torch.cat([torch.ones(batch_size, padding), torch.ones(*state.shape[:2])]).to(device)
        # state = torch.cat([torch.zeros(batch_size, padding, state.shape[2]), state])
        # actions = torch.cat([torch.zeros(batch_size, padding, actions.shape[2]), actions])
        # timestep = torch.cat([torch.zeros(batch_size, padding, timestep.shape[2]), timestep])
        # state = torch.cat([torch.zeros(batch_size, padding, state.shape[2]), state])
    
        # transform inputs to correct shape and type
        # reward = reward[...,None]
        # reward_to_go = reward_to_go[...,None]
        timestep = timestep.long()

        # use decision transformer model
        pred = self.model(
            states=state,
            actions=actions,
            rewards=reward,
            returns_to_go=reward_to_go,
            timesteps=timestep,
            attention_mask=attention_mask,
            return_dict=False,
        )
        # we are only interested in actions
        steer, acc, drift, brake = [k[...,0] for k in pred[1].split(1, dim=2)]

        # scale actions
        steer = torch.tanh(steer)
        acc = torch.sigmoid(acc)
        # drift = (drift > 0).int()
        # brake = (brake > 0).int()

        return steer, acc, drift, brake


class TransformerController(AbstractAgent):
    def __init__(self, 
        env, 
        options: Dict = None,
        model: AgentTransformerModel = None,
        target_reward:float = 400,
        use_cpu:bool = False,
        allow_drift:bool = False,
        fixed_velocity: float = 0.5,
        tracks: List[str] = ['lighthouse'],
        **kwargs
    ):
        super().__init__(env, options)
        self.act_dim = len(self.env.action_space)

        self.device = cpu_device if use_cpu else cuda_device
        self.model = model if model is not None else AgentTransformerModel()
        if not isinstance(model, AgentTransformerModel):
            raise Exception("Model is not a AgentTransformerModel")
        self.model = self.model.to(self.device)
        self.to_tensor = ToTensor()
        self.target_reward = target_reward

        # expand functionality
        self.allow_drift = allow_drift
        self.fixed_velocity = fixed_velocity
        self.tracks = tracks

    def reset(self, options: Dict = None):
        ret = super().reset(options)
        state = ret[0]

        # get starting values
        self.rew_to_go = torch.as_tensor(self.target_reward, dtype=torch.float32).reshape(1,1,1)
        self.imgs = self.to_tensor(state['img'])[None,None]
        self.vels = torch.from_numpy(state['vel']).float()[None,None]
        self.rots = torch.from_numpy(state['rot']).float()[None,None]
        self.timesteps = torch.as_tensor(0, dtype=torch.long).reshape(1,1)
        self.rews = torch.zeros(1,0,1).float()
        self.actions = torch.zeros(1,0,self.act_dim)

        return ret

    def clear(self):
        self.rew_to_go = None
        self.imgs = None
        self.vels = None
        self.rots = None
        self.timesteps = None
        self.rews = None
        self.actions = None

        torch.cuda.empty_cache()
    
    def step(self, action):
        ret = super().step(action)
        state, reward, _, _, _ = ret

        # update action taken and reward
        self.actions[0,-1] = torch.as_tensor([action['steer'], action['acceleration'], action['drift'], action['brake']])
        self.rews[0,-1] = reward

        # append new state info
        self.imgs = torch.cat([self.imgs, self.to_tensor(state['img'])[None,None]], dim=1)
        self.vels = torch.cat([self.vels, torch.from_numpy(state['vel']).float()[None,None]], dim=1)
        self.rots = torch.cat([self.rots, torch.from_numpy(state['rot']).float()[None,None]], dim=1)
        self.timesteps = torch.cat([self.timesteps, torch.as_tensor(self.cur_stats.steps, dtype=torch.long).reshape(1,1)], dim=1)

        # update reward to go
        # todo maybe scale reward to go
        self.rew_to_go = torch.cat([self.rew_to_go, (self.rew_to_go[0,-1] - reward).reshape(1,1,1)], dim=1)

        return ret

    @torch.no_grad()
    def act(self, state):
        super().act(state)

        # concatenate actions for prediction
        self.actions = torch.cat([self.actions, torch.zeros(1,1,self.act_dim)], dim=1)
        self.rews = torch.cat([self.rews, torch.zeros(1,1,1)], dim=1)

        # steer, acc, drift, brake
        steer,acc,drift,brake = self.model(
            timestep=self.timesteps.to(self.device),
            image=self.imgs.to(self.device),
            velocity=self.vels.to(self.device),
            rotation=self.rots.to(self.device),
            actions=self.actions.to(self.device),
            reward=self.rews.to(self.device),
            reward_to_go=self.rew_to_go.to(self.device),
            device=self.device,
        )

        # todo change brake
        return PyTux.Action(
            acceleration=acc[0,-1].item() if self.fixed_velocity is None else self.fixed_velocity,
            brake=False,
            steer=steer[0,-1].item(),
            drift=(drift[0,-1] > 0).bool().item() if self.allow_drift else False,
        )

    def eval_mode(self, eval:bool = True):
        self.model.train(not eval)

    def train(
        self,
        save_name:str = None,
        dataset_path: str = 'data',
        checkpoint_path:str = None,
        seed:int = 1234,
        log_dir: str = './logs',
        save_path: str = './saved/trans',
        lr: float = 1e-3,
        optimizer_name: str = "adamw",
        n_epochs: int = 60,
        batch_size: int = 1,
        num_workers: int = 0,
        scheduler_mode: str = 'min_val_loss',
        debug_mode: bool = False,
        steps_save: int = 1,
        steps_eval:int = 50,
        use_cpu: bool = False,
        device=None,
        scheduler_patience: int = 10,
    ):
        model = self.model
        dict_model = self.model.dict_model

        # cpu or gpu used for training if available (gpu much faster)
        if device is None:
            device = cuda_device if not (use_cpu or debug_mode) else cpu_device
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
        ]}

        valid_logger = tb.SummaryWriter(path.join(log_dir, str(type(model).__name__), save_name, f"valid"), flush_secs=1)
        train_logger = tb.SummaryWriter(path.join(log_dir, str(type(model).__name__), save_name, f"train"), flush_secs=1)

        # Model
        dict_model.update(dict_param)
        model = model.to(device)

        # Loss
        var_list = ["steer", "acc", "drift", "brake"]
        var_mask = torch.as_tensor([True, self.fixed_velocity is None, self.allow_drift, False]).int().to(device)
        loss_list = [
            torch.nn.MSELoss().to(device),
            torch.nn.MSELoss().to(device),
            torch.nn.BCEWithLogitsLoss().to(device),
            lambda x,y: torch.as_tensor(0).to(device),
        ]

        # load data
        dataset = LazySuperTuxDataset(path=dataset_path)
        # no test since test will be the environment itself
        train_loader, val_loader = load_data(dataset, batch_size, num_workers, lengths=[0.80,0.20])

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise Exception("Optimizer not configured")

        if scheduler_mode in ["min_loss", 'min_val_loss']:
            scheduler = get_linear_schedule_with_warmup(optimizer,num_training_steps=n_epochs, num_warmup_steps=100)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)
            scheduler_metric = torchmetrics.MinMetric()
        elif scheduler_mode in []:
            scheduler = get_linear_schedule_with_warmup(optimizer,num_training_steps=n_epochs, num_warmup_steps=100)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=scheduler_patience)
            scheduler_metric = torchmetrics.MaxMetric()
        else:
            raise Exception("Scheduler not configured")

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

        # Global best metrics
        eval_best = torchmetrics.MaxMetric()

        for epoch in range(old_epoch, n_epochs):
            # for epoch in (p_bar := trange(n_epochs, leave = True)):
            # p_bar.set_description(f"{name_model} -> best in {dict_model['epoch']}: {dict_model['val_acc']}")

            # Start training
            train_metric = []
            # Set training mode
            model.train()

            for d in train_loader:
                timestep,img,vel,rot,act,rew,rew_to_go = [k.to(device) for k in d]

                # Compute loss on training and update parameters
                # steer, acc, drift, brake
                pred = model(
                    timestep=timestep, 
                    image=img,
                    velocity=vel,
                    rotation=rot,
                    actions=act,
                    reward=rew[...,None],
                    reward_to_go=rew_to_go[...,None],
                )
                # calculate loss over steering only
                loss_train = torch.stack([l(pred[k],act[:,:,k]) for k,l in enumerate(loss_list)]) * var_mask
                loss = loss_train.sum()

                # Do back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Add train loss and accuracy
                train_metric.append(loss_train.cpu().detach())

            # Evaluate the model
            val_metric = []
            eval_results = None
            # Set evaluation mode
            model.eval()

            with torch.no_grad():
                for d in val_loader:
                    timestep,img,vel,rot,act,rew,rew_to_go = [k.to(device) for k in d]

                    # steer, acc, drift, brake
                    pred = model(
                        timestep=timestep, 
                        image=img,
                        velocity=vel,
                        rotation=rot,
                        actions=act,
                        reward=rew[...,None],
                        reward_to_go=rew_to_go[...,None],
                    )

                    # Add loss and accuracy
                    loss_val = torch.stack([l(pred[k],act[:,:,k]) for k,l in enumerate(loss_list)]) * var_mask
                    val_metric.append(loss_val.cpu().detach())
                
                if epoch % steps_eval == steps_eval - 1:
                    # run evaluate for each track
                    self.options.render_every = 1
                    eval_results = 0
                    for tr in self.tracks:
                        self.options.save_video=f"{save_path}/{save_name}_videos/{epoch}_{tr}.mp4"
                        self.options.track = tr
                        eval_results += self.evaluate().cum_reward
                    eval_results /= len(self.tracks)

                    # Update best eval
                    eval_best.update(eval_results)
                    # Clear to save memory
                    self.clear()

            # calculate mean metrics
            train_part_loss = torch.stack(train_metric).mean(0).numpy()
            train_loss = train_part_loss.sum().item()
            val_part_loss = torch.stack(val_metric).mean(0).numpy()
            val_loss = val_part_loss.sum().item()
            
            # calculate time/epoch
            toc = time.time()
            epoch_time_metric.update(toc - tic, weight=epoch)   # give more weight to recent values
            epoch_time = epoch_time_metric.compute()
            tic = toc

            # Get scheduler metric
            scheduler_vars = dict(
                min_loss=train_loss,
                min_val_loss=val_loss,
                eval_reward=eval_results,
            )
            met = scheduler_vars[scheduler_mode]
            # Update and get whether best result has been improved
            scheduler_metric.update(met)
            is_better = (met == scheduler_metric.compute()).item()
            # Step the scheduler to change the learning rate
            # scheduler.step(met)
            scheduler.step()

            # Log metrics
            global_step += 1
            if train_logger is not None:
                # train log
                suffix = 'train'
                train_logger.add_scalar(f'loss_{suffix}', train_loss, global_step=global_step)

                # validation log
                suffix = 'val'
                valid_logger.add_scalar(f'loss_{suffix}', val_loss, global_step=global_step)

                # log part metrics
                for k,v in enumerate(var_list):
                    train_logger.add_scalar(f'loss_train_{v}', train_part_loss[k].item(), global_step=global_step)
                    valid_logger.add_scalar(f'loss_val_{v}', val_part_loss[k].item(), global_step=global_step)

                # test log
                if eval_results is not None:
                    valid_logger.add_scalar(f'reward_{suffix}', eval_results, global_step=global_step)

                # learning rate log
                train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            # Save the model
            if (is_periodic := epoch % steps_save == steps_save - 1) or is_better:
                d = dict_model

                # training info
                d["epoch"] = epoch
                d['last_epoch_sec'] = epoch_time
                # metrics
                d["train_loss"] = train_loss
                d["val_loss"] = val_loss
                for k,v in enumerate(var_list):
                    d[f"train_loss_{v}"] = train_part_loss[k].item()
                    d[f"val_loss_{v}"] = val_part_loss[k].item()
                d['eval_reward'] = eval_results

                # name_path = str(list(name_dict.values()))[1:-1].replace(',', '_').replace("'", '').replace(' ', '')
                if save_name is None:
                    name_path = int((np.random.rand() + train_loss) * 10000)
                else:
                    name_path = save_name  

                # save model if improves best result
                if is_better:
                    save_model(model, save_path, f"{name_path}_best", MODEL_CLASS, param_dicts=d)
                    save_checkpoint(
                        path=save_path, 
                        name=f"{name_path}_best", 
                        epoch=epoch, 
                        optimizer=optimizer,
                    )
                    print(f"New best {epoch}/{n_epochs} in {epoch_time:0.1f}s: loss {train_loss:0.6f}, val loss {val_loss:0.6f}, best_val_reward {eval_best.compute()}")

                # save model periodically
                if is_periodic:
                    save_model(model, save_path, f"{name_path}_{epoch}", MODEL_CLASS, param_dicts=d)
                    save_checkpoint(
                        path=save_path, 
                        name=f"{name_path}_{epoch}", 
                        epoch=epoch, 
                        optimizer=optimizer,
                    )
                    print(f"{epoch}/{n_epochs} in {epoch_time:0.1f}s: loss {train_loss:0.6f}, val loss {val_loss:0.6f}, best_val_reward {eval_best.compute()}")
            
            # empty cuda cache
            torch.cuda.empty_cache()


MODEL_CLASS = {
    'dec_trans': AgentTransformerModel,
}


# todo train for a bit and then use combined to augment
# class GuidedTransformerController(TransformerController):
#     def __init__(self, env, options: Dict = None, model: AgentTransformerModel = None, target_reward: float = 400, device=None):
#         super().__init__(env, options, model, target_reward, device)

#     def act(self, state):
#         AimPointController

#         return super().act(state)


if __name__ == '__main__':
    import gym
    from environments.pytux import PyTux
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_drift', '-ndr', action='store_true', help="Model with drift disabled and fixed velocity")
    parser.add_argument('--fixed_velocity', '-fv', action='store_true', help="Model with fixed velocity")
    args = parser.parse_args()

    options = PyTux.default_params.copy()
    options.update(dict(
        track='lighthouse',
        ai=None,
        render_every=0,
        n_karts=1,
        n_laps=1,
        no_pause_render=True,
    ))
    env = gym.make('PyTux-v0', options=options)
    
    if args.no_drift:
        # Model with drift disabled and fixed velocity
        model = TransformerController(
            env, 
            options=options,
            target_reward=500,
            allow_drift=False,
            model=AgentTransformerModel(cnn_model_path = 'saved/ae/color1_best'),
        )
        model.train(
            dataset_path='dataNoDrift',

            save_name="decTransColor1", 
            save_path='./saved/trans/colorNoDrift',

            use_cpu=False, 
            num_workers=0,
            scheduler_patience=20,
            scheduler_mode = 'min_val_loss',
            steps_save=20,
            batch_size=1,
            lr=1e-4,
            n_epochs=300,
            steps_eval=20,
        )
    elif args.fixed_velocity:
        # Model with fixed velocity
        model = TransformerController(
            env, 
            options=options,
            target_reward=500,
            allow_drift=True,
            model=load_model('./saved/trans/colorNoDrift/decTransColor1_best', MODEL_CLASS)[0],
        )

        model.train(
            dataset_path='data',

            save_name="decTransColor_drift1", 
            save_path='./saved/trans/colorDrift',

            use_cpu=False, 
            num_workers=0,
            scheduler_patience=20,
            scheduler_mode = 'min_val_loss',
            steps_save=20,
            batch_size=1,
            lr=1e-4,
            # lr=6e-5,
            n_epochs=300,
            steps_eval=20,
        )
    else:
        # Model that predicts steer, drift, and acceleration
        model = TransformerController(
            env, 
            options=options,
            target_reward=500,
            allow_drift=True,
            fixed_velocity=None,

            model=AgentTransformerModel(cnn_model_path = 'saved/ae/color1_best'),
            # model=load_model('./saved/trans/colorDrift_tmp2/decTransColor_drift1_best', MODEL_CLASS)[0],
        )

        model.train(
            dataset_path='data',

            save_name="decTransColor_drift_acc1", 
            save_path='./saved/transMultiple/colorDriftAcc',

            # checkpoint_path='./saved/trans/decTrans1_best',
            use_cpu=False, 
            num_workers=0,
            scheduler_patience=20,
            scheduler_mode = 'min_val_loss',
            steps_save=20,
            batch_size=1,
            lr=1e-4,
            # n_epochs=300,
            n_epochs=500,
            steps_eval=20,
        )

        # model = TransformerController(
        #     env, 
        #     options=options,
        #     target_reward=500,
        #     allow_drift=True,
        #     fixed_velocity=None,
        #     tracks=pytux_tracks,

        #     model=AgentTransformerModel(cnn_model_path = 'saved/ae/color1_best'),
        #     # model=load_model('./saved/trans/colorDrift_tmp2/decTransColor_drift1_best', MODEL_CLASS)[0],
        # )

        # model.train(
        #     dataset_path='dataMultiple',

        #     save_name="decTransMultiple1", 
        #     save_path='./saved/transMultiple/colorDriftAcc',

        #     # checkpoint_path='./saved/transMultiple/decTrans1_best',
        #     use_cpu=False, 
        #     num_workers=0,
        #     scheduler_patience=20,
        #     scheduler_mode = 'min_val_loss',
        #     steps_save=20,
        #     batch_size=1,
        #     lr=1e-4,
        #     # n_epochs=300,
        #     n_epochs=500,
        #     steps_eval=20,
        # )

    env.close()
