import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
import os
from src.utils import get_parser, nondefault_trainer_args, load_model_from_config, instantiate_from_config, \
    WrappedDataset, DataModuleFromConfig, SetupCallback, CUDACallback, ImageLogger, LearningRateMonitor, ManualCheckpointSaver
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

def emergency_fix_sd_loading(config, ckpt_path):    
    # Load SD checkpoint
    sd_ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    
    # Create model
    model = instantiate_from_config(config.model)
    
    # Get model state dict
    model_state = model.state_dict()
    
    print(f"SD checkpoint: {len(sd_ckpt)} params")
    print(f"Your model: {len(model_state)} params")
    
    # Try common SD key mappings
    successful_loads = 0
    
    for model_key in model_state.keys():
        possible_sd_keys = [
            model_key,  # Direct match
            f"model.{model_key}",  # Add model prefix
            model_key.replace("model.diffusion_model.", "model.diffusion_model."),
        ]
        
        for sd_key in possible_sd_keys:
            if sd_key in sd_ckpt:
                if model_state[model_key].shape == sd_ckpt[sd_key].shape:
                    model_state[model_key].copy_(sd_ckpt[sd_key])
                    successful_loads += 1
                    break
    
    print(f"✅ Successfully loaded {successful_loads}/{len(model_state)} parameters")
    
    # Load back into model
    model.load_state_dict(model_state)
    
    if successful_loads == 0:
        print("❌ TOTAL FAILURE - No parameters loaded!")
        print("Your config is completely incompatible with SD 1.4")
        return None
    
    return model


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstripd("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""

        if opt.datadir_in_name:
            now = os.path.basename(os.path.normpath(opt.data_root)) + now
            
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp (distributed data parallel) if not set
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        config.model.params.personalization_config.params.embedding_manager_ckpt = opt.embedding_manager_ckpt
        config.model.params.personalization_config.params.placeholder_tokens = opt.placeholder_tokens

        if opt.init_word:
            config.model.params.personalization_config.params.initializer_words[0] = opt.init_word

        if opt.actual_resume:
            # model = load_model_from_config(config, opt.actual_resume)
            model = emergency_fix_sd_loading(config, opt.actual_resume)
        else:
            model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                    "project": "LSAST",
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }

        # default_logger_cfg = default_logger_cfgs["testtube"]
        default_logger_cfg = default_logger_cfgs["wandb"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # default checkpoint configs
        # default_modelckpt_cfg = {
        #     "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        #     "params": {
        #         "dirpath": ckptdir,
        #         "filename": "{epoch:06}",
        #         "verbose": True,
        #         "save_last": True,
        #     }
        # }
        # if hasattr(model, "monitor"):
        #     print(f"Monitoring {model.monitor} as checkpoint metric.")
        #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
        #     default_modelckpt_cfg["params"]["save_top_k"] = 1

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        # modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        
        # if version.parse(pl.__version__) < version.parse('1.4.0'):
        #     trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "manual_checkpoint_saver": {
                "target": "src.utils.ManualCheckpointSaver",
                "params": {
                    "save_dir": ckptdir,
                    "filename": "{epoch:06}.ckpt",
                }
            }
        }
        # if version.parse(pl.__version__) >= version.parse('1.4.0'):
        #     default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        # If define callbacks in config, this callbacks will overwrite the defaults
        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 1,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["max_steps"] = trainer_opt.max_steps

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        config.data.params.train.params.data_root = opt.data_root
        config.data.params.validation.params.data_root = opt.data_root
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            gpus = lightning_config.trainer.gpus
            if isinstance(gpus, str):
                ngpu = len(gpus.strip(",").split(','))
            elif isinstance(gpus, int):
                ngpu = gpus
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # # allow checkpointing via USR1
        # def melk(*args, **kwargs):
        #     # run all checkpoint hooks
        #     if trainer.global_rank == 0:
        #         print("Summoning checkpoint.")
        #         ckpt_path = os.path.join(ckptdir, "melk.ckpt")
        #         trainer.save_checkpoint(ckpt_path)


        # def divein(*args, **kwargs):
        #     if trainer.global_rank == 0:
        #         import pudb;
        #         pudb.set_trace()


        # import signal

        # signal.signal(signal.SIGUSR1, melk)
        # signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            # try:
            trainer.fit(model, data)
            # except Exception:
            #     melk()
            #     raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
            
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if Trainer.global_rank == 0:
            print(trainer.profiler.summary())

