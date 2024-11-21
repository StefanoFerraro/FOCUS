import os
from pathlib import Path

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import warnings
warnings.filterwarnings("ignore")

import hydra
import numpy as np
import torch
from dm_env import specs
import wandb

from env import RS_TASKS_OBJ, MS_TASKS_OBJ, MW_TASKS_OBJ, DMC_TASKS_OBJ, PRIMAL_TASKS 
from env.make import make

import utils
from logger import Logger
from replay_buffer import ReplayBuffer, make_replay_loader
from online_train import Workspace

class FinetuneWorkspace(Workspace):
    def __init__(self, cfg, savedir=None, workdir=None):
        super().__init__(cfg, maindir=None, workdir=None)

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()["agent"]
            self.agent.init_from(pretrained_agent)

    def train(self):
        super(FinetuneWorkspace, self).train()
        
        if self.cfg.save_ft_model:
            self.save_finetuned_model()

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir) 
        snapshot = (
            snapshot_base_dir
            / f"snapshot_{self.cfg.snapshot_ts}.pt"
        )

        def try_load(seed):
            if not snapshot.exists():
                return None
            with snapshot.open("rb") as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            print(f"Snapshot loaded from: {snapshot}")
            return payload
        else:
            raise Exception(f"Snapshot not found at: {snapshot}")

    @utils.retry
    def save_finetuned_model(self):
        root_dir = Path.cwd()
        snapshot = root_dir / "finetuned_snapshot.pt"
        keys_to_save = ["agent", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_last_model(self):
        root_dir = Path.cwd()
        try:
            snapshot = root_dir / "last_snapshot.pt"
            with snapshot.open("rb") as f:
                payload = torch.load(f)
        except:
            snapshot = root_dir / "second_last_snapshot.pt"
            with snapshot.open("rb") as f:
                payload = torch.load(f)
        for k, v in payload.items():
            setattr(self, k, v)
            if k == "wandb_run_id":
                assert wandb.run is None
                cfg = self.cfg
                exp_name = "_".join(
                    [
                        "Finetune",
                        cfg.agent.name,
                        cfg.env.name,
                        cfg.task,
                    ]
                )
                wandb.init(
                    project=cfg.project_name + "_finetune",
                    group=cfg.agent.name,
                    name=exp_name,
                    id=v,
                    resume="must",
                )

    def setup_wandb(self):
        cfg = self.cfg
        exp_name = "_".join(
            [   "Finetune",
                cfg.agent.name,
                cfg.env.name,
                cfg.task,
            ]
        )
        wandb.init(
            project=cfg.project_name,
            group=cfg.agent.name,
            name=exp_name,
        )
        wandb.config.update(dict(cfg))
        self.wandb_run_id = wandb.run.id

@hydra.main(config_path="configs", config_name="finetune")
def main(cfg):
    from online_finetune import FinetuneWorkspace as W
    root_dir = Path.cwd()

    workspace = W(cfg)
    workspace.root_dir = root_dir
    snapshot = root_dir / "last_snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_last_model()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
    workspace.train()


if __name__ == "__main__":
    main()
