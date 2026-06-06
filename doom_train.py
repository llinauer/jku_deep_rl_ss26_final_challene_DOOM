import argparse
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import onnx
import torch
from gymnasium.spaces import Box
from torch import nn

from doom_arena import VizdoomMPEnv
from doom_arena.reward import VizDoomReward
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import TrainingInfoInterface, register_env
from sample_factory.train import run_rl


USE_GRAYSCALE = False
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLAYER_CONFIG = {
    "algo_type": "QVALUE",
    "n_stack_frames": 4,
    "extra_state": ["labels", "depth"],
    "hud": "none",
    "crosshair": True,
    "screen_format": 8 if USE_GRAYSCALE else 0,
}

CHANNELS_PER_FRAME = (1 if USE_GRAYSCALE else 3) + 2
FLATTENED_CHANNELS = CHANNELS_PER_FRAME * PLAYER_CONFIG["n_stack_frames"]


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_training_steps: int
    num_bots: int
    episode_timeout: int


TRAINING_STAGES = [
    CurriculumStage("warmup", 1_000_000, 1, 400),
    CurriculumStage("midgame", 3_000_000, 2, 550),
    CurriculumStage("fullmatch", 10**18, 4, 700),
]


class SimpleCombatReward(VizDoomReward):
    def __init__(self, num_players: int):
        super().__init__(num_players)

    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Tuple[float, ...]:
        self._step += 1
        _ = vizdoom_reward, player_id

        frag_delta = game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"]
        death_delta = game_var["DEATHCOUNT"] - game_var_old["DEATHCOUNT"]
        hit_delta = game_var["HITCOUNT"] - game_var_old["HITCOUNT"]

        return (
            3.0 * frag_delta,
            -1.0 * death_delta,
            1.0 * hit_delta,
            -0.001,
        )


def _first_player_obs(obs: Any):
    if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
        obs = obs[0]
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    return obs


def _flatten_obs(obs: Any) -> np.ndarray:
    obs = _first_player_obs(obs)
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 4:
        c, t, h, w = obs.shape
        obs = obs.reshape(c * t, h, w)
    return obs


def get_stage(training_steps: int) -> CurriculumStage:
    for stage in TRAINING_STAGES:
        if training_steps < stage.max_training_steps:
            return stage
    return TRAINING_STAGES[-1]


class SampleFactoryDoomEnv(gym.Env, TrainingInfoInterface):
    metadata = {}

    def __init__(self, cfg, env_config, render_mode=None):
        gym.Env.__init__(self)
        TrainingInfoInterface.__init__(self)

        self.cfg = cfg
        self.env_config = env_config
        self.render_mode = render_mode
        self.base_seed = int(getattr(cfg, "seed", 0))
        self.stage = None
        self.env = None

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(FLATTENED_CHANNELS, 128, 128),
            dtype=np.float32,
        )
        self.action_space = None
        self._ensure_env(force=True)

    def _training_steps(self) -> int:
        return int(self.training_info.get("approx_total_training_steps", 0))

    def _seed_for_env(self) -> int:
        worker_index = int(getattr(self.env_config, "worker_index", 0))
        vector_index = int(getattr(self.env_config, "vector_index", 0))
        env_id = int(getattr(self.env_config, "env_id", 0))
        return self.base_seed + worker_index * 10_000 + vector_index * 100 + env_id

    def _make_env(self, stage: CurriculumStage) -> VizdoomMPEnv:
        env = VizdoomMPEnv(
            num_players=1,
            num_bots=stage.num_bots,
            bot_skill=0,
            doom_map="ROOM",
            extra_state=PLAYER_CONFIG["extra_state"],
            episode_timeout=stage.episode_timeout,
            n_stack_frames=PLAYER_CONFIG["n_stack_frames"],
            crosshair=PLAYER_CONFIG["crosshair"],
            hud=PLAYER_CONFIG["hud"],
            screen_format=PLAYER_CONFIG["screen_format"],
            reward_fn=SimpleCombatReward(num_players=1),
            seed=self._seed_for_env(),
        )
        for player_env in env.envs:
            player_env.frame_skip = int(self.cfg.env_frameskip)
        return env

    def _ensure_env(self, force: bool = False) -> None:
        next_stage = get_stage(self._training_steps())
        if not force and self.stage is not None and self.stage.name == next_stage.name:
            return
        if self.env is not None:
            self.env.close()
        self.stage = next_stage
        self.env = self._make_env(next_stage)
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.base_seed = int(seed)
        self._ensure_env()
        obs = _flatten_obs(self.env.reset())
        return obs, {"stage": self.stage.name}

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        obs = _flatten_obs(obs)
        reward = float(reward[0] if isinstance(reward, (list, tuple)) else reward)
        info = dict(info)
        info["num_frames"] = int(self.cfg.env_frameskip)
        info["stage"] = self.stage.name
        return obs, reward, bool(done), False, info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def make_sf_doom_env(full_env_name, cfg, env_config, render_mode=None):
    _ = full_env_name
    return SampleFactoryDoomEnv(cfg, env_config, render_mode=render_mode)


def register_sf_doom_env():
    register_env("doom_sf", make_sf_doom_env)


def add_onnx_metadata(onnx_path: Path):
    model = onnx.load(onnx_path)
    meta = model.metadata_props.add()
    meta.key = "config"
    meta.value = json.dumps(PLAYER_CONFIG)
    onnx.save(model, onnx_path)


class ExportableSFDoomPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros((FLATTENED_CHANNELS, 128, 128), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((FLATTENED_CHANNELS, 128, 128), dtype=torch.float32))

        self.conv1 = nn.Conv2d(FLATTENED_CHANNELS, 32, kernel_size=8, stride=4)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.act2 = nn.ELU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.act3 = nn.ELU()
        self.fc = nn.Linear(9216, 256)
        self.act4 = nn.ELU()
        self.policy = nn.Linear(256, 8)

        self.obs_scale = 1.0
        self.obs_subtract_mean = 0.0
        self.norm_eps = 1e-5
        self.norm_clip = 5.0

    def load_from_state_dict(self, state_dict: dict):
        self.running_mean.copy_(
            state_dict["obs_normalizer.running_mean_std.running_mean_std.obs.running_mean"].float()
        )
        self.running_var.copy_(
            state_dict["obs_normalizer.running_mean_std.running_mean_std.obs.running_var"].float()
        )
        mapping = {
            "conv1.weight": "encoder.encoders.obs.enc.conv_head.0.weight",
            "conv1.bias": "encoder.encoders.obs.enc.conv_head.0.bias",
            "conv2.weight": "encoder.encoders.obs.enc.conv_head.2.weight",
            "conv2.bias": "encoder.encoders.obs.enc.conv_head.2.bias",
            "conv3.weight": "encoder.encoders.obs.enc.conv_head.4.weight",
            "conv3.bias": "encoder.encoders.obs.enc.conv_head.4.bias",
            "fc.weight": "encoder.encoders.obs.enc.mlp_layers.0.weight",
            "fc.bias": "encoder.encoders.obs.enc.mlp_layers.0.bias",
            "policy.weight": "action_parameterization.distribution_linear.weight",
            "policy.bias": "action_parameterization.distribution_linear.bias",
        }
        own = self.state_dict()
        for dst, src in mapping.items():
            own[dst].copy_(state_dict[src])

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        if self.obs_subtract_mean != 0.0:
            obs = obs - self.obs_subtract_mean
        if self.obs_scale != 1.0:
            obs = obs / self.obs_scale
        obs = (obs - self.running_mean) / torch.sqrt(self.running_var + self.norm_eps)
        return torch.clamp(obs, -self.norm_clip, self.norm_clip)

    def forward(self, obs: torch.Tensor):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        elif obs.ndim == 4 and obs.shape[1] != FLATTENED_CHANNELS:
            obs = obs.unsqueeze(0)
        if obs.ndim == 5:
            b, c, t, h, w = obs.shape
            obs = obs.reshape(b, c * t, h, w)
        x = self.normalize_obs(obs)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.act4(self.fc(x))
        return self.policy(x)


def export_checkpoint_to_onnx(checkpoint_path: Path, output_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ExportableSFDoomPolicy()
    model.load_from_state_dict(ckpt["model"])
    model.eval()

    dummy_input = torch.zeros(
        (1, CHANNELS_PER_FRAME, PLAYER_CONFIG["n_stack_frames"], 128, 128),
        dtype=torch.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args=dummy_input,
        f=output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
        external_data=False,
    )
    add_onnx_metadata(output_path)
    print(f"Exported ONNX to {output_path}")
    print(f"Checkpoint source: {checkpoint_path}")


def export_best_watcher(checkpoint_dir: Path, output_path: Path, poll_interval_sec: float, stop_event: threading.Event):
    last_exported = None
    while not stop_event.is_set():
        candidates = sorted(glob(str(checkpoint_dir / "best_*")))
        if candidates:
            best_path = candidates[-1]
            if best_path != last_exported:
                try:
                    export_checkpoint_to_onnx(Path(best_path), output_path)
                    last_exported = best_path
                except Exception as exc:
                    print(f"Best ONNX export failed for {best_path}: {exc}")
        stop_event.wait(poll_interval_sec)


def build_sf_cfg(args, train_dir: Path, experiment: str):
    argv = [
        "--algo=APPO",
        "--env=doom_sf",
        f"--experiment={experiment}",
        f"--train_dir={str(train_dir.resolve())}",
        "--restart_behavior=overwrite",
        "--use_rnn=False",
        "--recurrence=32",
        "--serial_mode=False",
        "--async_rl=True",
        "--batched_sampling=True",
        f"--num_workers={args.num_workers}",
        f"--num_envs_per_worker={args.num_envs_per_worker}",
        f"--worker_num_splits={args.worker_num_splits}",
        f"--policy_workers_per_policy={args.policy_workers_per_policy}",
        "--max_policy_lag=1000",
        f"--rollout={args.rollout}",
        f"--batch_size={args.batch_size}",
        f"--num_batches_per_epoch={args.num_batches_per_epoch}",
        f"--num_epochs={args.num_epochs}",
        f"--gamma={args.gamma}",
        f"--gae_lambda={args.gae_lambda}",
        "--with_vtrace=False",
        f"--learning_rate={args.learning_rate}",
        "--exploration_loss=entropy",
        f"--exploration_loss_coeff={args.exploration_loss_coeff}",
        f"--value_loss_coeff={args.value_loss_coeff}",
        f"--max_grad_norm={args.max_grad_norm}",
        f"--reward_scale={args.reward_scale}",
        f"--reward_clip={args.reward_clip}",
        "--normalize_returns=False",
        "--encoder_conv_architecture=convnet_atari",
        f"--encoder_conv_mlp_layers={args.encoder_conv_mlp_layers}",
        "--obs_scale=1.0",
        "--env_gpu_observations=False",
        f"--env_frameskip={args.env_frameskip}",
        f"--save_every_sec={args.save_every_sec}",
        f"--keep_checkpoints={args.keep_checkpoints}",
        f"--experiment_summaries_interval={args.experiment_summaries_interval}",
        f"--train_for_env_steps={args.train_for_env_steps}",
        f"--device={args.device}",
        f"--seed={args.seed}",
    ]
    parser, _ = parse_sf_args(argv)
    return parse_full_cfg(parser, argv)


def write_run_config(run_dir: Path, args):
    run_config = {
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "player_config": PLAYER_CONFIG,
        "curriculum": [stage.__dict__ for stage in TRAINING_STAGES],
        "args": vars(args),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))


def train(args):
    register_sf_doom_env()
    root = Path(__file__).resolve().parent
    train_dir = root / args.train_dir
    experiment = args.experiment or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = train_dir / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Run directory: {run_dir}")
    write_run_config(run_dir, args)

    sf_cfg = build_sf_cfg(args, train_dir, experiment)
    checkpoint_dir = run_dir / "checkpoint_p0"

    stop_event = threading.Event()
    watcher = None
    if not args.no_export_during_training:
        watcher = threading.Thread(
            target=export_best_watcher,
            args=(checkpoint_dir, run_dir / "best_model.onnx", args.export_poll_interval_sec, stop_event),
            daemon=True,
        )
        watcher.start()

    try:
        return run_rl(sf_cfg)
    finally:
        stop_event.set()
        if watcher is not None:
            watcher.join(timeout=max(args.export_poll_interval_sec + 1.0, 5.0))
        candidates = sorted(glob(str(checkpoint_dir / "best_*")))
        if candidates:
            export_checkpoint_to_onnx(Path(candidates[-1]), run_dir / "best_model.onnx")


def export_only(args):
    root = Path(__file__).resolve().parent
    checkpoint = root / args.checkpoint
    output = root / args.output
    export_checkpoint_to_onnx(checkpoint, output)
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "export"], default="train")
    p.add_argument("--train-dir", default="runs")
    p.add_argument("--experiment", default="")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--env-frameskip", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-envs-per-worker", type=int, default=2)
    p.add_argument("--worker-num-splits", type=int, default=2)
    p.add_argument("--policy-workers-per-policy", type=int, default=1)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-batches-per-epoch", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--exploration-loss-coeff", type=float, default=0.01)
    p.add_argument("--value-loss-coeff", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=4.0)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--reward-clip", type=float, default=5.0)
    p.add_argument("--encoder-conv-mlp-layers", type=int, default=256)
    p.add_argument("--train-for-env-steps", type=int, default=10_000_000)
    p.add_argument("--save-every-sec", type=int, default=300)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--experiment-summaries-interval", type=int, default=30)
    p.add_argument("--device", default="gpu")

    p.add_argument("--no-export-during-training", action="store_true")
    p.add_argument("--export-poll-interval-sec", type=float, default=30.0)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output", default="")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        return train(args)
    return export_only(args)


if __name__ == "__main__":
    raise SystemExit(main())
