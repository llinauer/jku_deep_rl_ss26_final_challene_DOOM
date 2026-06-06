# `jku.wad`
<div align="center">
    <picture>
    <img src="media/doom_cover.jpg" alt="DOOM" width="300"/>
    </picture>
</div>

---

`jku.wad` is the final challenge for the 2025 deep reinforcement learning course at JKU.

The environment is based on the 1993 first person shooter __DOOM__.
It is set up as a deathmatch, and can be played as a single agent against bots, or with multiple agents against each other.
- __Actions:__ `Discrete(8)`, simplified game buttons
- __Observations:__ `192x256` RGB game frames

## Setup

### Using uv
```bash
git clone <your-repo-url> doom_challenge
cd doom_challenge
uv venv --python 3.11
uv sync
```

This creates `.venv/` and installs the exact dependencies from `pyproject.toml`.

## Training

Start training with the Sample Factory APPO setup:
```bash
uv run python doom_train.py --experiment <experiment_name> --seed 1337
```

Artifacts are written to:
```text
runs/<experiment_name>/
```

During training, the current best checkpoint is exported automatically to:
```text
runs/<experiment_name>/best_model.onnx
```


## Acknowledgements
- [VizDoom](https://github.com/Farama-Foundation/ViZDoom), game interface for RL.
- [Arena](https://github.com/tencent-ailab/Arena), adapted as a multi-agent DOOM environment.
