# SpiRob Vision Extension

A CNN-based visual reinforcement learning extension for the
[spirob-mjlab](https://github.com/YADlin/spirob-mjlab) egg-to-bucket task.

This package adds a camera to the SpiRob simulation and trains a policy to
perform the task **directly from RGB images** - no simulator segmentation, no
ground-truth object coordinates, no hand-computed object positions. It
registers a new task, Mjlab-SpiRob-EggToBucket-Vision, alongside the
existing Mjlab-SpiRob-EggToBucket-Stage1, which it leaves unmodified.

---

## Status

**The vision pipeline is complete and verified.** Camera → CNN → PPO runs
end-to-end on GPU; observations, action shapes, and checkpointing are all
confirmed working.

**The manipulation task is not solved.** Five reward configurations and an
extended-duration run were evaluated; none produced a successful egg
placement. The unmodified non-vision baseline exhibits the same zero-success
behaviour on the same machine, which locates the difficulty in the task
formulation rather than in the vision system.

See [docs/report.md](docs/report.md) for the full analysis, including each
reward configuration, its diagnosed failure mode, and the supporting
measurements.

---

## What this adds

| Aspect | Base Stage-1 task | This extension |
| --- | --- | --- |
| Object perception | egg_to_bucket vector from simulator state | 320×240 RGB camera image |
| Policy input | 93-dim state vector | (3, 240, 320) image + 48-dim proprioception |
| Policy network | MLP | Spatial-softmax CNN encoder + MLP |
| Actions | 2 cable-length targets | unchanged |
| Reward / terminations | base terms | base terms + tip-to-egg delta shaping |

The base task's egg_to_bucket and robot_keypoints_xy observation terms
are deliberately excluded: both supply object information that would remove
any need for the CNN to learn to perceive. Proprioceptive terms (tendon
length and velocity, touch sensors, last action) are retained - they describe
the robot's own state, which a physical robot also has access to.

Reward and termination terms continue to use privileged simulator state. That
is training infrastructure, not agent perception, and does not compromise the
constraint that the *policy* learn from pixels.

---

## Installation

This is an extension, not a standalone package. It requires the base
repository.

### 1. Set up the base repository

Follow the [base repository's](https://github.com/YADlin/spirob-mjlab)
instructions. In summary:

```bash
git clone https://github.com/YADlin/spirob-mjlab.git
cd spirob-mjlab
uv python install 3.12.3
uv sync --frozen
```

### 2. Add this extension

```bash
# from the spirob-mjlab root
cp -r /path/to/spirob-vision-extension/vision src/spirob_mjlab/
cp /path/to/spirob-vision-extension/test_vision_pipeline.py .
```

### 3. Register the task

Append one line to src/spirob_mjlab/__init__.py:

```bash
echo "from spirob_mjlab import vision  # noqa: F401" >> src/spirob_mjlab/__init__.py
```

This is the only modification made to any inherited file.

### 4. Verify

```bash
uv run list-envs | grep Vision          # expects Mjlab-SpiRob-EggToBucket-Vision
uv run python test_vision_pipeline.py   # expects PASS
```

The pre-flight test checks image shape [1, 3, 240, 320], values within
[0, 1], and action shape [1, 2]. It runs on CPU — no GPU required.

---

## Training

```bash
uv run train Mjlab-SpiRob-EggToBucket-Vision \
  --env.scene.num-envs 16 \
  --agent.max-iterations 500 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name vision-run
```

16 parallel environments rather than the base task's 64: per-environment
rendering plus CNN forward/backward is considerably heavier than state-based
training. Raise this if GPU memory allows.

For GPU training without a local setup, see
[notebooks/spirob_vision_cnn_colab.ipynb](notebooks/spirob_vision_cnn_colab.ipynb),
which clones the base repository, installs this extension, and runs training
on Colab.

### Checking a configuration before training

Reward weights are set programmatically, so it is worth confirming the active
configuration before spending GPU time - particularly after editing:

```bash
uv run python -c "
from spirob_mjlab.vision.vision_env_cfg import spirob_vision_env_cfg
cfg = spirob_vision_env_cfg(play=True)
for name, term in cfg.rewards.items():
    print(f'{name}: {term.weight}')
"
```

The committed configuration (v3) has four terms:

```
egg_delta_progress: 2.0
inside_bucket: 25.0
egg_fell_penalty: -10.0
tip_to_egg_progress: 0.3
```

---

## Reading results

**Do not judge progress by the summed mean reward.** With shaping terms
present, its sign and magnitude are dominated by term weights rather than task
progress. In the v1 configuration below, reward rose steadily to +2.38 while
task success remained at exactly zero throughout.

Use the per-term panels instead:

| Metric | Interpretation |
| --- | --- |
| Episode_Termination/success_egg_inside_bucket | The only direct measure of task success |
| Episode_Reward/egg_delta_progress | Signed egg movement toward the bucket - distinguishes correct from incorrect direction |
| Episode_Termination/egg_fell | Fraction of episodes ending with the egg displaced from the pedestal |
| Episode_Termination/time_out | Fraction reaching the 800-step limit; high with flat reward means inactivity |
| Entropy loss, action std | Whether learning is occurring at all. Values at initialisation (≈3.11, ≈1.15) indicate no effective gradient |

---

## Reward configurations evaluated

Each was run for 100 iterations at 16 environments, changing one aspect at a
time, and diagnosed from per-term panels.

| Ver | Change | Outcome |
| --- | --- | --- |
| v1 | Static proximity reward exp(-d/s) | Reward 0 → +2.38, entropy 3.11 → 0.06, episodes always 800 steps, success 0 - the policy held a fixed pose near the egg and farmed proximity reward without touching it |
| v2 | Delta progress + undirected contact bonus | Contact bonus → ~17, egg_delta_progress negative - any touch was rewarded regardless of direction, so the egg was pushed away as readily as toward |
| **v3** | Contact bonus removed, directed egg progress made dominant | egg_fell 77%, egg_delta_progress −0.19, entropy still moving - correct incentive structure, but contact forceful enough to topple the egg. **Committed configuration** |
| v4 | v3 + doubled fall penalty | Entropy and action std returned to initialisation - nearly all outcomes heavily negative, so no gradient distinguished better from worse |
| v5 | v3 + softened shaping + action-rate penalty | egg_fell → ~100%, egg_delta_progress −2.2 - weakened directional guidance without reducing contact force |

v3 is committed as the best configuration tested. A 500-iteration run of v3
(§4.4 of the report) did not improve on its 100-iteration result, indicating
the limitation is not sample budget alone.

---

## Layout

```
spirob-vision-extension/
├── vision/                       # copy into src/spirob_mjlab/
│   ├── __init__.py               # task registration
│   ├── vision_mdp.py             # image observation + shaping reward
│   ├── vision_env_cfg.py         # camera, observations, reward config
│   └── vision_rl_cfg.py          # CNN PPO configuration
├── test_vision_pipeline.py       # pre-flight check (CPU, no GPU needed)
├── notebooks/
│   └── spirob_vision_cnn_colab.ipynb
└── docs/
    ├── report.md                 # full technical report
    └── results/                  # training logs and TensorBoard captures
```

---

## Design notes

**Camera placement.** The camera is mounted on robot/robot_base rather than
the arm tip. The tip (link_021) has the smallest radius of curvature and
swings through large angles while curling, producing an unstable view; the
fixed base yields a consistent viewpoint in which scene objects remain in
approximately the same image region, which is considerably easier for a
convolutional encoder to exploit.

**Non-invasive extension.** spirob_vision_env_cfg() calls the base
configuration function and mutates the returned object rather than editing
source files. The base task remains registered and runnable; the total
footprint on inherited code is a single import line.

**End-to-end CNN training.** The CNN has no auxiliary perception objective
and no labelled supervision. Its filters are shaped entirely by PPO's task
reward - it learns to see because seeing improves return.

---

## Acknowledgements

Built on [spirob-mjlab](https://github.com/YADlin/spirob-mjlab) by YADlin,
which implements the Stage-1 egg-to-bucket task and the SpiRob model used
here.

The robot design follows Wang, Freris & Wei, *SpiRobs: Logarithmic
spiral-shaped robots for versatile grasping across scales*, Device 3, 100646
(2025). Note that the paper demonstrates grasping via hand-designed
piecewise-linear cable actuation sequences, not learned control.
