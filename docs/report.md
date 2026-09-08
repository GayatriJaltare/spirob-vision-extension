# Camera Integration and CNN-Based Visual Reinforcement Learning for SpiRob

---

**Task:** Mjlab-SpiRob-EggToBucket-Vision
**Base task:** Mjlab-SpiRob-EggToBucket-Stage1 (unmodified)
**Environment:** mjlab 1.4.0, MuJoCo / MuJoCo Warp 3.8.1, Warp 1.14.0, rsl-rl-lib 5.2.0, Python 3.12.3

---

## 1. Objective

Integrate and simulate a camera mounted on an existing SpiRob simulation in
mjlab, and establish a camera-vision pipeline usable for perception and
reinforcement learning.

The scope was subsequently extended, at the supervisor's direction, to a
CNN-based visual RL policy: the robot should learn the egg-to-bucket task
directly from camera RGB images, with no simulator segmentation, no
ground-truth object coordinates, and no manually computed egg position.

## 2. Background

SpiRob is a soft continuum manipulator whose body discretises a logarithmic
spiral into tapering units, actuated by cables (Wang, Freris & Wei, *Device*,
2025). The inherited simulation implements a Stage-1 manipulation task: a
fixed-base, two-cable SpiRob with 21 compliant elements must move an egg from
a fixed pedestal position into a fixed bucket.

The base task's policy observation is 93-dimensional and includes
egg_to_bucket - the egg-to-bucket displacement vector read directly from
simulator state. This is privileged information unavailable to a physical
robot. Replacing it with learned perception from camera images is the purpose
of this work.

## 3. Implementation

### 3.1 Code organisation

All work is contained in a separate vision/ package that layers onto the
base repository without modifying it:

```
src/spirob_mjlab/
├── __init__.py              # base file, one import line appended
├── env_cfgs.py              # base, unmodified
├── mdp.py                   # base, unmodified
├── rl_cfg.py                # base, unmodified
├── entities.py              # base, unmodified
└── vision/                  # this work
    ├── __init__.py          # task registration
    ├── vision_mdp.py        # image observation + shaping reward
    ├── vision_env_cfg.py    # camera, observations, reward configuration
    └── vision_rl_cfg.py     # CNN PPO configuration
```

The vision environment configuration calls the base configuration function
and mutates the returned object rather than editing source. The base task
remains registered and runnable; the total footprint on inherited code is one
import line.

### 3.2 Camera

A CameraSensorCfg named spirob_camera is attached to robot/robot_base
at pos=(0.0, -0.25, 0.0), quat=(0.301, 0.954, 0.0, 0.0), fovy=90,
producing 320×240 RGB frames.

Base mounting was selected over tip mounting after evaluation. The arm tip
(link_021) has the smallest radius of curvature and swings through large
angles during curling, producing an unstable view. The fixed base yields a
consistent viewpoint in which scene objects remain in approximately the same
image region - substantially easier for a convolutional encoder to exploit.
Camera pose was tuned iteratively until robot, egg, and bucket were all
reliably framed.

### 3.3 Observation design

| Group | Contents | Shape |
| --- | --- | --- |
| actor / critic | tendon length, tendon velocity, last action, 42 touch sensors | (48,) |
| camera | normalised RGB image, channel-first | (3, 240, 320) |

The base task's egg_to_bucket and robot_keypoints_xy terms are excluded.
Both supply object or shape information that would remove any need for the
CNN to learn to perceive. Proprioceptive terms are retained: they describe
the robot's own state, not the object's, and are available to a physical
robot.

Reward and termination terms continue to use privileged simulator state.
This is training infrastructure rather than agent perception, and does not
compromise the constraint that the policy learn from pixels.

### 3.4 Policy architecture

Actor and critic both use mjlab's SpatialSoftmaxCNNModel, configured with
obs_groups={"actor": ("actor", "camera"), "critic": ("critic", "camera")}.
The CNN (two convolutional layers, 16 and 32 channels, kernels 5 and 3,
stride 2, ELU activation, spatial softmax) encodes the image group; its
output is concatenated with the proprioceptive vector and passed to a
(128, 128) MLP. The CNN is trained end-to-end by PPO - its filters are shaped
by task reward alone, with no auxiliary perception objective or labelled
supervision.

### 3.5 Verification

A standalone pre-flight test (test_vision_pipeline.py) confirms the chain
camera → preprocessed tensor → environment step → two cable actions, checking
image shape [1, 3, 240, 320], value range within [0, 1], and action shape
[1, 2]. Training runs on Google Colab GPU; the vision/ package is
supplied from private storage rather than committed to the base repository.

**The pipeline is complete and functional.** All components verified: camera
renders correctly, observations have correct shapes, the CNN executes within
PPO without numerical failure, and checkpoints are written.

## 4. Training results

### 4.1 Initial finding: no learning signal

The first training runs produced mean reward of exactly 0.00, mean value loss
0.0000, entropy frozen at its initialisation value of 3.11, and mean episode
length 800.00 - every episode running the full 8-second timeout.

Diagnosis: the base reward pays out only when the egg moves toward the bucket
(egg_delta_progress), on success (inside_bucket), or on failure
(egg_fell_penalty). A random policy never contacts the egg, so all three
terms are identically zero, no gradient exists, the policy never changes, and
contact never occurs. The condition is self-sustaining.

Supporting measurements:

- 2,000 random-action steps produced zero touch-sensor activations; egg
  displacement 3 mm, consistent with settling rather than contact.
- A targeted contact test using a deliberately chosen pose achieved
  touch = 1.0 with 4.3 mm egg displacement, confirming contact is
  physically achievable.
- Summed link offsets total 0.445 m against an egg 0.158 m from the base;
  reach is not the limiting factor.

This failure mode was anticipated in the base task's own source comments,
which note that the sparse object-motion signal may prove insufficient and
that reach/contact shaping would then be required.

### 4.2 Baseline control

The unmodified, non-vision Stage-1 baseline was run under its documented
R002 conditions (256 environments, 500 iterations, seed 42) on the same
machine. It also produced mean reward 0.00, episode length 800, and zero
successes.

This is the most important control in the study: the failure to learn is
present in the inherited task independently of the vision pipeline, and is
therefore not attributable to camera integration or the CNN policy.

### 4.3 Reward configurations tested

Five configurations were evaluated, each at 100 iterations with 16 parallel
environments, changing one aspect at a time and diagnosing from per-term
TensorBoard panels rather than summed mean reward.

| Ver | Change | Outcome |
| --- | --- | --- |
| v1 | Static proximity reward exp(-d/s), weight 1.0 | Reward 0 → +2.38; entropy 3.11 → 0.06; episodes always 800 steps; success 0 |
| v2 | Delta progress (0.5) + undirected contact bonus (0.2) | Contact bonus → ~17; egg_delta_progress negative; egg moved for the first time |
| v3 | Contact bonus removed; egg_delta_progress 1.0 → 2.0; tip shaping 0.5 → 0.3 | egg_fell 77% of episodes; egg_delta_progress −0.19; entropy 3.00, std 1.09 |
| v4 | v3 + egg_fell_penalty −10 → −20 | Episode length 448 → 276; entropy 3.12, std 1.15 — both at initialisation |
| v5 | v3 + softened tip shaping + action-rate penalty | egg_fell → ~100%; egg_delta_progress −2.2 |

**v1** established that the pipeline can learn: reward rose and entropy
collapsed. However, TensorBoard showed inside_bucket and
egg_delta_progress both flat at zero while the proximity term saturated
the policy had converged on a fixed pose near the egg, accumulating proximity
reward over the full episode without ever touching it. This is reward
hacking: a continuous per-step reward for *being* close is more profitable
than a single terminal reward for succeeding.

**v2** replaced the static term with a delta formulation rewarding only
*reduction* in distance, mirroring the base task's own egg_delta_progress
design. Hovering ceased and the egg moved for the first time. The added
contact bonus rewarded any touch irrespective of direction, and
egg_delta_progress went negative - the policy pushed the egg away from the
bucket as readily as toward it.

**v3** removed the contact bonus and made directed egg progress dominant.
This is the correct incentive structure, and remains the best configuration
tested. Its failure mode is physical rather than incentive-based: contact
became forceful enough to topple the egg in 77% of episodes.

**v4** attempted to suppress this by doubling the fall penalty. Entropy and
action standard deviation returned to their initialisation values — learning
ceased entirely. When nearly every outcome is heavily negative, no gradient
distinguishes better from worse; this is the mirror image of the original
zero-reward problem.

**v5** attempted to address the cause rather than the symptom, softening the
reach reward and adding an action-rate smoothness penalty. egg_fell rose
toward 100% and egg_delta_progress fell to −2.2: weakening the directional
signal removed guidance without reducing contact force.

### 4.4 Training duration

Since all configurations were judged at 100 iterations, v3 was rerun for 500
iterations (256,000 steps, 61 minutes, 16 environments) to test whether the
limitation was sample budget.

Final iteration: mean reward 2.28, mean value loss 9.99, **entropy 3.1205,
action standard deviation 1.15** - both at initialisation values - and mean
episode length 278.68. The extended run did not improve on the 100-iteration
result; the policy did not converge toward any consistent behaviour.

## 5. Discussion

The camera and CNN pipeline meets its specification. Failure to solve the
manipulation task arises upstream of it.

Eliminated as explanations, each with supporting evidence: pipeline defects;
egg unreachability; absence of a functioning learning mechanism (v1
demonstrates the policy responds strongly when reward is available);
incorrect reward direction (isolated and corrected in v2–v3); and
insufficient training duration (§4.4). The vision components are further
exonerated by the baseline control in §4.2.

Two explanations remain open.

**Task difficulty.** Precise manipulation of a small object on a narrow
pedestal, using two cables driving a compliant 21-segment arm, and requiring
final placement within a 1.5 cm tolerance, may not be discoverable by
undirected exploration at accessible compute budgets. It is worth noting
that the SpiRob paper itself demonstrates grasping via hand-designed
piecewise-linear cable actuation sequences, not learned control.

**Reproducibility of the baseline result.** The R002 record documents a
successful run under conditions that did not reproduce here. The record
itself states that it is not evidence of multi-seed reliability, and
single-seed RL results are known to vary substantially. This warrants
independent investigation.

A general methodological point emerges from the five configurations: each
was diagnosed by examining what the policy *did*, via per-term reward and
termination panels, rather than by whether summed reward increased. In v1,
reward rose steadily while task progress was exactly zero. Summed reward is
not a reliable indicator of progress when shaping terms are present.

## 6. Conclusions

A camera-vision pipeline for SpiRob has been implemented and verified: a
base-mounted RGB camera feeding a spatial-softmax CNN encoder, integrated
with PPO through mjlab's model configuration, trained end-to-end on GPU, and
registered as a task independent of the unmodified base task. The policy
observes no privileged object state.

The egg-to-bucket task was not solved. Five reward configurations and an
extended-duration run were evaluated systematically, each failure mode
identified from evidence. The inherited non-vision baseline exhibits the same
zero-success behaviour on the same machine, locating the difficulty in the
task formulation rather than in the vision system.

## 7. Recommended next steps

1. **Reduce task difficulty to establish learnability.** Placing the egg on
   the ground rather than the pedestal removes the dominant egg_fell
   failure mode; reduced object separation and a relaxed success threshold
   would raise the probability of a first success from which learning can
   bootstrap. Demonstrating success in a simplified configuration and then
   tightening it constitutes a curriculum, and would determine whether the
   present difficulty is one of exploration or of fundamental feasibility.
   This changes the inherited task design and requires approval.

2. **Increase compute.** Vision-based RL commonly requires substantially more
   samples than state-based RL. All runs here used 16 parallel environments,
   constrained by per-environment rendering cost on a Colab T4. A run of
   several thousand iterations at a higher environment count would settle
   the sample-budget question definitively.

3. **Investigate baseline reproducibility.** The discrepancy between the
   documented R002 success and its non-reproduction here should be resolved
   independently of the vision work, across multiple seeds.

---

## Appendix A - Reproduction

Local verification (no GPU required):

```bash
uv run list-envs | grep Vision          # expects Mjlab-SpiRob-EggToBucket-Vision
uv run python test_vision_pipeline.py   # expects PASS
```

Training (Colab GPU):

```bash
uv run train Mjlab-SpiRob-EggToBucket-Vision \
  --env.scene.num-envs 16 \
  --agent.max-iterations 500 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name vision-v3-500
```

## Appendix B - Diagnostic metrics

For any run, the summed mean reward should not be used as the primary
indicator. The informative quantities are:

| Metric | Interpretation |
| --- | --- |
| Episode_Termination/success_egg_inside_bucket | The only direct measure of task success |
| Episode_Reward/egg_delta_progress | Signed egg movement toward the bucket; distinguishes correct from incorrect direction |
| Episode_Termination/egg_fell | Fraction of episodes ending in the egg being displaced from the pedestal |
| Episode_Termination/time_out | Fraction reaching the 800-step limit; high values with flat reward indicate inactivity |
| Entropy loss, action std | Whether learning is occurring at all; values at initialisation (≈3.11, ≈1.15) indicate no effective gradient |
