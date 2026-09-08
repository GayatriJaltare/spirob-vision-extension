"""Vision environment configuration for the SpiRob CNN task.

This does NOT modify the base Stage-1 task. It calls the base cfg function
(spirob_egg_to_bucket_stage1_env_cfg) to obtain a fully-built config object
and then mutates the returned copy to:

  1. add the spirob_camera sensor to the scene,
  2. replace the observation groups so the policy sees proprioception + RGB
     image and NO ground-truth egg/bucket position, and
  3. add tip-to-egg delta-progress shaping on top of the base rewards.


REWARD-DESIGN HISTORY - five configurations tested, v3 selected


Each was run for 100 iterations at 16 environments, changing one aspect at a
time, and diagnosed from the per-term TensorBoard panels. The summed mean
reward was not used: in v1 it rose steadily while task success stayed at
exactly zero.

v1  Static proximity reward, exp(-dist/scale).
    Reward hacking - the policy held a fixed pose near the egg, farming
    per-step proximity reward without ever touching it. Replaced with delta
    progress, which pays only for REDUCING distance, so standing still earns
    nothing. (Also the key breakthrough: it broke the dead-zero-reward
    deadlock and proved the pipeline could learn.)

v2  Delta progress + undirected contact bonus.
    Any touch was rewarded regardless of direction, so the policy pushed the
    egg away from the bucket as readily as toward it. Bonus removed.

v3  Contact bonus removed; egg_delta_progress raised 1.0 -> 2.0 (the only
    term distinguishing "pushed toward the bucket" from "pushed anywhere");
    tip shaping reduced 0.5 -> 0.3.
    Correct incentive structure and the best configuration tested - this
    file. Remaining issue is physical, not incentive-based: contact forceful
    enough to topple the egg in 77% of episodes.

v4  v3 + egg_fell_penalty doubled to -20.0.
    Regression. Nearly all outcomes became heavily negative, leaving no
    gradient to distinguish better from worse; entropy and action std
    returned to their initialisation values and learning stopped - the
    mirror image of the original zero-reward problem.

v5  v3 + softened tip shaping + action-rate smoothness penalty.
    Regression. Weakening the directional signal removed guidance without
    reducing contact force; egg_fell rose toward 100%.

v4 and v5 both regressed, so weight tuning had reached negative returns. A
500-iteration run of v3 did not improve on its 100-iteration result, ruling
out sample budget as the sole limitation.

Full analysis, including the non-vision baseline control: docs/report.md

"""


from __future__ import annotations

from mjlab.sensor import CameraSensorCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from spirob_mjlab.env_cfgs import spirob_egg_to_bucket_stage1_env_cfg
from spirob_mjlab.entities import CABLE_NAMES
from spirob_mjlab import mdp  # reuse the base task's proprioception terms
from spirob_mjlab.vision import vision_mdp  # image + shaping terms


def _spirob_camera_cfg() -> CameraSensorCfg:
    """Camera mounted on the fixed robot base, viewing the workspace.

    Mounted on the base rather than the arm tip: the tip swings through
    large angles as the arm curls, giving a chaotic view, whereas the fixed
    base gives a stable view in which objects stay in roughly the same part
    of the frame - much easier for a CNN to learn from.

    pos/quat/fovy are the values verified to frame the full workspace
    (robot, egg, and bucket all visible).
    """
    return CameraSensorCfg(
        name="spirob_camera",
        parent_body="robot/robot_base",
        pos=(0.0, -0.25, 0.0),
        quat=(0.301, 0.954, 0.0, 0.0),
        fovy=90.0,
        width=320,
        height=240,
        data_types=("rgb",),
    )


def _robot_tendon_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        tendon_names=CABLE_NAMES,
        preserve_order=True,
    )


def spirob_vision_env_cfg(play: bool = False):
    """Build the base Stage-1 config, then layer camera + image obs + shaping."""
    cfg = spirob_egg_to_bucket_stage1_env_cfg(play=play)

    # 1. Add the camera sensor to the scene.
    cfg.scene.sensors = (_spirob_camera_cfg(),)

    # 2. Replace observations: proprioception + RGB image only.
    # The base task's egg_to_bucket and robot_keypoints_xy terms are
    # deliberately dropped - both hand the policy object position directly,
    # which would remove any need for the CNN to learn to see.
    tendon_cfg = _robot_tendon_cfg()
    proprio_terms = {
        "tendon_len": ObservationTermCfg(
            func=mdp.tendon_length,
            params={"asset_cfg": tendon_cfg},
        ),
        "tendon_vel": ObservationTermCfg(
            func=mdp.tendon_velocity,
            params={"asset_cfg": tendon_cfg},
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action),
        "touch": ObservationTermCfg(func=mdp.touch_values),
    }
    camera_terms = {
        "camera_rgb": ObservationTermCfg(
            func=vision_mdp.camera_rgb_image,
            params={"camera_name": "spirob_camera", "resize": None},
        ),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(proprio_terms, enable_corruption=False),
        "camera": ObservationGroupCfg(camera_terms, enable_corruption=False),
        "critic": ObservationGroupCfg({**proprio_terms}, enable_corruption=False),
    }

    # 3a. Make directed egg progress dominant. This base-task term measures
    # the EGG's signed movement toward the bucket - the only term able to
    # distinguish "pushed the right way" from "pushed any way" (see v2).
    cfg.rewards["egg_delta_progress"].weight = 2.0

    # 3b. Delta tip-to-egg shaping. v1's static proximity reward, v2's
    # undirected contact bonus, v4's doubled fall penalty and v5's
    # smoothness penalty are all intentionally absent - see the history in
    # the module docstring for what each did.
    cfg.rewards = {
        **cfg.rewards,
        "tip_to_egg_progress": RewardTermCfg(
            func=vision_mdp.tip_to_egg_delta_progress,
            weight=0.3,
            params={"scale": 0.005},
        ),
    }

    return cfg
