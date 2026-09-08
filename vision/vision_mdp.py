"""Vision MDP terms for the SpiRob CNN task.

Kept separate from the base task's mdp.py so the existing Stage-1 task is
never modified. This module adds a camera-image observation term and a
dense tip-to-egg DELTA progress reward.

No segmentation, no depth, no ground-truth object position is used for the
image term: the policy perceives the egg/bucket/workspace purely through
RGB pixels.

This is the v3 configuration, restored. See vision_env_cfg.py for the full
reward-design history and why v3 was selected over v1/v2/v4/v5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def camera_rgb_image(
    env: "ManagerBasedRlEnv",
    camera_name: str = "spirob_camera",
    resize: tuple[int, int] | None = None,
) -> torch.Tensor:
    """RGB image from spirob_camera, normalized to [0, 1], channel-first.

    Returns [num_envs, 3, H, W] float32 tensor in [0, 1].
    """
    cam = env.scene[camera_name].data
    img = cam.rgb.permute(0, 3, 1, 2).float() / 255.0
    if resize is not None:
        img = torch.nn.functional.interpolate(
            img, size=resize, mode="bilinear", align_corners=False,
        )
    return img


_TIP_SITE_ID_CACHE: dict[int, int] = {}


def _tip_site_id(env: "ManagerBasedRlEnv") -> int:

    key = id(env.sim.model)
    if key not in _TIP_SITE_ID_CACHE:
        cfg = SceneEntityCfg("robot", site_names=("tip_site",))
        cfg.resolve(env.scene)
        site_ids = cfg.site_ids
        assert len(site_ids) == 1, f"expected 1 site, resolved {len(site_ids)}: {site_ids}"
        _TIP_SITE_ID_CACHE[key] = site_ids[0]
    return _TIP_SITE_ID_CACHE[key]


def _tip_to_egg_distance(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    site_id = _tip_site_id(env)
    tip = robot.data.site_pos_w[:, site_id, :]
    egg = env.scene["egg"].data.root_link_pos_w
    return torch.linalg.norm(tip - egg, dim=-1)


def tip_to_egg_delta_progress(
    env: "ManagerBasedRlEnv",
    scale: float = 0.005,
) -> torch.Tensor:
   
    current_distance = _tip_to_egg_distance(env)

    buffer_name = "_vision_prev_tip_egg_distance"
    previous_distance = getattr(env, buffer_name, None)

    if previous_distance is None or previous_distance.shape != current_distance.shape:
        setattr(env, buffer_name, current_distance.detach().clone())
        return torch.zeros_like(current_distance)

    progress = previous_distance - current_distance

    # Do not interpret a newly reset environment as tip movement.
    new_episode = env.episode_length_buf <= 1
    progress = torch.where(new_episode, torch.zeros_like(progress), progress)

    setattr(env, buffer_name, current_distance.detach().clone())

    # RewardManager multiplies terms by step_dt. Dividing here makes the
    # final contribution equal to the intended normalized progress.
    normalized_progress = progress / scale
    return torch.clamp(normalized_progress, min=-1.0, max=1.0) / env.step_dt
