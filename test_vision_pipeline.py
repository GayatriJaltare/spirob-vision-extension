"""Standalone pre-training sanity check for the SpiRob vision task.

Run this BEFORE any training (locally, no GPU needed) to confirm the chain
camera RGB -> preprocessed tensor -> env step -> 2 cable actions works.

    uv run python test_vision_pipeline.py

Expected: prints image shape [1, 3, 240, 320], value range within [0, 1],
action shape [1, 2], and finally "PASS".
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from spirob_mjlab.vision.vision_env_cfg import spirob_vision_env_cfg
from spirob_mjlab.vision import vision_mdp


def main() -> None:
    cfg = spirob_vision_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg=cfg)
    obs, info = env.reset()

    img = vision_mdp.camera_rgb_image(env)
    print("image:", tuple(img.shape), img.dtype,
          "range", round(img.min().item(), 3), "-", round(img.max().item(), 3))
    assert img.ndim == 4 and img.shape[1] == 3, f"expected [B,3,H,W], got {tuple(img.shape)}"
    assert 0.0 <= img.min().item() and img.max().item() <= 1.0, "expected [0,1] normalized"

    zero = torch.zeros(
        env.action_manager.action.shape,
        device=env.action_manager.action.device,
    )
    env.step(zero)
    action_shape = tuple(env.action_manager.action.shape)
    print("action shape:", action_shape, "(expect [num_envs, 2])")
    assert action_shape[-1] == 2, f"expected 2 cable actions, got {action_shape}"

    print("PASS")


if __name__ == "__main__":
    main()
