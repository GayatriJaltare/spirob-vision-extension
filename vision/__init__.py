"""Registers the SpiRob CNN vision task, separate from the base Stage-1 task.

Importing this module registers `Mjlab-SpiRob-EggToBucket-Vision` alongside the
existing `Mjlab-SpiRob-EggToBucket-Stage1`. The base task is unaffected.

To make this task discoverable, add the following single line to the end of
src/spirob_mjlab/__init__.py:

    from spirob_mjlab import vision  # noqa: F401  (registers the vision task)
"""

from mjlab.tasks.registry import register_mjlab_task

from spirob_mjlab.vision.vision_env_cfg import spirob_vision_env_cfg
from spirob_mjlab.vision.vision_rl_cfg import spirob_vision_ppo_runner_cfg

VISION_TASK_ID = "Mjlab-SpiRob-EggToBucket-Vision"

register_mjlab_task(
    task_id=VISION_TASK_ID,
    env_cfg=spirob_vision_env_cfg(play=False),
    play_env_cfg=spirob_vision_env_cfg(play=True),
    rl_cfg=spirob_vision_ppo_runner_cfg(),
)
