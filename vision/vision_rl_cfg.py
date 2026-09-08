"""CNN PPO configuration for the SpiRob vision task.

Separate from the base task's rl_cfg.py. Uses mjlab's built-in
SpatialSoftmaxCNNModel to encode the RGB camera group, concatenate the
resulting features with the low-dimensional proprioception group, and feed
the combined vector to the actor/critic MLPs. The CNN is trained end-to-end
with PPO (its weights are shaped by the task reward, not a separate step).

The CNN architecture (_VISION_CNN_CFG) is reused from mjlab's own validated
vision example rather than invented from scratch.
"""

from __future__ import annotations

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

# CNN encoder config reused from mjlab's validated vision example.
_VISION_CNN_CFG = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
}


def spirob_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        # Route both the proprioception group ("actor"/"critic") and the image
        # group ("camera") into the actor and critic models.
        obs_groups={
            "actor": ("actor", "camera"),
            "critic": ("critic", "camera"),
        },
        actor=RslRlModelCfg(
            class_name="mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel",
            cnn_cfg=_VISION_CNN_CFG,
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name="mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel",
            cnn_cfg=_VISION_CNN_CFG,
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        # Separate experiment name so vision checkpoints never mix with the
        # base MLP task's checkpoints (different architectures).
        experiment_name="spirob_vision",
        save_interval=50,
        num_steps_per_env=32,
        max_iterations=1000,
    )
