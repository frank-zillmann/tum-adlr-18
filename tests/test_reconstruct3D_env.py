"""
Simple test script to verify the ExplorationEnv environment works correctly.
"""

import numpy as np
import robosuite as suite
from robosuite.utils.camera_utils import get_real_depth_map
from robosuite.controllers import load_composite_controller_config
import matplotlib.pyplot as plt
import os


def test_reconstruct3D_env():
    """Test the Reconstruct3D environment"""

    # # Create environment with BASIC composite controller
    # controller_config = load_composite_controller_config(
    #     controller="BASIC",
    #     robot="Panda",
    # )

    # Create environment with WHOLE_BODY_MINK_IK composite controller
    controller_config = load_composite_controller_config(
        controller="WHOLE_BODY_MINK_IK",
        robot="Panda",
    )

    env = suite.make(
        env_name="Reconstruct3D",
        robots="Panda",
        controller_configs=controller_config,
        horizon=100,
        camera_names=[
            "frontview",
            "birdview",
            # "agentview",
            "sideview",
            # "robot0_robotview",
            "robot0_eye_in_hand",
        ],
    )

    print("Environment created successfully!")
    print(f"Action space: {env.action_spec}")
    print(f"Action dim: {env.action_dim}")

    # Reset environment
    obs = env.reset()
    print(f"\nObservation keys: {obs.keys()}")

    path_to_save = "./data/test_reconstruct3D_env/"
    os.makedirs(path_to_save, exist_ok=True)

    # Print some observation shapes
    for key in obs:
        if isinstance(obs[key], np.ndarray):
            print(f"  {key}: shape {obs[key].shape}, dtype {obs[key].dtype}")
            if obs[key].shape.__len__() == 1 and obs[key].shape[0] <= 50:
                print(f"    Values: {obs[key]}")
            if obs[key].shape.__len__() == 3 and obs[key].shape[2] == 3:
                # Save RGB image
                plt.figure(figsize=(6, 6))
                plt.imshow(obs[key])
                plt.title(f"{key} - RGB Image")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(path_to_save + f"initial_{key}.png")
                print(f"    Saved to '{path_to_save}initial_{key}.png'")
                plt.close()
            elif obs[key].shape.__len__() == 3 and obs[key].shape[2] == 1:
                # Convert normalized depth to real depth values (in meters)
                depth_normalized = obs[key][:, :, 0]
                depth_real = get_real_depth_map(env.sim, depth_normalized)

                # Print depth statistics
                print(
                    f"    Depth range (normalized): [{depth_normalized.min():.3f}, {depth_normalized.max():.3f}]"
                )
                print(
                    f"    Depth range (real meters): [{depth_real.min():.3f}, {depth_real.max():.3f}]"
                )

                # Save depth map with real values and fixed scale
                max_displayed_depth = None  # meters
                plt.figure(figsize=(6, 6))
                im = plt.imshow(
                    depth_real, cmap="viridis", vmin=0, vmax=max_displayed_depth
                )
                plt.title(f"{key} - Depth Map (meters)")
                plt.axis("off")
                plt.colorbar(im, label="Depth (m)")
                plt.tight_layout()
                plt.savefig(path_to_save + f"initial_{key}.png")
                print(f"    Saved to '{path_to_save}initial_{key}.png'")
                plt.close()
            else:
                print(f"    (omitted values)")

    # Run a few steps with random actions
    n_steps = 3
    print(f"\nRunning {n_steps} random steps...")
    total_reward = 0
    for i in range(n_steps):
        # Random action
        # action = np.random.uniform(env.action_spec[0], env.action_spec[1])
        action = np.array([i / n_steps] * env.action_dim)

        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"\nTotal reward: {total_reward:.4f}")

    # Close environment
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_reconstruct3D_env()
