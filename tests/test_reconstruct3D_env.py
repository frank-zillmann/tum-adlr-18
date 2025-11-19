"""
Simple test script to verify the ExplorationEnv environment works correctly.
"""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import matplotlib.pyplot as plt


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
    )

    print("Environment created successfully!")
    print(f"Action space: {env.action_spec}")
    print(f"Action dim: {env.action_dim}")

    # Reset environment
    obs = env.reset()
    print(f"\nObservation keys: {obs.keys()}")

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
                plt.savefig(f"initial_{key}.png")
                print(f"    Saved to 'initial_{key}.png'")
                plt.close()
            elif obs[key].shape.__len__() == 3 and obs[key].shape[2] == 1:
                # Save depth map
                plt.figure(figsize=(6, 6))
                depth = obs[key][:, :, 0]
                im = plt.imshow(depth, cmap="viridis")
                plt.title(f"{key} - Depth Map")
                plt.axis("off")
                plt.colorbar(im, label="Depth (m)")
                plt.tight_layout()
                plt.savefig(f"initial_{key}.png")
                print(f"    Saved to 'initial_{key}.png'")
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
