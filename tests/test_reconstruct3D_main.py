"""
Main test script for Reconstruct3D environment.
Runs all test functions in sequence.
"""

import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# Import test functions
from test_visualize_observations import test_visualize_observations
from test_compute_sdf import test_compute_sdf
from test_render_sdf_mesh import test_render_sdf_mesh
from test_run_dummy_steps import test_run_dummy_steps


def main():
    """Run all Reconstruct3D tests."""

    print("=" * 60)
    print("RECONSTRUCT3D ENVIRONMENT TEST SUITE")
    print("=" * 60)

    # Create environment with WHOLE_BODY_MINK_IK composite controller
    print("\nCreating environment...")
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
            "sideview",
            "robot0_eye_in_hand",
        ],
    )

    print("Environment created successfully!")

    # Reset environment
    print("\nResetting environment...")
    obs = env.reset()
    print("Environment reset complete!")

    # Run all tests
    try:
        # Test 1: Visualize observations
        test_visualize_observations(env)

        # Test 2: Compute SDF
        sdf_grid, bbox_center, bbox_size = test_compute_sdf(env, sdf_size=16)

        # Test 3: Render mesh from SDF
        test_render_sdf_mesh(env, sdf_grid, bbox_center, bbox_size)

        # Test 4: Run dummy steps
        test_run_dummy_steps(env, n_steps=3)

    finally:
        # Close environment
        env.close()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
