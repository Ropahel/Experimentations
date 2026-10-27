import gymnasium as gym


def test_cartpole_can_reset_and_render_an_image() -> None:
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=0)
        image = env.render()

        assert observation.shape == (4,)
        assert image is not None
        assert image.ndim == 3
        assert image.shape[2] == 3
    finally:
        env.close()
