import gymnasium as gym


def test_lunar_lander_can_reset() -> None:
    env = gym.make("LunarLander-v3")
    try:
        observation, _ = env.reset(seed=0)

        assert observation.shape == (8,)
    finally:
        env.close()
