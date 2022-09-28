from gops.env.env_wrapper.shaping_reward import ShapingRewardData, ShapingRewardModel
from gops.env.env_wrapper.scale_observation import ScaleObservationData, ScaleObservationModel


def all_none(a, b):
    if (a is None) and (b is None):
        return True
    else:
        return False


def wrapping_env(env, reward_shift=None, reward_scale=None, obs_shift=None, obs_scale=None):
    if not all_none(reward_scale, reward_shift):
        reward_scale = 1.0 if reward_scale is None else reward_scale
        reward_shift = 0.0 if reward_shift is None else reward_shift
        env = ShapingRewardData(env, reward_shift, reward_scale)

    if not all_none(obs_shift, obs_scale):
        obs_scale = 1.0 if obs_scale is None else obs_scale
        obs_shift = 0.0 if obs_shift is None else obs_shift
        env = ScaleObservationData(env, obs_shift, obs_scale)

    return env


def wrapping_model(model, reward_shift=None, reward_scale=None, obs_shift=None, obs_scale=None):
    if not all_none(reward_scale, reward_shift):
        reward_scale = 1.0 if reward_scale is None else reward_scale
        reward_shift = 0.0 if reward_shift is None else reward_shift
        model = ShapingRewardModel(model, reward_shift, reward_scale)

    if not all_none(obs_shift, obs_scale):
        obs_scale = 1.0 if obs_scale is None else obs_scale
        obs_shift = 0.0 if obs_shift is None else obs_shift
        model = ScaleObservationModel(model, obs_shift, obs_scale)

    return model

if __name__ == "__main__":
    from gops.env.env_gym.gym_cartpoleconti_data import env_creator
    env = env_creator()
    env.reset()
    a = env.action_space.sample()
    s, r, d, info = env.step(a)
    print(s, r)

    env = wrapping_env(env, 100, 1.0, 100., 2.0)
    env.reset()
    a = env.action_space.sample()
    s, r, d, info = env.step(a)
    print(s, r)