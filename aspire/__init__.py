from gym.envs.registration import register

register(
    id='Aspire-v1',
    entry_point='aspire.envs:AspireEnv',
)
