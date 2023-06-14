from gym.envs.registration import register

register(
    id='SocNavGym-v1',
    entry_point='socnavgym.envs:SocNavEnv_v1',
)