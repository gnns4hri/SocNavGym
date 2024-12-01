from gymnasium.envs.registration import register

register(
    id='SocNavGym-v1',
    entry_point='socnavgym.envs:SocNavEnv_v1',
)

register(
    id='SocNavGym-v2',
    entry_point='socnavgym.envs:SocNavEnv_v2',
)


