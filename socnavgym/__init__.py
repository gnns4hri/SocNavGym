from gymnasium.envs.registration import register

# SocNavGym-v1 was removed (socnavgym.envs has no SocNavEnv_v1); use SocNavGym-v2.
register(
    id='SocNavGym-v2',
    entry_point='socnavgym.envs:SocNavEnv_v2',
)


