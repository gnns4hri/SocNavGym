import pytest
import sys
import os
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/..")
import gym
import socnavgym
import yaml

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_shape():
    env = gym.make("SocNavGym-v1", config=os.path.dirname(os.path.abspath(__file__)) + "/../environment_configs/test_env.yaml")
    env.set_padded_observations(True)
    obs, _ = env.reset()

    config_file = os.path.dirname(os.path.abspath(__file__)) + "/../environment_configs/test_env.yaml"
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    total_max_humans = (
        config["env"]["max_static_humans"] + config["env"]["max_dynamic_humans"]+  
        config["env"]["max_h_h_dynamic_interactions"]*config["env"]["max_human_in_h_h_interactions"] + 
        config["env"]["max_h_h_dynamic_interactions_non_dispersing"]*config["env"]["max_human_in_h_h_interactions"] + 
        config["env"]["max_h_h_static_interactions"]*config["env"]["max_human_in_h_h_interactions"]+ 
        config["env"]["max_h_h_static_interactions_non_dispersing"]*config["env"]["max_human_in_h_h_interactions"]+ 
        config["env"]["max_h_l_interactions"] + config["env"]["max_h_l_interactions_non_dispersing"] 
    )
    assert(obs["humans"].shape == (total_max_humans *  env.entity_obs_dim, ))
    assert(obs["plants"].shape == (config["env"]["max_plants"]* env.entity_obs_dim, ))
    assert(obs["tables"].shape == (config["env"]["max_tables"]* env.entity_obs_dim, ))
    assert(obs["laptops"].shape == ((config["env"]["max_laptops"] + config["env"]["max_h_l_interactions"] + config["env"]["max_h_l_interactions_non_dispersing"])* env.entity_obs_dim, ))

    env.set_padded_observations(False)
    obs, _ = env.reset()
    if env.total_humans > 0: assert(obs["humans"].shape == (env.total_humans *  env.entity_obs_dim, ))
    if env.NUMBER_OF_PLANTS > 0: assert(obs["plants"].shape == (env.NUMBER_OF_PLANTS * env.entity_obs_dim, ))
    if env.NUMBER_OF_TABLES > 0: assert(obs["tables"].shape == (env.NUMBER_OF_TABLES * env.entity_obs_dim, ))
    if env.NUMBER_OF_LAPTOPS > 0: assert(obs["laptops"].shape == ((env.NUMBER_OF_LAPTOPS + env.TOTAL_H_L_INTERACTIONS)* env.entity_obs_dim, ))