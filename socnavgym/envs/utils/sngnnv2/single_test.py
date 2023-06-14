import sys
import time
import json

# if len(sys.argv) != 3:
#     print(f"Usage: 'python3 {sys.argv[0]} model_directory file'")
#     print(f'e.g.: python3 {sys.argv[0]} example_model/ jsons_test/S1_000003.json')
#     sys.exit(0)

sys.path.append("..")
from socnav import *
from socnav_V2_API import *


torch.set_grad_enabled(False)

sngnn = SocNavAPI(device="cpu", params_dir="example_model")  # change to cpu when no gpu


scenario = "jsons_test/tp.json"
with open(scenario, "r") as f:
    data_sequence = json.loads(f.read())

timea = time.time()

sn_sequence = []
for data_structure in reversed(data_sequence):
    sn = SNScenario(data_structure["timestamp"])
    sn.add_goal(data_structure["goal"][0]["x"], data_structure["goal"][0]["y"])
    sn.add_command(data_structure["command"])
    for human in data_structure["people"]:
        sn.add_human(Human(
            human["id"],
            human["x"],
            human["y"],
            human["a"],
            human["vx"],
            human["vy"],
            human["va"],
        ))
    for objectt in data_structure["objects"]:
        sn.add_object(Object(
            objectt["id"],
            objectt["x"],
            objectt["y"],
            objectt["a"],
            objectt["vx"],
            objectt["vy"],
            objectt["va"],
            objectt["size_x"],
            objectt["size_y"],
        ))

    sn.add_room(data_structure["walls"])
    
    for interaction in data_structure["interaction"]:
        sn.add_interaction([interaction["dst"], interaction["src"]])
        print(interaction["dst"])
    sn_sequence.append(sn.to_json())

graph = SocNavDataset(sn_sequence, "1", "test", verbose=False)
ret_gnn = sngnn.predictOneGraph(graph)[0]

timeb = time.time()

print(f'Result for a *single* query: {ret_gnn}. It took {timeb-timea} seconds')
