import torch
import random
from data_conversions import clone_sequence

def mirror_tDic_sequence(tDict_sequence):
    new_tDict_sequence = tDict_sequence

    new_tDict_sequence['robot']['y'] = -new_tDict_sequence['robot']['y']
    new_tDict_sequence['robot']['a'] = -new_tDict_sequence['robot']['a']
    new_tDict_sequence['robot']['vy'] = -new_tDict_sequence['robot']['vy']
    new_tDict_sequence['robot']['va'] = -new_tDict_sequence['robot']['va']

    new_tDict_sequence['people']['y'] = -new_tDict_sequence['people']['y']
    new_tDict_sequence['people']['a'] = -new_tDict_sequence['people']['a']
    new_tDict_sequence['objects']['y'] = -new_tDict_sequence['objects']['y']
    new_tDict_sequence['objects']['a'] = -new_tDict_sequence['objects']['a']
    new_tDict_sequence['walls']['y'] = -new_tDict_sequence['walls']['y']

    return new_tDict_sequence


def tensor_transform_with_random_mirroring(tDict_sequence):
    if random.randint(0,1)==1:
        tensor_dict_clone = clone_sequence(tDict_sequence)                        
        tDict_mirrored = mirror_tDic_sequence(tensor_dict_clone)
        return tDict_mirrored
    else:
        return tDict_sequence
    



