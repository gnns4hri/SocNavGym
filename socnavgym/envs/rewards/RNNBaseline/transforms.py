from data_mirroring import tensor_transform_with_random_mirroring
from data_normalization import tensor_transform_to_goal_fr
from data_random_orientation import tensor_transform_with_random_orientation
from data_random_noise import tensor_transform_with_random_noise


class NormalizeTrajectory:
    def __call__(self, trajectory):
        return tensor_transform_to_goal_fr(trajectory)

class ApplyRandomMirroring:
    def __call__(self, trajectory):
        return tensor_transform_with_random_mirroring(trajectory)

class ApplyRandomOrientation:
    def __call__(self, trajectory):
        return tensor_transform_with_random_orientation(trajectory)

class ApplyRandomNoise:
    def __call__(self, trajectory):
        return tensor_transform_with_random_noise(trajectory)
