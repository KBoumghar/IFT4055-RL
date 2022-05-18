from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List
import random

ENV_LENGTH = 100000000


class ENVB(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'ENVTEST-v0'

        super().__init__(*args, max_episode_steps=ENV_LENGTH, reward_threshold=100, **kwargs)

    # def create_server_world_generators(self) -> List[Handler]:

    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [  # Coordinates and lifestats
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats()]

    def create_server_initial_conditions(self) -> List[Handler]:
        idx = random.randint(0, 3)
        # Check if these work / are only options
        weatherlist = ["rain", "thunder", ""]
        weather = weatherlist[idx]

        # Start with a random time, spawning point and weather
        time = random.randint(0, 24000)
        return [
            handlers.TimeInitialCondition(True, time),
            handlers.SpawningInitialCondition(True),
            handlers.WeatherInitialCondition(weather)
        ]
