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

    def create_server_world_generators(self) -> List[Handler]:
        return [
            # Basic FlatWorldGenerator for now, could use different biomes.
            # Check how generator string works in future
            handlers.FlatWorldGenerator(True)
        ]

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

    def create_agent_handlers(self) -> List[Handler]:
        # For now, agent doesn't stop from nothing but number of steps
        return []

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            handlers.KeybasedCommandAction("use"),

            # Do we need to allow it to equip each item individually?
            handlers.EquipAction()
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return []

    def create_server_decorators(self) -> List[Handler]:
        return []

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'Minecraft'

    def get_docstring(self):
        return ""

