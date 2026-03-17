import os
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson, opendata

class DiffusionPlannerData(Dataset):
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len, skill_map_path=None):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        self.skill_map = openjson(skill_map_path) if skill_map_path is not None else None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        sample_key = self.data_list[idx]
        data = opendata(os.path.join(self.data_dir, sample_key))

        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future']

        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]

        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects = data['static_objects']

        if self.skill_map is None:
            skill_id = 0
        else:
            if sample_key not in self.skill_map:
                raise KeyError(
                    f"skill_map missing key for sample '{sample_key}'. "
                    f"Please ensure skill_map.json keys exactly match entries in data_list."
                )
            skill_id = int(self.skill_map[sample_key])

        data = {
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects,
            "skill_id": skill_id,
        }

        return (
            ego_current_state,
            ego_agent_future,
            neighbor_agents_past,
            neighbor_agents_future,
            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,
            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,
            static_objects,
            skill_id,
        )