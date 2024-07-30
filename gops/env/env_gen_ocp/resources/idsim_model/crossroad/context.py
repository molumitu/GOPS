import numpy as np
import torch

# from idsim.envs.env import CrossRoad
from gops.env.env_gen_ocp.pyth_base import Env as CrossRoad
from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext, State, Parameter
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig, ego_mean_array, ego_std_array, ego_lower_bound_array, ego_upper_bound_array

from gops.env.env_gen_ocp.resources.idsim_model.crossroad.ref import get_ref_param
from gops.env.env_gen_ocp.resources.idsim_model.crossroad.traffic import get_traffic_light_param
from gops.env.env_gen_ocp.resources.idsim_model.crossroad.sur import get_sur_param


class CrossRoadContext(BaseContext):
    def advance(self, x: State) -> "CrossRoadContext":
        return CrossRoadContext(x=x, p=self.p, t=self.t, i=self.i + 1,)
    @classmethod
    def from_env(cls, env: CrossRoad, model_config: ModelConfig, ref_index_param: int = None):
        rng = env.engine.context.rng  # random number generator
        ego_state = env.engine.context.vehicle.state
        last_last_action = env.engine.context.vehicle.last_action  # real value
        last_action = env.engine.context.vehicle.action  # real value
        env_time = env.engine.context.simulation_time
        light_param = get_traffic_light_param(env, model_config)
        ref_param = get_ref_param(env, model_config, light_param)
        sur_param = get_sur_param(env, model_config, rng)
        if ref_index_param is None:
            ref_index_param = np.array(0)
        else:
            ref_index_param = np.array(ref_index_param)

        # add noise to ego_state
        ego_noise = rng.normal(
            ego_mean_array, ego_std_array, size=ego_state.shape)
        ego_noise = np.clip(ego_noise, ego_lower_bound_array,
                            ego_upper_bound_array)
        # add noise only for vx >= 1m/s
        ego_state += ego_noise * (ego_state[2] >= 1)

        # numpy to tensor
        ego_state = torch.from_numpy(ego_state).float()
        last_last_action = torch.from_numpy(last_last_action).float()
        last_action = torch.from_numpy(last_action).float()
        ref_param = torch.from_numpy(ref_param).float()
        sur_param = torch.from_numpy(sur_param).float()
        light_param = torch.from_numpy(light_param).float()
        ref_index_param = torch.from_numpy(ref_index_param).long()

        return CrossRoadContext(
            x=State(ego_state=ego_state,
                    last_last_action=last_last_action, last_action=last_action),
            p=Parameter(ref_param=ref_param, sur_param=sur_param,
                        light_param=light_param, ref_index_param=ref_index_param,
                        ),
            t=env_time,
            i=0,
        )
