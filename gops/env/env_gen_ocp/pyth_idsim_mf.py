from dataclasses import dataclass
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Generic, Optional, Tuple, Union

import gym
import numpy as np
import torch
import grpc
import json
import pickle
from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, State, stateType)
from gops.env.env_gen_ocp.resources.idsim_tags import reward_tags
from gops.env.env_gen_ocp.resources.idsim_var_type import Config
from gops.env.env_gen_ocp.resources.idsim_model.model_context import Parameter, BaseContext, State as ModelState
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig
from gops.env.env_gen_ocp.resources.idsim_model.model import IdSimModel
from gops.env.env_gen_ocp.resources.model_free_reward import model_free_reward_multilane
from gops.env.env_gen_ocp.resources.model_free_reward import reward_function_multilane

from gops.env.env_gen_ocp.resources.idsim_server import cloudserver_pb2 
from gops.env.env_gen_ocp.resources.idsim_server import cloudserver_pb2_grpc
import sys
sys.path.append(str(Path(__file__).with_name("resources")))

def dict_to_message(data: Dict):
    message: str = json.dumps(data)
    return message

def message_to_dict(message: str):
    message_dict: Dict = json.loads(message)
    return message_dict

class CloudServer:
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:50051')
        # self.channel = grpc.insecure_channel('47.95.66.102:50051')
        self.stub = cloudserver_pb2_grpc.IdSimServiceStub(self.channel)
        self.idsim_id  = str

    def init_idsim(self, env_config: Config, model_config: ModelConfig, scenario_id: str):
        env_config_dict = asdict(env_config)
        model_config_dict = asdict(model_config)
        message = pickle.dumps((env_config_dict, model_config_dict, scenario_id))
        request = cloudserver_pb2.InitIdSimRequest(
            message = message,
        )
        response = self.stub.InitIdSim(request)
        handle = response.handle
        self.idsim_id = response.idsim_id
        return handle
    
    def reset_idsim(self):
        request = cloudserver_pb2.ResetIdSimRequest(
            idsim_id = self.idsim_id
        )
        response = self.stub.ResetIdSim(request)
        message = pickle.loads(response.message)
        return message
        
    def get_multilane_idsimcontext(self, model_config, ref_index_param, handle=0):
        message = pickle.dumps((ref_index_param,))
        request = cloudserver_pb2.GetMultilaneIdsimcontextRequest(
            message = message,
            idsim_id = self.idsim_id
        )
        response = self.stub.GetMultilaneIdsimcontext(request)
        message = pickle.loads(response.message)
        return message

    def get_crossroad_idsimcontext(self, model_config: ModelConfig, ref_index_param, handle=0):
        message = pickle.dumps((ref_index_param,))
        request = cloudserver_pb2.GetCrossroadIdsimcontextRequest(
            message = message,
            idsim_id = self.idsim_id
        )
        response = self.stub.GetCrossroadIdsimcontext(request)
        message = pickle.loads(response.message)
        return message
    
    def step_idsim(self, action, handle=0):
        message = pickle.dumps((action,))
        request = cloudserver_pb2.StepIdsimRequest(
            message = message,
            idsim_id = self.idsim_id
        )
        response = self.stub.StepIdsim(request)
        message = pickle.loads(response.message)
        return message

    def close_idsim(self):
        request = cloudserver_pb2.CloseRequest(
            idsim_id=self.idsim_id
        )
        response = self.stub.Close(request)

@dataclass
class idSimContextState(ContextState[stateType], Generic[stateType]):
    light_param: Optional[stateType] = None
    ref_index_param: Optional[stateType] = None
    boundary_param: Optional[stateType] = None
    real_t: Union[int, stateType] = 0


class idSimContext(Context):
    def reset(self) -> idSimContextState[np.ndarray]:
        pass

    def step(self) -> idSimContextState[np.ndarray]:
        pass

    def get_zero_state(self) -> idSimContextState[np.ndarray]:
        pass
    

class idSimEnv(Env): 
    """
    env_config: Config, 
    model_config: ModelConfig, 
    scenario: str, 
    rou_config: Dict[str, Any]=None
    """
    def __init__(self, env_config: Config, model_config: ModelConfig, 
                 scenario: str, rou_config: Dict[str, Any]=None,env_idx: int=None, scenerios_list: List[str]=None):
        self.env_idx = env_idx  
        print('env_idx:', env_idx)
        self.rou_config = rou_config
        self.env_config = env_config
        self.server = CloudServer()
        self.server.init_idsim(env_config, model_config, scenario)

        self.model = IdSimModel(env_config, model_config)
        obs_dim = self.model.obs_dim
        self.model_config = model_config
        self.scenario = scenario

        self._state = None
        self._info = {"reward_comps": np.zeros(len(model_config.reward_comps), dtype=np.float32)}
        self._reward_comp_list = model_config.reward_comps
        self.use_random_ref_param = env_config.use_multiple_path_for_multilane
        self.random_ref_probability = env_config.random_ref_probability
        self.random_ref_v = env_config.random_ref_v
        self.ref_v_range = env_config.ref_v_range

        if self.use_random_ref_param > 0.0:
            print(f'INFO: randomly choosing reference when resetting env')
        if self.random_ref_probability > 0.0:
            print(f'INFO: randomly choosing reference when stepping at P={self.random_ref_probability}')
        if env_config.takeover_bias:
            print('INFO: using takeover bias True')
        if env_config.use_random_acc:
            print('INFO: using random acceleration')
        if model_config.track_closest_ref_point:
            print('INFO: tracking closest reference point')

        self.lc_cooldown = self.env_config.random_ref_cooldown
        self.lc_cooldown_counter = 0

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.context = idSimContext() # fake idsim context
        
        self.ref_index = None
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.lc_cooldown_counter = 0
        obs, info = self.server.reset_idsim()
        self.ref_index = np.random.choice(
            np.arange(self.model_config.num_ref_lines)
        ) if self.use_random_ref_param else None
        self._state = self._get_state_from_idsim(ref_index_param=self.ref_index)
        self._info = self._get_info(info)
        self._info.update(info)
        return self._get_obs(), self._info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, _, terminated, truncated, info = self.server.step_idsim(action)
        vehicle = info["vehicle"]
        reward, reward_info = reward_function_multilane(asdict(self.env_config), vehicle)
        info.update(reward_info)
        # ----- cal the next_obs, reward -----
        self.lc_cooldown_counter += 1
        if self.lc_cooldown_counter > self.lc_cooldown:
            # lane change is allowable
            if self.use_random_ref_param and np.random.rand() < self.random_ref_probability :
                new_ref_index = np.random.choice(np.arange(self.model_config.num_ref_lines))
                if new_ref_index != self.ref_index:
                    # lane change
                    self.ref_index = new_ref_index
                    self.lc_cooldown_counter = 0
        _, reward_details, next_state = self._get_reward(action)
        self._state = next_state
        reward_model_free, mf_info = self._get_model_free_reward(action)
        info.update(mf_info)

        info["reward_details"] = dict(
            zip(reward_tags, [i.item() for i in reward_details])
        )
        done = terminated or truncated
        if truncated:
            info["TimeLimit.truncated"] = True # for gym

        self._info = self._get_info(info)
        self._info.update(info)
        total_reward = reward + reward_model_free
        obs = self._get_obs()
        return obs, total_reward, done, self._info

    def set_ref_index(self, ref_index: int):
        self.ref_index = ref_index
    
    def _get_info(self, info) -> dict:
        info.update(Env._get_info(self))
        if "env_reward_step" in info.keys():
            info["reward_comps"] = np.array([info[i] for i in self._reward_comp_list], dtype=np.float32)
        else:
            info["reward_comps"] = np.zeros(len(self._reward_comp_list), dtype=np.float32)
        return info
    
    @property
    def additional_info(self) -> dict:
        print('{}')
        return {}
    
    def _get_obs(self) -> np.ndarray:
        idsim_context = get_idsimcontext(
            State.stack([self._state.array2tensor()]), mode="batch", scenario=self.scenario)
        model_obs = self.model.observe(idsim_context)
        return model_obs.numpy().squeeze(0)

    def _get_reward(self, action: np.ndarray) -> Tuple[float, Tuple]:
        cur_state = self._state.array2tensor()
        next_state = self._get_state_from_idsim(ref_index_param=self.ref_index)
        idsim_context = get_idsimcontext(State.stack([next_state.array2tensor()]), mode="batch", scenario=self.scenario)
        action = torch.tensor(action)
        reward_details = self.model.reward_nn_state(
            context=idsim_context,
            last_last_action=cur_state.robot_state[..., -4:-2].unsqueeze(0), # absolute action
            last_action=cur_state.robot_state[..., -2:].unsqueeze(0), # absolute action
            action=action.unsqueeze(0) # incremental action
        )
        return reward_details[0].item(), reward_details, next_state
    
    def _get_model_free_reward(self, action: np.ndarray) -> float:
        idsim_context = get_idsimcontext(
            State.stack([self._state]), 
            mode="batch", 
            scenario=self.scenario
        )
        last_last_action=self._state.robot_state[..., -4:-2][None, :] # absolute action
        last_action=self._state.robot_state[..., -2:][None, :] # absolute action
        action=action[None, :] # incremental action
        reward, info = model_free_reward_multilane(
                                    asdict(self.env_config),
                                    idsim_context, last_last_action, 
                                    last_action, action)
        return reward, info
    
    def _get_terminated(self) -> bool:
        """abandon this function, use terminated from idsim instead"""
        ...
    
    def _get_state_from_idsim(self, ref_index_param=None) -> State:
        if self.scenario == "crossroad":
            idsim_context = self.server.get_crossroad_idsimcontext(self.model_config, ref_index_param)
        elif self.scenario == "multilane":
            idsim_context = self.server.get_multilane_idsimcontext(self.model_config, ref_index_param)
        else:
            raise NotImplementedError
        return State(
            robot_state=torch.concat([
                idsim_context.x.ego_state, 
                idsim_context.x.last_last_action, 
                idsim_context.x.last_action],
            dim=-1),
            context_state=idSimContextState(
                reference=idsim_context.p.ref_param, 
                constraint=idsim_context.p.sur_param,
                light_param=idsim_context.p.light_param, 
                ref_index_param=idsim_context.p.ref_index_param,
                boundary_param=idsim_context.p.boundary_param,
                real_t = torch.tensor(idsim_context.t).int(),
                t = torch.tensor(idsim_context.i).int()
            )
        ).tensor2array()

    def get_zero_state(self) -> State[np.ndarray]:
        if self._state is None:
            self.reset()
        return State(
            robot_state=np.zeros_like(self._state.robot_state, dtype=np.float32),
            context_state=idSimContextState(
                reference=np.zeros_like(self._state.context_state.reference, dtype=np.float32),
                constraint=np.zeros_like(self._state.context_state.constraint, dtype=np.float32),
                t=np.zeros_like(self._state.context_state.t, dtype=np.int64),
                light_param=np.zeros_like(self._state.context_state.light_param, dtype=np.float32),
                ref_index_param=np.zeros_like(self._state.context_state.ref_index_param, dtype=np.int64),
                boundary_param=np.zeros_like(self._state.context_state.boundary_param, dtype=np.float32),
                real_t=np.zeros_like(self._state.context_state.real_t, dtype=np.int64)
            )
        )
    # close
    def close(self) -> None:
        self.server.close_idsim()

def get_idsimcontext(state: State, mode: str, scenario: str) -> BaseContext:
    Context = BaseContext
    if mode == "full_horizon":
        context = Context(
            x = ModelState(
                ego_state = state.robot_state[..., :-4],
                last_last_action = state.robot_state[..., -4:-2],
                last_action = state.robot_state[..., -2:]
            ),
            p = Parameter(
                ref_param = state.context_state.reference,
                sur_param = state.context_state.constraint,
                light_param = state.context_state.light_param,
                ref_index_param = state.context_state.ref_index_param,
                boundary_param=state.context_state.boundary_param
            ),
            t = state.context_state.real_t,
            i = state.context_state.t[0]
        )
    elif mode == "batch":
        if isinstance(state.context_state.t, np.ndarray):
            assert np.unique(state.context_state.t).shape[0] == 1, "batch mode only support same t"
        elif isinstance(state.context_state.t, torch.Tensor):
            assert state.context_state.t.unique().shape[0] == 1, "batch mode only support same t"
        else:
            raise NotImplementedError
        context = Context(
            x = ModelState(
                ego_state = state.robot_state[..., :-4],
                last_last_action = state.robot_state[..., -4:-2],
                last_action = state.robot_state[..., -2:]
            ),
            p = Parameter(
                ref_param = state.context_state.reference,
                sur_param = state.context_state.constraint,
                light_param = state.context_state.light_param,
                ref_index_param = state.context_state.ref_index_param,
                boundary_param=state.context_state.boundary_param
            ),
            t = state.context_state.real_t,
            i = state.context_state.t[0]
        )
    else:
        raise NotImplementedError
    return context

def env_creator(**kwargs):
    """
    make env `pyth_idsim`
    """
    assert "env_config" in kwargs.keys(), "env_config must be specified"
    env_config = deepcopy(kwargs["env_config"])

    assert "env_scenario" in kwargs.keys(), "env_scenario must be specified"
    env_scenario = kwargs["env_scenario"]

    assert 'scenario_root' in env_config, "scenario_root must be specified in env_config"
    env_config['scenario_root'] = Path(env_config['scenario_root'])
    env_config = Config.from_partial_dict(env_config)

    assert "env_model_config" in kwargs.keys(), "env_model_config must be specified"
    model_config = deepcopy(kwargs["env_model_config"])
    model_config = ModelConfig.from_partial_dict(model_config)

    rou_config = kwargs["rou_config"] if "rou_config" in kwargs.keys() else None

    env_idx = kwargs["env_idx"] if "env_idx" in kwargs.keys() else 0

    scenerios_list = kwargs["scenerios_list"] if "scenerios_list" in kwargs.keys() else None
    env = idSimEnv(env_config, model_config, env_scenario, rou_config, env_idx, scenerios_list)
    return env