import pathlib
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import torch
from idsim.config import Config
from idsim.envs.env import CrossRoad
from idsim.utils.fs import TEMP_ROOT
from idsim_model.model_context import ModelContext
from idsim_model.params import model_config as default_model_config

from gops.env.env_gen_ocp.pyth_base import (Context, ContextState, Env, Robot,
                                            State, stateType)


@dataclass
class idSimContextState(ContextState):
    light_param: stateType
    ref_index_param: stateType
    real_t: stateType


@dataclass
class idSimState(State):
    context_state: idSimContextState
    CONTEXT_STATE_TYPE = idSimContextState


class idSimEnv(CrossRoad, Env):
    def __init__(self, config: Config):
        super(idSimEnv, self).__init__(config)
        # TODO: add model config
        self.model_config = None
    
    def set_model_config(self, model_config):
        self.model_config = model_config

    def reset(self) -> Tuple[np.ndarray, dict]:
        obs, info = super(idSimEnv, self).reset()
        self._get_state_from_idsim()
        return obs, self._get_info(info)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super(idSimEnv, self).step(action)
        self._get_state_from_idsim()
        done = terminated or truncated
        return obs, reward, done, self._get_info(info)
    
    def _get_info(self, info) -> dict:
        info.update({'state': self._state})
        try:
            info['cost'] = self._get_constraint()
        except NotImplementedError:
            pass
        return info
    
    def _get_obs(self) -> np.ndarray:
        """abandon this function, use obs from idsim instead"""
        ...

    def _get_reward(self, action: np.ndarray) -> float:
        """abandon this function, use reward from idsim instead"""
        ...
    
    def _get_terminated(self) -> bool:
        """abandon this function, use terminated from idsim instead"""
        ...
    
    def _get_state_from_idsim(self, ref_index_param=None) -> State:
        idsim_context = ModelContext.from_env(self, self.model_config, ref_index_param)
        self._state = idSimState(
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
                real_t = torch.tensor(idsim_context.t).int(),
                t = torch.tensor(idsim_context.i).int()
            )
        )
        self._state = idSimState.tensor2array(self._state)

    def get_state_from_idsim(self, ref_index_param=None) -> State:
        self._get_state_from_idsim(ref_index_param=ref_index_param)
        return self.state

def env_creator(**kwargs):
    """
    make env `pyth_idsim`
    """
    env_config = deepcopy(kwargs["env_config"])
    if 'scenario_root' in env_config:
        env_config['scenario_root'] = Path(env_config['scenario_root'])
    env_config = Config.from_partial_dict(env_config)
    if "env_model_config" in kwargs.keys():
        model_config = kwargs["env_model_config"]
    else:
        model_config = default_model_config
    env = idSimEnv(env_config)
    env.set_model_config(model_config)
    return env