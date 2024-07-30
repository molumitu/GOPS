import torch
from typing import NamedTuple, Union

from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig


class State(NamedTuple):
    ego_state: torch.Tensor
    last_last_action: torch.Tensor
    last_action: torch.Tensor


class Parameter(NamedTuple):
    ref_param: torch.Tensor
    sur_param: torch.Tensor
    light_param: torch.Tensor
    ref_index_param: torch.Tensor
    boundary_param: torch.Tensor = torch.tensor([0.0, 0.0])


class BaseContext(NamedTuple):
    x: State
    p: Parameter
    t: Union[int, torch.Tensor]
    i: Union[int, torch.Tensor]

    def advance(self, x: State):
        raise NotImplementedError

    @classmethod
    def from_env(cls, env, model_config: ModelConfig, ref_index_param: int = None):
        raise NotImplementedError
