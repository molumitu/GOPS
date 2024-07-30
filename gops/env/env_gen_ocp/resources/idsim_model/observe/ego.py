import torch

from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext


def get_ego_obs(context: BaseContext) -> torch.Tensor:
    # [batch, feature]
    ego_state = context.x.ego_state
    last_last_action = context.x.last_last_action
    last_action = context.x.last_action
    x, y, vx, vy, phi, r = ego_state.unbind(dim=-1)
    last_last_acc, last_last_steer = last_last_action.unbind(dim=-1)
    last_acc, last_steer = last_action.unbind(dim=-1)
    ego_obs = torch.stack(
        (vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer), dim=-1)
    return ego_obs
