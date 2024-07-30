import torch

from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext
from gops.env.env_gen_ocp.resources.idsim_model.utils.math_utils import convert_ground_coord_to_ego_coord


def get_single_step_sur_obs(context: BaseContext, sur_obs_cur) -> torch.Tensor:
    # transform to ego coord
    ego_state = context.x.ego_state
    ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state.unbind(dim=-1)
    sur_x, sur_y, sur_phi, sur_vx, sur_length, sur_width, sur_mask = sur_obs_cur.unbind(
        dim=-1)
    rel_x_ego_coord, rel_y_ego_coord, rel_phi_ego_coord = \
        convert_ground_coord_to_ego_coord(sur_x, sur_y, sur_phi,
                                          ego_x.unsqueeze(dim=-1), ego_y.unsqueeze(dim=-1), ego_phi.unsqueeze(dim=-1))

    sur_obs = torch.stack((
        rel_x_ego_coord,
        rel_y_ego_coord,
        torch.cos(rel_phi_ego_coord),
        torch.sin(rel_phi_ego_coord),
        sur_obs_cur[:, :, 3],
    ), dim=-1)
    return sur_obs


def get_sur_obs(context: BaseContext, n: int) -> torch.Tensor:
    sur_obs_list = []
    sur_param = context.p.sur_param
    for i in range(context.i, context.i + n):
        sur_obs_cur = sur_param[:, i]
        sur_obs = get_single_step_sur_obs(context, sur_obs_cur)  # [B, M, d]
        sur_obs_list.append(sur_obs)
    mask = sur_obs_cur[:, :, -1]
    sur_obs = torch.stack(sur_obs_list, dim=-1)  # [B, M, d, N]
    sur_obs = sur_obs.reshape(sur_obs.shape[0], sur_obs.shape[1], -1)

    sur_info_obs = torch.stack((
        sur_obs_cur[:, :, 4],
        sur_obs_cur[:, :, 5],
        mask
    ), dim=-1)  # [B, M, 3]
    sur_obs = torch.cat((sur_obs, sur_info_obs), dim=-1)
    # [B, M, d]
    batch_size = sur_obs.shape[0]
    sur_obs = sur_obs.reshape(batch_size, -1)
    return sur_obs
