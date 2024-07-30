import torch

from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext
from gops.env.env_gen_ocp.resources.idsim_model.utils.math_utils import convert_ground_coord_to_ego_coord, convert_ref_to_ego_coord, convert_ego_to_ref_coord


def select_ref_by_index(multi_ref: torch.Tensor, ref_index: torch.Tensor) -> torch.Tensor:
    # multi_ref [batch, num_ref_lines, feature]
    # ref_index [batch]
    ref_index_for_gather = ref_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, multi_ref.size(-1))
    single_ref = multi_ref.gather(1, ref_index_for_gather).squeeze(1)
    return single_ref


def compute_onref_mask(
    context: BaseContext,
    *,
    aggregation: str = 'or',
) -> torch.Tensor:
    # B = context.x.ego_state.shape[0]
    # return torch.ones(B, self.M).bool()
    ego_state = context.x.ego_state  # [B, 6] # x, y, vx, vy, phi, r
    ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state.unbind(dim=-1)
    sur_param = context.p.sur_param
    sur_state = sur_param[:, context.i]  # [B, M, 7]
    sur_x, sur_y, sur_phi, sur_vx, sur_length, sur_width, sur_mask = sur_state.unbind(
        dim=-1)
    # [B, M]
    rel_x_ego_coord, rel_y_ego_coord, rel_phi_ego_coord = \
        convert_ground_coord_to_ego_coord(sur_x, sur_y, sur_phi,
                                          ego_x.unsqueeze(dim=-1), ego_y.unsqueeze(dim=-1), ego_phi.unsqueeze(dim=-1))
    onref_mask = ~(
        # (
        #     (rel_y_ego_coord > 3)
        #     * (torch.abs(rel_phi_ego_coord) > (torch.pi * 175/180))
        # )
        # +
        (
            (rel_x_ego_coord < -2.5)
            * (torch.abs(rel_y_ego_coord) < 1)
            * (torch.abs(rel_phi_ego_coord) < (torch.pi / 18))
        )
    )
    return onref_mask


def get_ref_obs(context: BaseContext, num_ref_points, num_ref_lines) -> torch.Tensor:
    # ref_param [batch, num_ref_lines, num_ref_points, feature[delta_x, delta_y, delta_phi]]
    # ego coordinate
    ego_state = context.x.ego_state
    ref_param = context.p.ref_param
    multi_ref_obs_absolute = ref_param[:, :, context.i:context.i + num_ref_points]
    multi_ref_obs = []
    for i in range(num_ref_lines):
        ref_obs_absolute = multi_ref_obs_absolute[:, i, :, :3]
        ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = convert_ref_to_ego_coord(ref_obs_absolute, ego_state)
        vx = ego_state[:, 2].unsqueeze(-1)
        ref_vx = multi_ref_obs_absolute[:, i, :, -1]
        vx_error = ref_vx - vx
        ref_obs = torch.concat((ref_x_ego_coord, ref_y_ego_coord,
                                torch.cos(ref_phi_ego_coord), torch.sin(ref_phi_ego_coord), vx_error), axis=-1)
        multi_ref_obs.append(ref_obs)
    # [batch, num_ref_lines, feature]
    return torch.stack(multi_ref_obs, dim=1)

def get_ref_obs_frenet_coord(context: BaseContext, num_ref_points, num_ref_lines) -> torch.Tensor:
    # ref_param [batch, num_ref_lines, num_ref_points, feature[vx_error, lat_error, phi_error]]
    # frenet coordinate, transform_ego_to_ref_coord
    ego_state = context.x.ego_state
    ref_param = context.p.ref_param
    multi_ref_obs_absolute = ref_param[:, :, context.i:context.i + num_ref_points]
    multi_ref_obs = []
    for i in range(num_ref_lines):
        ref_obs_absolute = multi_ref_obs_absolute[:, i, :, :3]
        ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = convert_ego_to_ref_coord(ref_obs_absolute, ego_state)  # frenet coordinate
        vx = ego_state[:, 2].unsqueeze(-1)
        ref_vx = multi_ref_obs_absolute[:, i, :, -1]
        vx_error = vx - ref_vx  # frenet coordinate
        ref_obs = torch.concat((ref_x_ego_coord, ref_y_ego_coord,
                                torch.cos(ref_phi_ego_coord), torch.sin(ref_phi_ego_coord), vx_error), axis=-1)
        multi_ref_obs.append(ref_obs)
    # [B, R, d]
    return torch.stack(multi_ref_obs, dim=1)
