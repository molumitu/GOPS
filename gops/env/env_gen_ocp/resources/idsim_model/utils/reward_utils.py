import torch
import numpy as np
from typing import Tuple

def separate_obs4toyota(o: torch.Tensor,
                        pre_horizon: int,
                        ego_feat_dim: int,
                        per_ref_feat_dim: int,
                        per_sur_feat_dim: int,
                        sur_num: int
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start_idx = 0
    interval_list = [
        ego_feat_dim,
        per_ref_feat_dim * (pre_horizon + 1)]
    separated_obs = []
    for i in range(2):
        end_idx = start_idx + interval_list[i]
        separated_obs.append(o[:, start_idx:end_idx])
        start_idx = end_idx
    separated_obs.append(o[:, start_idx:])
    ego_obs, ref_obs, sur_full_obs = separated_obs
    batch_size = ego_obs.shape[0]
    ref_obs = ref_obs.reshape(batch_size, per_ref_feat_dim, pre_horizon + 1)
    sur_full_obs = sur_full_obs.reshape(batch_size, sur_num, -1)
    sur_obs = sur_full_obs[:, :, :-3]
    sur_obs = sur_obs.reshape(batch_size, sur_num, per_sur_feat_dim, -1)
    sur_info_obs = sur_full_obs[:, :, -3:]
    sur_info_obs = sur_info_obs.reshape(batch_size, sur_num, 3)
    ## sur_obs [B, M, d, N]
    ## sur_info_obs [B, M, 3]
    return ego_obs, ref_obs, sur_obs, sur_info_obs

def closest_dist_2_circles(
    rela_x: torch.Tensor,
    rela_y: torch.Tensor,
    rela_phi_cos: torch.Tensor,
    rela_phi_sin: torch.Tensor,
    other_length: torch.Tensor,
    other_width: torch.Tensor,
    ego_length: float,
    ego_width: float
    ):
    other_radius = 0.5 * torch.sqrt(torch.square(other_width) + torch.square(other_length / 2))
    ego_radius = 0.5 * np.sqrt(ego_width ** 2 + (ego_length / 2) ** 2)
    other_circle_rela_x = torch.concat([rela_x.unsqueeze(-1) for _ in range(2 * 2)], dim=-1) + (other_length / 4 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 1, -1, -1])
    other_circle_rela_y =  torch.concat([rela_y.unsqueeze(-1) for _ in range(2 * 2)], dim=-1) + (other_length / 4 * rela_phi_sin).unsqueeze(-1) * torch.tensor([1, 1, -1, -1])
    dist_between_circles = torch.square(other_circle_rela_x - ego_length/4 * torch.tensor([1, -1, 1, -1])) + torch.square(other_circle_rela_y)
    # dist_closest = torch.sqrt(torch.min(dist_between_circles, dim=-1).values) - other_radius - ego_radius
    dist_closest = torch.sqrt(torch.min(dist_between_circles, dim=-1).values)
    return dist_closest, other_radius + ego_radius


def closest_dist_3_circles(
    rela_x: torch.Tensor,
    rela_y: torch.Tensor,
    rela_phi_cos: torch.Tensor,
    rela_phi_sin: torch.Tensor,
    other_length: torch.Tensor,
    other_width: torch.Tensor,
    ego_length: float,
    ego_width: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    other_radius = 0.5 * other_width
    ego_radius = 0.5 * np.sqrt(ego_width ** 2 + (ego_length / 2) ** 2)

    other_circle_rela_x = torch.concat([rela_x.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_x += (other_length / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_x += (other_radius * rela_phi_cos).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    other_circle_rela_y = torch.concat([rela_y.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_y += (other_length / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_y += (other_radius * rela_phi_sin).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    ego_c1_x = 3/4*ego_length * torch.ones((3))
    ego_c2_x = 1/4*ego_length * torch.ones((3))
    ego_c3_x = -1/4*ego_length * torch.ones((3))

    dists_to_c1 = torch.square(other_circle_rela_x - ego_c1_x) + torch.square(other_circle_rela_y)
    dists_to_c2 = torch.square(other_circle_rela_x - ego_c2_x) + torch.square(other_circle_rela_y)
    dists_to_c3 = torch.square(other_circle_rela_x - ego_c3_x) + torch.square(other_circle_rela_y)
    dists_to_circles = torch.cat([dists_to_c1, dists_to_c2, dists_to_c3], dim=-1)
    dist_closest = torch.sqrt(torch.min(dists_to_circles, dim=-1).values)

    return dist_closest, other_radius + ego_radius


def closest_dist_4_circles(
    rela_x: torch.Tensor,
    rela_y: torch.Tensor,
    rela_phi_cos: torch.Tensor,
    rela_phi_sin: torch.Tensor,
    other_length: torch.Tensor,
    other_width: torch.Tensor,
    ego_length: float,
    ego_width: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    other_radius = 0.5 * other_width
    # ego_radius = 0.5 * np.sqrt(ego_width ** 2 + (ego_length / 3) ** 2)
    ego_radius = 0.5 * ego_width

    other_circle_rela_x = torch.concat([rela_x.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_x += (other_length / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_x += (other_radius * rela_phi_cos).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    other_circle_rela_y = torch.concat([rela_y.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_y += (other_length / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_y += (other_radius * rela_phi_sin).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    ego_c1_x = (ego_length/2 + ego_width/2) * torch.ones((3))
    ego_c2_x = (ego_length/2 - ego_width/2) * torch.ones((3))
    ego_c3_x = 0*ego_length * torch.ones((3))
    ego_c4_x = (-ego_length/2 + ego_width/2) * torch.ones((3))

    dists_to_c1 = torch.square(other_circle_rela_x - ego_c1_x) + torch.square(other_circle_rela_y)
    dists_to_c2 = torch.square(other_circle_rela_x - ego_c2_x) + torch.square(other_circle_rela_y)
    dists_to_c3 = torch.square(other_circle_rela_x - ego_c3_x) + torch.square(other_circle_rela_y)
    dists_to_c4 = torch.square(other_circle_rela_x - ego_c4_x) + torch.square(other_circle_rela_y)
    dists_to_circles = torch.cat([dists_to_c1, dists_to_c2, dists_to_c3, dists_to_c4], dim=-1)
    dist_closest = torch.sqrt(torch.min(dists_to_circles, dim=-1).values)

    return dist_closest, other_radius + ego_radius


def dist_4to3_circles(
    rela_x: torch.Tensor,
    rela_y: torch.Tensor,
    rela_phi_cos: torch.Tensor,
    rela_phi_sin: torch.Tensor,
    other_length: torch.Tensor,
    other_width: torch.Tensor,
    ego_length: float,
    ego_width: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    other_radius = 0.5 * other_width
    ego_radius = 0.5 * ego_width

    other_circle_rela_x = torch.concat([rela_x.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_x += (other_length / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_x += (other_radius * rela_phi_cos).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    other_circle_rela_y = torch.concat([rela_y.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_y += (other_length / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_y += (other_radius * rela_phi_sin).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    ego_c1_x = (ego_length/2 + ego_width/2) * torch.ones((3))
    ego_c2_x = (ego_length/2 - ego_width/2) * torch.ones((3))
    ego_c3_x = 0*ego_length * torch.ones((3))
    ego_c4_x = (-ego_length/2 + ego_width/2) * torch.ones((3))

    dists_to_c1 = torch.square(other_circle_rela_x - ego_c1_x) + torch.square(other_circle_rela_y)
    dists_to_c2 = torch.square(other_circle_rela_x - ego_c2_x) + torch.square(other_circle_rela_y)
    dists_to_c3 = torch.square(other_circle_rela_x - ego_c3_x) + torch.square(other_circle_rela_y)
    dists_to_c4 = torch.square(other_circle_rela_x - ego_c4_x) + torch.square(other_circle_rela_y)
    dists_to_circles = torch.cat([dists_to_c1, dists_to_c2, dists_to_c3, dists_to_c4], dim=-1)
    dists_to_circles = torch.sqrt(dists_to_circles)
    # [Batch, num_obs, num_circle]
    return dists_to_circles, (other_radius + ego_radius).unsqueeze(-1)


def dist_3to2_circles(
    rela_x: torch.Tensor,
    rela_y: torch.Tensor,
    rela_phi_cos: torch.Tensor,
    rela_phi_sin: torch.Tensor,
    other_length: torch.Tensor,
    other_width: torch.Tensor,
    ego_length: float,
    ego_width: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    other_radius = 0.5 * other_width
    ego_radius = 0.5 * ego_width
    other_bias = (other_length - other_width) / 2
    ego_bias = (ego_length - ego_width) / 2

    device = rela_x.device

    other_circle_rela_x = rela_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 2)
    other_circle_rela_x += (other_bias * rela_phi_cos).unsqueeze(-1).unsqueeze(-1) * \
                           torch.tensor([1, -1], device=device).repeat(1, 1, 3, 1)

    other_circle_rela_y = rela_y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 2)
    other_circle_rela_y += (other_bias * rela_phi_sin).unsqueeze(-1).unsqueeze(-1) * \
                           torch.tensor([1, -1], device=device).repeat(1, 1, 3, 1)

    ego_x = ego_bias * torch.tensor([[1, 1], [-1, -1], [2.5, 2.5]], device=device)

    dist_x = other_circle_rela_x - ego_x
    dist_y = other_circle_rela_y
    dists_to_circles = torch.sqrt(torch.square(dist_x) + torch.square(dist_y)+1e-8)

    # [Batch, num_obs, ego_circle_num(3), sur_circle_num(2)]
    return dists_to_circles, (other_radius + ego_radius).unsqueeze(-1).unsqueeze(-1)

def dist_3to3_circles(
    rela_x: torch.Tensor,
    rela_y: torch.Tensor,
    rela_phi_cos: torch.Tensor,
    rela_phi_sin: torch.Tensor,
    other_length: torch.Tensor,
    other_width: torch.Tensor,
    ego_length: float,
    ego_width: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    other_radius = 0.5 * other_width
    ego_radius = 0.5 * ego_width

    other_circle_rela_x = torch.concat([rela_x.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_x += (other_length / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_x += (other_radius * rela_phi_cos).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    other_circle_rela_y = torch.concat([rela_y.unsqueeze(-1) for _ in range(3)], dim=-1)
    other_circle_rela_y += (other_length / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([1, 0, -1])
    other_circle_rela_y += (other_radius * rela_phi_sin).unsqueeze(-1) * torch.tensor([-1, 0, 1])

    ego_c2_x = (ego_length/2 - ego_width/2) * torch.ones((3))
    ego_c3_x = 0*ego_length * torch.ones((3))
    ego_c4_x = (-ego_length/2 + ego_width/2) * torch.ones((3))

    dists_to_c2 = torch.square(other_circle_rela_x - ego_c2_x) + torch.square(other_circle_rela_y)
    dists_to_c3 = torch.square(other_circle_rela_x - ego_c3_x) + torch.square(other_circle_rela_y)
    dists_to_c4 = torch.square(other_circle_rela_x - ego_c4_x) + torch.square(other_circle_rela_y)
    dists_to_circles = torch.cat([dists_to_c2, dists_to_c3, dists_to_c4], dim=-1)
    dists_to_circles = torch.sqrt(dists_to_circles)
    # [Batch, num_obs, num_circle]
    return dists_to_circles, (other_radius + ego_radius).unsqueeze(-1)


# def closest_dist_3_circles(
#     rela_x: torch.Tensor,
#     rela_y: torch.Tensor,
#     rela_phi_cos: torch.Tensor,
#     rela_phi_sin: torch.Tensor,
#     other_length: torch.Tensor,
#     other_width: torch.Tensor,
#     ego_length: float,
#     ego_width: float
#     ):
#     ego_radius = 0.5 * np.sqrt(ego_width ** 2 + (ego_length / 2) ** 2)
#     ## recover 4 corners of the other vehicle
#     other_corners_x = torch.concat([rela_x.unsqueeze(-1) for _ in range(8)], dim=-1)
#     other_corners_x += (other_length / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 0, -1, -1, -1, 0, 1, 1])
#     other_corners_x += (other_width / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([-1, -1, -1, 0, 1, 1, 1, 0])

#     other_corners_y = torch.concat([rela_y.unsqueeze(-1) for _ in range(8)], dim=-1)
#     other_corners_y += (other_length / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([1, 0, -1, -1, -1, 0, 1, 1])
#     other_corners_y += (other_width / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 1, 1, 0, -1, -1, -1, 0])

#     ego_c1_x = 3/4*ego_length * torch.ones((8))
#     ego_c2_x = 1/4*ego_length * torch.ones((8))
#     ego_c3_x = -1/4*ego_length * torch.ones((8))

#     dists_to_c1 = torch.square(other_corners_x - ego_c1_x) + torch.square(other_corners_y)
#     dists_to_c2 = torch.square(other_corners_x - ego_c2_x) + torch.square(other_corners_y)
#     dists_to_c3 = torch.square(other_corners_x - ego_c3_x) + torch.square(other_corners_y)
#     dists_to_circles = torch.cat([dists_to_c1, dists_to_c2, dists_to_c3], dim=-1)
#     dist_closest = torch.sqrt(torch.min(dists_to_circles, dim=-1).values)
#     return dist_closest, ego_radius

def closest_dist_ellipse(rela_x: torch.Tensor,
    rela_y: torch.Tensor,
    rela_phi_cos: torch.Tensor,
    rela_phi_sin: torch.Tensor,
    other_length: torch.Tensor,
    other_width: torch.Tensor,
    ego_length: float,
    ego_width: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    ellipse_a = torch.tensor(6.0, dtype=torch.float32)
    ellipse_b = torch.tensor(2.6, dtype=torch.float32)
    ellipse_c = torch.sqrt(torch.square(ellipse_a) - torch.square(ellipse_b))
    ## recover 4 corners of the other vehicle
    other_corners_x = torch.concat([rela_x.unsqueeze(-1) for _ in range(8)], dim=-1)
    other_corners_x += (other_length / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 0, -1, -1, -1, 0, 1, 1])
    other_corners_x += (other_width / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([-1, -1, -1, 0, 1, 1, 1, 0])

    other_corners_y = torch.concat([rela_y.unsqueeze(-1) for _ in range(8)], dim=-1)
    other_corners_y += (other_length / 2 * rela_phi_sin).unsqueeze(-1) * torch.tensor([1, 0, -1, -1, -1, 0, 1, 1])
    other_corners_y += (other_width / 2 * rela_phi_cos).unsqueeze(-1) * torch.tensor([1, 1, 1, 0, -1, -1, -1, 0])

    ellipse_c1_x = torch.tensor([ellipse_c, ellipse_c, ellipse_c, ellipse_c, ellipse_c, ellipse_c, ellipse_c, ellipse_c])
    ellipse_c1_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    ellipse_c2_x = torch.tensor([-ellipse_c, -ellipse_c, -ellipse_c, -ellipse_c, -ellipse_c, -ellipse_c, -ellipse_c, -ellipse_c])
    ellipse_c2_y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])

    dists_to_c1 = torch.square(other_corners_x - ellipse_c1_x) + torch.square(other_corners_y - ellipse_c1_y)
    dists_to_c2 = torch.square(other_corners_x - ellipse_c2_x) + torch.square(other_corners_y - ellipse_c2_y)
    dists_sum = torch.sqrt(dists_to_c1) + torch.sqrt(dists_to_c2)
    dist_closest = torch.min(dists_sum, dim=-1).values
    return dist_closest, 2 * ellipse_a

def square_loss(error: torch.Tensor) -> torch.Tensor:
    return torch.square(error)

def log_loss(error: torch.Tensor) -> torch.Tensor:
    return torch.log10(error + 1)

def tanh_loss(error: torch.Tensor) -> torch.Tensor:
    return torch.tanh(error)
