import torch
from typing import List

from gops.env.env_gen_ocp.resources.idsim_model.model_context import Parameter, State
from gops.env.env_gen_ocp.resources.idsim_model.multilane.context import MultiLaneContext


def stack_samples(samples: List[MultiLaneContext]) -> MultiLaneContext:
    ego_state = torch.stack([s.x.ego_state for s in samples])
    last_action = torch.stack([s.x.last_action for s in samples])
    last_last_action = torch.stack([s.x.last_last_action for s in samples])
    ref_param = torch.stack([s.p.ref_param for s in samples])
    sur_param = torch.stack([s.p.sur_param for s in samples])
    light_param = torch.stack([s.p.light_param for s in samples])
    ref_index_param = torch.stack([s.p.ref_index_param for s in samples])
    boundary_param = torch.stack([s.p.boundary_param for s in samples]) 
    t = torch.tensor([s.t for s in samples])  # [B, ]

    return MultiLaneContext(
        x=State(ego_state, last_last_action, last_action),
        p=Parameter(ref_param=ref_param, sur_param=sur_param,
                    light_param=light_param, ref_index_param=ref_index_param,
                    boundary_param=boundary_param,
                    ),
        t=t,
        i=0,
    )


def stack_samples_full_horizon(samples: List[MultiLaneContext]) -> MultiLaneContext:
    ego_state = torch.stack([s.x.ego_state for s in samples], dim=1)
    last_action = torch.stack([s.x.last_action for s in samples], dim=1)
    last_last_action = torch.stack(
        [s.x.last_last_action for s in samples], dim=1)
    ref_param = torch.stack([s.p.ref_param for s in samples], dim=1)
    sur_param = torch.stack([s.p.sur_param for s in samples], dim=1)
    light_param = torch.stack([s.p.light_param for s in samples], dim=1)
    ref_index_param = torch.stack(
        [s.p.ref_index_param for s in samples], dim=1)
    boundary_param = torch.stack([s.p.boundary_param for s in samples], dim=1)  # [B, 2]
    t = torch.stack([s.t for s in samples], dim=1)
    return MultiLaneContext(
        x=State(ego_state, last_last_action, last_action),
        p=Parameter(ref_param=ref_param, sur_param=sur_param,
                    light_param=light_param, ref_index_param=ref_index_param,
                    boundary_param=boundary_param,
                    ),
        t=t,
        i=torch.tensor([s.i for s in samples])
    )


def select_sample(context: MultiLaneContext, index: int) -> MultiLaneContext:
    return MultiLaneContext(
        x=State(
            ego_state=context.x.ego_state[:, index],
            last_last_action=context.x.last_last_action[:, index],
            last_action=context.x.last_action[:, index],
        ),
        p=Parameter(
            ref_param=context.p.ref_param[:, index],
            sur_param=context.p.sur_param[:, index],
            light_param=context.p.light_param[:, index],
            ref_index_param=context.p.ref_index_param[:, index],
            boundary_param=context.p.boundary_param[:, index],
        ),
        t=context.t[:, index],
        i=context.i[index],
    )
