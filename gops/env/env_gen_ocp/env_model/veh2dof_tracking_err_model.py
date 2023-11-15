import torch

from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.env_model.veh2dof_tracking_model import Veh2DoFTrackingModel


class Veh2DoFTrackingErrModel(Veh2DoFTrackingModel):
    def get_constraint(self, state: State) -> torch.Tensor:
        y = state.robot_state[:, 0]
        y_ref = state.context_state.index_by_t().reference[:, 1]
        y_error_tol = state.context_state.constraint[:, 0]
        return (torch.abs(y - y_ref) - y_error_tol).unsqueeze(1)


def env_model_creator(**kwargs) -> Veh2DoFTrackingErrModel:
    return Veh2DoFTrackingErrModel(**kwargs)


if __name__ == "__main__":
    from gops.env.env_gen_ocp.veh2dof_tracking_err import Veh2DoFTrackingErr
    from gops.env.inspector.consistency_checker import check_env_model_consistency

    env = Veh2DoFTrackingErr()
    model = Veh2DoFTrackingErrModel()
    check_env_model_consistency(env, model)
