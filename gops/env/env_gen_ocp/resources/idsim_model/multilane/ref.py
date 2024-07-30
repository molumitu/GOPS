import numpy as np
from typing import Dict, List, Tuple

# from idsim.lib import point_project_to_line, compute_waypoints_by_intervals
from gops.env.env_gen_ocp.resources.lib import point_project_to_line, compute_waypoints_by_intervals
# from idsim.envs.env import CrossRoad
from gops.env.env_gen_ocp.pyth_base import Env as CrossRoad
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig


def get_ref_param(env: CrossRoad, model_config: ModelConfig, light_param: np.ndarray) -> np.ndarray:
    # [num_ref_lines, num_ref_points+N, per_point_dim]
    N = model_config.N
    num_ref_lines = model_config.num_ref_lines
    num_ref_points = model_config.num_ref_points
    ref_list = env.engine.context.vehicle.reference_list
    ref_info_list = env.engine.context.vehicle.reference_info_list
    ref_v_lane = model_config.ref_v_lane
    dt = env.config.dt

    traffic_light = light_param[0, 0]
    path_planning_mode = "green"
    if traffic_light == 0:
        am = model_config.dec_before_junction_green
        path_planning_mode = "green"
    else:
        am = model_config.dec_before_junction_red
        path_planning_mode = "red"
        min_ahead_lane_length = model_config.ahead_lane_length_min
    
    driving_task = env.engine.context.vehicle.direction.lower()
    if driving_task == "s":
        ref_v_junction = ref_v_lane * model_config.v_discount_in_junction_straight
    elif driving_task == "l":
        ref_v_junction = ref_v_lane * model_config.v_discount_in_junction_left_turn
    elif driving_task == "r":
        ref_v_junction = ref_v_lane * model_config.v_discount_in_junction_right_turn
    else:
        raise ValueError("Error driving task: {}".format(driving_task))
    
    if env.engine.context.ref_acc is not None:
        cur_v = ref_v_lane + env.engine.context.ref_acc * env.engine.context.acc_time * dt
        cur_v = np.clip(cur_v, a_min=0, a_max=ref_v_lane+10) # TODO: remove this
    else:
        cur_v = ref_v_lane

    ref_param = []
    for ref_line in ref_list:
        ref_info = ref_info_list[ref_list.index(ref_line)]
        current_part = ref_info[0]
        position_on_ref = point_project_to_line(
            ref_line, *env.engine.context.vehicle.ground_position)
        
        if current_part['destination'] == True:
            ref_info = ref_info_list[ref_list.index(ref_line)]
            position_on_ref = point_project_to_line(ref_line, *env.engine.context.vehicle.ground_position)
            intervals, ref_v = compute_intervals(ref_info, num_ref_points + N -1, cur_v, ref_v_lane, dt, env.engine.context.ref_acc, )
        elif current_part['destination'] == False and current_part["in_junction"] == True:
            intervals, ref_v = compute_intervals_in_junction(
                num_ref_points + N - 1, ref_v_junction, dt)
        elif current_part["in_junction"] == False and current_part['destination'] == False:
            if path_planning_mode == "green":
                intervals, ref_v = compute_intervals_initsegment_green(
                    position_on_ref, current_part, num_ref_points + N - 1, ref_v_lane, ref_v_junction, dt, am)
            elif path_planning_mode == "red":
                intervals, ref_v = compute_intervals_initsegment_red(
                    position_on_ref, current_part, num_ref_points + N - 1, ref_v_lane, dt, am, min_ahead_lane_length)
            else:
                raise ValueError("Error path_planning_mode")
        else:
            raise ValueError("Error ref_line")
        # repeat the last v
        ref_v = np.append(ref_v, ref_v[-1])
        ref_v = np.expand_dims(ref_v, axis=1)
        
        ref_array = compute_waypoints_by_intervals(ref_line, position_on_ref, intervals)
        ref_array = np.concatenate((ref_array, ref_v), axis=-1)
        ref_param.append(ref_array)
    # padding
    for _ in range(num_ref_lines - len(ref_list)):
        ref_param.append(ref_param[0])
    return np.array(ref_param)


def update_ref_param(env: CrossRoad,
                     ref_param: np.ndarray,
                     light_param: np.ndarray,
                     model_config: ModelConfig) -> np.ndarray:
    #TODO: implement this
    return ref_param
def compute_intervals_destination(total_num: int,
                      ref_v_lane: float,
                      dt: float,) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    # ref_v keeps 8.0
    intervals[:] = ref_v_lane * dt
    ref_v[:] = ref_v_lane
    return intervals, ref_v

def compute_intervals_in_junction(total_num: int,
                      ref_v_junction: float,
                      dt: float,) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    # ref_v from ref_v_junction to ref_v_lane
    # ahead_length = current_part['length'] - position_on_ref
    # n1 = int(ahead_length / ref_v_junction)
    # intervals[:n1] = ref_v_junction * dt
    # intervals[n1:] = ref_v_lane * dt
    # ref_v[:n1] = ref_v_junction
    # ref_v[n1:] = ref_v_lane

    # ref_v keeps ref_v_junction for all points if ego is in the junction
    intervals[:] = ref_v_junction * dt
    ref_v[:] = ref_v_junction
    return intervals, ref_v


def compute_intervals_initsegment_green(position_on_ref: float,
                      current_part: Dict[str, float],
                      total_num: int,
                      ref_v_lane: float,
                      ref_v_junction: float,
                      dt: float,
                      am: float) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    position_on_ref = np.clip(position_on_ref, a_min=0,
                              a_max=current_part['length'])
    # ref_v from ref_v_lane to ref_v_junction
    ahead_length = current_part['length'] - position_on_ref
    v0 = np.sqrt(2 * am * ahead_length + ref_v_junction ** 2)
    ref_v = [v0 - am * dt * i for i in range(total_num)]
    # when v < v_junction and v > v_lane, a = am
    a_list = [am if v > ref_v_junction and v <
                    ref_v_lane else 0 for v in ref_v]
    ref_v = np.clip(ref_v, a_min=ref_v_junction, a_max=ref_v_lane)
    intervals = [v * dt - 0.5 * a * dt * dt for v, a in zip(ref_v, a_list)]
    intervals = np.clip(
        intervals, a_min=ref_v_junction * dt, a_max=ref_v_lane * dt)
    return intervals, ref_v


def compute_intervals_initsegment_red(position_on_ref: float,
                      current_part: Dict[str, float],
                      total_num: int,
                      ref_v_lane: float,
                      dt: float,
                      am: float,
                      min_ahead_lane_length: float) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for red mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    position_on_ref = np.clip(position_on_ref, a_min=0,
                              a_max=current_part['length'])
    # ref_v from ref_v_lane to 0
    ahead_length = current_part['length'] - position_on_ref - (min_ahead_lane_length - 4) # FIXME: 4 is 0.8*ego_length, which has been considered in current_part['length'] for red light
    if ahead_length < 0.01:
        intervals = np.zeros((total_num, ))
        ref_v = np.zeros((total_num, ))
    else:
        v0 = np.sqrt(2 * am * ahead_length)
        ref_v = [v0 - am * dt * i for i in range(total_num)]
        # when v < v_junction and v > v_lane, a = am
        a_list = [am if v > 0 and v <
                        ref_v_lane else 0 for v in ref_v]
        ref_v = np.clip(ref_v, a_min=0, a_max=ref_v_lane)
        intervals = [v * dt for v, a in zip(ref_v, a_list)]
        intervals = np.clip(
            intervals, a_min=0 * dt, a_max=ref_v_lane * dt)
    return intervals, ref_v


def compute_intervals(ref_info: List[Dict[str, float]],
                      total_num: int,
                      cur_v: float,
                      ref_v_lane: float,
                      dt: float,
                      acc: float,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    # total_num = total_num - 1
    ref_v = np.ones(total_num) * ref_v_lane
    current_part = ref_info[0]

    assert current_part['destination'] == True, "Error ref_line"
    if acc is not None: # calculate ref_v using acc
        ref_v = [cur_v + acc * dt * i for i in range(total_num)]
        ref_v = np.clip(ref_v, a_min=0, a_max=ref_v_lane+10)

    intervals = ref_v * dt

    return intervals, ref_v
