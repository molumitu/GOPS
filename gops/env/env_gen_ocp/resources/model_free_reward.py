from gops.env.env_gen_ocp.resources.lib import point_project_to_line, compute_waypoint
import numpy as np
import math
from typing import Tuple, Dict

def deal_with_phi_rad(phi: float):
    return (phi + math.pi) % (2*math.pi) - math.pi


def reward_function_multilane(config: Dict, vehicle):
    # tracking_error cost
    position_on_ref = point_project_to_line(vehicle["reference"], *vehicle["ground_position"])
    current_first_ref_x, current_first_ref_y, current_first_ref_phi = compute_waypoint(vehicle["reference"], position_on_ref)

    ego_x, ego_y = vehicle["ground_position"]
    tracking_error = np.sqrt((ego_x - current_first_ref_x) **2 + (ego_y - current_first_ref_y) **2 ) 

    delta_phi = deal_with_phi_rad(vehicle["heading"] - current_first_ref_phi)

    out_of_range = tracking_error > 8 or np.abs(delta_phi) > np.pi/4 
    in_junction = vehicle["in_junction"]

    # collision risk cost
    ego_vx = vehicle["vx"]
    ego_W = vehicle["width"]
    ego_L = vehicle["length"]
    head_ang_error = delta_phi


    safety_lat_margin_front = config["safety_lat_margin_front"]
    safety_lat_margin_rear = safety_lat_margin_front # TODO: safety_lat_margin_rear
    safety_long_margin_front = config["safety_long_margin_front"]
    safety_long_margin_side = config["safety_long_margin_side"]
    front_dist_thd = config["front_dist_thd"]
    space_dist_thd = config["space_dist_thd"]
    rel_v_thd = config["rel_v_thd"]
    rel_v_rear_thd = config["rel_v_rear_thd"]
    time_dist = config["time_dist"]

    punish_done = config["P_done"]
    
    pun2front = 0.
    pun2side = 0.
    pun2space = 0.
    pun2rear = 0.

    pun2front_sum = 0.
    pun2side_sum = 0.
    pun2space_sum = 0.
    pun2rear_sum = 0.

    min_front_dist = np.inf

    sur_info = vehicle["surrounding_veh_info"]
    ego_edge = vehicle["edge"]
    ego_lane = vehicle["lane"]
    if config["ignore_opposite_direction"]:  # FIXME: hardcoded scenario_id
        sur_info = [s for s in sur_info if s[4] == ego_edge]


    for sur_vehicle in sur_info:
        rel_x, rel_y, sur_vx, sur_lane, sur_road, sur_W, sur_L = sur_vehicle
        # [1 - tanh(x)]: 0.25-> 75%  0.5->54%, 1->24%, 1.5->9.5% 2->3.6%, 3->0.5% 
        if np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_front and rel_x > 0:
            min_front_dist = min(min_front_dist, rel_x - (ego_L + sur_L) / 2)  
    
        pun2front_cur = np.where( (sur_lane == ego_lane or  np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_front)
                and rel_x >= 0 and rel_x < front_dist_thd and ego_vx > sur_vx,
            np.clip(1. - np.tanh((rel_x-(ego_L + sur_L) / 2 - safety_long_margin_front) / (time_dist*(np.max(ego_vx,0) + 0.1))), 0., 1.),
            0,
        )
        pun2front = np.maximum(pun2front, pun2front_cur)
        pun2front_sum += pun2front_cur

        pun2side_cur =  np.where(
            np.abs(rel_x) < (ego_L + sur_L) / 2 + safety_long_margin_side and rel_y*head_ang_error > 0 and  rel_y > (ego_W + sur_W) / 2, 
            np.clip(1. - np.tanh((np.abs(rel_y)- (ego_W + sur_W) / 2) / (np.abs(ego_vx*np.sin(head_ang_error))+0.01)), 0., 1.),
            0,
        )
        pun2side = np.maximum(pun2side, pun2side_cur)
        pun2side_sum += pun2side_cur

        pun2space_cur = np.where(
            np.abs(rel_y) < (ego_W + sur_W) / 2 and rel_x >= 0 and rel_x < space_dist_thd and ego_vx > sur_vx + rel_v_thd,
            np.clip(1. - (rel_x - (ego_L + sur_L) / 2) / (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
            0,) + np.where(
            np.abs(rel_x) < (ego_L + sur_L) / 2 and np.abs(rel_y) > (ego_W + sur_W) / 2,
            np.clip(1. - np.tanh(3.0*(np.abs(rel_y) - (ego_W + sur_W) / 2)), 0., 1.),
            0,)
        pun2space = np.maximum(pun2space, pun2space_cur)
        pun2space_sum += pun2space_cur
        
        pun2rear_cur = np.where(
            (sur_lane == ego_lane or  np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_rear) and rel_x < 0 and rel_x > -space_dist_thd and ego_vx < sur_vx - rel_v_rear_thd,
            np.clip(1. - (-1)*(rel_x + (ego_L + sur_L) / 2) / (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
            0,)
        pun2rear = np.maximum(pun2rear, pun2rear_cur)
        pun2rear_sum += pun2rear_cur
        
    if config["punish_sur_mode"] == "sum":
        pun2front = pun2front_sum
        pun2side = pun2side_sum
        pun2space = pun2space_sum
        pun2rear = pun2rear_sum
    elif config["punish_sur_mode"] == "max":
        pass
    else:
        # raise ValueError(f"Invalid punish_sur_mode: {config["punish_sur_mode"]}")
        pass
    scaled_pun2front = pun2front * config["P_front"]
    scaled_pun2side = pun2side * config["P_side"]
    scaled_pun2space = pun2space * config["P_space"]
    scaled_pun2rear = pun2rear * config["P_rear"]
    braking_mode = (min_front_dist < 4)

    punish_collision_risk = scaled_pun2front + scaled_pun2side + scaled_pun2space + scaled_pun2rear

    # out of driving area cost
    if in_junction or config["P_boundary"] == 0: # TODO: boundary cost = 0  when boundary info is not available
        punish_boundary = 0.
    else:
        rel_angle = np.abs(delta_phi)
        left_distance =  np.abs(vehicle["left_distance"])
        right_distance = np.abs(vehicle["right_distance"])
        min_left_distance = left_distance - (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
        min_right_distance = right_distance - (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
        boundary_safe_margin = 0.5
        boundary_distance = np.clip(np.minimum(min_left_distance, min_right_distance), 0.,None)

        punish_boundary = np.where(
            boundary_distance < boundary_safe_margin,
            np.clip((1. - boundary_distance/boundary_safe_margin), 0., 1.),
            0.0,
        )
    scaled_punish_boundary = punish_boundary * config["P_boundary"]
    
    # action related reward

    reward = - scaled_punish_boundary

    if config["penalize_collision"]:
        reward -= punish_collision_risk
    
    event_flag = 0 # nomal driving (on lane, stop)
    # Event reward: target reached, collision, out of driving area
    if ego_vx < 1 and  not braking_mode:  # start to move from stop
        event_flag = 1
    if braking_mode:  # start to brake
        event_flag = 2
    if vehicle["collision"]:   # collision
        reward -= punish_done if config["penalize_collision"] else 0.
        event_flag = 3
    if vehicle["out_of_driving_area"] or out_of_range:  # out of driving area
        reward -= punish_done
        event_flag = 4

    return reward, {
            "category": event_flag,
            "env_pun2front": pun2front,
            "env_pun2side": pun2side,
            "env_pun2space": pun2space,
            "env_pun2rear": pun2rear,
            "env_scaled_reward_part1": reward,
            "env_reward_collision_risk": - punish_collision_risk,
            "env_scaled_pun2front": scaled_pun2front,
            "env_scaled_pun2side": scaled_pun2side,
            "env_scaled_pun2space": scaled_pun2space,
            "env_scaled_pun2rear": scaled_pun2rear,
            "env_scaled_punish_boundary": scaled_punish_boundary,
        }

def model_free_reward_multilane(config: Dict,
                            context, # S_t
                            last_last_action, # absolute action, A_{t-2}
                            last_action, # absolute action, A_{t-1}
                            action # normalized incremental action, （A_t - A_{t-1}） / Z
                            ) -> Tuple[float, dict]:
        # all inputs are batched
        # vehicle state: context.x.ego_state
        ego_state = context.x.ego_state[0] # [6]: x, y, vx, vy, phi, r
        ref_param = context.p.ref_param[0] # [R, 2N+1, 4] ref_x, ref_y, ref_phi, ref_v
        ref_index = context.p.ref_index_param[0]
        ref_state = ref_param[ref_index, context.i, :] # 4
        next_ref_state = ref_param[ref_index, context.i + 1, :] # 4
        ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state
        ref_x, ref_y, ref_phi, ref_v = ref_state
        next_ref_v = next_ref_state[3]
        last_acc, last_steer = last_action[0][0], last_action[0][1]*180/np.pi
        last_last_acc, last_last_steer = last_last_action[0][0], last_last_action[0][1]*180/np.pi
        delta_steer =  (last_steer - last_last_steer)/config["dt"]
        jerk = (last_acc - last_last_acc)/config["dt"]
        # print(f'last_acc: {last_acc}, last_steer: {last_steer}, delta_steer: {delta_steer}, jerk: {jerk}')

        
        # live reward
        rew_step =  1.0 
        
        tracking_error = np.sqrt((ego_x - ref_x) ** 2 + (ego_y - ref_y) ** 2) 
        delta_phi = deal_with_phi_rad(ego_phi - ref_phi)*180/np.pi # degree
        ego_r = ego_r*180/np.pi # degree
        speed_error = ego_vx - ref_v

        # tracking_error
        punish_dist_lat = 5*np.where(
            np.abs(tracking_error) < 0.3,
            np.square(tracking_error),
           0.02* np.abs(tracking_error) + 0.084,
        ) # 0~1 0~6m 50% 0~0.3m

        punish_vel_long = 0.5*np.where(
            np.abs(speed_error) < 1,
            np.square(speed_error),
            0.1*np.abs(speed_error)+0.85,
        ) # 0~1 0~11.5m/s 50% 0~1m/s
        punish_head_ang = 0.05*np.where(
            np.abs(delta_phi) < 3,
            np.square(delta_phi),
            np.abs(delta_phi)+ 8,
        ) # 0~1  0~12 degree 50% 0~3 degree

        punish_yaw_rate = 0.1*np.where(
            np.abs(ego_r) < 2,
            np.square(ego_r),
            np.abs(ego_r)+ 2,
        ) # 0~1  0~8 degree/s 50% 0~2 degree/s

        scaled_punish_overspeed = 3*np.clip(
        np.where(
            ego_vx > 1.1*ref_v,
            1 + np.abs(ego_vx - 1.1*ref_v),
            0,),
        0, 2)

        scaled_punish_dist_lat = punish_dist_lat * config["P_lat"]
        scaled_punish_vel_long = punish_vel_long * config["P_long"]
        scaled_punish_head_ang = punish_head_ang * config["P_phi"]
        scaled_punish_yaw_rate = punish_yaw_rate * config["P_yaw"]

        # # reward related to action
        # nominal_steer = self._get_nominal_steer_by_state(
        #     ego_state, ref_param, ref_index)*180/np.pi
        # # print(f'nominal_steer: {nominal_steer}')
        nominal_steer = 0.0
            
        abs_steer = np.abs(last_steer- nominal_steer)   
        reward_steering = -np.where(abs_steer < 4, np.square(abs_steer), 2*abs_steer+8)

        # self.out_of_action_range = abs_steer > 20

        if ego_vx < 1 and config["enable_slow_reward"]:
            reward_steering = reward_steering * 5

        abs_ax = np.abs(last_acc)
        reward_acc_long = -np.where(abs_ax < 2, np.square(abs_ax), 2*abs_ax)

        reward_delta_steer = -np.where(np.abs(delta_steer) < 4, np.square(delta_steer), 2*np.abs(delta_steer)+8)
        reward_jerk = -np.where(np.abs(jerk) < 2, np.square(jerk), 2*np.abs(jerk)+8)

        # turning = np.abs(nominal_steer) > 5 and in_junction
        # if turning:
        #     scaled_punish_dist_lat = scaled_punish_dist_lat * 0.5
        #     scaled_punish_head_ang = scaled_punish_head_ang 
        #     scaled_punish_yaw_rate = scaled_punish_yaw_rate * 0.2
        #     # scaled_punish_vel_long = scaled_punish_vel_long * 0.2
        #     # reward_steering = reward_steering * 0.2
        #     rew_step = np.clip(ego_vx, 0, 1.0)*2

        # if ego_vx < 1 and config["enable_slow_reward:
        #     scaled_punish_dist_lat = scaled_punish_dist_lat * 0.1
        #     scaled_punish_head_ang = scaled_punish_head_ang * 0.1
        break_condition = (ref_v < 2 and (next_ref_v - ref_v) < -0.1) or (ref_v < 1.0)
        if break_condition and config["nonimal_acc"]:
            nominal_acc = -2.5
            # scaled_punish_vel_long = 0  # remove the effect of speed error
            scaled_punish_dist_lat = 0  # remove the effect of tracking error
            scaled_punish_head_ang = 0
            reward_acc_long = 0
        else:
            nominal_acc = 0
            punish_nominal_acc = 0

        # if braking_mode and config["nonimal_acc"]:
        #     nominal_acc = -2.5
        #     scaled_punish_vel_long = 0

        delta_acc = np.abs(nominal_acc - last_acc)
        punish_nominal_acc =(nominal_acc != 0)* 4 * np.where(delta_acc < 0.5, np.square(delta_acc),  delta_acc-0.25)
            
        # action related reward
        scaled_reward_steering = reward_steering * config["P_steer"]
        scaled_reward_acc_long = reward_acc_long * config["P_acc"]
        scaled_reward_delta_steer = reward_delta_steer * config["P_delta_steer"]
        scaled_reward_jerk = reward_jerk * config["P_jerk"]

        # live reward
        scaled_rew_step = rew_step * config["R_step"]

        reward_ego_state = scaled_rew_step - \
            (scaled_punish_dist_lat + 
             scaled_punish_vel_long + 
             scaled_punish_head_ang + 
             scaled_punish_yaw_rate + 
             punish_nominal_acc + 
             scaled_punish_overspeed) + \
            (scaled_reward_steering + 
             scaled_reward_acc_long + 
             scaled_reward_delta_steer + 
             scaled_reward_jerk)
        
        reward_ego_state = np.clip(reward_ego_state, -10, 20)

        return reward_ego_state, {
            "env_tracking_error": tracking_error,
            "env_speed_error": np.abs(speed_error),
            "env_delta_phi": np.abs(delta_phi),

            "env_reward_step": rew_step,

            "env_reward_steering": reward_steering,
            "env_reward_acc_long": reward_acc_long,
            "env_reward_delta_steer": reward_delta_steer,
            "env_reward_jerk": reward_jerk,

            "env_reward_dist_lat": -punish_dist_lat,
            "env_reward_vel_long": -punish_vel_long,
            "env_reward_head_ang": -punish_head_ang,
            "env_reward_yaw_rate": -punish_yaw_rate,

            "env_scaled_reward_part2": reward_ego_state,
            "env_scaled_reward_step": scaled_rew_step,
            "env_scaled_reward_dist_lat": -scaled_punish_dist_lat,
            "env_scaled_reward_vel_long": -scaled_punish_vel_long,
            "env_scaled_reward_head_ang": -scaled_punish_head_ang,
            "env_scaled_reward_yaw_rate": -scaled_punish_yaw_rate,
            "env_scaled_reward_steering": scaled_reward_steering,
            "env_scaled_reward_acc_long": scaled_reward_acc_long,
            "env_scaled_reward_delta_steer": scaled_reward_delta_steer,
            "env_scaled_reward_jerk": scaled_reward_jerk,
        }