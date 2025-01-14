reward_tags = (
    "reward",
    "reward_tracking_lon",
    "reward_tracking_lat",
    "reward_tracking_phi",
    "reward_tracking_vx",
    "reward_tracking_vy",
    "reward_tracking_yaw_rate",
    "reward_action_acc",
    "reward_action_steer",
    "reward_cost_acc_rate_1",
    "reward_cost_steer_rate_1",
    "reward_cost_steer_rate_2_min",
    "reward_cost_steer_rate_2_max",
    "reward_cost_vx_min",
    "reward_cost_vx_max",
    "reward_cost_vy_min",
    "reward_cost_vy_max",
    "reward_penalty_lat_error",
    "reward_penalty_sur_dist",
    "reward_penalty_road",
    "collision_flag",

    "env_tracking_error",
    "env_delta_phi",
    "env_speed_error",
    "env_pun2front",
    "env_pun2side",
    "env_pun2space",
    "env_pun2rear",
    "env_reward_vel_long",
    "env_reward_yaw_rate",
    "env_reward_dist_lat",
    "env_reward_head_ang",
    "env_reward_steering",
    "env_reward_acc_long",
    "env_reward_delta_steer",
    "env_reward_jerk",
    "env_reward_step",
    "env_punish_collision_risk",
    "env_scaled_reward_part1",
    "env_scaled_reward_part2",
    "env_scaled_reward_step",
    "env_scaled_reward_vel_long",
    "env_scaled_reward_yaw_rate",
    "env_scaled_reward_dist_lat",
    "env_scaled_reward_head_ang",
    "env_scaled_reward_steering",
    "env_scaled_reward_acc_long",
    "env_scaled_reward_delta_steer",
    "env_scaled_reward_jerk",
    "env_scaled_pun2front",
    "env_scaled_pun2side",
    "env_scaled_pun2space",
    "env_scaled_pun2rear",
    "env_scaled_punish_boundary",
)


done_tags = (
    "done/arrival",
    "done/red_violation",
    "done/yellow_violation",
    "done/out_of_driving_area",
    "done/collision",
    "done/max_steps")

idsim_tb_tags = (
    "Evaluation/Arrival rate-RL iter",
    "Evaluation/Red violation rate-RL iter",
    "Evaluation/Yellow violation rate-RL iter",
    "Evaluation/Out of driving area rate-RL iter",
    "Evaluation/Collision rate-RL iter",
    "Evaluation/Max steps rate-RL iter",
    "Evaluation/total_reward",
    "Evaluation/reward_tracking_lon",
    "Evaluation/reward_tracking_lat",
    "Evaluation/reward_tracking_phi",
    "Evaluation/reward_tracking_vx",
    "Evaluation/reward_tracking_vy",
    "Evaluation/reward_tracking_yaw_rate",
    "Evaluation/reward_action_acc",
    "Evaluation/reward_action_steer",
    "Evaluation/reward_cost_acc_rate_1",
    "Evaluation/reward_cost_steer_rate_1",
    "Evaluation/reward_cost_steer_rate_2_min",
    "Evaluation/reward_cost_steer_rate_2_max",
    "Evaluation/reward_cost_vx_min",
    "Evaluation/reward_cost_vx_max",
    "Evaluation/reward_cost_vy_min",
    "Evaluation/reward_cost_vy_max",
    "Evaluation/reward_penalty_lat_error",
    "Evaluation/reward_penalty_sur_dist",
    "Evaluation/reward_penalty_road",
    "Evaluation/collision_flag",
    
    "Evaluation/env_tracking_error",
    "Evaluation/env_delta_phi",
    "Evaluation/env_speed_error",
    "Evaluation/env_pun2front",
    "Evaluation/env_pun2side",
    "Evaluation/env_pun2space",
    "Evaluation/env_pun2rear",
    "Evaluation/env_reward_vel_long",
    "Evaluation/env_reward_yaw_rate",
    "Evaluation/env_reward_dist_lat",
    "Evaluation/env_reward_head_ang",
    "Evaluation/env_reward_steering",
    "Evaluation/env_reward_acc_long",
    "Evaluation/env_reward_delta_steer",
    "Evaluation/env_reward_jerk",
    "Evaluation/env_reward_step",
    "Evaluation/env_punish_collision_risk",
    "Evaluation/env_scaled_reward_part1",
    "Evaluation/env_scaled_reward_part2",
    "Evaluation/env_scaled_reward_step",
    "Evaluation/env_scaled_reward_vel_long",
    "Evaluation/env_scaled_reward_yaw_rate",
    "Evaluation/env_scaled_reward_dist_lat",
    "Evaluation/env_scaled_reward_head_ang",
    "Evaluation/env_scaled_reward_steering",
    "Evaluation/env_scaled_reward_acc_long",
    "Evaluation/env_scaled_reward_delta_steer",
    "Evaluation/env_scaled_reward_jerk",
    "Evaluation/env_scaled_pun2front",
    "Evaluation/env_scaled_pun2side",
    "Evaluation/env_scaled_pun2space",
    "Evaluation/env_scaled_pun2rear",
    "Evaluation/env_scaled_punish_boundary",
)

idsim_tb_keys = (*done_tags, *reward_tags)
assert len(idsim_tb_keys) == len(idsim_tb_tags) == len(set(idsim_tb_keys)) == len(set(idsim_tb_tags)), "idsim_tb_keys and idsim_tb_tags should have the same length and no duplicates"
idsim_tb_tags_dict = dict(zip(idsim_tb_keys, idsim_tb_tags))