#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Mobile Robot Environment
#  Paper: https://ieeexplore.ieee.org/document/9785377
#  Update Date: 2022-06-05, Baiyu Peng: create environment


import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding
from gym.wrappers.time_limit import TimeLimit

gym.logger.setLevel(gym.logger.ERROR)


class PythMobilerobot:
    def __init__(self, **kwargs):
        self.n_obstacle = 1

        self.robot = Robot()
        self.obses = [Robot() for _ in range(self.n_obstacle)]
        self.dt = 0.4  # seconds between state updates
        self.state_dim = (1 + self.n_obstacle) * 5 + 3
        self.action_dim = 2
        self.use_constraint = kwargs.get("use_constraint", True)
        self.constraint_dim = self.n_obstacle

        lb_state = np.array(
            [-30, -30, -np.pi, -1, -np.pi / 2]
            + [-30, -np.pi, -2]
            + [-30, -30, -np.pi, -1, -np.pi / 2] * self.n_obstacle
        )
        hb_state = np.array(
            [30, 30, np.pi, 1, np.pi / 2] + [30, np.pi, 2] + [30, 30, np.pi, 1, np.pi / 2] * self.n_obstacle
        )
        lb_action = np.array([-0.4, -np.pi / 3])
        hb_action = np.array([0.4, np.pi / 3])

        self.action_space = spaces.Box(low=lb_action, high=hb_action)
        self.observation_space = spaces.Box(lb_state, hb_state)

        self.seed()
        self.state = self.reset()

        self.steps = 0

    @staticmethod
    def additional_info():
        info_dict = dict(
            constraint={"shape": (1,), "dtype": np.float64}
        )
        return info_dict
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray):
        ################################################################################################################
        #  define your forward function here: the format is just like: state_next = f(state,action)
        veh2vehdist = np.zeros((self.state.shape[0], self.n_obstacle))
        action = action.reshape(1, -1)  # TODO is right
        for i in range(1 + self.n_obstacle):
            if i == 0:
                robot_state = self.robot.f_xu(self.state[:, :5], action.reshape(1, -1), self.dt, "ego")
                tracking_error = self.robot.tracking_error(robot_state)
                state_next = np.concatenate((robot_state, tracking_error), 1)

            else:
                obs_state = self.robot.f_xu(
                    self.state[:, 3 + i * 5 : 3 + i * 5 + 5],
                    self.state[:, 3 + i * 5 + 3 : 3 + i * 5 + 5],
                    self.dt,
                    "obs",
                )
                state_next = np.concatenate((state_next, obs_state), 1)

                safe_dis = self.robot.robot_params["radius"] + self.obses[i - 1].robot_params["radius"] + 0.15  # 0.35
                veh2vehdist[:, i - 1] = (
                    safe_dis
                    - (
                        (
                            (state_next[:, 3 + i * 5] - state_next[:, 0]) ** 2
                            + (state_next[:, 3 + i * 5 + 1] - state_next[:, 1]) ** 2
                        )
                    )
                    ** 0.5
                )

        self.state = state_next
        ############################################################################################
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        r_tracking = -1.4 * np.abs(tracking_error[:, 0]) - 1 * np.abs(tracking_error[:, 1]) - 16 * np.abs(tracking_error[:, 2])
        r_action = -0.2 * np.abs(action[:, 0]) - 0.5 * np.abs(action[:, 1])
        reward = r_tracking + r_action
        ############################################################################################
        # define the constraint here

        constraint = veh2vehdist
        dead = veh2vehdist > 0
        ################################################################################################################
        # define the ending condition here the format is just like isdone = l(next_state)

        isdone = bool(
            dead.all(1)
            + (self.state[:, 0] < -2)
            + (self.state[:, 0] > 13)
            + (self.state[:, 1] > 3)
            + (self.state[:, 1] < -1)
        )
        ############################################################################################
        self.steps += 1
        info = {"TimeLimit.truncated": self.steps > 170, "constraint": constraint.reshape(-1)} # TODO is right
        return np.array(self.state.reshape(-1), dtype=np.float32), float(reward), isdone, info  # TODO is right

    def reset(self, n_agent=1):
        def uniform(low, high):
            return self.np_random.random([n_agent]) * (high - low) + low

        state = np.zeros([n_agent, self.state_dim])
        for i in range(1 + self.n_obstacle):
            if i == 0:
                state[:, 0] = uniform(0, 2.7)
                state[:, 1] = uniform(-1, 1)
                state[:, 2] = uniform(-0.6, 0.6)
                state[:, 3] = uniform(0, 0.3)
                state[:, 4] = state[:, 4]
                state[:, 4:7] = self.robot.tracking_error(state[:, :5])
            else:
                state[:, 3 + 5 * i] = uniform(3.5, 6)
                state[:, 3 + 5 * i + 1] = uniform(-3, 3)
                state[:, 3 + 5 * i + 2] = np.where(
                    state[:, 3 + 5 * i + 1] > 0,
                    state[:, 3 + 5 * i + 2] - np.pi / 2,
                    state[:, 3 + 5 * i + 2] + np.pi / 2,
                ) + uniform(-0.8, 0.8)
                state[:, 3 + 5 * i + 3] = uniform(0.0, 0.5)
                state[:, 3 + 5 * i + 4] = 0

        self.steps_beyond_done = None
        self.steps = 0
        self.state = state

        return np.array(self.state.reshape(-1), dtype=np.float32)  # TODO is right

    def render(self, n_window=1):

        if not hasattr(self, "artists"):
            self.render_init(n_window)
        state = self.state
        r_rob = self.robot.robot_params["radius"]
        r_obs = self.obses[0].robot_params["radius"]

        def arrow_pos(state):
            x, y, theta = state[0], state[1], state[2]
            return [x, x + np.cos(theta) * r_rob], [y, y + np.sin(theta) * r_rob]

        for i in range(n_window):
            for j in range(n_window):
                idx = i * n_window + j
                circles, arrows = self.artists[idx]
                circles[0].center = state[idx, :2]
                arrows[0].set_data(arrow_pos(state[idx, :5]))
                for k in range(self.n_obstacle):
                    circles[k + 1].center = state[idx, 3 + (k + 1) * 5 : 3 + (k + 1) * 5 + 2]
                    arrows[k + 1].set_data(arrow_pos(state[idx, 3 + (k + 1) * 5 : 3 + (k + 1) * 5 + 5]))
            plt.pause(0.02)

    def render_init(self, n_window=1):

        fig, axs = plt.subplots(n_window, n_window, figsize=(9, 9))
        artists = []

        r_rob = self.robot.robot_params["radius"]
        r_obs = self.obses[0].robot_params["radius"]
        for i in range(n_window):
            for j in range(n_window):
                if n_window == 1:
                    ax = axs
                else:
                    ax = axs[i, j]
                ax.set_aspect(1)
                ax.set_ylim(-3, 3)
                x = np.linspace(0, 6, 1000)
                y = np.sin(1 / 30 * x)
                ax.plot(x, y, "k")
                circles = []
                arrows = []
                circles.append(plt.Circle([0, 0], r_rob, color="red", fill=False))
                arrows.append(ax.plot([], [], "red")[0])
                ax.add_artist(circles[-1])
                ax.add_artist(arrows[-1])
                for k in range(self.n_obstacle):
                    circles.append(plt.Circle([0, 0], r_obs, color="blue", fill=False))
                    ax.add_artist(circles[-1])

                    arrows.append(ax.plot([], [], "blue")[0])
                artists.append([circles, arrows])
        self.artists = artists
        plt.ion()

    def close(self):
        plt.close("all")


class Robot:
    def __init__(self):
        self.robot_params = dict(
            v_max=0.4, w_max=np.pi / 2, v_delta_max=1.8, w_delta_max=0.8, v_desired=0.3, radius=0.74 / 2  # per second
        )
        self.path = ReferencePath()

    def f_xu(self, states, actions, T, type):
        v_delta_max = self.robot_params["v_delta_max"]
        v_max = self.robot_params["v_max"]
        w_max = self.robot_params["w_max"]
        w_delta_max = self.robot_params["w_delta_max"]
        std_type = {"ego": [0.03, 0.02], "obs": [0.03, 0.02], "none": [0, 0], "explore": [0.3, 0.3]}
        stds = std_type[type]

        x, y, theta, v, w = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
        v_cmd, w_cmd = actions[:, 0], actions[:, 1]

        delta_v = np.clip(v_cmd - v, -v_delta_max * T, v_delta_max * T)
        delta_w = np.clip(w_cmd - w, -w_delta_max * T, w_delta_max * T)
        v_cmd = np.clip(v + delta_v, -v_max, v_max) + np.random.normal(0, stds[0], [states.shape[0]]) * 0.5
        w_cmd = np.clip(w + delta_w, -w_max, w_max) + np.random.normal(0, stds[1], [states.shape[0]]) * 0.5
        next_state = [
            x + T * np.cos(theta) * v_cmd,
            y + T * np.sin(theta) * v_cmd,
            np.clip(theta + T * w_cmd, -np.pi, np.pi),
            v_cmd,
            w_cmd,
        ]

        return np.stack(next_state, 1)

    def tracking_error(self, x):
        error_position = x[:, 1] - self.path.compute_path_y(x[:, 0])
        error_head = x[:, 2] - self.path.compute_path_phi(x[:, 0])

        error_v = x[:, 3] - self.robot_params["v_desired"]
        tracking = np.concatenate((error_position.reshape(-1, 1), error_head.reshape(-1, 1), error_v.reshape(-1, 1)), 1)
        return tracking

class ReferencePath(object):
    def __init__(self):
        pass

    def compute_path_y(self, x):
        y = np.sin(1/30 * x)
        return y

    def compute_path_phi(self, x):
        deriv = 1/30 * np.cos(1/30 * x)
        return np.arctan(deriv)


def env_creator(**kwargs):
    """
    make env `pyth_mobilerobot`
    """
    return PythMobilerobot(**kwargs)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min) * t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result


if __name__ == "__main__":
    env = env_creator()
    env.reset()
    a = env.action_space.sample()
    d, r, d, info = env.step(a)
    print(info["constraint"].dtype, type(info["constraint"]))