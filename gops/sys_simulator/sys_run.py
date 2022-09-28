import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
from gym import wrappers

from gops.create_pkg.create_env import create_env
from gops.utils.plot_evaluation import cm2inch
from gops.utils.common_utils import get_args_from_json, mp4togif

default_cfg = dict()
default_cfg["fig_size"] = (12, 9)
default_cfg["dpi"] = 300
default_cfg["pad"] = 0.5

default_cfg["tick_size"] = 8
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",
    "size": "8",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "family": "Times New Roman",
    "size": "9",
    "weight": "normal",
}

default_cfg["img_fmt"] = "png"


class PolicyRuner():
    def __init__(self, log_policy_dir_list, trained_policy_iteration_list, save_render=False, plot_range=[],
                 is_init_state=False, init_state=[], legend_list=[], use_opt=False, constrained_env=False, is_tracking=False) -> None:
        self.log_policy_dir_list = log_policy_dir_list
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_state:
            self.init_state = init_state
        else:
            self.init_state = []
        self.legend_list = legend_list
        self.use_opt = use_opt
        self.constrained_env = constrained_env
        self.is_tracking = is_tracking
        if self.use_opt:
            self.legend_list.append('LQ')
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError("The lenth of policy number is not equal to the number of policy iteration")

        # data for plot

        #####################################################
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "policy_result")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(path, algs_name + self.env_id, datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
        os.makedirs(self.save_path, exist_ok=True)

    def run_an_episode(self, env, controller, init_state, is_opt, render=True):
        obs_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        step = 0
        step_list = []
        obs = env.reset()
        # plot tracking
        t_list = []
        x_list = []
        x_ref_list = []
        y_list = []
        y_ref_list = []
        if len(init_state) == 0:
            pass
        elif len(obs) == len(init_state):
            obs = np.array(init_state)
            env.env.reset(obs)
        else:
            raise NotImplementedError("The dimension of Initial state is wrong!")
        done = False
        info = {"TimeLimit.truncated": False}
        while not (done or info["TimeLimit.truncated"]):
            if is_opt:
                action = self.compute_action_lqr(obs, controller)
            else:
                action = self.compute_action(obs, controller)

            # action = env.control_policy(obs)
            step_list.append(step)
            next_obs, reward, done, info = env.step(action)
            step = step + 1
            print("step:", step)
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()
            reward_list.append(reward)
            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                t_list.append(info["t"])
                x_list.append(info["x"])
                x_ref_list.append(info["x_ref"])
                y_list.append(info["y"])
                y_ref_list.append(info["y_ref"])

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
            "step_list": step_list,
        }
        if self.constrained_env:
            eval_dict.update({
                "constrain_list": constrain_list,
            })

        if self.is_tracking:
            tracking_dict = {
                "t_list": t_list,
                "x_list": x_list,
                "x_ref_list": x_ref_list,
                "y_list": y_list,
                "y_ref_list": y_ref_list,
            }
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def compute_action(self, obs, networks):
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action
    def compute_action_lqr(self, obs, K):
        action = -K @ obs
        return action
    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        obs_dim = self.eval_list[0]["obs_list"][0].shape[0]
        policy_num = len(self.algorithm_list)
        if self.use_opt:
            policy_num += 1

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        constrain_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["obs_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            if self.constrained_env:
                constrain_list.append(self.eval_list[i]["constrain_list"])

        if len(self.plot_range) == 0:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = np.min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range: end_range]
                action_list[i] = action_list[i][start_range: end_range]
                state_list[i] = state_list[i][start_range: end_range]
                step_list[i] = step_list[i][start_range: end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range: end_range]
        else:
            raise NotImplementedError("The setting of plot range is wrong")

        # Convert List to Array
        reward_array = np.array(reward_list)
        action_array = np.array(action_list)
        state_array = np.array(state_list)
        step_array = np.array(step_list)
        if self.constrained_env:
            constrain_array = np.array(constrain_list)[:, :, 0]  # todo:special for SPIL

        # Plot reward
        path_reward_fmt = os.path.join(self.save_path, "reward.{}".format(default_cfg["img_fmt"]))
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_array)
        reward_data.to_csv('{}\\reward.csv'.format(self.save_path), encoding='gbk')

        for i in range(policy_num):
            legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
            sns.lineplot(x=step_array[i], y=reward_array[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("Time step", default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(self.save_path, "action-{}.{}".format(j, default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=action_array[:, :, j])
            action_data.to_csv('{}\\action-{}.csv'.format(self.save_path, j), encoding='gbk')

            for i in range(policy_num):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                sns.lineplot(x=step_array[i], y=action_array[i, :, j],
                             label="{}".format(legend))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Time step", default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot state
        for j in range(obs_dim):
            path_state_fmt = os.path.join(self.save_path, "state-{}.{}".format(j, default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=state_array[:, :, j])
            state_data.to_csv('{}\\state-{}.csv'.format(self.save_path, j), encoding='gbk')

            for i in range(policy_num):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                sns.lineplot(x=step_array[i], y=state_array[i, :, j],
                             label="{}".format(legend))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Time step", default_cfg["label_font"])
            plt.ylabel("State_{}".format(j), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

            # plot tracking
            if self.is_tracking:
                # Creat inital array
                t_array_list = []
                x_array_list = []
                x_ref_array_list = []
                y_array_list = []
                y_ref_array_list = []
                marker_list = ['*', 'p', 's', 'D', 'h', '^', '8', 'x']
                for i in range(policy_num):
                    t_array_list.append(np.array(self.tracking_list[i]["t_list"]))
                    x_array_list.append(np.array(self.tracking_list[i]["x_list"]))
                    x_ref_array_list.append(np.array(self.tracking_list[i]["x_ref_list"]))
                    y_array_list.append(np.array(self.tracking_list[i]["y_list"]))
                    y_ref_array_list.append(np.array(self.tracking_list[i]["y_ref_list"]))

                # plot x_compare
                path_x_compare_fmt = os.path.join(self.save_path, "x_compare.{}".format(default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
                for i in range(policy_num):
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=t_array_list[i], y=x_array_list[i], label="{}".format(legend))
                sns.lineplot(x=t_array_list[0], y=x_ref_array_list[0], label="x_ref")
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel("Time", default_cfg["label_font"])
                plt.ylabel("x_compare", default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_x_compare_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
                # plt.show()

                # plot y_compare
                path_y_compare_fmt = os.path.join(self.save_path, "y_compare.{}".format(default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
                for i in range(policy_num):
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=t_array_list[i], y=y_array_list[i], label="{}".format(legend))
                sns.lineplot(x=t_array_list[0], y=y_ref_array_list[0], label="y_ref")
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel("Time", default_cfg["label_font"])
                plt.ylabel("y_compare", default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_y_compare_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

                # plot x_y
                path_x_y_fmt = os.path.join(self.save_path, "x_y.{}".format(default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
                for i in range(policy_num):
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    plt.scatter(x=x_array_list[i], y=y_array_list[i], c=t_array_list[i], s=2,
                                label="{}".format(legend), cmap='plasma', marker=marker_list[i])
                sc = plt.scatter(x=x_ref_array_list[0], y=y_ref_array_list[0], c=t_array_list[0], label="x_y_ref",
                                 s=2, cmap='plasma')
                plt.colorbar(sc)
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel("X", default_cfg["label_font"])
                plt.ylabel("Y", default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_x_y_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot constraint value
        if self.constrained_env:
            path_constraint_fmt = os.path.join(self.save_path, "constraint.{}".format(default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward data to csv
            constrain_data = pd.DataFrame(data=constrain_array)
            constrain_data.to_csv('{}\\constrain.csv'.format(self.save_path), encoding='gbk')

            for i in range(policy_num):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                sns.lineplot(x=step_array[i], y=constrain_array[i], label="{}".format(legend))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Time step", default_cfg["label_font"])
            plt.ylabel("Constraint value", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_constraint_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot error
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(self.save_path, "reward_error.{}".format(default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_array = reward_array[:-1] - reward_array[-1]
            reward_error_data = pd.DataFrame(data=reward_error_array)
            reward_error_data.to_csv('{}\\reward_error.csv'.format(self.save_path), encoding='gbk')

            for i in range(policy_num - 1):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                sns.lineplot(x=step_array[i], y=reward_error_array[i], label="{}".format(legend))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Time step", default_cfg["label_font"])
            plt.ylabel("Reward_error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_reward_error_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

            # action error
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(self.save_path, "action-{}_error.{}".format(j, default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

                action_error_array = np.zeros_like(action_array[:-1])

                for i in range(policy_num - 1):
                    action_error_array[i] = action_array[i] - action_array[-1]
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=step_array[i], y=action_error_array[i, :, j],
                                 label="{}".format(legend))
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel("Time step", default_cfg["label_font"])
                plt.ylabel("Action-{}_error".format(j), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_action_error_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

                # save action error data to csv
                action_error_data = pd.DataFrame(data=action_error_array[:, :, j])
                action_error_data.to_csv('{}\\action-{}_error.csv'.format(self.save_path, j), encoding='gbk')

            # state error
            for j in range(obs_dim):
                path_state_error_fmt = os.path.join(self.save_path, "state-{}_error.{}".format(j, default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

                state_error_array = np.zeros_like(state_array[:-1])

                for i in range(policy_num - 1):
                    state_error_array[i] = state_array[i] - state_array[-1]
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=step_array[i], y=state_error_array[i, :, j],
                                 label="{}".format(legend))
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel("Time step", default_cfg["label_font"])
                plt.ylabel("State_{}_error".format(j), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_state_error_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

                # save state data to csv
                state_error_data = pd.DataFrame(data=state_error_array[:, :, j])
                state_error_data.to_csv('{}\\state-{}_error.csv'.format(self.save_path, j), encoding='gbk')

            # compute relative error
            error_result = {}
            # action error
            for i in range(self.policy_num):
                for j in range(action_dim):
                    action_error = {}
                    error_list = []
                    for q in range(100):
                        error = np.abs(self.error_dict["policy_{}".format(i)]["action"][q, j] - self.error_dict["opt"]["action"][q, j]) \
                                / (np.max(self.error_dict["opt"]["action"][:, j]) - np.min(self.error_dict["opt"]["action"][:, j]))
                        error_list.append(error)
                    action_error["max_error"] = max(error_list)
                    action_error["mean_error"] = sum(error_list) / len(error_list)
                    error_result["policy_{}-action_{}_error".format(i, j)] = action_error
            # state error
            for i in range(self.policy_num):
                for j in range(obs_dim):
                    state_error = {}
                    error_list = []
                    for q in range(100):
                        error = np.abs(self.error_dict["policy_{}".format(i)]["next_obs"][q, j] - self.error_dict["opt"]["next_obs"][q, j]) \
                                / (np.max(self.error_dict["opt"]["next_obs"][:, j]) - np.min(self.error_dict["opt"]["next_obs"][:, j]))
                        error_list.append(error)
                    state_error["max_error"] = max(error_list)
                    state_error["mean_error"] = sum(error_list) / len(error_list)
                    error_result["policy_{}-state_{}_error".format(i, j)] = state_error
            print(error_result)




    @staticmethod
    def __load_args(log_policy_dir):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            self.args_list.append(args)
            env_id = args["env_id"]
            self.env_id_list.append(env_id)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self):
        env = create_env(**self.args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            env = wrappers.RecordVideo(env, video_path,
                                       name_prefix="{}_video".format(self.args["algorithm"]))
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        # self.args["has_controller"] = hasattr(env, 'has_optimal_controller') & env.has_optimal_controller
        return env

    def __load_policy(self, log_policy_dir, trained_policy_iteration):
        # Create policy
        alg_name = self.args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        networks = ApproxContainer(**self.args)
        print("Create {}-policy successfully!".format(alg_name))

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
        networks.load_state_dict(torch.load(log_path))
        print("Load {}-policy successfully!".format(alg_name))
        return networks

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            env = self.__load_env()
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(env, networks, self.init_state, is_opt=False, render=False)
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)
        if self.use_opt:
            K = env.control_matrix
            eval_dict_lqr, _ = self.run_an_episode(env, K, self.init_state, is_opt=True, render=False)
            self.eval_list.append(eval_dict_lqr)
            for i in range(self.policy_num):
                if i == 0:
                    self.obs_list = self.__get_init_obs(env, 100)
                    self.error_dict = {}
                net_error_dict = self.__error_compute(env, self.obs_list, networks, 100, is_opt=False)
                LQ_error_dict = self.__error_compute(env, self.obs_list, K, 100, is_opt=True)
                self.error_dict["policy_{}".format(i)] = net_error_dict
                self.error_dict["opt"] = LQ_error_dict


                

    def __get_init_obs(self, env, init_state_nums):
        obs_list = []
        for i in range(init_state_nums):
            obs = env.reset()
            obs_list.append(obs)
        return obs_list
    def __error_compute(self, env, obs_list, controller, init_state_nums, is_opt):
        action_list = []
        next_obs_list = []
        for i in range(init_state_nums):
            obs = obs_list[i]
            if is_opt:
                action = self.compute_action_lqr(obs, controller)
            else:
                action = self.compute_action(obs, controller)

            next_obs, reward, done, info = env.step(action)
            action_list.append(action)
            next_obs_list.append(next_obs)
        action_array = np.array(action_list)
        next_obs_array = np.array(next_obs_list)
        error_dict = {"action": action_array, "next_obs": next_obs_array}

        return error_dict

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert env_id == eid, "policy {} and policy 0 is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()