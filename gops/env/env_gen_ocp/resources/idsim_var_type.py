from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Dict, Optional, Tuple, Union

from omegaconf import OmegaConf

Color = Any  # Tuple[int, int, int, int]
PolyOption = Tuple[Color, int, float, bool]  # color, layer, width, filled
ObsNumSur = Any  # Union[int, Dict[str, int]]

@dataclass(frozen=False)
class Config:
    """Configuration for the idSim environment."""

    seed: Optional[int] = None
    debug: bool = False

    # ===== Env =====
    dt: float = 0.1  # Do not change this value.
    actuator: str = "ExternalActuator"
    max_steps: int = 1000
    use_pose_reward: bool = False
    penalize_collision: bool = True
    no_done_at_collision: bool = False
    takeover_bias: bool = False
    takeover_bias_prob: float = 0.0
    random_ref_v: bool = False
    ref_v_range: Tuple[float, float] = (2,12)
    nonimal_acc: bool = False
    # [mean, std]
    takeover_bias_x: Tuple[float, float] = (0.0, 0.1)
    takeover_bias_y: Tuple[float, float] = (0.0, 0.1)
    takeover_bias_phi: Tuple[float, float] = (0.0, 0.05)
    takeover_bias_vx: Tuple[float, float] = (0.6, 0.2)
    takeover_bias_ax: Tuple[float, float] = (0.0, 0.1)
    takeover_bias_steer: Tuple[float, float] = (0.0, 0.01)
    # prioritize take over in junction vehicle
    prioritize_in_junction_veh: bool = False
    # model free reward config
    punish_sur_mode: str = "sum"
    enable_slow_reward: bool = False
    R_step: float = 5.0
    P_lat: float = 5.0
    P_long: float = 2.5
    P_phi: float = 20.0
    P_yaw: float = 10.0
    P_front: float = 5.0
    P_side: float = 5.0
    P_space: float = 5.0
    P_rear: float = 5.0
    P_steer: float = 50.0
    P_acc: float = 0.2
    P_delta_steer: float = 50.0
    P_jerk: float = 0.1
    P_boundary: float = 0.0
    P_done: float = 2000.0


    safety_lat_margin_front: float = 0.0
    safety_long_margin_front: float = 0.0
    safety_long_margin_side: float = 0.0
    front_dist_thd: float = 50.0
    space_dist_thd: float = 8.0
    rel_v_thd: float = 1.0
    rel_v_rear_thd: float = 0.0
    time_dist: float = 2.0
    # Since libsumo/traci only allows a single running instance,
    # we need to use a singleton mode to avoid multiple instances.
    # The following modes are available:
    #   "raise": raise an error if multiple instances are created (default).
    #   "reuse": reuse the previous instance.
    #   "invalidate": invalidate the previous instance.
    singleton_mode: str = "raise"

    # ===== Ego veh dynamics =====
    ego_id: Optional[str] = None
    action_repeat: int = 1
    incremental_action: bool = False
    action_lower_bound: Tuple[float, float] = (-2.0, -0.35)
    action_upper_bound: Tuple[float, float] = ( 1.0,  0.35)
    real_action_lower_bound: Tuple[float, float] = (-3.0, -0.065)
    real_action_upper_bound: Tuple[float, float] = ( 0.8,  0.065)
    vehicle_spec:Tuple[float, float, float, float, float, float, float, float] = \
        (1412.0, 1536.7, 1.06, 1.85, -128915.5, -85943.6, 20.0, 0.0)
    # m: float  # mass
    # Iz: float  # moment of inertia
    # lf: float  # distance from front axle to center of gravity
    # lr: float  # distance from rear axle to center of gravity
    # Cf: float  # cornering stiffness of front tires (negative)
    # Cr: float  # cornering stiffness of rear tires (negative)
    # vx_max: float  # maximum longitudinal velocity
    # vx_min: float  # minimum longitudinal velocity

    # ===== Observation =====
    obs_components: Tuple[str, ...] = (
        "EgoState", "Waypoint", "TrafficLight", "DrivingArea", "SurState"
    )
    obs_flatten: bool = True
    obs_normalize: bool = False
    # For Waypoint
    obs_num_ref_points: int = 5
    obs_ref_interval: float = 5.0
    obs_ref_candidate_set: bool = False
    random_ref_cooldown: int = 30
    # For DrivingArea
    obs_driving_area_lidar_num_rays: int = 16
    obs_driving_area_lidar_range: float = 20.0
    # For SurState
    obs_num_surrounding_vehicles: ObsNumSur = 0
    # For SurLidar
    obs_surrounding_lidar_num_rays: int = 16
    # For SurEncoding
    obs_surrounding_encoding_model_path: Path = Path('/')

    # ===== Scenario =====
    # NOTE: consider using idscene config?
    scenario_root: Path = Path('/')
    scenario_name_template: str = "{id}"
    num_scenarios: int = 10
    scenario_selector: Optional[str] = None  # should be in `:a,b,c:d,e:` format
    scenario_filter_surrounding_selector: Optional[str] = None  # should be a list

    # ===== Traffic =====
    step_length: float = dt
    extra_sumo_args: Tuple[str, ...] = ()  # e.g. ("--start", "--quit-on-end", "--delay", "100")
    warmup_time: float = 5.0  # Select ego vehicle after `warmup_time` seconds for fresh (un-reused) reset
    scenario_reuse: int = 1
    detect_range: float = 5.0
    ignore_traffic_lights: bool = False
    persist_traffic_lights: bool = False
    native_collision_check: bool = False
    skip_waiting: bool = False
    skip_waiting_after_episode_steps: int = 20
    v_class_selector: str = "passenger"
    direction_selector: Optional[str] = None
    choose_vehicle_retries: int = 3
    choose_vehicle_step_time: float = 5.0
    ignore_surrounding: bool = False
    ignore_opposite_direction: bool = False
    minimum_clearance_when_takeover: float = -1.0
    keep_route_mode: int = 0b011  # See https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#move_to_xy_0xb4

    # ===== Navigation =====
    ref_length: float = 20.0  # May be less than expected if cut
    ref_v: float = 8.0
    reference_selector: int = 0  # NOTE: will add more options in the future
    use_left_turn_waiting_area: bool = False

    use_multiple_path_for_multilane: bool = False
    random_ref_probability: float = 0.0
    random_ref_cooldown: int = 30

    use_random_acc: bool = False # if false, ref_v will be calculated by default.
    random_acc_cooldown: Tuple[int] = (0, 50, 50) # cooldown for acceleration, deceleration and ref_v, respectively
    random_acc_prob: Tuple[float] = (0.0, 0.5) # probability to accelerate and decelerate, respectively
    random_acc_range: Tuple[float] = (0.0, 0.0) # (m/s^2), used for acceleration (now useless)
    random_dec_range: Tuple[float] = (-3.0, -1.0) # (m/s^2), used for deceleration

    # ===== Trajectory =====
    use_trajectory: bool = False
    trajectory_deque_capacity: int = 20

    # ===== Rendering =====
    use_render: bool = False  # False - sumo + libsumo; True - sumo-gui + libtraci
    gui_setting_file: Optional[Path] = Path('/')
    ego_color: Color = (255, 0, 255, 255)
    ref_poly_option: PolyOption = ((0, 145, 247, 255), 15, 0.2, False)
    show_detect_range: bool = True
    detection_poly_option: PolyOption = ((255, 255, 255, 255), 10, 0.2, False)
    detection_detail: int = 8
    show_driving_area: bool = True
    driving_area_poly_option: PolyOption = ((56, 255, 65, 255), 15, 0.4, False)
    vehicle_info_position: Tuple[float, float] = (-50.0, 50.0)  # Left-bottom corner of the vehicle info
    vehicle_info_template: str = textwrap.dedent("""
    step: {context.episode_step}
    time: {context.simulation_time:.2f}
    minor: {minor}

    x: {vehicle.state[0]:.2f}
    y: {vehicle.state[1]:.2f}
    vx: {vehicle.state[2]:.2f}
    vy: {vehicle.state[3]:.2f}
    phi: {vehicle.state[4]:.2f}
    omega: {vehicle.state[5]:.2f}

    ax: {vehicle.action[0]:5.2f}
    steer: {vehicle.action[1]:5.2f}

    route: {vehicle.route}
    edge: {vehicle.edge}
    lane: {vehicle.lane}

    on_lane: {vehicle.on_lane}
    waiting_well_positioned: {vehicle.waiting_well_positioned}
    waiting_ill_positioned: {vehicle.waiting_ill_positioned}
    speed_limit: {vehicle.speed_limit:5.2f}

    traffic_light: {vehicle.traffic_light}
    ahead_lane_length: {vehicle.ahead_lane_length:5.2f}
    remain_phase_time: {vehicle.remain_phase_time:5.2f}
    """)
    use_screenpeek: bool = False
    video_root: Path = Path("video")
    video_name_template: str = "{context.created_at:%Y-%m-%d_%H-%M-%S}_{context.id}/{context.scenario_count:03d}/{context.episode_count:04d}.mp4"
    video_width: int = 960
    video_height: int = 540
    video_zoom: float = 300.0
    video_output: bool = True  # Only when use_screenpeek is True, requires ffmpeg
    video_cleanup: bool = True

    # ===== Logging =====
    use_logging: bool = False
    logging_root: Path = Path("logs")
    logging_name_template: str = "{context.created_at:%Y-%m-%d_%H-%M-%S}_{context.id}/{context.scenario_count:03d}/{context.episode_count:04d}.pkl"
    logging_context: bool = False
    output_fcd: bool = True  # Also requires use_logging=True
    fcd_name_template: str = "{context.created_at:%Y-%m-%d_%H-%M-%S}_{context.id}/{context.scenario_count:03d}/fcd.xml"

    @staticmethod
    def from_partial_dict(partial_dict) -> "Config":
        base = OmegaConf.structured(Config)
        merged = OmegaConf.merge(base, partial_dict)
        return OmegaConf.to_object(merged)  # type: ignore
    
# fmt: off
from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple, Optional

from shapely.geometry import LineString, Polygon


class AbstractNetwork(ABC):
    # ----- Lane API -----
    @abstractmethod
    def get_lane_center_line(self, lane_id: str) -> LineString: ...

    @abstractmethod
    def get_lane_polygon(self, lane_id: str) -> Polygon: ...

    # ----- Edge API -----
    @abstractmethod
    def get_edge_polygon(self, edge_id: str) -> Polygon: ...

    @abstractmethod
    def get_edge_lane(self, edge_id: str, lane_index: int) -> str: ...

    @abstractmethod
    def get_edge_lanes(self, edge_id: str, v_class: Optional[str] = None) -> List[str]: ...

    @abstractmethod
    def is_edge_internal(self, edge_id: str) -> bool: ...

    @abstractmethod
    def get_last_normal_edge_and_lane(self, edge_id: str, lane_id: str) -> Tuple[str, str]: ...

    # NOTE: should pass v_class to get_connection_direction
    @abstractmethod
    def get_connection_direction(self, from_edge_id: str, to_edge_id: str, v_class: Optional[str] = None) -> Optional[str]: ...

    @abstractmethod
    def get_connection_tl_info(self, from_edge_id: str, to_edge_id: str, v_class: Optional[str] = None) -> Tuple[str, int]: ...

    @abstractmethod
    def has_connection_line(self, from_lane:str, to_lane:str) -> bool: ...

    @abstractmethod
    def get_connection_line(self, from_lane:str, to_lane:str) -> LineString: ...

    # ----- Junction API -----
    @abstractmethod
    def get_junction_polygon(self, junction_id: str) -> Polygon: ...

    @abstractmethod
    def get_junction_by_edge_hint(self, vehicle_shape: Polygon, edge_id: str) -> Tuple[str, bool]:
        """
        Returns (junction_id, if junction_id is to_node of edge_id)
        """

    @abstractmethod
    def get_upcoming_junction(self, edge_id: str) -> str: ...

    @abstractmethod
    def get_junction_incoming_edges(self, junction_id: str) -> Tuple[str, ...]: ...

    @abstractmethod
    def get_driving_area_polygon(self, route: Sequence[str]) -> Polygon: ...

    @abstractmethod
    def get_next_tls_switch(self, tls_id: str, tls_index: int, tls_phase: int, current_time: float) -> float: ...

