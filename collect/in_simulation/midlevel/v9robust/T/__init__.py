from scipy import linalg
from scipy.stats import norm
import cvxpy as cp
"""v9 does contingency planning and safe region

    - option to do contingency planning
        + multiple coinciding control
        + randomized multiple coinciding control
    - bicycle model with steering as control
    - LTV approximation about nominal path
    - Applies curved road boundaries using segmented polytopes.
"""
# Built-in libraries
import os
import logging
import collections
import weakref
import copy
import numbers
import math
import random
import time
import pickle

# PyPI libraries
import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
import matplotlib.patches as patches
import torch
import docplex.mp
import docplex.mp.model

try:
    from utils.trajectory_utils import prediction_output_to_trajectories
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from ..plotting_cont import (
    get_ovehicle_color_set,
    PlotPredictiveControl,
    PlotSimulation,
    PlotPIDController
)
from ..util import compute_L4_outerapproximation
from ..ovehicle import OVehicle
from ..prediction import generate_vehicle_latents
from ...dynamics.bicycle_v2 import VehicleModel
from ...lowlevel.v4 import VehiclePIDController
from ....generate import AbstractDataCollector
from ....generate import create_semantic_lidar_blueprint
from ....generate.map import MapQuerier
from ....generate.scene import OnlineConfig
from ....generate.scene.v3_2.trajectron_scene import TrajectronPlusPlusSceneBuilder
from ....profiling import profile
from ....exception import InSimulationException

# Local libraries
import carla
import utility as util
import utility.npu
import carlautil
import carlautil.debug

class MidlevelAgent(AbstractDataCollector):

    Z_SENSOR_REL = 2.5

    def __create_segmentation_lidar_sensor(self):
        return self.__world.spawn_actor(
                create_semantic_lidar_blueprint(self.__world),
                carla.Transform(carla.Location(z=self.Z_SENSOR_REL)),
                attach_to=self.__ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid)

    def __make_global_params(self):
        """Get scenario wide parameters used across all loops"""
        params = util.AttrDict()
        # Slack variable for solver
        params.M_big = 10_000
        # Control variable for solver, setting max/min acceleration/speed
        params.max_a = 4 # 3.5
        params.min_a = -7 # -7
        params.max_v = 10 # 10
        # objective : util.AttrDict
        #   Parameters in objective function. 
        params.objective = util.AttrDict(
            w_final=300.0,
            w_ch_accel=0.05,
            w_ch_turning=0.20,
            w_ch_joint=0.01,
            w_accel=0.05,
            w_turning=0.10,
            w_joint=0.02,
        )
        # Maximum steering angle
        physics_control = self.__ego_vehicle.get_physics_control()
        wheels = physics_control.wheels
        params.limit_delta = np.deg2rad(wheels[0].max_steer_angle)
        # Max steering
        #   We fix max turning angle to make reasonable planned turns.
        params.max_delta = 0.5*params.limit_delta

        # longitudinal and lateral dimensions of car are normally 3.70 m, 1.79 m resp.
        bbox = util.AttrDict()
        bbox.lon, bbox.lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        params.bbox = bbox
        # Number of faces of obstacle sets
        params.L = 4
        # Minimum distance from vehicle to avoid collision.
        #   Assumes that car is a circle.
        # TODO: remove this. Improve bounds instead
        params.diag = np.sqrt(bbox.lon**2 + bbox.lat**2) / 2.
        return params

    def __setup_rectangular_boundary_conditions(self):
        # __road_segment_enclosure : np.array
        #   Array of shape (4, 2) enclosing the road segment
        # __road_seg_starting : np.array
        #   The position and the heading angle of the starting waypoint
        #   of the road of form [x, y, angle] in (meters, meters, radians).
        (
            self.__road_seg_starting,
            self.__road_seg_enclosure,
            self.__road_seg_params,
        ) = self.__map_reader.road_segment_enclosure_from_actor(self.__ego_vehicle)
        self.__road_seg_starting[1] *= -1  # need to flip about x-axis
        self.__road_seg_starting[2] = util.npu.reflect_radians_about_x_axis(
            self.__road_seg_starting[2]
        )  # need to flip about x-axis
        self.__road_seg_enclosure[:, 1] *= -1  # need to flip about x-axis
        # __goal
        #   Goal destination the vehicle should navigates to.
        self.__goal = util.AttrDict(x=50, y=0, is_relative=True)

    def __setup_curved_road_segmented_boundary_conditions(
        self, turn_choices, max_distance
    ):
        # __turn_choices : list of int
        #   List of choices of turns to make at intersections,
        #   starting with the first intersection to the last.
        self.__turn_choices = turn_choices
        # __max_distance : number
        #   Maximum distance from road
        self.__max_distance = max_distance
        # __road_segs : util.AttrDict
        #   Container of road segment properties.
        self.__road_segs = self.__map_reader.curved_road_segments_enclosure_from_actor(
            self.__ego_vehicle,
            self.__max_distance,
            choices=self.__turn_choices,
            flip_y=True,
        )
        logging.info(
            f"max curvature of planned path is {self.__road_segs.max_k}; "
            f"created {len(self.__road_segs.polytopes)} polytopes covering "
            f"a distance of {np.round(self.__max_distance, 2)} m in total."
        )
        x, y = self.__road_segs.spline(self.__road_segs.distances[-1])
        # __goal
        #   Not used for motion planning when using this BC.
        self.__goal = util.AttrDict(x=x, y=y, is_relative=False)

    def __setup_road_boundary_conditions(
        self, turn_choices, max_distance
    ):
        """Set up a generic road boundary configuration.

        TODO: extend RoadBoundaryConstraint so it takes over the role of
        __setup_curved_road_segmented_boundary_conditions() and
        __setup_rectangular_boundary_conditions()
        """
        # __turn_choices : list of int
        #   List of choices of turns to make at intersections,
        #   starting with the first intersection to the last.
        self.__turn_choices = turn_choices
        # __max_distance : number
        #   Maximum distance from road
        self.__max_distance = max_distance
        # __road_boundary : RoadBoundaryConstraint
        #   Factory for road boundary constraints.
        self.__road_boundary = self.__map_reader.road_boundary_constraints_from_actor(
            self.__ego_vehicle, self.__max_distance,
            choices=self.__turn_choices, flip_y=True,
        )
        n_polytopes = len(self.__road_boundary.road_segs.polytopes)
        logging.info(
            f"created {n_polytopes} polytopes covering "
            f"a distance of {np.round(self.__max_distance, 2)} m in total."
        )
        x, y = self.__road_boundary.points[-1]
        # __goal
        #   Not used for motion planning when using this BC.
        self.__goal = util.AttrDict(x=x, y=y, is_relative=False)

    def __init__(
        self,
        ego_vehicle,
        map_reader: MapQuerier,
        other_vehicle_ids,
        eval_stg,
        scene_builder_cls=TrajectronPlusPlusSceneBuilder,
        scene_config=OnlineConfig(),
        ##########################
        # Motion Planning settings
        n_burn_interval=4,
        n_predictions=100,
        prediction_horizon=8,
        control_horizon=6,
        step_horizon=1,
        n_coincide=1,
        random_mcc=False, # ??
        road_boundary_constraints=True,
        angle_boundary_constraints=False,
        #######################
        # Logging and debugging
        log_cplex=False,
        log_agent=False,
        plot_simulation=False,
        plot_boundary=False,
        plot_scenario=False,
        plot_vertices=False,
        plot_overapprox=False,
        #######################
        # Planned path settings
        turn_choices=[],
        max_distance=100,
        #######################
        **kwargs,
    ):
        assert control_horizon <= prediction_horizon
        assert n_coincide <= step_horizon
        # __ego_vehicle : carla.Vehicle
        #   The vehicle to control in the simulator.
        self.__ego_vehicle = ego_vehicle
        # __map_reader : MapQuerier
        #   To query map data.
        self.__map_reader = map_reader
        #   Prediction Model to generate multi-agent forecasts.
        self.__eval_stg = eval_stg
        # __n_burn_interval : int
        #   Interval in prediction timesteps to skip prediction and control.
        self.__n_burn_interval = n_burn_interval
        # __n_predictions : int
        #   Number of predictions to generate on each control step.
        self.__n_predictions = n_predictions
        # __prediction_horizon : int
        #   Number of predictions timesteps to predict other vehicles over.
        self.__prediction_horizon = prediction_horizon
        # __control_horizon : int
        #   Number of predictions steps to optimize control over.
        self.__control_horizon = control_horizon
        # __step_horizon : int
        #   Number of predictions steps to execute at each iteration of MPC.
        self.__step_horizon = step_horizon
        # __n_coincide : int
        #   Number of control steps to coincide.
        self.__n_coincide = n_coincide
        # __random_mcc : bool
        #   Whether to use MCC or random MCC
        self.__random_mcc = random_mcc
        self.__scene_builder_cls = scene_builder_cls
        self.__scene_config = scene_config
        # __first_frame : int
        #   First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__world = self.__ego_vehicle.get_world()
        vehicles = self.__world.get_actors(other_vehicle_ids)
        # __other_vehicles : list of carla.Vehicle
        #     List of IDs of vehicles not including __ego_vehicle.
        #     Use this to track other vehicles in the scene at each timestep. 
        self.__other_vehicles = dict(zip(other_vehicle_ids, vehicles))
        # __steptime : float
        #   Time in seconds taken to complete one step of MPC.
        self.__steptime = (
            self.__scene_config.record_interval
            * self.__world.get_settings().fixed_delta_seconds
        )
        # __sensor : carla.Sensor
        #     Segmentation sensor. Data points will be used to construct overhead.
        self.__sensor = self.__create_segmentation_lidar_sensor()
        # __lidar_feeds : collections.OrderedDict
        #     Where int key is frame index and value
        #     is a carla.LidarMeasurement or carla.SemanticLidarMeasurement
        self.__lidar_feeds = collections.OrderedDict()
        # __U_warmstarting : ndarray
        #   Controls computed from last MPC step for warmstarting.
        self.__U_warmstarting = None
        self.__local_planner = VehiclePIDController(self.__ego_vehicle)
        # __params : util.AttrDict
        #   Global parameters for optimization.
        self.__params = self.__make_global_params()
        lon = self.__params.bbox.lon
        self.__vehicle_model = VehicleModel(
            self.__control_horizon, self.__steptime, l_r=0.5 * lon, L=lon
        )
        self.__setup_road_boundary_conditions(
            turn_choices, max_distance
        )
        self.road_boundary_constraints = road_boundary_constraints
        self.angle_boundary_constraints = angle_boundary_constraints
        self.log_cplex = log_cplex
        self.log_agent = log_agent
        self.plot_simulation = plot_simulation
        self.plot_boundary = plot_boundary
        self.plot_scenario   = plot_scenario
        self.plot_vertices   = plot_vertices
        self.plot_overapprox = plot_overapprox
        if self.plot_simulation:
            self.__plot_simulation_data = util.AttrDict(
                actual_trajectory=collections.OrderedDict(),
                planned_trajectories=collections.OrderedDict(),
                planned_controls=collections.OrderedDict(),
                goals=collections.OrderedDict(),
                lowlevel=collections.OrderedDict(),
            )

    def get_vehicle_state(self, flip_x=False, flip_y=False):
        """Get the vehicle state as an ndarray. State consists of
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
        length, width, height, pitch, yaw, roll] where pitch, yaw, roll are in
        radians."""
        return carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(
            self.__ego_vehicle, flip_x=flip_x, flip_y=flip_y
        )

    def get_goal(self):
        return copy.copy(self.__goal)

    
    def set_goal(self, x=None, y=None, distance=None, is_relative=True, **kwargs):
        if x is not None and y is not None:
            self.__goal = util.AttrDict(x=x, y=y, is_relative=is_relative)
        elif distance is not None:
            point = self.__road_boundary.get_point_from_start(distance)
            self.__goal = util.AttrDict(x=point[0], y=point[1], is_relative=False)
        else:
            raise NotImplementedError("Unknown method of setting motion planner goal.")

    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.__sensor.listen(lambda image: type(self).parse_image(weak_self, image))
    
    def stop_sensor(self):
        """Stop the sensor."""
        self.__sensor.stop()

    @property
    def sensor_is_listening(self):
        return self.__sensor.is_listening

    def __plot_simulation(self):
        if len(self.__plot_simulation_data.planned_trajectories) == 0:
            return
        filename = f"agent{self.__ego_vehicle.id}_simulation"
        bbox = self.__params.bbox
        PlotSimulation(
            self.__scene_builder.get_scene(),
            self.__map_reader.map_data,
            self.__plot_simulation_data.actual_trajectory,
            self.__plot_simulation_data.planned_trajectories,
            self.__plot_simulation_data.planned_controls,
            self.__plot_simulation_data.goals,
            self.__plot_simulation_data.lowlevel,
            self.__road_boundary.road_segs,
            np.array([bbox.lon, bbox.lat]),
            self.__step_horizon,
            self.__steptime,
            T_coin=self.__n_coincide,
            filename=filename,
            road_boundary_constraints=self.road_boundary_constraints
        ).plot_mcc()
        PlotPIDController(
            self.__plot_simulation_data.lowlevel,
            self.__world.get_settings().fixed_delta_seconds,
            filename=filename
        ).plot()

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.__sensor.destroy()
        self.__sensor = None
        if self.plot_simulation:
            self.__plot_simulation()

    def do_prediction(self, frame):
        """Get processed scene object from scene builder,
        input the scene to a model to generate the predictions,
        and then return the predictions and the latents variables."""

        """Construct online scene"""
        scene = self.__scene_builder.get_scene()

        """Extract Predictions"""
        frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
        timestep = frame_id # we use this as the timestep
        timesteps = np.array([timestep])
        with torch.no_grad():
            z, predictions, nodes, predictions_dict, latent_probs = generate_vehicle_latents(
                    self.__eval_stg, scene, timesteps,
                    num_samples=self.__n_predictions,
                    ph=self.__prediction_horizon,
                    z_mode=False, gmm_mode=False, full_dist=False, all_z_sep=False)

        _, past_dict, ground_truth_dict = \
                prediction_output_to_trajectories(
                    predictions_dict, dt=scene.dt, max_h=10,
                    ph=self.__prediction_horizon, map=None)
        return util.AttrDict(scene=scene, timestep=timestep, nodes=nodes,
                predictions=predictions, z=z, latent_probs=latent_probs,
                past_dict=past_dict, ground_truth_dict=ground_truth_dict)

    def make_ovehicles(self, result):
        scene, timestep, nodes = result.scene, result.timestep, result.nodes
        predictions, latent_probs, z = result.predictions, result.latent_probs, result.z
        past_dict, ground_truth_dict = result.past_dict, result.ground_truth_dict

        """Preprocess predictions"""
        minpos = np.array([scene.x_min, scene.y_min])
        ovehicles = []
        for idx, node in enumerate(nodes):
            if node.id == 'ego':
                continue
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(
                self.__other_vehicles[int(node.id)]
            )
            veh_bbox = np.array([lon, lat])
            veh_gt         = ground_truth_dict[timestep][node] + minpos
            veh_past       = past_dict[timestep][node] + minpos
            veh_predict    = predictions[idx] + minpos
            veh_latent_pmf = latent_probs[idx]
            n_states = veh_latent_pmf.size
            zn = z[idx]
            veh_latent_predictions = [[] for x in range(n_states)]
            for jdx, p in enumerate(veh_predict):
                veh_latent_predictions[zn[jdx]].append(p)
            for jdx in range(n_states):
                veh_latent_predictions[jdx] = np.array(veh_latent_predictions[jdx])
            ovehicle = OVehicle.from_trajectron(node,
                    self.__prediction_horizon, veh_gt, veh_past,
                    veh_latent_pmf, veh_latent_predictions, bbox=veh_bbox)
            ovehicles.append(ovehicle)
        
        return ovehicles

    def get_current_velocity(self):
        """Get current velocity of vehicle in m/s."""
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        return np.sqrt(v_0_x ** 2 + v_0_y ** 2)

    def make_local_params(self, frame, ovehicles,Tsh):
        """Get the local optimization parameters used for current MPC step."""

        """Get parameters to construct control and state variables."""
        params = util.AttrDict()
        params.frame = frame
        p_0_x, p_0_y, _ = carlautil.to_location_ndarray(self.__ego_vehicle, flip_y=True)
        _, psi_0, _ = carlautil.actor_to_rotation_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        v_0_mag = self.get_current_velocity()
        x_init = np.array([p_0_x, p_0_y, psi_0, v_0_mag])
        params.x_init = x_init # hyeontae
        initial_state = util.AttrDict(world=x_init, local=np.array([0, 0, 0, v_0_mag]))
        params.initial_state = initial_state
        # try:
        #     u_init = self.__U_warmstarting[self.__step_horizon]
        # except TypeError:
        #     u_init = np.array([0., 0.])
        # Using previous control doesn't work
        u_init = np.array([0.0, 0.0])

        lon = self.__params.bbox.lon

        self.__vehicle_model = VehicleModel(
            Tsh, self.__steptime, l_r=0.5 * lon, L=lon
        )
        x_bar, u_bar, Gamma, nx, nu = self.__vehicle_model.get_optimization_ltv(
            x_init, u_init
        )
        params.x_bar, params.u_bar, params.Gamma = x_bar, u_bar, Gamma
        params.nx, params.nu = nx, nu

        """Get parameters for other vehicles."""
        # O - number of obstacles
        params.O = len(ovehicles)
        # K - for each o=1,...,O K[o] is the number of outer approximations for vehicle o
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states

        """Get parameters for (random) multiple coinciding control."""
        # N_traj : int 
        #   Number of planned trajectories possible to compute
        params.N_traj = np.prod(params.K)
        if params.O == 0:
            params.sublist_joint_decisions = [[]]
            params.N_select = 1
        elif self.__random_mcc:
            """How does random multiple coinciding control work?
            each item in the product set S_1 X S_2 X S_3 X S_4
            represents the particular choices each vehicles make 
            get the subset of the product set S_1 X S_2 X S_3 X S_4
            such that for each i = 1..4, for each j in S_i there is
            a tuple in the subset with j in the i-th place.
            """
            vehicle_n_states = [ovehicle.n_states for ovehicle in ovehicles]            
            n_states_max = max(vehicle_n_states)
            vehicle_state_ids = [
                util.range_to_list(n_states) for n_states in vehicle_n_states
            ]
            def preprocess_state_ids(state_ids):
                state_ids = state_ids + random.choices(
                    state_ids, k=n_states_max - len(state_ids)
                )
                random.shuffle(state_ids)
                return state_ids
            vehicle_state_ids = util.map_to_list(preprocess_state_ids, vehicle_state_ids)
            # sublist_joint_decisions : list of (list of int)
            #   Subset of S_1 X S_2 X S_3 X S_4 of joint decisions
            params.sublist_joint_decisions = np.array(vehicle_state_ids).T.tolist()
            # N_select : int
            #   Number of planned trajectories to compute
            params.N_select = len(params.sublist_joint_decisions)

        else:
            # sublist_joint_decisions : list of (list of int)
            #   Entire set of S_1 X S_2 X S_3 X S_4 of joint decision outcomes
            params.sublist_joint_decisions = util.product_list_of_list(
                [util.range_to_list(ovehicle.n_states) for ovehicle in ovehicles]
            )
            # N_select : int
            #   Number of planned trajectories to compute, equal to N_traj.
            params.N_select = len(params.sublist_joint_decisions)

        return params

    def __plot_segs_polytopes(self, params, segments, goal):
        fig, ax = plt.subplots(figsize=(7, 7))
        x_min, y_min = np.min(self.__road_boundary.points, axis=0)
        x_max, y_max = np.max(self.__road_boundary.points, axis=0)
        self.__map_reader.render_map(
            ax, extent=(x_min - 20, x_max + 20, y_min - 20, y_max + 20)
        )
        x, y, _ = carlautil.to_location_ndarray(self.__ego_vehicle, flip_y=True)
        ax.scatter(x, y, c="r", zorder=10)
        x, y = goal
        ax.scatter(x, y, c="g", marker="*", zorder=10)
        for (A, b), in_junction in zip(segments.polytopes, segments.mask):
            if in_junction:
                util.npu.plot_h_polyhedron(ax, A, b, fc='r', ec='r', alpha=0.3)
            else:
                util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.3)
        filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_boundary"
        fig.savefig(os.path.join("out", f"{filename}.png"))
        fig.clf()

    def compute_segs_polytopes_and_goal(self, params, Tsh):
        """Compute the road boundary constraints and goal.

        Returns
        =======
        util.AttrDict
            Payload of segment polytopes for optimization. 
        ndarray
            Global (x, y) coordinates for car's destination at for the MPC step.
        """
        position = params.initial_state.world[:2]
        v_lim = min(self.__ego_vehicle.get_speed_limit() * 0.28, self.__params.max_v)
        distance = v_lim * self.__steptime * Tsh + 1
        segments = self.__road_boundary.collect_segs_polytopes_and_goal(
            position, distance
        )
        goal = segments.goal
        if self.plot_boundary:
            self.__plot_segs_polytopes(params, segments, goal)
        return segments, goal
    
    def compute_state_constraints(self, params, X):
        """Set velocity magnitude constraints.
        Usually street speed limits are 30 km/h == 8.33.. m/s.
        Speed limits can be 30, 40, 60, 90 km/h

        Parameters
        ==========
        params : util.AttrDict
        X : np.array of docplex.mp.vartype.VarType
            State space variables of shape (?, T, nx)
        """
        v = X[:, 3]
        max_v = self.__params.max_v
        constraints = []
        constraints.extend([z <= max_v for z in v])
        constraints.extend([z >= 0 for z in v])
        return constraints



    def __compute_vertices(self, params, ovehicles):
        """Compute verticles from predictions."""
        K, n_ov = params.K, params.O
        T = self.__prediction_horizon
        vertices = np.empty((T, np.max(K), n_ov,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    ps = ovehicle.pred_positions[latent_idx][:,t]
                    yaws = ovehicle.pred_yaws[latent_idx][:,t]
                    vertices[t][latent_idx][ov_idx] = util.npu.vertices_of_bboxes(
                            ps, yaws, ovehicle.bbox)

        return vertices

    def __plot_overapproximations(self, params, ovehicles, vertices, A_union, b_union):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        ax = axes[1]
        X = vertices[-1][0][0][:,0:2].T
        ax.scatter(X[0], X[1], color='r', s=2)
        X = vertices[-1][0][0][:,2:4].T
        ax.scatter(X[0], X[1], color='b', s=2)
        X = vertices[-1][0][0][:,4:6].T
        ax.scatter(X[0], X[1], color='g', s=2)
        X = vertices[-1][0][0][:,6:8].T
        ax.scatter(X[0], X[1], color='m', s=2)
        A = A_union[-1][0][0]
        b = b_union[-1][0][0]
        try:
            util.npu.plot_h_polyhedron(ax, A, b, fc='none', ec='k')
        except scipy.spatial.qhull.QhullError as e:
            print(f"Failed to plot polyhedron of OV")

        x, y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        ax = axes[0]
        ax.scatter(x, y, marker='*', c='k', s=100)
        ovehicle_colors = get_ovehicle_color_set([ov.n_states for ov in ovehicles])
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                logging.info(f"Plotting OV {ov_idx} latent value {latent_idx}.")
                color = ovehicle_colors[ov_idx][latent_idx]
                for t in range(self.__prediction_horizon):
                    X = vertices[t][latent_idx][ov_idx][:,0:2].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,2:4].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,4:6].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,6:8].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    A = A_union[t][latent_idx][ov_idx]
                    b = b_union[t][latent_idx][ov_idx]
                    try:
                        util.npu.plot_h_polyhedron(ax, A, b, fc='none', ec=color)#, alpha=0.3)
                    except scipy.spatial.qhull.QhullError as e:
                        print(f"Failed to plot polyhedron of OV {ov_idx} latent value {latent_idx} timestep t={t}")
        
        for ax in axes:
            ax.set_aspect('equal')
        filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_overapprox"
        fig.savefig(os.path.join('out', f"{filename}.png"))
        fig.clf()

    def __compute_overapproximations(self, params, ovehicles, vertices):
        """Compute the approximation of the union of obstacle sets.

        Parameters
        ==========
        vertices : ndarray
            Verticles to for overapproximations.
        params : util.AttrDict
            Parameters of motion planning problem.
        ovehicles : list of OVehicle
            Vehicles to compute overapproximations from.

        Returns
        =======
        ndarray
            Collection of A matrices of shape (N_traj, T, O, L, 2).
            Axis 2 (zero-based) is sorted by ovehicle.
        ndarray
            Collection of b vectors of shape (N_traj, T, O, L).
            Axis 2 (zero-based) is sorted by ovehicle.
        """

        """Compute overapproximations across all vehicles, latents and timesteps"""
        T = self.__prediction_horizon
        K, O = params.K, params.O
        A_union = np.empty((T, np.max(K), O,), dtype=object).tolist()
        b_union = np.empty((T, np.max(K), O,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    yaws = ovehicle.pred_yaws[latent_idx][:,t]
                    vertices_k = vertices[t][latent_idx][ov_idx]
                    mean_theta_k = np.mean(yaws)
                    A_union_k, b_union_k = compute_L4_outerapproximation(
                        mean_theta_k, vertices_k
                    )
                    A_union[t][latent_idx][ov_idx] = A_union_k
                    b_union[t][latent_idx][ov_idx] = b_union_k

        """Assign overapproximations to subsampling of joint decisions."""
        N_select = params.N_select
        A_unions = np.empty((N_select, T, O,), dtype=object).tolist()
        b_unions = np.empty((N_select, T, O,), dtype=object).tolist()
        for traj_idx, latent_indices in enumerate(params.sublist_joint_decisions):
            for t in range(T):
                for ov_idx, ovehicle in enumerate(ovehicles):
                    latent_idx = latent_indices[ov_idx]
                    A_unions[traj_idx][t][ov_idx] = A_union[t][latent_idx][ov_idx]
                    b_unions[traj_idx][t][ov_idx] = b_union[t][latent_idx][ov_idx]
    
        """Plot the overapproximation"""
        if self.plot_overapprox:
            raise NotImplementedError()
            # self.__plot_overapproximations(params, ovehicles, vertices, A_union, b_union)

        return np.array(A_unions), np.array(b_unions)
    def compute_road_boundary_constraints(self, params, X, Omicron, segments, Tsh):
        boundconstraints = []
        T = Tsh
        M_big = self.__params.M_big
        N_select = params.N_select
        # diag = self.__params.diag
        for n in range(N_select):
            for t in range(T):
                #Omicron_nt = []
                sum_Omicron = cp.Constant(0)
                for seg_idx, (A, b) in enumerate(segments.polytopes):
                    #lhs = util.obj_matmul(A, X[n, t, :2]) - np.array(
                    lhs = util.obj_matmul(A, X[n][t, :2]) - np.array(
                        M_big * (1 - Omicron[(n, seg_idx, t)])
                    )
                    rhs = b  # - diag
                    """Constraints on road boundaries"""
                    boundconstraints.extend([l <= r for (l, r) in zip(lhs, rhs)])
                    sum_Omicron = cp.sum([sum_Omicron, Omicron[(n, seg_idx, t)]])
                    #Omicron_nt.append(Omicron[(n, seg_idx, t)])
                #boundconstraints.append(cp.sum(Omicron_nt) >= 1)
                boundconstraints.append(sum_Omicron >= 1)
        return boundconstraints

    def compute_obstacle_constraints(self, params, ovehicles, X, Delta, Omicron, segments):
        """Compute obstacle constraints.

        Parameters
        ==========
        X : np.array of docplex.mp.vartype.ContinuousVarType
            State space plan of shape (N_select, T, nx)
        Delta : np.array of docplex.mp.vartype.BinaryVarType
            OV obstacle slack variables of shape (N_select, T, O, L)
        Omicron : np.array of docplex.mp.vartype.BinaryVarType
            Road boundary slack variables of shape (N_select, I, T)
        """
        constraints = []
        N_select, M_big = params.N_select, self.__params.M_big
        T = self.__control_horizon
        L = self.__params.L
        diag = self.__params.diag
        if self.road_boundary_constraints:
            S_big = M_big * np.sum(Omicron[:, ~segments.mask], axis=1)
            S_big = np.repeat(S_big[..., None], L, axis=2)
        else:
            S_big = np.zeros(N_select, T, L, dtype=float)
        vertices = self.__compute_vertices(params, ovehicles)
        A_unions, b_unions = self.__compute_overapproximations(params, ovehicles, vertices)
        for n in range(N_select):
            # select outerapprox. by index n
            A_union, b_union = A_unions[n], b_unions[n]
            for t in range(T):
                As, bs = A_union[t], b_union[t]
                for o, (A, b) in enumerate(zip(As, bs)):
                    lhs = util.obj_matmul(A, X[n,t,:2]) + M_big*(1 - Delta[n,t,o]) + S_big[n,t]
                    rhs = b + diag
                    constraints.extend([l >= r for (l,r) in zip(lhs, rhs)])
                    constraints.extend([np.sum(Delta[n,t,o]) >= 1])
        return constraints, vertices, A_unions, b_unions
    
    def compute_obstacle_constraints_GMM(
        self, params, ovehicles, Delta2, temp_x, eps_ura, Omicron, segments, Tsh 
    ):
        GMMconstraints = []
        N_select, M_big = params.N_select, self.__params.M_big
        T = Tsh
        L = self.__params.L
        I = len(segments.polytopes)
        
        truck_d = np.array([self.__params.bbox.lon, self.__params.bbox.lat])
        diag = self.__params.diag
        CAR_R = diag
        truck = {'d': np.array([3.7, 1.79])}
        truck_d = truck['d']
        CAR_R = 4.47213
        # Ignoring impossible modes of the OV
        if self.road_boundary_constraints:
            # Initializing Z with all possible keys
            Z = {(n, i, t): cp.Variable(boolean=True) 
                for n in range(N_select) 
                for i in range(I) 
                for t in range(T)}

            # Then apply the mask
            mask_constraints = [
                Z[n, i, t] == (Omicron[n, i, t] if not segments.mask[i] else 0)
                for n in range(N_select) 
                for i in range(I) 
                for t in range(T)
            ]
            # Summing along axis=1 for Omicron's second dimension: # of segments
            Z_sum = { 
                (n, t): cp.sum([Z[(n, i, t)] for i in range(I)]) 
                for n in range(N_select)
                for t in range(T)
            }
            # Create S_big to accommodate the 3rd dimension
            S_big = {
                (n, t): M_big * Z_sum[(n, t)] 
                for n in range(N_select) 
                for t in range(T)
            }
            # Expand S_big into S_big_repeated with the L dimension
            S_big_repeated = {
                (n, t, l): S_big[(n, t)] 
                for n in range(N_select) 
                for t in range(T) 
                for l in range(L)
            }
            GMMconstraints += mask_constraints
        else:
            S_big_repeated = {(n, t, l): 0 for n in range(N_select) for t in range(T) for l in range(L)}
        O = params.O
        K = params.K

        # Let's check if the first prediction passes the intersection or not.
        Tpred = self.__control_horizon

        OV_injuction = np.empty((O,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):   
            for latent_idx in range(ovehicle.n_states):
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                posemean_x = np.mean(poseData[0::Tpred,0])
                posemean_y = np.mean(poseData[0::Tpred,1])
                if posemean_x >= 190 or posemean_y <=-80: # in Town03_scene4
                #if posemean_x <= -98.122 or (posemean_x <= -83.122 and posemean_y <= -21.178): # in Town03_scene3
                #    OV_injuction[ov_idx] = False
                #elif (posemean_x >= -80.122 and posemean_y >= 10.178): 
                    OV_injuction[ov_idx] = False
                else:
                    OV_injuction[ov_idx] = True
        # make constraints
        #if OV_injuction:
        OVconstraint = False
        for ov_idx, ovehicle in enumerate(ovehicles):
            OVconstraint = OVconstraint or OV_injuction[ov_idx]


        N_select = 0
        for ov_idx, ovehicle in enumerate(ovehicles): 
            N_select = np.max([ovehicle.n_states, N_select])


        # Save pose, angle [tau+1|tau] of OV
        posemean_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posemean_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawmean_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        posecov_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
        yawcov_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                # load data
                poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                yawData = np.vstack(ovehicle.pred_yaws[latent_idx])

                posemean_x_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,0])
                posemean_y_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,1])
                yawmean_save[ov_idx][latent_idx] = np.mean(yawData[:,0])
                posecov_x_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,0])
                posecov_y_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,1])
                yawcov_save[ov_idx][latent_idx] = np.cov(yawData[:,0])

        logging.info(f"K:{params.K}")
        for n in range(N_select):
            for ov_idx, ovehicle in enumerate(ovehicles):
                if OV_injuction[ov_idx]:
                    #num_mode[ov_idx] = ovehicle.n_states
                    if n+1 > ovehicle.n_states: # N_select > ovehicle.n_states:
                        continue
                    # latent_idx = N_select % ovehicle.n_states # latent_idx = N_select % ovehicle.n_states 
                    latent_idx = n
                    for t in range(T):
                        yawData = np.vstack(ovehicle.pred_yaws[latent_idx])
                        poseData = np.vstack(ovehicle.pred_positions[latent_idx])

                        coeff1 = [-np.cos(yawData[:,t]), np.sin(yawData[:,t]), np.cos(yawData[:,t]) * poseData[t::Tpred,0] - np.sin(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[1]/2]
                        coeff2 = [-np.sin(yawData[:,t]), -np.cos(yawData[:,t]), np.sin(yawData[:,t]) * poseData[t::Tpred,0] + np.cos(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[0]/2]
                        coeff3 = [np.cos(yawData[:,t]), -np.sin(yawData[:,t]), -np.cos(yawData[:,t]) * poseData[t::Tpred,0] + np.sin(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[1]/2]
                        coeff4 = [np.sin(yawData[:,t]), np.cos(yawData[:,t]), -np.sin(yawData[:,t]) * poseData[t::Tpred,0] - np.cos(yawData[:,t]) * poseData[t::Tpred,1] + truck_d[0]/2]
                                
                        delta_nit = Delta2[(latent_idx, ov_idx, t)]          
                        eps_ijt = eps_ura[ov_idx, latent_idx] / Tpred
                        Gamma_ijt = norm.ppf(1-eps_ijt)
                    
                        mean1 = np.mean(coeff1, axis=1)
                        mean2 = np.mean(coeff2, axis=1)
                        mean3 = np.mean(coeff3, axis=1)
                        mean4 = np.mean(coeff4, axis=1)
                        
                        cov1 = linalg.sqrtm(np.cov(coeff1))
                        cov2 = linalg.sqrtm(np.cov(coeff2))
                        cov3 = linalg.sqrtm(np.cov(coeff3))
                        cov4 = linalg.sqrtm(np.cov(coeff4))

                        # save covs and means
                        # cov1_fro[ov_idx][latent_idx][t] = np.linalg.norm(cov1,'fro')
                        # cov2_fro[ov_idx][latent_idx][t] = np.linalg.norm(cov2,'fro')
                        # cov3_fro[ov_idx][latent_idx][t] = np.linalg.norm(cov3,'fro')
                        # cov4_fro[ov_idx][latent_idx][t] = np.linalg.norm(cov4,'fro')
                        
                        # mean1_save[ov_idx][latent_idx][t] = mean1
                        # mean2_save[ov_idx][latent_idx][t] = mean2
                        # mean3_save[ov_idx][latent_idx][t] = mean3
                        # mean4_save[ov_idx][latent_idx][t] = mean4

                        GMMconstraints += [
                            mean1.T @ temp_x[n,t] + Gamma_ijt * cp.norm(cov1 @ temp_x[n,t], p=2) + 0.5 * CAR_R - M_big * (1-delta_nit[0]) <= S_big_repeated[n, t, 0] ,
                            mean2.T @ temp_x[n,t] + Gamma_ijt * cp.norm(cov2 @ temp_x[n,t], p=2) + 0.5 * CAR_R - M_big * (1-delta_nit[1]) <= S_big_repeated[n, t, 1] ,
                            mean3.T @ temp_x[n,t] + Gamma_ijt * cp.norm(cov3 @ temp_x[n,t], p=2) + 0.5 * CAR_R - M_big * (1-delta_nit[2]) <= S_big_repeated[n, t, 2] ,
                            mean4.T @ temp_x[n,t] + Gamma_ijt * cp.norm(cov4 @ temp_x[n,t], p=2) + 0.5 * CAR_R - M_big * (1-delta_nit[3]) <= S_big_repeated[n, t, 3] ,
                        ]

                        GMMconstraints += [cp.sum(delta_nit) == 1]
                else:
                    GMMconstraints += []


        ovStateMean_tau_1 = (posemean_x_save, posemean_y_save, yawmean_save)
        ovStateCov_tau_1 = (posecov_x_save, posecov_y_save, yawcov_save)

        # cov_to_save = (cov1_fro, cov2_fro, cov3_fro, cov4_fro)
        # mean_to_save = (mean1_save, mean2_save, mean3_save, mean4_save)

        # data_save = {'cov': cov_to_save, 'mean': mean_to_save, 'num_mode':num_mode}
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(
            params, ovehicles, vertices
        )
        return GMMconstraints, vertices, A_union, b_union, OVconstraint, ovStateMean_tau_1, ovStateCov_tau_1 #, data_save
    
    def compute_objective(self, X, U, goal, N_select, Tsh):
        """Set the objective."""
        obj = self.__params.objective
        T = Tsh
        R1 = cp.Constant([[obj.w_accel, obj.w_joint], [obj.w_joint, obj.w_turning]])
        R2 = cp.Constant([[obj.w_ch_accel, obj.w_ch_joint], [obj.w_ch_joint, obj.w_ch_turning]])
        '''
        # final destination objective
        cost = obj.w_final * cp.sum([(X[n, -1, 0] - goal[0]) ** 2 
                    + (X[n, -1, 1] - goal[1]) ** 2 for n in range(N_select)])
        '''
        cost = 0
        # change in acceleration objective, turning objective
        for n in range(N_select):
            # Control effort
            cost += cp.sum([cp.quad_form(cp.hstack([U[n, t, 0], U[n, t, 1]]), R1) for t in range(T)])

            # Control change
            cost += cp.sum([cp.quad_form(cp.hstack([U[n, t, 0], U[n, t, 1]]) - cp.hstack([U[n, t-1, 0], U[n, t-1, 1]]), R2) for t in range(1, T)])
        

        for n in range(N_select):
            cost += (
                obj.w_final * (X[n][-1, 0] - goal[0]) ** 2
                + obj.w_final * (X[n][-1, 1] - goal[1]) ** 2
            )
        return cost
    
    def compute_mean_objective(self, X, U, goal):
        cost = 0
        for _U, _X in zip(U, X):
            cost += self.compute_objective(_X, _U, goal)
        return cost
    
    def save_data(self, data_save, params, ego_vehicle_id):
        filename_cov = f"agent{ego_vehicle_id}_frame{params.frame}_cov"
        #filename_cov = f"100_frame{params.frame}_cov"
        filepath_cov = os.path.join("out/data",filename_cov)
        try:
            with open(filepath_cov,"wb") as f:
                pickle.dump(data_save,f)
        except Exception as e:
            print(f"Error while saving: {str(e)}")

    def do_highlevel_control(self, params, ovehicles, Tsh, shrinking):
        """Decide parameters."""
        timeLimit = 600

        """Compute road segments and goal."""
        segments, goal = self.compute_segs_polytopes_and_goal(params, Tsh)

        """Apply motion planning problem"""
        N_select = params.N_select
        x_bar, u_bar, Gamma = params.x_bar, params.u_bar, params.Gamma
        nx, nu = params.nx, params.nu
        T =  Tsh #self.__control_horizon
        max_a, min_a = self.__params.max_a, self.__params.min_a
        max_delta = self.__params.max_delta

        constraints = []
        N_select = 0
        for ov_idx, ovehicle in enumerate(ovehicles): 
            N_select = np.max([ovehicle.n_states, N_select])
        params.N_select = N_select

        logging.info(f"number of trajectories:{N_select}")
        
        """Model, control and state variables"""
        min_u = np.vstack((np.full(T, min_a), np.full(T, -max_delta))).T.ravel()
        max_u = np.vstack((np.full(T, max_a), np.full(T, max_delta))).T.ravel()

        # Slack variables for control
        u = {(n): cp.Variable(shape=(nu * T), name="u") for n in range(N_select)}
        # input bound
        for n in range(N_select):
            constraints += [u[n] >= min_u, u[n] <= max_u]


        # Constructing state variable
        u_delta = {(n): u[n] - u_bar[:nu*T] for n in range(N_select)}

        # State variables x, y, psi, v
        x = {(n):util.obj_matmul(Gamma[:nx*T,:nu*T], u_delta[n]) + x_bar[:nx*T] for n in range(N_select)}
        X = {(n):x[n].reshape(T, nx) for n in range(N_select)}

        # Control variables a, delta
        U = {(n, t, i): u[n][ t * nu + i] for n in range(N_select) for t in range(T) for i in range(nu)}
        state_constraints = []
        """Apply state constraints"""
        for n in range(N_select):
            state_constraints += self.compute_state_constraints(params, X[n])
        constraints.extend(state_constraints)

        """Apply road boundary constraints"""
        if self.road_boundary_constraints:
            N_select = params.N_select
            I = len(segments.polytopes)
            # Slack variables from road obstacles
            Omicron = {(n,i,t): cp.Variable(boolean=True) for n in range(N_select) for i in range(I) for t in range(T)}
            constraints.extend(
                self.compute_road_boundary_constraints(params, X, Omicron, segments, T)
            )
        else:
            Omicron = None
        """Apply vehicle collision constraints"""
        if params.O > 0:
            N_select = params.N_select
            O, L = params.O, self.__params.L

            # These are for Kai's planner from here
            eps = 0.05
            eps_ura = np.zeros((O, N_select))
            for i in range(O):
                for k in range(N_select):
                    eps_ura[i, k] = eps / (O) 

            # Slack variables for vehicle obstacles
            Delta2 = {(n, i, t): cp.Variable((L), boolean=True
            , name="delta2") for n in range(N_select) for i in range(params.O) for t in range(T)}
            
            # time horizon 0~7, T = 8
            temp_x = {(n, t): cp.vstack([X[n][t,0], X[n][t,1], 1]) for n in range(N_select) for t in range(T)}    
            
            if T < self.__control_horizon:
                logging.info(f"Shrinking horizon, T = {T}")
            else:
                logging.info(f"Receding horizon, T = {T}")

            # GMM constraints computation
            (
                GMMconstraints, vertices, A_unions, b_unions, OVconstraint, ovStateMean_tau_1, ovStateCov_tau_1
            ) = self.compute_obstacle_constraints_GMM(
                params, ovehicles, Delta2, temp_x, eps_ura, Omicron, segments, T 
            )
            saveData = True # save data indicator
            data_save = {}
            data_save["ovStateMean_tau_1"] = ovStateMean_tau_1
            data_save["ovStateCov_tau_1"] = ovStateCov_tau_1
            data_save["MeanCov"] = True # save True when param.O >0
            data_save["OVconstraint"] = OVconstraint # save True when it really considers OV constraints
            constraints.extend(GMMconstraints)
        else:
            saveData = False # save data indicator
            data_save = {}
            data_save["MeanCov"] = False
            Delta2 = None
            vertices = A_unions = b_unions = None
        
        """Set up coinciding constraints"""
        for t in range(0, self.__n_coincide):
            for i in range(nu):
                for n in range(N_select-1):
                    constraints.extend([U[n,t,i] == U[n+1,t,i]])

        """Set up coinciding constraints by states"""
        #for t in range(0, self.__n_coincide):
        #    for n in range(N_select-1):
        #        constraints.extend([temp_x[n,t] == temp_x[n+1,t]])
    
        """Compute and minimize objective"""
        cost = self.compute_objective(X, U, goal, N_select, T)
        objective = cp.Minimize(cost)
        cplex_params = {"timelimit": timeLimit}
        X = {(n, t, i):x[n][ t * nx + i] for n in range(N_select) for t in range(T) for i in range(nx)}
        data_save["x_init"] = params.x_init
        try:
            start = time.time()
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CPLEX, cplex_params=cplex_params, verbose=True)
            logging.info("finished solving problem!")
            process_time = time.time() - start

            """Extract solution"""
            cost = objective.value
            #f = lambda x: x if isinstance(x, numbers.Number) else x.value
            U_star = {
                key: var.value 
                for key, var in U.items()
            }
            X_star = {
                key: var.value 
                for key, var in X.items()
            }
            # make this an array
            X_star=list(X_star.values())
            X_star = np.array(X_star).reshape(N_select,T,nx)
            solver_time = prob.solver_stats.solve_time
            data_save['process_time'] = process_time
            data_save['solve_time'] = solver_time
            data_save['cost'] = cost
            data_save["U_star"] = U_star
            data_save["X_star"] = X_star
            data_save["goal"] = goal
            data_save["shrinking"] = shrinking
            data_save["ovehicles"] = ovehicles

            didTimeout = solver_time >= timeLimit-0.1

            if didTimeout:
                data_save["infeasible"] = True # False
                data_save["timeout"] = True
                if saveData:
                   self.save_data(data_save, params, self.__ego_vehicle.id)
                return util.AttrDict(
                    cost=cost, U_star=U_star, X_star=X_star,
                    goal=goal, A_unions=A_unions, b_unions=b_unions,
                    vertices=vertices, segments=segments
                ), True, None
            else:
                data_save["infeasible"] = False
                data_save["timeout"] = False
                if saveData:
                   self.save_data(data_save, params, self.__ego_vehicle.id) 
                return util.AttrDict(
                    cost=cost, U_star=U_star, X_star=X_star,
                    goal=goal, A_unions=A_unions, b_unions=b_unions,
                    vertices=vertices, segments=segments
                ), False, None
        except:
            data_save["infeasible"] = True
            data_save["timeout"] = False
            if saveData:
                self.save_data(data_save, params, self.__ego_vehicle.id) 
            # Handle CVXPY-specific errors here
            return util.AttrDict(
                cost=None, U_star=None, X_star=None,
                goal=goal, A_unions=A_unions, b_unions=b_unions,
                vertices=vertices, segments=segments
            ), False, InSimulationException("Optimizer failed to find a solution")

    def __plot_scenario(
        self, pred_result, ovehicles, params, ctrl_result, error=None
    ):
        lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        ego_bbox = np.array([lon, lat])
        params.update(self.__params)
        if error:
            filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_mcc_fail"
            PlotPredictiveControl(
                pred_result, ovehicles, params, ctrl_result,
                self.__control_horizon, ego_bbox, T_coin=self.__n_coincide
            ).plot_mcc_failure(filename=filename)
        else:
            filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_mcc_predict"
            PlotPredictiveControl(
                pred_result, ovehicles, params, ctrl_result,
                self.__control_horizon, ego_bbox, T_coin=self.__n_coincide
            ).plot_mcc_prediction(filename=filename)

    @profile(sort_by='cumulative', lines_to_print=50, strip_dirs=True)
    def __compute_prediction_controls(self, frame, Tsh, shrinking):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_local_params(frame, ovehicles,Tsh)
        ctrl_result, timeout, error = self.do_highlevel_control(params, ovehicles, Tsh, shrinking)
        if self.plot_scenario:
            """Plot scenario"""
            self.__plot_scenario(
                pred_result, ovehicles, params, ctrl_result, error=error
            )
        if error:
            raise error
        
        """use control input next round for warm starting."""
        self.__U_warmstarting = ctrl_result.U_star

        if self.plot_simulation:
            """Save planned trajectory for final plotting"""
            X_init = np.repeat(params.initial_state.world[None], params.N_select, axis=0)
            X = np.concatenate((X_init[:, None], ctrl_result.X_star), axis=1)
            self.__plot_simulation_data.planned_trajectories[frame] = X
            self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star

        """Get trajectory and velocity"""
        try:
            angles = util.npu.reflect_radians_about_x_axis(
                ctrl_result.X_star[0, :self.__n_coincide, 2]
            )
            speeds = ctrl_result.X_star[0, :self.__n_coincide, 3]
        except:
            raise InSimulationException("Optimizer failed to find a solution")
            
        return speeds, angles, timeout

    def do_first_step(self, frame):
        self.__first_frame = frame
        self.__scene_builder = self.__scene_builder_cls(
            self,
            self.__map_reader,
            self.__ego_vehicle,
            self.__other_vehicles,
            self.__lidar_feeds,
            "test",
            self.__first_frame,
            scene_config=self.__scene_config,
            debug=False)

    def run_step(self, frame, Tsh=6, shrinking=False,control=None):
        """Run motion planner step. Should be called whenever carla.World.click() is called.

        Parameters
        ==========
        frame : int
            Current frame of the simulation.
        control: carla.VehicleControl (optional)
            Optional control to apply to the motion planner. Used to move the vehicle
            while burning frames in the simulator before doing motion planning.
        """
        #logging.debug(f"In LCSSHighLevelAgent.run_step() with frame = {frame}")
        if self.__first_frame is None:
            self.do_first_step(frame)
        
        self.__scene_builder.capture_trajectory(frame)
        timeout = False
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            """We only motion plan every `record_interval` frames
            (e.g. every 0.5 seconds of simulation)."""
            frame_id = int(
                (frame - self.__first_frame) / self.__scene_config.record_interval
            )
            if frame_id < self.__n_burn_interval:
                """Initially collect data without doing any control to the vehicle."""
                pass
            elif (frame_id - self.__n_burn_interval) % self.__step_horizon == 0:
                speeds, angles, timeout = self.__compute_prediction_controls(frame, Tsh, shrinking)
                self.__local_planner.set_plan(
                    speeds, angles, self.__scene_config.record_interval
                )
            if self.plot_simulation:
                """Save actual trajectory for final plotting"""
                payload = carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(
                    self.__ego_vehicle, flip_y=True
                )
                payload = np.array(
                    [
                        payload[0],
                        payload[1],
                        payload[13],
                        self.get_current_velocity(),
                    ]
                )
                self.__plot_simulation_data.actual_trajectory[frame] = payload
                self.__plot_simulation_data.goals[frame] = self.get_goal()

        if not control:
            control = self.__local_planner.step()
        self.__ego_vehicle.apply_control(control)
        if self.plot_simulation:
            payload = self.__local_planner.get_current()
            self.__plot_simulation_data.lowlevel[frame] = payload
        return timeout

    def remove_scene_builder(self, first_frame):
        raise Exception(f"Can't remove scene builder from {util.classname(first_frame)}.")
    
    @staticmethod
    def parse_image(weak_self, image):
        """Pass sensor image to each scene builder.

        Parameters
        ==========
        image : carla.SemanticLidarMeasurement
        """
        self = weak_self()
        if not self:
            return
        #logging.debug(f"in DataCollector.parse_image() player = {self.__ego_vehicle.id} frame = {image.frame}")
        self.__lidar_feeds[image.frame] = image
        if self.__scene_builder:
            self.__scene_builder.capture_lidar(image)    