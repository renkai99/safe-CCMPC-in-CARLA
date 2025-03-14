"""Plot the predictions of Trajectron++
This can be use to plot either predictions on CARLA or NuScenes data.
It works for 
"""
import sys
import os
import json
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import torch

import utility as util

# Import to generate latents
# from model.dataset import *
from model.dataset import get_timesteps_data
# from model.components import *
from model.model_utils import ModeKeys
# from model.model_utils import *
# import visualization

from helper import load_model, prediction_output_to_trajectories
from helper import *

# Colors
LATENT_CM = cm.nipy_spectral
LATENT_COLORS = LATENT_CM(np.linspace(0, 1, 25))
AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'olive', 'gold', 'orange', 'red', 'deeppink']
NCOLORS = len(AGENT_COLORS)
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 3) % NCOLORS for i in range(NCOLORS)], 0)

## Datasets.

approot = os.environ['APPROOT']

dataset_0 = util.AttrDict(
    test_set_path='../processed/nuScenes_test_full.pkl',
    name='NuScenes test set',
    desc="Preprocessed test set from NuScenes data.")

dataset_4 = util.AttrDict(
        test_set_path=f"{approot}/carla_v3_0_1_dataset/carla_test_v3_0_1_full.pkl",
        name='carla_test_v3_0_1',
        desc="CARLA synthesized dataset smaller sized with heading fix.",
        v2_bitmap=True)
    
dataset_5 = util.AttrDict(
        test_set_path=f"{approot}/carla_v3-1_dataset/v3-1_split1_test.pkl",
        name='v3-1_split1_test',
        desc="CARLA synthesized dataset with heading fix, occlusion fix, and 32 timesteps.")

## Models

model_3 = util.AttrDict(
        path='models/20210622/models_19_Mar_2021_22_14_19_int_ee_me_ph8',
        desc="Base model +Dynamics Integration, Maps with K=25 latent values "
             "(on NuScenes dataset)",
        ph=8)

model_11 = util.AttrDict(
        path='models/models_20_Jul_2021_11_48_11_carla_v3_0_1_base_distmap_ph8',
        desc="Base +Map model with heading fix, PH=8 "
             "(trained on small carla v3_0_1 dataset)",
        ph=8, v2_bitmap=True)

model_12 = util.AttrDict(
        path='models/20210804/models_25_Jul_2021_15_29_29_carla_v3-1_base_distmap_ph8',
        desc="Base +Map model with heading and occlusion fix, PH=8 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_13 = util.AttrDict(
        path='models/20210725/models_25_Jul_2021_15_37_17_carla_v3-1_base_distmap_K20_ph8',
        desc="Base +Map model with heading fix, PH=8, K=20 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_14 = util.AttrDict(
        path='models/20210725/models_25_Jul_2021_15_38_46_carla_v3-1_base_distmap_K15_ph8',
        desc="Base +Map model with heading fix, PH=8, K=15 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_15 = util.AttrDict(
        path='models/20210725/models_25_Jul_2021_15_38_20_carla_v3-1_base_distmap_K10_ph8',
        desc="Base +Map model with heading fix, PH=8, K=10 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_16 = util.AttrDict(
        path='models/20210725/models_25_Jul_2021_15_39_52_carla_v3-1_base_distmap_K5_ph8',
        desc="Base +Map model with heading fix, PH=8, K=5 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

def render_roads(ax, scene, is_v2_bitmap=False, is_white=False):
    road_color = 'white' if is_white else 'grey'
    map_mask = scene.map['VEHICLE'].as_image()
    # map_mask has shape (y, x, c)
    road_bitmap = np.max(map_mask, axis=2)
    road_div_bitmap = map_mask[..., 1]
    lane_div_bitmap = map_mask[..., 0]

    # extent = (scene.x_min, scene.x_max, scene.y_min, scene.y_max)
    extent = (0, scene.x_size, 0, scene.y_size)

    if is_v2_bitmap:
        # map_mask has shape (y, x, c)
        road_bitmap = np.max(map_mask, axis=2)
        road_div_bitmap = map_mask[..., 1]
        lane_div_bitmap = map_mask[..., 0]

        # Axes.imshow() expects (y, x, c)
        ax.imshow(road_bitmap,     extent=extent, origin='lower', cmap=colors.ListedColormap(['none', road_color]))
        ax.imshow(road_div_bitmap, extent=extent, origin='lower', cmap=colors.ListedColormap(['none', 'yellow']))
        ax.imshow(lane_div_bitmap, extent=extent, origin='lower', cmap=colors.ListedColormap(['none', 'silver']))
    else:
        """
        NuScenes bitmap format
        scene.map[...].as_image() has shape (y, x, c)
        Channel 1: lane, road_segment, drivable_area
        Channel 2: road_divider
        Channel 3: lane_divider
        """
        # NuScenes
        road_bitmap = np.max(map_mask, axis=2)
        road_bitmap = map_mask[..., 0]
        road_div_bitmap = map_mask[..., 1]
        lane_div_bitmap = map_mask[..., 2]
        ax.imshow(road_bitmap,     extent=extent, origin='lower', cmap=colors.ListedColormap(['none', road_color]))
        ax.imshow(road_div_bitmap, extent=extent, origin='lower', cmap=colors.ListedColormap(['none', 'yellow']))
        ax.imshow(lane_div_bitmap, extent=extent, origin='lower', cmap=colors.ListedColormap(['none', 'silver']))

def generate_vehicle_latents(eval_stg,
            scene, t, ph, num_samples=200,
            z_mode=False, gmm_mode=False,
            full_dist=False, all_z_sep=False):
    """Generate predicted trajectories and their corresponding
    latent variables.

    Returns
    =======
    scene : Scene
        The nuScenes scene
    t : int
        The timestep in the scene
    z : ndarray
        Has shape (number of vehicles, number of samples)
    zz : ndarray
        Has shape (number of vehicles, number of samples, number of latent values)
    predictions : ndarray
        Has shape (number of vehicles, number of samples, prediction horizon, D)
    nodes : list of Node
        Has size (number of vehicles)
        List of vehicle nodes
    predictions_dict : dict
        Contains map of predictions by timestep, by vehicle node
    """
    # Trajectron.predict() arguments
    timesteps = np.array([t])
    min_future_timesteps = 0
    min_history_timesteps = 1

    # In Trajectron.predict() scope
    node_type = eval_stg.env.NodeType.VEHICLE
    if node_type not in eval_stg.pred_state:
        raise Exception("fail")

    model = eval_stg.node_models_dict[node_type]

    # Get Input data for node type and given timesteps
    batch = get_timesteps_data(
            env=eval_stg.env, scene=scene,
            t=timesteps, node_type=node_type,
            state=eval_stg.state, pred_state=eval_stg.pred_state,
            edge_types=model.edge_types,
            min_ht=min_history_timesteps, max_ht=eval_stg.max_ht,
            min_ft=min_future_timesteps, max_ft=min_future_timesteps,
            hyperparams=eval_stg.hyperparams)
    
    # There are no nodes of type present for timestep
    if batch is None:
        raise Exception("fail")

    (first_history_index,
    x_t, y_t, x_st_t, y_st_t,
    neighbors_data_st,
    neighbors_edge_value,
    robot_traj_st_t,
    map), nodes, timesteps_o = batch

    x = x_t.to(eval_stg.device)
    x_st_t = x_st_t.to(eval_stg.device)
    if robot_traj_st_t is not None:
        robot_traj_st_t = robot_traj_st_t.to(eval_stg.device)
    if type(map) == torch.Tensor:
        map = map.to(eval_stg.device)

    # MultimodalGenerativeCVAE.predict() arguments
    inputs = x
    inputs_st = x_st_t
    first_history_indices = first_history_index
    neighbors = neighbors_data_st
    neighbors_edge_value = neighbors_edge_value
    robot = robot_traj_st_t
    prediction_horizon = ph

    # In MultimodalGenerativeCVAE.predict() scope
    mode = ModeKeys.PREDICT

    x, x_nr_t, _, y_r, _, n_s_t0 = model.obtain_encoded_tensors(mode=mode,
                                                            inputs=inputs,
                                                            inputs_st=inputs_st,
                                                            labels=None,
                                                            labels_st=None,
                                                            first_history_indices=first_history_indices,
                                                            neighbors=neighbors,
                                                            neighbors_edge_value=neighbors_edge_value,
                                                            robot=robot,
                                                            map=map)

    model.latent.p_dist = model.p_z_x(mode, x)
    z, num_samples, num_components = model.latent.sample_p(num_samples,
                                                        mode,
                                                        most_likely_z=z_mode,
                                                        full_dist=full_dist,
                                                        all_z_sep=all_z_sep)
    
    _, predictions = model.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)

    z = z.cpu().detach().numpy()
    zz = z
    # z has shape (number of samples, number of vehicles, number of latent values)
    # z[i,j] gives the latent for sample i of vehicle j
    # print(z.shape)

    # Back to Trajectron.predict() scope    
    predictions = predictions.cpu().detach().numpy()
    # predictions has shape (number of samples, number of vehicles, prediction horizon, D)
    # print(predictions.shape)

    predictions_dict = dict()
    for i, ts in enumerate(timesteps_o):
        if ts not in predictions_dict.keys():
            predictions_dict[ts] = dict()
        predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))
    
    z = np.swapaxes(np.argmax(z, axis=-1), 0, 1)
    predictions = np.swapaxes(predictions, 0, 1)
        
    return z, zz, predictions, nodes, predictions_dict

class TrajectronPlotter(object):
    def __init__(self):
        self.plotdir = 'plots'

        dataset = dataset_5
        self.is_v2_bitmap = hasattr(dataset, 'v2_bitmap')
        with open(dataset.test_set_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')
        self.eval_scenes = eval_env.scenes
        logging.info(dataset.desc)

        model = model_12
        self.ph = model.ph
        self.eval_stg, self.hyp = load_model(model.path, eval_env, ts=20, device='cuda')
        logging.info(model.desc)

        self.ptype = ''
        self.num_samples = 500
        self.dpi = 180

    def plot_scene_timestep_overhead(self, scene, t, predictions, nodes,
            predictions_dict, histories_dict, futures_dict):
        """Plot nuScense map with the forecasts on top of it for each timestep
        in each scene.

        Parameters
        ==========
        scene : Scene
            The Trajectron++ input scene
        t : int
            The timestep in the scene
        predictions
            TBD
        nodes : list of Node
            List of vehicle nodes
        predictions_dict : dict
            Contains map of predictions by timestep, by agent node
        histories_dict : dict
            Past trajectories by timestep, by agent node
        futures_dict : dict
            Ground truth trajectories by timestep, by agent node
        """

        # v_nodes = list(filter(lambda k: 'VEHICLE' in repr(k), predictions[t].keys()))

        fig, ax = plt.subplots(figsize=(12,15))
        render_roads(ax, scene)

        for idx, node in enumerate(nodes):
            player_future = futures_dict[t][node]
            player_past = histories_dict[t][node]
            player_predict = predictions_dict[t][node]

            ax.plot(player_future[:,0], player_future[:,1],
                        marker='s', color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1, markersize=8, markerfacecolor='none')
            ax.plot(player_past[:,0], player_past[:,1],
                        marker='d', color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1, markersize=8, markerfacecolor='none')
            for row in player_predict[0]:
                ax.plot(row[:,0], row[:,1],
                        marker='o', color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1, alpha=0.1, markersize=4)

        ax.set_xlim([0, scene.x_size])
        ax.set_ylim([0, scene.y_size])
        fig.tight_layout()
        scene_id = scene.name.replace('/', '_')
        savepath = 'plots/predict_{}_t{}_overhead.png'.format(scene_id, t)
        fig.savefig(savepath, dpi=self.dpi)
        plt.close('all')

    def plot_combined_latents(self,
            scene, t,
            z, zz, predictions, nodes, predictions_dict, histories_dict, futures_dict):
        """Plot latents and forecast coded by latents.

        Parameters
        ==========
        scene : Scene
            The nuScenes scene
        t : int
            The timestep in the scene
        z : ndarray
            Has shape (number of vehicles, number of samples)
        zz : ndarray
            Has shape (number of vehicles, number of samples, number of latent values)
        predictions : ndarray
            Has shape (number of vehicles, number of samples, prediction horizon, D)
        nodes : list of Node
            Has size (number of vehicles)
            List of vehicle nodes
        predictions_dict : dict
            Contains map of predictions by timestep, by vehicle node
        """
        z_counts = np.sum(zz, axis=0) / z.shape[1]
        n_vehicles = len(nodes)

        fig, (ax1, ax) = plt.subplots(2, 1, figsize=(11, 16))
        fig.tight_layout()

        ax1.set_facecolor("grey")
        for idx, zz in enumerate(z_counts):
            ax1.plot(range(25), zz, c=AGENT_COLORS[idx % NCOLORS])
        ax1.set_xlim([0, 24])
        ax1.set_ylim([0, 1])
        ax1.set_aspect(7)
        ax1.set_ylabel("p(z = value)")
        ax1.set_title(f"Latent distribution of {n_vehicles} vehicles, {z.shape[1]} samples each")
        ax1.get_xaxis().set_ticks([])

        scalarmappaple = cm.ScalarMappable(cmap=LATENT_CM)
        scalarmappaple.set_array(range(0, 25))
        plt.colorbar(scalarmappaple, ax=ax1, orientation='horizontal',
                    pad=0.01, ticks=range(0, 25), label="z = value")
        ax.set_facecolor("grey")
        ax.set_title(f"Latent plot of {n_vehicles} vehicles")
        ax.set_aspect('equal')

        for idx, node in enumerate(nodes):
            player_past = histories_dict[t][node]
            player_past = np.vstack((player_past, player_past[-1][None] + 5))
            zn = z[idx]
            pn = predictions[idx]

            ax.plot(player_past[:,0], player_past[:,1],
                    color=AGENT_COLORS[idx % NCOLORS],
                    linewidth=1)
            for jdx in range(zn.shape[0]):
                color = LATENT_COLORS[zn[jdx]]
                ax.plot(pn[jdx, :, 0], pn[jdx, :, 1],
                        marker='o',
                        color=color,
                        linewidth=1, alpha=0.2, markersize=4)
        
        scene_id = scene.name.replace('/', '_')
        savepath = 'plots/predict_{}_t{}_comb_latents.png'.format(scene_id, t)
        fig.savefig(savepath, dpi=self.dpi)
        plt.close('all')

    def plot_each_latents(self,
            scene, t,
            z, zz, predictions, nodes, predictions_dict, histories_dict, futures_dict):
        """Plot latents and forecast coded by latents.

        Parameters
        ==========
        scene : Scene
            The nuScenes scene
        t : int
            The timestep in the scene
        z : ndarray
            Has shape (number of vehicles, number of samples)
        zz : ndarray
            Has shape (number of vehicles, number of samples, number of latent values)
        predictions : ndarray
            Has shape (number of vehicles, number of samples, prediction horizon, D)
        nodes : list of Node
            Has size (number of vehicles)
            List of vehicle nodes
        predictions_dict : dict
            Contains map of predictions by timestep, by vehicle node
        """
        coords = predictions.reshape(-1, 2)
        x_max, y_max = np.amax(coords, axis=0)
        x_min, y_min = np.amin(coords, axis=0)
        figwidth  = 20.
        figheight = figwidth*(13. / 2.)*(y_max - y_min)/(x_max - x_min)
        figheight = np.min([(13. / 2.)*figwidth, figheight])
        
        n_vehicles = len(nodes)
        fig, axes = plt.subplots(13, 2,
                figsize=(figwidth, figheight))
        fig.tight_layout()
        
        for behavior in range(0, 25):
            has_plotted = False
            ax = axes[behavior // 2][behavior % 2]
            color = LATENT_COLORS[behavior]
            label=f"z = {behavior}"
            for idx, node in enumerate(nodes):
                zn = z[idx]
                pn = predictions[idx]
                mask = zn == behavior
                zn = zn[mask]
                pn = pn[mask]
                player_past = histories_dict[t][node]
                player_past = np.vstack(
                        (player_past, player_past[-1][None] + 5))
                
                ax.plot(player_past[:,0], player_past[:,1],
                        color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1)
                
                for jdx in range(zn.shape[0]):
                    ax.plot(pn[jdx, :, 0], pn[jdx, :, 1],
                            marker='o', color=color,
                            linewidth=1, alpha=0.2, markersize=4,
                            label=label)
                    label=None
                    has_plotted = True
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_aspect('equal')
            if has_plotted:
                ax.legend(loc='upper right')
            ax.set_facecolor("grey")

        axes[-1,-1].set_xlim([x_min, x_max])
        axes[-1,-1].set_ylim([y_min, y_max])
        axes[-1,-1].set_visible(False)
        
        scene_id = scene.name.replace('/', '_')
        savepath = 'plots/predict_{}_t{}_each_latents.pdf'.format(scene_id, t)
        fig.savefig(savepath, dpi=self.dpi)
        plt.close('all')

    def run(self):
        with torch.no_grad():
            for scene in self.eval_scenes:
                print(f"Plotting scene {scene.name} ({scene.timesteps} timesteps)")
                for t in range(2, scene.timesteps - 6, 3):
                    print(f"    timestep {t}")
                    z,zz, predictions, nodes, predictions_dict = generate_vehicle_latents(
                            self.eval_stg, scene, t, self.ph, num_samples=100)

                    # Obtain, past, predict and ground truth predictions
                    _, histories_dict, futures_dict = \
                            prediction_output_to_trajectories(
                                predictions_dict, dt=scene.dt, max_h=10, ph=self.ph, map=None)

                    self.plot_scene_timestep_overhead(scene, t, predictions, nodes,
                            predictions_dict, histories_dict, futures_dict)

                    z,zz, predictions, nodes, predictions_dict = generate_vehicle_latents(
                            self.eval_stg, scene, t, self.ph, num_samples=300)

                    self.plot_combined_latents(scene, t,
                            z, zz, predictions, nodes, predictions_dict, histories_dict, futures_dict)
                    self.plot_each_latents(scene, t,
                            z, zz, predictions, nodes, predictions_dict, histories_dict, futures_dict)
        
        logging.info("Done plotting.")

if __name__ == '__main__':
    plotter = TrajectronPlotter()
    plotter.run()
