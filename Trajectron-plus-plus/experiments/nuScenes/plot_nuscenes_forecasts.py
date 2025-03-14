"""Plot the predictions of Trajectron++ on NuScenes.
This only works for the NuScenes dataset."""

import sys
import os
import numpy as np
import torch
import dill
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from helper import *
import visualization

# Import to generate latents
from model.dataset import *
from model.components import *
from model.model_utils import *

# Colors
LATENT_CM = cm.nipy_spectral
LATENT_COLORS = LATENT_CM(np.linspace(0, 1, 25))
AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'olive', 'gold', 'orange', 'red', 'deeppink']
NCOLORS = len(AGENT_COLORS)
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 3) % NCOLORS for i in range(NCOLORS)], 0)

# Load nuScenes SDK
nuScenes_data_path = "/home/fireofearth/code/robotics/trajectron-plus-plus/experiments/nuScenes/v1.0"
# Data Path to nuScenes data set
nuScenes_devkit_path = './devkit/python-sdk/'
sys.path.append(nuScenes_devkit_path)
from nuscenes.map_expansion.map_api import NuScenesMap
nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name='boston-seaport')


class TrajectronPlotter(object):
    def __init__(self):
        # ph : int
        #    Prediction horizon
        self.ph = 6
        self.viewport_hw = 60
        self.ptype = ''
        self.num_samples = 500
        self.dpi = 180

        ## Load dataset
        with open('../processed/nuScenes_test_full.pkl', 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')
        self.eval_scenes = eval_env.scenes

        ## Load model
        log_dir = './models'
        model_dir = os.path.join(log_dir, 'int_ee_me')
        self.eval_stg, hyp = load_model(
            model_dir, eval_env, ts=12)

        ## To select specific scenes to plot
        # scenes_to_use = ['329']
        # eval_scenes = list(filter(lambda s : s.name in scenes_to_use, eval_scenes))

    def generate_vehicle_latents(self, scene, t,
                num_samples=200,
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
        node_type = self.eval_stg.env.NodeType.VEHICLE
        if node_type not in self.eval_stg.pred_state:
            raise Exception("fail")

        model = self.eval_stg.node_models_dict[node_type]

        # Get Input data for node type and given timesteps
        batch = get_timesteps_data(
                env=self.eval_stg.env, scene=scene,
                t=timesteps, node_type=node_type,
                state=self.eval_stg.state, pred_state=self.eval_stg.pred_state,
                edge_types=model.edge_types,
                min_ht=min_history_timesteps, max_ht=self.eval_stg.max_ht,
                min_ft=min_future_timesteps, max_ft=min_future_timesteps,
                hyperparams=self.eval_stg.hyperparams)
        
        # There are no nodes of type present for timestep
        if batch is None:
            raise Exception("fail")

        (first_history_index,
        x_t, y_t, x_st_t, y_st_t,
        neighbors_data_st,
        neighbors_edge_value,
        robot_traj_st_t,
        map), nodes, timesteps_o = batch

        x = x_t.to(self.eval_stg.device)
        x_st_t = x_st_t.to(self.eval_stg.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.eval_stg.device)
        if type(map) == torch.Tensor:
            map = map.to(self.eval_stg.device)

        # MultimodalGenerativeCVAE.predict() arguments
        inputs = x
        inputs_st = x_st_t
        first_history_indices = first_history_index
        neighbors = neighbors_data_st
        neighbors_edge_value = neighbors_edge_value
        robot = robot_traj_st_t
        prediction_horizon = self.ph

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

    def plot_scene_timestep_overhead(self,
            scene, t, nodes, predictions_dict, histories_dict, futures_dict):
        """Plot nuScense map with the forecasts on top of it for each timestep
        in each scene.

        Parameters
        ==========
        scene : Scene
            The nuScenes scene
        t : int
            The timestep in the scene
        predictions_dict : dict
            Contains map of predictions by timestep, by agent node
        histories_dict : dict
            Past trajectories by timestep, by agent node
        futures_dict : dict
            Ground truth trajectories by timestep, by agent node
        nodes : list of Node
            List of vehicle nodes
        """
        global nusc_map
        timesteps = np.array([t])

        # when using generate_vehicle_latents() then all nodes are already vehicle nodes
        # nodes = list(filter(lambda k: 'VEHICLE' in repr(k), predictions[t].keys()))
        # nodes.sort(key=lambda k: 0 if 'ego' in repr(k) else 1)
        ego_node = next(filter(lambda k: 'ego' in repr(k), nodes))

        minpos = np.array([scene.x_min, scene.y_min])
        ego_lastpos = histories_dict[t][ego_node][-1]
        ego_lastx = ego_lastpos[0]
        ego_lasty = ego_lastpos[1]
        center = np.array([
            scene.ego_initx + ego_lastx,
            scene.ego_inity + ego_lasty])

        center = minpos + ego_lastpos
        my_patch = (center[0] - self.viewport_hw, center[1] - self.viewport_hw,
                    center[0] + self.viewport_hw, center[1] + self.viewport_hw)
        if scene.map_name != nusc_map.map_name:
            nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name=scene.map_name)
        fig, ax = nusc_map.render_map_patch(my_patch, scene.layer_names,
                figsize=(10, 10), alpha=0.2, render_egoposes_range=False)
        
        for idx, node in enumerate(nodes):
            player_future = futures_dict[t][node]
            player_past = histories_dict[t][node]
            player_predict = predictions_dict[t][node]
            minpos_player_future  = player_future + minpos
            minpos_player_past    = player_past + minpos
            minpos_player_predict = player_predict + minpos

            ax.plot(minpos_player_future[:,0], minpos_player_future[:,1],
                        marker='s', color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1, markersize=8, markerfacecolor='none')
            ax.plot(minpos_player_past[:,0], minpos_player_past[:,1],
                        marker='d', color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1, markersize=8, markerfacecolor='none')
            for row in minpos_player_predict[0]:
                ax.plot(row[:,0], row[:,1],
                        marker='o', color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1, alpha=0.1, markersize=4)
        savepath = 'plots/predict_scene{}_t{}{}_overhead.png'.format(scene.name, t, self.ptype)
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
        savepath = 'plots/predict_scene{}_t{}{}_comb_latents.png'.format(scene.name, t, self.ptype)
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
        
        savepath = 'plots/predict_scene{}_t{}{}_each_latents.pdf'.format(scene.name, t, self.ptype)
        fig.savefig(savepath, dpi=self.dpi)
        plt.close('all')

    def run(self):
        with torch.no_grad():
            for scene in self.eval_scenes:
                print(f"Plotting scene {scene.name} ({scene.timesteps} timesteps)")
                for t in range(2, scene.timesteps - 6, 3):
                    print(f"    timestep {t}")
                    z,zz, predictions, nodes, predictions_dict = self.generate_vehicle_latents(
                            scene, t, num_samples=100)

                    # Obtain, past, predict and ground truth predictions
                    _, histories_dict, futures_dict = \
                            prediction_output_to_trajectories(
                                predictions_dict, dt=scene.dt, max_h=10, ph=self.ph, map=None)

                    self.plot_scene_timestep_overhead(scene, t, nodes, predictions_dict, histories_dict, futures_dict)

                    z,zz, predictions, nodes, predictions_dict = self.generate_vehicle_latents(
                            scene, t, num_samples=300)

                    self.plot_combined_latents(scene, t,
                            z, zz, predictions, nodes, predictions_dict, histories_dict, futures_dict)
                    self.plot_each_latents(scene, t,
                            z, zz, predictions, nodes, predictions_dict, histories_dict, futures_dict)

plotter = TrajectronPlotter()
plotter.run()
