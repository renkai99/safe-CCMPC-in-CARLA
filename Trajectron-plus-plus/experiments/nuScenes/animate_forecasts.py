import os
import sys
import json
from glob import glob
from timeit import default_timer as timer
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

from tqdm import tqdm
import numpy as np
import torch
import dill
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.animation as animation

import utility as util
from helper import load_model, prediction_output_to_trajectories
from model.dataset import get_timesteps_data
from model.model_utils import ModeKeys

AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'gold', 'orange', 'red', 'deeppink']
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 5) % len(AGENT_COLORS) for i in range(17)], 0)
NCOLORS = len(AGENT_COLORS)

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
        path='models/20210725/models_25_Jul_2021_15_29_29_carla_v3-1_base_distmap_ph8',
        desc="Base +Map model with heading and occlusion fix, PH=8 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_13 = util.AttrDict(
        path='models/models_25_Jul_2021_15_37_17_carla_v3-1_base_distmap_K20_ph8',
        desc="Base +Map model with heading fix, PH=8, K=20 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_14 = util.AttrDict(
        path='models/models_25_Jul_2021_15_38_46_carla_v3-1_base_distmap_K15_ph8',
        desc="Base +Map model with heading fix, PH=8, K=15 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_15 = util.AttrDict(
        path='models/models_25_Jul_2021_15_38_20_carla_v3-1_base_distmap_K10_ph8',
        desc="Base +Map model with heading fix, PH=8, K=10 "
             "(trained on small carla v3-1 dataset)",
        ph=8)

model_16 = util.AttrDict(
        path='models/models_25_Jul_2021_15_39_52_carla_v3-1_base_distmap_K5_ph8',
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

def generate_vehicle_latents(
            eval_stg, scene, t, ph, num_samples=200,
            z_mode=False, gmm_mode=False, full_dist=False, all_z_sep=False):
    """Generate predicted trajectories and their corresponding
    latent variables.

    Parameters
    ==========
    eval_stg : torch.nn.Module or object
        The model to do predictions on
    scene : Scene
        The nuScenes scene
    t : int
        The timestep in the scene
    ph : int
        The prediction horizon
    
    Returns
    =======
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
    latent_probs : ndarray
        The latent PMF of shape (number of latent values,).
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
    latent_probs = model.latent.get_p_dist_probs() \
            .cpu().detach().numpy()
    latent_probs = np.squeeze(latent_probs)
    
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
        
    return z, zz, predictions, nodes, predictions_dict, latent_probs

class TrajectronForecastAnimator(object):

    def __init__(self):
        self.plotdir = 'plots/forecasts'
        self.num_samples_per_timestep = 100

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

    def get_predictions_across_scene(self, scene):
        results = {}
        for t in tqdm(range(1, scene.timesteps)):
            z, zz, zpredictions, nodes, predictions_dict, latent_probs = generate_vehicle_latents(
                    self.eval_stg, scene, t, self.ph, num_samples=self.num_samples_per_timestep)

            prediction_dict, histories_dict, futures_dict = \
                    prediction_output_to_trajectories(
                        predictions_dict, dt=scene.dt, max_h=10, ph=self.ph, map=None)

            result = util.AttrDict(z=z, zz=zz, zpredictions=zpredictions, nodes=nodes, latent_probs=latent_probs,
                    prediction_dict=prediction_dict, histories_dict=histories_dict, futures_dict=futures_dict)

            results[t] = result
        return results

    def plot_scene(self, scene):

        logging.info(f"Computing forecasts across scene.")
        results = self.get_predictions_across_scene(scene)

        logging.info(f"Plotting forecasts across scene.")
        fig, axes = plt.subplots(1, 2, figsize=(30,15))
        axes = axes.ravel()

        plot_artists = []
        plot_artists_across_scenes = {}
        def generate_plots_for_timestep(t):
            nonlocal plot_artists
            plot_artists_across_scenes[t] = []
            if t < 1 or t >= scene.timesteps:
                return
            
            result = results[t]
            z, zz, zpredictions, nodes, latent_probs, prediction_dict, histories_dict, futures_dict \
                    = result.z, result.zz, result.zpredictions, result.nodes, result.latent_probs, \
                        result.prediction_dict, result.histories_dict, result.futures_dict
            n_vehicles = len(nodes)

            # Prediction on map
            ax = axes[0]
            render_roads(ax, scene)
            for idx, node in enumerate(nodes):
                player_future = futures_dict[t][node]
                player_past = histories_dict[t][node]
                player_predict = prediction_dict[t][node]

                plot = ax.plot(player_future[:,0], player_future[:,1],
                            marker='s', color=AGENT_COLORS[idx % NCOLORS],
                            linewidth=1, markersize=8, markerfacecolor='none')
                plot_artists.extend(plot)
                plot_artists_across_scenes[t].extend(plot)
                plot = ax.plot(player_past[:,0], player_past[:,1],
                            marker='d', color=AGENT_COLORS[idx % NCOLORS],
                            linewidth=1, markersize=8, markerfacecolor='none')
                plot_artists.extend(plot)
                plot_artists_across_scenes[t].extend(plot)
                for row in player_predict[0]:
                    plot = ax.plot(row[:,0], row[:,1],
                            marker='o', color=AGENT_COLORS[idx % NCOLORS],
                            linewidth=1, alpha=0.1, markersize=4)
                    plot_artists.extend(plot)
                    plot_artists_across_scenes[t].extend(plot)

            ax.set_title(f"Plot of p(y|x) at timestep {t} of {n_vehicles} vehicles on bitmap")

            # Prediction by latent
            ax = axes[1]
            latent_colors = cm.nipy_spectral(np.linspace(0, 1, self.hyp['K']))
            # z_counts = np.sum(zz, axis=0) / z.shape[1]

            for idx, node in enumerate(nodes):
                player_past = histories_dict[t][node]
                player_past = np.vstack((player_past, player_past[-1][None] + 5))
                plot = ax.plot(player_past[:,0], player_past[:,1],
                        color=AGENT_COLORS[idx % NCOLORS],
                        linewidth=1)
                plot_artists.extend(plot)
                plot_artists_across_scenes[t].extend(plot)

                # latent variable z corresponding to vehicle.
                zn = z[idx]
                pn = zpredictions[idx]
                for jdx in range(zn.shape[0]):
                    color = latent_colors[zn[jdx]]
                    plot = ax.plot(pn[jdx, :, 0], pn[jdx, :, 1],
                            marker='o',
                            color=color,
                            linewidth=1, alpha=0.2, markersize=4)
                    plot_artists.extend(plot)
                    plot_artists_across_scenes[t].extend(plot)

            ax.set_title(f"Plot of p(y|x) at timestep {t} of {n_vehicles} vehicles coded by latent")

        generate_plots_for_timestep(0)
        scalarmappaple = cm.ScalarMappable(cmap=cm.nipy_spectral)
        scalarmappaple.set_array(range(0, self.hyp['K']))
        plt.colorbar(scalarmappaple, ax=axes[1], orientation='horizontal',
                    pad=0.01,
                    ticks=range(0, self.hyp['K']),
                    label="z = value")
        axes[1].set_facecolor("grey")
        axes[1].set_aspect('equal')
        for ax in axes:
            ax.set_xlim([0, scene.x_size])
            ax.set_ylim([0, scene.y_size])
        fig.tight_layout()

        def predictions_of_frame(t, *args):
            nonlocal plot_artists
            for a in plot_artists:
                    a.remove()
            plot_artists = []
            if t not in plot_artists_across_scenes:
                generate_plots_for_timestep(t)
            return plot_artists_across_scenes[t]

        # Construct the animation, using the update function as the animation director.
        anim = animation.FuncAnimation(fig, predictions_of_frame, frames=scene.timesteps,
                                    interval=500, blit=False, repeat=False)
        writer = animation.FFMpegWriter(fps=1, codec="h264", extra_args=["-preset", "veryslow","-crf","0"])
        scene_label = scene.name.replace('/','_')
        savepath = os.path.join(self.plotdir, f"forecasts_{scene_label}.mp4")
        logging.info("Saving video of forecasts.")
        with tqdm(total=scene.timesteps + 2) as pbar:
            anim.save(savepath, writer=writer,
                    progress_callback=lambda *args: pbar.update(1))
        logging.info("Done saving video of forecasts.")
        plt.close('all')
    
    def run(self):
        os.makedirs(self.plotdir, exist_ok=True)
        logging.info("Starting to plot forecasts as animation.")
        eval_scenes = self.eval_scenes[9:]
        n_scenes = len(eval_scenes)
        with torch.no_grad():
            for idx, scene in enumerate(eval_scenes):
                logging.info(f"Plotting scene {idx + 1} / {n_scenes} : {scene.name}.")
                self.plot_scene(scene)
        logging.info("Done.")

if __name__ == '__main__':
    animator = TrajectronForecastAnimator()
    animator.run()
