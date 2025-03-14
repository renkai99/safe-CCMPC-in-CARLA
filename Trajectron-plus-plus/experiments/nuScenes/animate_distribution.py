import sys
import os
import json
from glob import glob
import logging
from threading import local

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

import numpy as np
import torch
import dill
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.animation as animation

import utility as util
from helper import load_model, prediction_output_to_trajectories
import visualization
from model.dataset import *
from model.components import *
from model.model_utils import *
from model.components.discrete_latent import DiscreteLatent

AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'olive', 'gold', 'orange', 'red', 'deeppink']
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 3) % len(AGENT_COLORS) for i in range(17)], 0)

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
        path='models/models_25_Jul_2021_15_29_29_carla_v3-1_base_distmap_ph8',
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


def generate_vehicle_gmms(
        eval_stg, scene, timesteps, ph, num_samples=100,
        z_mode=False, gmm_mode=False, full_dist=False, all_z_sep=False):
    # Trajectron.predict() arguments
    min_future_timesteps = 0
    min_history_timesteps = 1

    node_type = eval_stg.env.NodeType.VEHICLE
    if node_type not in eval_stg.pred_state:
        raise Exception("fail")

    model = eval_stg.node_models_dict[node_type]

    # Get Input data for node type and given timesteps
    batch = get_timesteps_data(env=eval_stg.env, scene=scene, t=timesteps, node_type=node_type,
                               state=eval_stg.state,
                               pred_state=eval_stg.pred_state, edge_types=model.edge_types,
                               min_ht=min_history_timesteps, max_ht=eval_stg.max_ht,
                               min_ft=min_future_timesteps,
                               max_ft=min_future_timesteps, hyperparams=eval_stg.hyperparams)
    
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
    
    dist, _ = model.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)
    
    return dist


def cholesky_mv_gaussian(pos, mu, L, is_log_prob=False):
    """Compute the discretized PDF of a multivariate Gaussian given its Cholesky decomposition.
    Parameters
    ==========
    pos : np.array
        The coordinate points of a multidmensional cartesian grid with shape (n_x,n_y,2)
        used to retrieve the discretized PDF at those points.
        For example the pos array from a 2D cartesian grid is:
        ```
        X = np.linspace(extent[0,0], extent[0,1], N)
        Y = np.linspace(extent[1,0], extent[1,1], N)
        X, Y = np.meshgrid(X, Y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        ```
    mu : np.array
        Mean of multivariate Gaussian.
    L : np.array
        Cholesky matrix of covariance matrix Sigma of multivariate Gaussian, namely Sigma = L L^T.
    
    Returns
    =======
    np.array
        The values of the PDF at the coordinate points with shape (n_x,n_y).
    """
    n = mu.size
    L_inv = np.linalg.inv(L)
    _p = -0.5*n*np.log(2*np.pi) - np.sum(np.log(np.diag(L)))
    _w = np.einsum("jk, ...k ->...j", L_inv, pos - mu)
    _w = -0.5*np.linalg.norm(_w, axis=-1)**2
    _p = _p + _w
    if is_log_prob:
        return _p
    else:
        return np.exp(_p)


def cholesky_mv_gaussian_mixture(pos, log_pis, mus, Ls):
    _p = np.zeros(pos.shape[:2])
    for (log_pi, mu, L) in zip(log_pis, mus, Ls):
        _w = cholesky_mv_gaussian(pos, mu, L, is_log_prob=True)
        _p += np.exp(log_pi + _w)
    return _p


class DistributionMaker(object):
    TOL = 1e-4
    N = 300

    def __init__(self):
        self.plotdir = 'plots/contours'

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

    def get_gmm_params_at_timestep(self, scene, t):
        _timesteps = np.array([t])
        dist = generate_vehicle_gmms(
                self.eval_stg, scene, _timesteps, self.ph, num_samples=1,
                z_mode=False, gmm_mode=False,
                full_dist=True, all_z_sep=False)

        if not self.hyp['dynamic']['VEHICLE']['distribution']:
            node_type = self.eval_stg.env.NodeType.VEHICLE
            dynamic = self.eval_stg.node_models_dict[node_type].dynamic
            dist = dynamic.integrate_distribution(dist)

        # Log Mixing Proportions
        # not used with this setting
        log_pis = dist.log_pis.cpu().detach().numpy()
        # Parameters of Gaussians
        mus = dist.mus.cpu().detach().numpy()
        sigmas = dist.sigmas.cpu().detach().numpy()
        covs = dist.get_covariance_matrix().cpu().detach().numpy()
        Ls = dist.L.cpu().detach().numpy()

        # assume that we are now using a mixture of Gaussians
        # log_pis has shape (# vehicles, prediction_horizon, 2)
        log_pis = log_pis[0]
        # mus, sigmas has shape (# vehicles, prediction_horizon, 2)
        mus = mus[0]
        sigmas = sigmas[0]
        # covs, Ls has shape (# vehicles, prediction_horizon, 2, 2)
        covs = covs[0]
        Ls = Ls[0]
        return log_pis, mus, sigmas,covs, Ls
    
    def get_gmm_distribution_across_scene(self, pos, scene):
        # Generate GMMs for timesteps 1,2,..,scene.timesteps
        gmm_parameters = {}
        gmm_discretize = {t: [] for t in range(1, scene.timesteps)}
        def gen():
            for t in range(1, scene.timesteps):
                log_pis, mus, sigmas,covs, Ls = self.get_gmm_params_at_timestep(scene, t)
                gmm_parameters[t] = (log_pis, mus, sigmas,covs, Ls)
                n_vehicles = mus.shape[0]
                prediction_horizon = mus.shape[1]
                iter_vehicles = range(n_vehicles)
                iter_ph = range(prediction_horizon)
                for v_idx in iter_vehicles:
                    Zs = []
                    for p_idx in iter_ph:
                        vp_log_pis = log_pis[v_idx,p_idx]
                        vp_mus = mus[v_idx,p_idx]
                        vp_Ls = Ls[v_idx,p_idx]
                        Z = cholesky_mv_gaussian_mixture(pos, vp_log_pis, vp_mus, vp_Ls)
                        Zs.append(Z)
                    Z = np.max(Zs, axis=0)
                    # Z has shape (N, N) where N is discretization
                    Z = np.ma.masked_where(Z < self.TOL, Z)
                    gmm_discretize[t].append(Z)
                yield
        for _ in tqdm(gen(), total=scene.timesteps - 1):
            pass
        return gmm_parameters, gmm_discretize
    
    def plot_scene(self, scene):
        """Plot GMM distributions from scene.
        
        Based on:
        https://stackoverflow.com/questions/43074828/remove-precedent-scatterplot-while-updating-python-animation
        """
        logging.info(f"Computing GMM distribution across scene.")
        extent = np.array([[0, scene.x_size], [0, scene.y_size]])
        X = np.linspace(extent[0,0], extent[0,1], self.N)
        Y = np.linspace(extent[1,0], extent[1,1], self.N)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        gmm_parameters, gmm_discretize = self.get_gmm_distribution_across_scene(pos, scene)

        logging.info("Plotting GMM distribution across scene.")
        fig, ax = plt.subplots(figsize=(15, 15))
        render_roads(ax, scene, is_white=True, is_v2_bitmap=self.is_v2_bitmap)
        extent = (0, scene.x_size, 0, scene.y_size)

        contours = []
        contour_across_scenes = {}
        def generate_cfs_for_timestep(t):
            nonlocal contours
            ax.set_title(f"Contour plot of p(y|x) at timestep {t} with TOL={self.TOL}")
            contour_across_scenes[t] = []
            if t == 0:
                return
            for v_idx, Z in enumerate(gmm_discretize[t]):
                cmap = colors.LinearSegmentedColormap.from_list('', ['none', AGENT_COLORS[v_idx % NCOLORS]])
                cf = ax.contourf(X, Y, Z, cmap=cmap, extent=extent, extend='max', locator=ticker.LogLocator())
                contours.append(cf)
                contour_across_scenes[t].append(cf)
        generate_cfs_for_timestep(0)

        ax.set_aspect('equal')
        ax.set_facecolor("grey")

        def contourf_of_frame(t, *args):
            nonlocal contours
            for cf in contours:
                for coll in cf.collections:
                    coll.remove()
            contours = []
            if t not in contour_across_scenes:
                generate_cfs_for_timestep(t)
            return contour_across_scenes[t]

        # Construct the animation, using the update function as the animation director.
        anim = animation.FuncAnimation(fig, contourf_of_frame, frames=scene.timesteps, interval=500, blit=False, repeat=False)
        writer = animation.FFMpegWriter(fps=1, codec="h264", extra_args=["-preset", "veryslow","-crf","0"])
        scene_label = scene.name.replace('/','_')
        savepath = os.path.join(self.plotdir, f"contour_{scene_label}.mp4")
        logging.info("Saving video of GMM distribution.")
        anim.save(savepath, writer=writer)
        logging.info("Done saving video of GMM distribution.")
        plt.close('all')

    def run(self):
        os.makedirs(self.plotdir, exist_ok=True)
        logging.info("Starting to plot distributions.")
        n_scenes = len(self.eval_scenes)
        with torch.no_grad():
            for idx, scene in enumerate(self.eval_scenes):
                logging.info(f"Plotting scene {idx + 1} / {n_scenes} : {scene.name}.")
                self.plot_scene(scene)
        logging.info("Done.")


if __name__ == '__main__':
    dm = DistributionMaker()
    dm.run()
