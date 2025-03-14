"""Evaluate multiple models in multiple experiments, or evaluate baseline on multiple datasets

TODO: use hydra or another model to manage the experiments
"""

import os
import sys
import json
import argparse
import logging
from glob import glob
import time
import string

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

import numpy as np
import pandas as pd
import h5py
import scipy
import scipy.interpolate
import scipy.stats
import torch
import dill
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.animation as animation
from tqdm import tqdm
from tabulate import tabulate

import utility as util
from helper import load_model, prediction_output_to_trajectories

pd.set_option('io.hdf.default_format','table')

############################
# Bag-of-N (BoN) FDE metrics
############################

def compute_min_FDE(predict, future):
    return np.min(np.linalg.norm(predict[...,-1,:] - future[-1], axis=-1))

def compute_min_ADE(predict, future):
    mean_ades = np.mean(np.linalg.norm(predict - future, axis=-1), axis=-1)
    return np.min(mean_ades)

def evaluate_scene_BoN(scene, ph, eval_stg, hyp, n_predictions=20, min_fde=True, min_ade=True):
    predictconfig = util.AttrDict(ph=ph, num_samples=n_predictions, z_mode=False, gmm_mode=False,
            full_dist=False, all_z_sep=False)
    max_hl = hyp['maximum_history_length']
    with torch.no_grad():
        predictions = eval_stg.predict(scene,
                np.arange(scene.timesteps), predictconfig.ph,
                num_samples=predictconfig.num_samples,
                min_future_timesteps=predictconfig.ph,
                z_mode=predictconfig.z_mode,
                gmm_mode=predictconfig.gmm_mode,
                full_dist=predictconfig.full_dist,
                all_z_sep=predictconfig.all_z_sep)

    prediction_dict, histories_dict, futures_dict = \
        prediction_output_to_trajectories(
            predictions, dt=scene.dt, max_h=max_hl, ph=predictconfig.ph, map=None)

    batch_metrics = {'min_ade': list(), 'min_fde': list()}
    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            if min_ade:
                batch_metrics['min_ade'].append(compute_min_ADE(prediction_dict[t][node], futures_dict[t][node]))
            if min_fde:
                batch_metrics['min_fde'].append(compute_min_FDE(prediction_dict[t][node], futures_dict[t][node]))
    return batch_metrics

def evaluate_BoN(env, ph, eval_stg, hyp, n_predictions=20, min_fde=True, min_ade=True):
    batch_metrics = {'min_ade': list(), 'min_fde': list()}
    prefix = f"Evaluate Bo{n_predictions} (ph = {ph}): "
    for scene in tqdm(env.scenes, desc=prefix, dynamic_ncols=True, leave=True):
        _batch_metrics = evaluate_scene_BoN(scene, ph, eval_stg, hyp,
                n_predictions=n_predictions, min_fde=min_fde, min_ade=min_ade)
        batch_metrics['min_ade'].extend(_batch_metrics['min_ade'])
        batch_metrics['min_fde'].extend(_batch_metrics['min_fde'])
    return batch_metrics

###############
# Other metrics
###############

def make_interpolate_map(scene):
    map =  scene.map['VEHICLE']
    obs_map = 1 - np.max(map.data[..., :, :, :], axis=-3) / 255
    interp_obs_map = scipy.interpolate.RectBivariateSpline(
            range(obs_map.shape[0]),
            range(obs_map.shape[1]),
            obs_map, kx=1, ky=1)
    return interp_obs_map

def compute_num_offroad_viols(interp_map, scene_map, predicted_trajs):
    """Count the number of predicted trajectories that go off the road.
    Note this does not count trajectories that go over road/lane dividers.
    
    Parameters
    ==========
    interp_map : scipy.interpolate.RectBivariateSpline
        Interpolation to get road obstacle indicator value from predicted points.
    scene_map : trajectron.environment.GeometricMap
        Map transform the predicted points to map coordinates.
    predicted_trajs : ndarray
        Predicted trajectories of shape (number of predictions, number of timesteps, 2).
    
    Returns
    =======
    int
        A value between [0, number of predictions].
    """
    old_shape = predicted_trajs.shape
    pred_trajs_map = scene_map.to_map_points(predicted_trajs.reshape((-1, 2)))
    traj_values = interp_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    # traj_values has shape (1, num_samples, ph).
    traj_values = traj_values.reshape((old_shape[0], old_shape[1], old_shape[2]))
    # num_viol_trajs is an integer in [0, num_samples].
    return np.sum(traj_values.max(axis=2) > 0, dtype=float)

def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]
    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = scipy.stats.gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = kde.logpdf(gt_traj[timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll

def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()

def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()

########################
# Most Likely Evaluation
########################

def evaluate_scene_most_likely(scene, ph, eval_stg, hyp,
            ade=True, fde=True):
    predictconfig = util.AttrDict(ph=ph, num_samples=1,
            z_mode=True, gmm_mode=True, full_dist=False, all_z_sep=False)
    max_hl = hyp['maximum_history_length']
    with torch.no_grad():
        predictions = eval_stg.predict(scene,
                np.arange(scene.timesteps), predictconfig.ph,
                num_samples=predictconfig.num_samples,
                min_future_timesteps=predictconfig.ph,
                z_mode=predictconfig.z_mode,
                gmm_mode=predictconfig.gmm_mode,
                full_dist=predictconfig.full_dist,
                all_z_sep=predictconfig.all_z_sep)

    prediction_dict, histories_dict, futures_dict = \
            prediction_output_to_trajectories(
                predictions, dt=scene.dt, max_h=max_hl, ph=predictconfig.ph, map=None)

    batch_metrics = {'ade': list(), 'fde': list()}
    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            if ade:
                batch_metrics['ade'].extend(
                        compute_ade(prediction_dict[t][node], futures_dict[t][node]) )
            if fde:
                batch_metrics['fde'].extend(
                        compute_fde(prediction_dict[t][node], futures_dict[t][node]) )
    return batch_metrics

def evaluate_most_likely(env, ph, eval_stg, hyp,
        ade=True, fde=True):
    batch_metrics = {'ade': list(), 'fde': list()}
    prefix = f"Evaluate Most Likely (ph = {ph}): "
    for scene in tqdm(env.scenes, desc=prefix, dynamic_ncols=True, leave=True):
        _batch_metrics = evaluate_scene_most_likely(scene, ph, eval_stg, hyp,
                ade=ade, fde=fde)
        batch_metrics['ade'].extend(_batch_metrics['ade'])
        batch_metrics['fde'].extend(_batch_metrics['fde'])
    return batch_metrics

#################
# Full Evaluation
#################

def evaluate_scene_full(scene, ph, eval_stg, hyp,
            ade=True, fde=True, kde=True, offroad_viols=True):
    num_samples = 2000
    predictconfig = util.AttrDict(ph=ph, num_samples=num_samples,
            z_mode=False, gmm_mode=False, full_dist=False, all_z_sep=False)
    max_hl = hyp['maximum_history_length']
    with torch.no_grad():
        predictions = eval_stg.predict(scene,
                np.arange(scene.timesteps), predictconfig.ph,
                num_samples=predictconfig.num_samples,
                min_future_timesteps=predictconfig.ph,
                z_mode=predictconfig.z_mode,
                gmm_mode=predictconfig.gmm_mode,
                full_dist=predictconfig.full_dist,
                all_z_sep=predictconfig.all_z_sep)

    prediction_dict, histories_dict, futures_dict = \
            prediction_output_to_trajectories(
                predictions, dt=scene.dt, max_h=max_hl, ph=predictconfig.ph, map=None)

    interp_map = make_interpolate_map(scene)
    map =  scene.map['VEHICLE']
    batch_metrics = {'ade': list(), 'fde': list(), 'kde': list(), 'offroad_viols': list()}
    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            if ade:
                batch_metrics['ade'].extend(
                        compute_ade(prediction_dict[t][node], futures_dict[t][node]) )
            if fde:
                batch_metrics['fde'].extend(
                        compute_fde(prediction_dict[t][node], futures_dict[t][node]) )
            if offroad_viols:
                batch_metrics['offroad_viols'].extend(
                        [ compute_num_offroad_viols(interp_map, map, prediction_dict[t][node]) / float(num_samples) ])
            if kde:
                batch_metrics['kde'].extend(
                        [ compute_kde_nll(prediction_dict[t][node], futures_dict[t][node]) ])
    return batch_metrics

def evaluate_full(env, ph, eval_stg, hyp,
        ade=True, fde=True, kde=True, offroad_viols=True):
    batch_metrics = {'ade': list(), 'fde': list(), 'kde': list(), 'offroad_viols': list()}
    prefix = f"Evaluate Full (ph = {ph}): "
    for scene in tqdm(env.scenes, desc=prefix, dynamic_ncols=True, leave=True):
        _batch_metrics = evaluate_scene_full(scene, ph, eval_stg, hyp,
                ade=ade, fde=fde, kde=kde, offroad_viols=offroad_viols)
        batch_metrics['ade'].extend(_batch_metrics['ade'])
        batch_metrics['fde'].extend(_batch_metrics['fde'])
        batch_metrics['kde'].extend(_batch_metrics['kde'])
        batch_metrics['offroad_viols'].extend(_batch_metrics['offroad_viols'])
    return batch_metrics

##########
# Datasets
##########

dataset_dir = "../../.."

dataset_1 = util.AttrDict(
        test_set_path=f"{ dataset_dir }/carla_v3-1_dataset/v3-1_split1_test.pkl",
        name='v3-1_split1_test',
        desc="CARLA synthesized dataset with heading fix, occlusion fix, and 32 timesteps.")

dataset_2 = util.AttrDict(
        test_set_path=f"{ dataset_dir }/carla_v3-1-1_dataset/v3-1-1_split1_test.pkl",
        name='v3-1-1_split1_test',
        desc="CARLA synthesized dataset with heading fix, occlusion fix, and 32 timesteps.")

dataset_3 = util.AttrDict(
        test_set_path=f"{ dataset_dir }/carla_v3-1-2_dataset/v3-1-2_split1_test.pkl",
        name='v3-1-2_split1_test',
        desc="CARLA synthesized dataset with heading fix, occlusion fix, and 32 timesteps.")

DATASETS = [dataset_1, dataset_2, dataset_3]

def load_dataset(dataset):
    logging.info(f"Loading dataset: {dataset.name}: {dataset.desc}")
    with open(dataset.test_set_path, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')
    return eval_env

#############
# Experiments
#############

"""
The experiments to evaluate are:

- 20210621 one model trained on NuScenes to use as baseline for other evaluation
- 20210801 have models trained from v3-1-1 (train set has 200 scenes). Compare MapV2, MapV3.
- 20210802 have models trained from v3-1-1. MapV5 squeezes map encoding to size 32 using FC.
- 20210803 have models trained from v3-1-1. Compare map, mapV4. MapV4 with multi K values. MapV4 does not apply FC. May have size 100 or 150.
- 20210804 have models trained from v3-1 (train set has 300 scenes). Compare map with mapV4.
- 20210805 have models trained from v3-1 (train set has 300 scenes). MapV4 with multi K values.
- 20210812 have models trained from v3-1-1 rebalanced. Models are trained 20 epochs.
- 20210815 have models trained from v3-1-1 rebalanced. Models are trained 40 epochs.
- 20210816 have models trained from v3-1-2 (train set has 600 scenes) rebalanced.
"""

model_dir = "models"

baseline_model = util.AttrDict(
        path=f"{ model_dir }/20210621/models_19_Mar_2021_22_14_19_int_ee_me_ph8",
        desc="Base model +Dynamics Integration, Maps with K=25 latent values "
             "(on NuScenes dataset)")

experiment_1 = util.AttrDict(
        models_dir=f"{ model_dir }/20210801",
        dataset=dataset_2,
        desc="20210801 have models trained from v3-1-1 (train set has 200 scenes). Compare MapV2, MapV3.")

experiment_2 = util.AttrDict(
        models_dir=f"{ model_dir }/20210802",
        dataset=dataset_2,
        desc="20210802 have models trained from v3-1-1. MapV5 squeezes map encoding to size 32 using FC.")

experiment_3 = util.AttrDict(
        models_dir=f"{ model_dir }/20210803",
        dataset=dataset_2,
        desc="20210803 have models trained from v3-1-1. Compare map, mapV4. MapV4 with multi K values. "
             "MapV4 does not apply FC. May have size 100 or 150.")

experiment_4 = util.AttrDict(
        models_dir=f"{ model_dir }/20210804",
        dataset=dataset_1,
        desc="20210804 have models trained from v3-1 (train set has 300 scenes). Compare map with mapV4.")

experiment_5 = util.AttrDict(
        models_dir=f"{ model_dir }/20210805",
        dataset=dataset_1,
        desc="20210805 have models trained from v3-1 (train set has 300 scenes). MapV4 with multi K values.")

experiment_6 = util.AttrDict(
        models_dir=f"{ model_dir }/20210812",
        dataset=dataset_2,
        desc="20210812 have models trained from v3-1-1 rebalanced. Models are trained 20 epochs.")

experiment_7 = util.AttrDict(
        models_dir=f"{ model_dir }/20210815",
        dataset=dataset_2, ts=40,
        desc="20210815 have models trained from v3-1-1 rebalanced. Models are trained 40 epochs.")

experiment_8 = util.AttrDict(
        models_dir=f"{ model_dir }/20210816",
        dataset=dataset_3,
        desc="20210816 have models trained from v3-1-2 (train set has 600 scenes) rebalanced.")

EXPERIMENTS = [experiment_1, experiment_2, experiment_3, experiment_4, experiment_5, experiment_6, experiment_7, experiment_8]

def _load_model(model_path, eval_env, ts=20):
    eval_stg, hyp = load_model(model_path, eval_env, ts=ts)#, device='cuda')
    return eval_stg, hyp

PREDICTION_HORIZONS = [2,4,6,8]

def run_evaluate_experiments(config):
    if config.experiment_index is not None and config.experiment_index >= 1:
        experiments = [EXPERIMENTS[config.experiment_index - 1]]
    else:
        experiments = EXPERIMENTS
    
    ######################
    # Evaluate experiments
    ######################

    # results_filename = f"results_{time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime())}.h5"
    logging.info("Evaluating each experiment")
    for experiment in experiments:
        results_key = experiment.models_dir.split('/')[-1]
        results_filename = f"results_{results_key}.h5"
        logging.info(f"Evaluating models in experiment: {experiment.desc}")
        logging.info(f"Writing to: {results_filename}")
        eval_env = load_dataset(experiment.dataset)

        # need hyper parameters to do this, but have to load models first
        has_computed_scene_graph = False
        
        for model_path in glob(f"{experiment.models_dir}/*"):
            model_key = '/'.join(model_path.split('/')[-2:])
            
            ts = getattr(experiment, 'ts', 20)
            eval_stg, hyp = _load_model(model_path, eval_env, ts=ts)

            if not has_computed_scene_graph:
                prefix = f"Preparing Node Graph: "
                for scene in tqdm(eval_env.scenes, desc=prefix, dynamic_ncols=True, leave=True):
                    scene.calculate_scene_graph(eval_env.attention_radius,
                            hyp['edge_addition_filter'], hyp['edge_removal_filter'])
                has_computed_scene_graph = True

            logging.info(f"Evaluating: {model_key}")

            BoN_results_key = '/'.join([experiment.dataset.name] + model_path.split('/')[-2:] + ['BoN'])
            with pd.HDFStore(results_filename, 'a') as s:
                for ph in PREDICTION_HORIZONS:
                    batch_metrics = evaluate_BoN(eval_env, ph, eval_stg, hyp)
                    df = pd.DataFrame(batch_metrics)
                    df['ph'] = ph
                    s.put(BoN_results_key, df, format='t', append=True, data_columns=True)
            
            ML_results_key = '/'.join([experiment.dataset.name] + model_path.split('/')[-2:] + ['ML'])
            with pd.HDFStore(results_filename, 'a') as s:
                for ph in PREDICTION_HORIZONS:
                    batch_metrics = evaluate_most_likely(eval_env, ph, eval_stg, hyp)
                    df = pd.DataFrame(batch_metrics)
                    df['ph'] = ph
                    s.put(ML_results_key, df, format='t', append=True, data_columns=True)
            
            full_results_key = '/'.join([experiment.dataset.name] + model_path.split('/')[-2:] + ['Full'])
            other_results_key = '/'.join([experiment.dataset.name] + model_path.split('/')[-2:] + ['Other'])
            for ph in PREDICTION_HORIZONS:
                batch_metrics = evaluate_full(eval_env, ph, eval_stg, hyp)
                with pd.HDFStore(results_filename, 'a') as s:
                    df = pd.DataFrame({'fde': batch_metrics['fde'], 'ade': batch_metrics['ade'], 'ph': ph })
                    s.put(full_results_key, df, format='t', append=True, data_columns=True)
                    df = pd.DataFrame({'kde': batch_metrics['kde'], 'offroad_viols': batch_metrics['offroad_viols'], 'ph': ph })
                    s.put(other_results_key, df, format='t', append=True, data_columns=True)
            
            del eval_stg
            del hyp
        del eval_env
    
    logging.info("done evaluating experiments")

def run_evaluate_baselines(config):

    ####################
    # Evaluate baselines
    ####################

    results_filename = "results_baseline.h5"
    logging.info("Evaluating baselines")
    for dataset in DATASETS:
        logging.info(f"Evaluating baseline on dataset: {dataset.name}: {dataset.desc}")
        eval_env = load_dataset(dataset)
        logging.info(f"Using baseline: {baseline_model.desc}")
        eval_stg, hyp = _load_model(baseline_model.path, eval_env)

        prefix = f"Preparing Node Graph: "
        for scene in tqdm(eval_env.scenes, desc=prefix, dynamic_ncols=True, leave=True):
            scene.calculate_scene_graph(eval_env.attention_radius,
                    hyp['edge_addition_filter'], hyp['edge_removal_filter'])

        BoN_results_key = '/'.join([dataset.name] + baseline_model.path.split('/')[-2:] + ['BoN'])
        with pd.HDFStore(results_filename, 'a') as s:
            for ph in PREDICTION_HORIZONS:
                batch_metrics = evaluate_BoN(eval_env, ph, eval_stg, hyp)
                df = pd.DataFrame(batch_metrics)
                df['ph'] = ph
                s.put(BoN_results_key, df, format='t', append=True, data_columns=True)
        
        ML_results_key = '/'.join([dataset.name] + baseline_model.path.split('/')[-2:] + ['ML'])
        with pd.HDFStore(results_filename, 'a') as s:
            for ph in PREDICTION_HORIZONS:
                batch_metrics = evaluate_most_likely(eval_env, ph, eval_stg, hyp)
                df = pd.DataFrame(batch_metrics)
                df['ph'] = ph
                s.put(ML_results_key, df, format='t', append=True, data_columns=True)
        
        full_results_key = '/'.join([dataset.name] + baseline_model.path.split('/')[-2:] + ['Full'])
        other_results_key = '/'.join([dataset.name] + baseline_model.path.split('/')[-2:] + ['Other'])
        for ph in PREDICTION_HORIZONS:
            batch_metrics = evaluate_full(eval_env, ph, eval_stg, hyp)
            with pd.HDFStore(results_filename, 'a') as s:
                df = pd.DataFrame({'fde': batch_metrics['fde'], 'ade': batch_metrics['ade'], 'ph': ph })
                s.put(full_results_key, df, format='t', append=True, data_columns=True)
                df = pd.DataFrame({'kde': batch_metrics['kde'], 'offroad_viols': batch_metrics['offroad_viols'], 'ph': ph })
                s.put(other_results_key, df, format='t', append=True, data_columns=True)

        del hyp
        del eval_stg
        del eval_env
    
    logging.info("Done evaluating baselines")

RESULTS_FILENAMES = [ 'results_20210801.h5', 'results_20210802.h5', 'results_20210803.h5',
        'results_20210804.h5', 'results_20210805.h5', 'results_20210812.h5',
        'results_20210815.h5', 'results_20210816.h5', 'results_baseline.h5',]
ALPHABET = tuple(k for k in string.ascii_uppercase)

def run_summarize_experiments(config):
    scores = {}

    logging.info("Run summarize experiments.")
    for h5_filename in RESULTS_FILENAMES:
        logging.info(f"Reading scores from H5 file {h5_filename}.")
        with pd.HDFStore(h5_filename, 'r') as store:
            for key in store.keys():
                logging.info(f"Getting scores for {key}.")
                key_fragments = key.split('/')[1:]
                # for example
                # dataset_name v3-1-1_split1_test
                # experiment_name 20210816
                # model_name models_17_Aug_2021_13_25_38_carla_v3-1-2_base_distmapV4_modfm_K15_ph8
                # score_type BoN
                dataset_name, experiment_name, model_name, score_type = key_fragments
                _experiment_name = f"{experiment_name}/{dataset_name}"
                if _experiment_name not in scores:
                    scores[_experiment_name] = {}
                if model_name not in scores[_experiment_name]:
                    scores[_experiment_name][model_name] = {}
                fig1, axes1 = plt.subplots(2, 2, figsize=(15,15))
                fig2, axes2 = plt.subplots(2, 2, figsize=(15,15))
                for fig in [fig1, fig2]:
                    if score_type == 'BoN':
                        fig.suptitle("Bag-of-20 Metrics", fontsize=14)
                    elif score_type == 'ML':
                        fig.suptitle("Most Likely (ML) Metrics", fontsize=14)
                    elif score_type == 'Full' or score_type == 'Other':
                        fig.suptitle("Metrics aggregated over 2000 samples", fontsize=14)
                axes1 = axes1.ravel()
                axes2 = axes2.ravel()
                for idx, ph in enumerate(PREDICTION_HORIZONS):
                    logging.info(f"    scores for PH = {ph}.")
                    if ph not in scores[_experiment_name][model_name]:
                        scores[_experiment_name][model_name][ph] = {}
                    df = store[key][store[key]['ph'] == ph]
                    if score_type == 'BoN':
                        scores[_experiment_name][model_name][ph]['min_ade_mean'] = df['min_ade'].mean()
                        scores[_experiment_name][model_name][ph]['min_fde_mean'] = df['min_fde'].mean()
                        scores[_experiment_name][model_name][ph]['min_ade_var'] = df['min_ade'].var()
                        scores[_experiment_name][model_name][ph]['min_fde_var'] = df['min_fde'].var()
                        axes1[idx].hist(df['min_ade'], bins=100)
                        axes1[idx].set_title(f"Min ADE @{ph / 2.}s")
                        axes2[idx].hist(df['min_fde'], bins=100)
                        axes2[idx].set_title(f"Min FDE @{ph / 2.}s")
                    elif score_type == 'ML':
                        scores[_experiment_name][model_name][ph]['ade_ml_mean'] = df['ade'].mean()
                        scores[_experiment_name][model_name][ph]['fde_ml_mean'] = df['fde'].mean()
                        scores[_experiment_name][model_name][ph]['ade_ml_var'] = df['ade'].var()
                        scores[_experiment_name][model_name][ph]['fde_ml_var'] = df['fde'].var()
                        axes1[idx].hist(df['ade'], bins=100)
                        axes1[idx].set_title(f"ADE ML @{ph / 2.}s")
                        axes2[idx].hist(df['fde'], bins=100)
                        axes2[idx].set_title(f"FDE ML @{ph / 2.}s")
                    elif score_type == 'Full':
                        scores[_experiment_name][model_name][ph]['ade_mean'] = df['ade'].mean()
                        scores[_experiment_name][model_name][ph]['fde_mean'] = df['fde'].mean()
                        scores[_experiment_name][model_name][ph]['ade_var'] = df['ade'].var()
                        scores[_experiment_name][model_name][ph]['fde_var'] = df['fde'].var()
                        axes1[idx].hist(df['ade'], bins=100)
                        axes1[idx].set_title(f"ADE @{ph / 2.}s")
                        axes2[idx].hist(df['fde'], bins=100)
                        axes2[idx].set_title(f"FDE @{ph / 2.}s")
                    elif score_type == 'Other':
                        scores[_experiment_name][model_name][ph]['kde_mean'] = df['kde'].mean()
                        scores[_experiment_name][model_name][ph]['offroad_viols_mean'] = df['offroad_viols'].mean()
                        scores[_experiment_name][model_name][ph]['kde_var'] = df['kde'].mean()
                        scores[_experiment_name][model_name][ph]['offroad_viols_var'] = df['offroad_viols'].mean()
                        axes1[idx].hist(df['kde'], bins=100)
                        axes1[idx].set_title(f"KDE NLL @{ph / 2.}s")
                        axes2[idx].hist(df['offroad_viols'], bins=100)
                        axes2[idx].set_title(f"Off-road violations @{ph / 2.}s")
                    else:
                        raise Exception(f"Score type {score_type} is not recognized.")
                savekey = f"{experiment_name}_{dataset_name}"
                logging.info("    saving plots.")
                if score_type == 'BoN':
                    fig1.savefig(f"plots/{savekey}_min_ade.png", dpi=180)
                    fig2.savefig(f"plots/{savekey}_min_fde.png", dpi=180)
                elif score_type == 'ML':
                    fig1.savefig(f"plots/{savekey}_ade_ml.png", dpi=180)
                    fig2.savefig(f"plots/{savekey}_fde_ml.png", dpi=180)
                elif score_type == 'Full':
                    fig1.savefig(f"plots/{savekey}_ade.png", dpi=180)
                    fig2.savefig(f"plots/{savekey}_fde.png", dpi=180)
                elif score_type == 'Other':
                    fig1.savefig(f"plots/{savekey}_kde.png", dpi=180)
                    fig2.savefig(f"plots/{savekey}_offroad_viols.png", dpi=180)
                plt.close('all')

    logging.info("Generating tables.")
    with open('plots/tables.txt', 'w') as f:
        for _experiment_name, v in scores.items():
            experiment_name, dataset_name = _experiment_name.split('/')
            print(f"Writing (experiment {experiment_name} using dataset {dataset_name})")
            print(f"Experiment {experiment_name} using dataset {dataset_name}", file=f)
            
            # Get model legend table
            headers = [f"Model Legend\nName (for experiment {experiment_name})", "\nIndex"]
            model_df = pd.DataFrame(columns=headers)
            for idx, (model_name, vv) in enumerate(v.items()):
                model_data = pd.Series({ headers[0]: model_name, headers[1]: ALPHABET[idx] })
                model_df = model_df.append(model_data, ignore_index=True)
            print(file=f)
            print(tabulate(model_df, headers='keys', showindex=False, tablefmt="simple"), file=f)
            
            score_tup = [
                    ("Min ADE", "min_ade"),
                    ("Min FDE", "min_fde"),
                    ("ADE ML",  "ade_ml"),
                    ("FDE ML",  "fde_ml"),
                    ("ADE",     "ade"),
                    ("FDE",     "fde"),
                    ("KDE NLL", "kde"),
                    ("Offroad Viols", "offroad_viols"),]
            for score_name, score_key in score_tup:
                headers  = ["\nModel", f"{score_name}\n@1s", "\n@2s", "\n@3s", "\n@4s"]
                mean_key = f"{score_key}_mean"
                var_key  = f"{score_key}_var"
            
                # Get score table
                score_df = pd.DataFrame(columns=headers)
                for idx, (model_name, vv) in enumerate(v.items()):
                    score_data = { headers[0]: ALPHABET[idx] }
                    for ph, vvv in vv.items():
                        kdx = ph // 2
                        mean_val = np.around(vvv[mean_key], decimals=2 )
                        dev_val  = np.around( np.sqrt(vvv[var_key]), decimals=2 )
                        score_data[headers[kdx]] = f"{mean_val} +/- {dev_val}"
                    model_df = model_df.append(model_data, ignore_index=True)
                    score_df = score_df.append(score_data, ignore_index=True)
                print(file=f)
                print(tabulate(score_df, headers='keys', showindex=False, tablefmt="simple"), file=f)
            
            # newline
            print(file=f)

def parse_arguments():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    subparsers = argparser.add_subparsers(dest="task")
    compute_parser = subparsers.add_parser('compute', help="Compute evaluation scores.")
    compute_parser.add_argument(
                '--experiment-index',
                default=0,
                type=int,
                help="Experiment index number to run")
    summarize_parser = subparsers.add_parser('summarize', help="Summarize evaluation scores and generate plots.")
    return argparser.parse_args()

if __name__ == "__main__":
    config = parse_arguments()
    if config.task == 'compute':
        if config.experiment_index == 0:
            run_evaluate_baselines(config)
        else:
            run_evaluate_experiments(config)
    elif config.task == 'summarize':
        run_summarize_experiments(config)
    else:
        logging.warning("No task selected.")