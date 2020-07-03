import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde
from Dataloader import Dataset_from_pkl

sys.path.append("../../")
import Utils
from Param import ROS_Param
param = ROS_Param()
DT = param.DT
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
args = parser.parse_args()


def load_model(model_path):
    print('Loading model from \n' + model_path)
    model = torch.load(model_path)
    print('Loaded!\n')
    return model

def calc_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade

def calc_fde(predicted, gt):
    final_error = np.linalg.norm(predicted - gt)
    return final_error

def calc_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]

    for timestep in range(num_timesteps):
        try:
            kde = gaussian_kde(predicted_trajs[timestep].T)
            pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
            kde_ll += pdf / (num_timesteps)
        except np.linalg.LinAlgError:
            kde_ll = np.nan

    return -kde_ll


def is_data_enough(data):
    for step in data:
        for coord in step:
            if coord == 0:
                return False
    return True
    

def compare_prediction_gt(prediction, gt):
    # gt         -> 1, num_ped, 20, 2
    # prediction -> 1, num_ped, 12, 2
    error_dict =  {'id':list(),'ade': list(), 'fde': list(), 'kde': list()}
    for num_ped in range(len(prediction)):
        # check data
        
        if not is_data_enough(gt[num_ped]):
            continue
        # id 
        error_dict['id'].append(num_ped)
        # ade
        ade = calc_ade(prediction[num_ped],gt[num_ped,8:])
        error_dict['ade'].append(ade)
        # fde
        fde = calc_fde(prediction[num_ped,-1],gt[num_ped,-1])
        error_dict['fde'].append(fde)
        # kde_nll
        kde = calc_kde_nll(prediction[num_ped],gt[num_ped,8:])
        error_dict['kde'].append(kde)
    # error_dict -> {'id':[num_ped],'ade': [num_ped], 'fde': [num_ped], 'kde': [num_ped]}
    return error_dict



if __name__ == "__main__":
    import sys
    from Param import ROS_Param
    param = ROS_Param()
    dataset = Dataset_from_pkl("/home/pazuzu/catkin_ws/src/Backprop-MPDM_torch/scripts/NN/datasets/processed/with_forces/", data_files=["eth_train.pkl"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        num_peds = batch.shape[1]
        print(batch.shape)
        # # create gt
        gt = batch[0,:,:,2:4]
        # # create prediction
        pred = torch.rand(num_peds,12,2)
        errors_dict = compare_prediction_gt(pred,gt)
        print(errors_dict)
        pass












# if __name__ == "__main__":
#     with open(args.data, 'rb') as f:
#         dataset = dill.load(f)[0]

#     model = load_model(args.model)
#     ph = 12       #prediction_horizon
#     max_hl = 8    #maximum_history_length
#     # dataloader =  TODO: insert dataloared here
#     with torch.no_grad():
#         ############### MOST LIKELY ###############
#         eval_ade_batch_errors = np.array([])
#         eval_fde_batch_errors = np.array([])
#         print("-- Evaluating GMM Grid Sampled (Most Likely)")
#         for i, data in enumerate(dataset):
#             print(f"---- Evaluating data {i + 1}/{len(dataset)}")
#             timesteps = np.arange(data)
#             predictions = model.predict(data, timesteps, ph,num_samples=1,
#                                            min_history_timesteps=7,
#                                            min_future_timesteps=12,
#                                            )
#             batch_error_arr = Utils.compute_batch_statistics(predictions,
#                                                                    DT,
#                                                                    max_hl=max_hl,
#                                                                    ph=ph,
#                                                                    prune_ph_to_future=True,
#                                                                    kde=False)

            # predictions = eval_stg.predict(scene,
            #                                timesteps,
            #                                ph,
            #                                num_samples=1,
            #                                min_history_timesteps=7,
            #                                min_future_timesteps=12,
            #                                z_mode=False,
            #                                gmm_mode=True,
            #                                full_dist=True)  # This will trigger grid sampling

            # batch_error_arr = evaluation.compute_batch_statistics(predictions,
            #                                                        DT,
            #                                                        max_hl=max_hl,
            #                                                        ph=ph,
            #                                                        node_type_enum=env.NodeType,
            #                                                        prune_ph_to_future=True,
            #                                                        kde=False)

        #     eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_arr['ade']))
        #     eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_arr['fde']))

        # print(np.mean(eval_fde_batch_errors))
        # pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_most_likely.csv'))
        # pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_most_likely.csv'))


        ############### MODE Z ###############
        # eval_ade_batch_errors = np.array([])
        # eval_fde_batch_errors = np.array([])
        # eval_kde_nll = np.array([])
        # print("-- Evaluating Mode Z")
        # for i, scene in enumerate(scenes):
        #     print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
        #     for t in tqdm(range(0, scene.timesteps, 10)):
        #         timesteps = np.arange(t, t + 10)
        #         predictions = eval_stg.predict(scene,
        #                                        timesteps,
        #                                        ph,
        #                                        num_samples=2000,
        #                                        min_history_timesteps=7,
        #                                        min_future_timesteps=12,
        #                                        z_mode=True,
        #                                        full_dist=False)

        #         if not predictions:
        #             continue

        #         batch_error_dict = evaluation.compute_batch_statistics(predictions,
        #                                                                scene.dt,
        #                                                                max_hl=max_hl,
        #                                                                ph=ph,
        #                                                                node_type_enum=env.NodeType,
        #                                                                map=None,
        #                                                                prune_ph_to_future=True)
        #         eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
        #         eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
        #         eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        # pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'z_mode'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_z_mode.csv'))
        # pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'z_mode'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_z_mode.csv'))
        # pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'z_mode'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_z_mode.csv'))


        # ############### BEST OF 20 ###############
        # eval_ade_batch_errors = np.array([])
        # eval_fde_batch_errors = np.array([])
        # eval_kde_nll = np.array([])
        # print("-- Evaluating best of 20")
        # for i, scene in enumerate(scenes):
        #     print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
        #     for t in tqdm(range(0, scene.timesteps, 10)):
        #         timesteps = np.arange(t, t + 10)
        #         predictions = eval_stg.predict(scene,
        #                                        timesteps,
        #                                        ph,
        #                                        num_samples=20,
        #                                        min_history_timesteps=7,
        #                                        min_future_timesteps=12,
        #                                        z_mode=False,
        #                                        gmm_mode=False,
        #                                        full_dist=False)

        #         if not predictions:
        #             continue

        #         batch_error_dict = evaluation.compute_batch_statistics(predictions,
        #                                                                scene.dt,
        #                                                                max_hl=max_hl,
        #                                                                ph=ph,
        #                                                                node_type_enum=env.NodeType,
        #                                                                map=None,
        #                                                                best_of=True,
        #                                                                prune_ph_to_future=True)
        #         eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
        #         eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
        #         eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        # pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_best_of.csv'))
        # pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_best_of.csv'))
        # pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_best_of.csv'))


        # ############### FULL ###############
        # eval_ade_batch_errors = np.array([])
        # eval_fde_batch_errors = np.array([])
        # eval_kde_nll = np.array([])
        # print("-- Evaluating Full")
        # for i, scene in enumerate(scenes):
        #     print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
        #     for t in tqdm(range(0, scene.timesteps, 10)):
        #         timesteps = np.arange(t, t + 10)
        #         predictions = eval_stg.predict(scene,
        #                                        timesteps,
        #                                        ph,
        #                                        num_samples=2000,
        #                                        min_history_timesteps=7,
        #                                        min_future_timesteps=12,
        #                                        z_mode=False,
        #                                        gmm_mode=False,
        #                                        full_dist=False)

        #         if not predictions:
        #             continue

        #         batch_error_dict = evaluation.compute_batch_statistics(predictions,
        #                                                                scene.dt,
        #                                                                max_hl=max_hl,
        #                                                                ph=ph,
        #                                                                node_type_enum=env.NodeType,
        #                                                                map=None,
        #                                                                prune_ph_to_future=True)

        #         eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
        #         eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
        #         eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        # pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_full.csv'))
        # pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_full.csv'))
        # pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_full.csv'))
