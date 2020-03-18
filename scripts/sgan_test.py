import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs

import matplotlib as mpl
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


if __name__ == '__main__':
    args = parser.parse_args()
    model = "./scripts/models/sgan-models/eth_8_model.pt"
    test_data_path = "./scripts/datasets/test/test"
    # model = "./models/sgan-models/eth_8_model.pt"
    # test_data_path = "./datasets/test"
    checkpoint = torch.load(model)
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])
    _, loader = data_loader(_args, test_data_path)
    print("loading is ok")
    fig = plt.figure()
    ax = plt.subplot(111)
    num_samples = 1

    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                print(pred_traj_fake_rel[0])
                # traj_fake = pred_traj_fake_rel.detach().cpu()
                # ax.plot(traj_fake[:, 0], traj_fake[:, 1],
                #         '*', label="pred_traj_fake")
                # traj_gt = pred_traj_gt.detach().cpu()
                # ax.plot(traj_gt[:, 0], traj_gt[:, 1],
                #         "-", label="pred_traj_gt")
                # ax.legend()
                # plt.show()
