import numpy as np
import pandas as pd
import seaborn as sns


def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {'position': ['x', 'y']}

            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]

            future = node.get(np.array([t + 1, t + ph]), position_state)
            future = future[~np.isnan(future.sum(axis=1))]

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :, :future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output

            if map is None:
                histories_dict[t][node] = history
                output_dict[t][node] = trajectory
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = map.to_map_points(trajectory)
                futures_dict[t][node] = map.to_map_points(future)

    return output_dict, histories_dict, futures_dict


def plot_boxplots(ax, perf_dict_for_pd, x_label, y_label):
    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)
    our_mean_color = sns.color_palette("muted")[9]
    marker_size = 7
    mean_markers = 'X'
    with sns.color_palette("muted"):
        sns.boxplot(x=x_label, y=y_label, data=perf_df, ax=ax, showfliers=False)
        ax.plot([0], [np.mean(perf_df[y_label])], color=our_mean_color, marker=mean_markers,
                markeredgecolor='#545454', markersize=marker_size, zorder=10)


def plot_barplots(ax, perf_dict_for_pd, x_label, y_label):
    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)
    with sns.color_palette("muted"):
        sns.barplot(x=x_label, y=y_label, ax=ax, data=perf_df)