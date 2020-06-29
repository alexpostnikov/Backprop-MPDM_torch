import sys
import os
import numpy as np
import pandas as pd
import dill
import torch


def devide_by_steps(data):
    # find first/last frame
    min_frame = min([x['frame']["id"][0] for x in data])
    max_frame = max([max(x['frame']["id"]) for x in data])
    # 
    new_data = []
    for n in range(min_frame, max_frame+1):
        frame = []
        for ped in data:
            if n in ped.values[:,1]:
                frame.append(ped.values[ped.values[:,1]==n]) 
        print("frame "+ str(n)+" from " + str(max_frame))
        new_data.append(frame)
    return new_data

def postproccess(dataset):
    arr = []
    for f in dataset:
        arr.append(devide_by_steps(f))
        print("dataset proccessed")
    # tarr = torch.tensor(arr)
    return arr

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    if x[~np.isnan(x)].shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)

    return dx


dt = 0.4

maybe_makedirs('../processed')
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
data_columns = data_columns.insert(0,('frame','id'))
data_columns = data_columns.insert(0,('ped','id'))
for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:
        data_dict_path = os.path.join('./processed', '_'.join([desired_source, data_class]) + '.pkl')
        processed_data_class = []
        for subdir, dirs, files in os.walk(os.path.join('trajnet', desired_source, data_class)):
            for file in files:
                if not file.endswith('.txt'):
                    continue
                input_data_dict = dict()
                full_data_path = os.path.join(subdir, file)
                print('At', full_data_path)

                data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                data['frame_id'] = data['frame_id'] // 10

                data['frame_id'] -= data['frame_id'].min()

                data['node_id'] = data['track_id'].astype(str)
                data.sort_values('frame_id', inplace=True)

                # Mean Position
                data['pos_x'] = data['pos_x'] - data['pos_x'].mean()
                data['pos_y'] = data['pos_y'] - data['pos_y'].mean()

                max_timesteps = data['frame_id'].max()

                processed_data = []
                
                for node_id in pd.unique(data['node_id']):

                    node_df = data[data['node_id'] == node_id]
                    assert np.all(np.diff(node_df['frame_id']) == 1)

                    node_values = node_df[['pos_x', 'pos_y']].values

                    if node_values.shape[0] < 2:
                        continue

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, dt)
                    vy = derivative_of(y, dt)
                    ax = derivative_of(vx, dt)
                    ay = derivative_of(vy, dt)

                    data_dict = {   
                                    ('ped','id'): int(node_id),
                                    ('frame','id'): node_df["frame_id"].values,
                                    ('position', 'x'): x,
                                    ('position', 'y'): y,
                                    ('velocity', 'x'): vx,
                                    ('velocity', 'y'): vy,
                                    ('acceleration', 'x'): ax,
                                    ('acceleration', 'y'): ay}

                    ped_dataframe = pd.DataFrame(data_dict, columns=data_columns)

                    processed_data.append(ped_dataframe)

                processed_data_class.append(processed_data)
        print(f'Processed {len(processed_data_class):.2f} files for data class {data_class}')
        if len(processed_data_class) > 0:
            processed_data_class = postproccess(processed_data_class)
            with open(data_dict_path, 'wb') as f:
                dill.dump(processed_data_class, f, protocol=dill.HIGHEST_PROTOCOL)
