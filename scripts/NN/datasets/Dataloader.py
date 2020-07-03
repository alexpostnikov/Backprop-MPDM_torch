from torch.utils.data import Dataset
import os
import torch
import dill
from typing import Union, List


class Dataset_from_pkl(Dataset):
    """
        Class for loading data in torch format from preprocessed pkl
        files with pedestrian poses, velocities and accelerations
    """

    def __init__(self, data_folder: str, data_files: Union[str, List[str]] = "all",
                 train: bool = True, test: bool = False, validate: bool = False):
        """
        :param data_folder: path to folder with preprocessed pkl files
        :param data_files: list of files to be used or "all"
        :param train: if data_files is all, if train is false -> all *train*.pkl files will be ignored
        :param test: if data_files is all, if train is false -> all *test*.pkl files will be ignored
        :param validate: if data_files is all, if train is false -> all *val*.pkl files will be ignored
        """

        super().__init__()
        self.train_dataset = torch.tensor([])
        file_list = []
        if "all" not in data_files:
            file_list = data_files
        else:
            dd = os.listdir(data_folder)
            for file in dd:
                if train and "train" in file:
                    file_list.append(file)
                if test and "test" in file:
                    file_list.append(file)
                if validate and "val" in file:
                    file_list.append(file)
        data_dict = {}
        for x in file_list:
            data_dict[x] = data_folder + "/" + x
        self.data = {}
        for file_name in data_dict.keys():
            with open(data_dict[file_name], 'rb') as f:
                print("loading " + file_name)
                self.data[file_name] = dill.load(f)

        self.dataset_indeces = {}
        self.data_length = 0

        for key in self.data.keys():
            for id, sub_dataset in enumerate(self.data[key]):
                self.data_length += len(sub_dataset) - 20
                self.dataset_indeces[self.data_length] = [key, id]

    def get_ped_data_in_time(self, start: int, end: int, dataset: List):
        """
        return stacked torch tensor of scene in specified timestamps from specified dataset.
        if at any given timestamp person is not found at dataset, but later (previously) will appear,
        that its tensor data is tensor of zeros.
        :param start: timestamp start
        :param end:  timestamp end
        :param dataset: list of data. shapesa are: 1: timestamp 1: num_peds, 2: RFU, 3: data np.array of 8 floats
        :return: torch tensor of shape : end-start, max_num_peds, , 20 , 8
        """
        indexes = self.get_peds_indexes_in_range_timestamps(start, end, dataset)

        max_num_of_peds = 0
        for key in indexes.keys():
            if len(indexes[key]) > max_num_of_peds:
                max_num_of_peds = len(indexes[key])
        prepared_data = torch.zeros(end - start, max_num_of_peds, 20, dataset[0].shape[1])
        for start_timestamp in range(start, end):
            for duration in range(0, 20):
                for ped in dataset[start_timestamp + duration]:
                    ped_id = indexes[start_timestamp].index(ped[0])
                    prepared_data[start_timestamp-start][ped_id][duration] = ped # torch.tensor(ped) if dataset not in tensor
        return prepared_data

    def limit_len(self, new_len):
        self.data_length = new_len


    def get_peds_indexes_in_range_timestamps(self, start: int, end: int, dataset: List):
        """
        :param start: timestamp start
        :param end:  timestamp end
        :param dataset: list of data. shapes are: 0: timestamp 1: num_peds, 2: RFU, 3: data np.array of 8 floats
        :return: dict of  predestrians ids at each scene (one scene is 20 timestamps)
        """
        indexes = {}
        for time_start in range (start, end):
            time_start_indexes = []
            for duration in range(0, 20):
                peoples = dataset[time_start + duration]
                time_start_indexes += list(self.get_peds_indexes_in_timestamp(peoples))
            indexes[time_start] = list(set(time_start_indexes))
        return indexes

    def get_peds_indexes_in_timestamp(self, timestamp_data: List):
        """
        :param timestamp_data: list of data. shapes are: 0: num_peds, 2: RFU, 3: data np.array of 8 floats
        :return: Set of  peds ids at scene
        """
        indexes = []
        for person in timestamp_data:
            indexes.append(person[0].item())
        return set(indexes)

    def get_dataset_from_index(self, data_index: int):
        """
        given index return dataset name and sub_dataset id, corresponding index in sub_dataset
        :param data_index: data sample number
        :return: file_name, sub_dataset id, corresponding index in sub_dataset
        """
        upper_bounds = list(self.dataset_indeces.keys())
        upper_bounds.append(0)
        upper_bounds.sort()
        index = [upper_bound > data_index for upper_bound in upper_bounds].index(True)
        index_in_sub_dataset = data_index - upper_bounds[index-1]
        return self.dataset_indeces[upper_bounds[index]], index_in_sub_dataset

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        [file, sub_dataset], index_in_sub_dataset = self.get_dataset_from_index(index)
        self.packed_data = []  # num_samples * 20 * numped * data_shape(14)
        data = self.get_ped_data_in_time(index_in_sub_dataset, index_in_sub_dataset+1, self.data[file][sub_dataset])
        return data[0]


def is_filled(data):
    return (data[:, 1] == 0.0).any().item()


if __name__ == "__main__":
    dataset = Dataset_from_pkl("/home/pazuzu/catkin_ws/src/Backprop-MPDM_torch/scripts/NN/datasets/processed/with_forces/", data_files=["eth_train.pkl", "zara2_test.pkl"])
    print(len(dataset))
    print(dataset[1272].shape)
    training_set = Dataset_from_pkl("/home/pazuzu/catkin_ws/src/Backprop-MPDM_torch/scripts/NN/datasets/processed/with_forces/", data_files=["eth_train.pkl"])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=1)
    #
    # for local_batch in training_generator:
    #     print(local_batch.shape)
    #     for ped in range(local_batch.shape[1]):
    #         observed_pose = local_batch[0, ped, 0:8, :]
    #         if is_filled(observed_pose):
    #             print("ped: ", ped, "observed_pose: ", observed_pose.shape)
    #         else:
    #             print("unfilled")
    #             break


