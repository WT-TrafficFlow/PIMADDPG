import sys
import os
sys.path.append("..")
# 获取当前脚本的绝对路径
current_path = os.path.abspath(__file__)

# 获取上级目录的绝对路径
parent_path = os.path.dirname(current_path)

# 获取上上级目录的绝对路径
grandparent_path = os.path.dirname(parent_path)
grandgrandparent_path = os.path.dirname(grandparent_path)
print('grandgrandparent_path',parent_path)
# 将上上级目录添加到系统路径
sys.path.append(parent_path)
import numpy as np
import traci
import torch
import argparse
import configparser
import ast
from model.PhySTGNN import make_model
import torch.optim as optim

def load_adjacent(pickle_file):
    adj_mx = np.load(pickle_file, allow_pickle=True)['data']
    return adj_mx

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()
class TrafficDetectorMerger:
    def __init__(self, T):
        """
        Initializes the history container and detector merger.

        Args:
            T (int): The number of time steps to keep in history.
        """
        self.T = T  # Number of time steps to store
        self.history = np.zeros((T, 12, 4))  # History container with shape (T, 12, 4)
        self.current_step = 0  # Current time step

    def get_detector_data(self, loop_ids):
        """
        Reads the flow, occupancy, and speed data from all detectors using traci.

        Args:
            loop_ids (list): A list of induction loop IDs.

        Returns:
            detectors (dict): A dictionary where the key is the group ID (e.g., 'e1down') and the value is
                               a dictionary containing lists of flow, occupancy, and speed.
        """
        detectors = {}
        for loop_id in loop_ids:
            # Read the required data from the current induction loop (detector)
            flow = traci.inductionloop.getLastIntervalVehicleNumber(loop_id)
            occupancy = traci.inductionloop.getLastIntervalOccupancy(loop_id)
            speed = traci.inductionloop.getLastIntervalMeanSpeed(loop_id)

            # Get the detector group (e.g., 'e1down', 'e1up' from 'e1down_0', 'e1down_1')
            group_id = loop_id.rsplit('_', 1)[0]

            if group_id not in detectors:
                detectors[group_id] = {'flow': [], 'occupancy': [], 'speed': []}

            # Append the current detector data
            detectors[group_id]['flow'].append(flow)
            detectors[group_id]['occupancy'].append(occupancy)
            detectors[group_id]['speed'].append(speed)

        return detectors

    def merge_detectors(self, detectors):
        """
        Merges the detectors by averaging the occupancy and speed, summing the flow for each group.

        Args:
            detectors (dict): A dictionary where the key is the group ID (e.g., 'e1down') and the value is
                               a dictionary containing lists of flow, occupancy, and speed.

        Returns:
            merged_detectors (list): A list of merged detector states for 12 detectors.
        """
        merged_detectors = []

        # Process each detector group
        for group_id in detectors:
            flow = int(sum(detectors[group_id]['flow']))  # Sum flow and divide by 60
            occupancy = sum(detectors[group_id]['occupancy']) / len(detectors[group_id]['occupancy']) # Average occupancy
            speed = sum(detectors[group_id]['speed']) / len(detectors[group_id]['speed']) # Average speed

            merged_detectors.append([flow, occupancy, speed])

        # Ensure we have exactly 12 detectors, pad if necessary
        while len(merged_detectors) < 12:
            merged_detectors.append([0, 0, 0])  # Fill with zeros if less than 12 detectors

        return merged_detectors

    def update(self, loop_ids, duration, speed_limit):
        """
        Updates the history container with the current state of the detectors after merging.

        Args:
            loop_ids (list): A list of induction loop IDs (detectors).
            duration (list): A list containing 4 values representing the duration for the current time step.
        """
        # Get the raw detector data
        detectors = self.get_detector_data(loop_ids)

        # Merge the detectors into 12 combined detectors
        merged_detectors = self.merge_detectors(detectors)

        # Create a temporary list to hold the state of all detectors
        detector_state = np.zeros((12, 5))  # (12, 4) where each detector has 3 values + 2 duration
        duration = 1 - np.array(duration) / 60
        speed_limit = np.array(speed_limit)
        # Loop over each detector and assign the corresponding duration value
        for i in range(12):
            # Get the flow, occupancy, and speed for this merged detector
            flow, occupancy, speed = merged_detectors[i]

            # Each detector should correspond to one of the four duration values (indexing is cyclic)
            duration_value = duration[i // 3]  # Cycle through the 4 duration values
            speed_limit_value = speed_limit[i // 3]
            # Combine the flow, occupancy, speed with the corresponding duration value
            detector_state[i] = [flow, occupancy, speed, duration_value, speed_limit_value]

        # Implementing sliding window behavior: store the latest data and move the window forward
        # Move the previous history down one step and add the new detector state at the end
        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1] = detector_state

        # Update the current time step
        self.current_step += 1

        return self.history

#
# # Example usage:
# T = 10  # Number of time steps to keep in history
# detector_merger = TrafficDetectorMerger(T)
#
# # Define loop IDs (in your case, these should be the actual induction loop IDs from your simulation)
# loop_ids = [
#     'e1down_0', 'e1down_1', 'e1ramp_0', 'e1up_0', 'e1up_1',
#     'e2down_0', 'e2down_1', 'e2ramp_0', 'e2up_0', 'e2up_1',
#     'e3down_0', 'e3down_1', 'e3ramp_0', 'e3up_0', 'e3up_1',
#     'e4down_0', 'e4down_1', 'e4ramp_0', 'e4up_0', 'e4up_1'
# ]
#
# # Simulate a few time steps
# for _ in range(15):  # Simulating 15 time steps
#     # Example duration list (it could be time-dependent, this is just a placeholder)
#     duration = [1.5, 2.0, 2.5, 3.0]
#
#     # Update history with the current detectors state
#     history = detector_merger.update(loop_ids, duration)
#     print(f"History after step {_ + 1}:\n", history)
class TrafficStatePred:
    def __init__(self):
        """
        Initializes the history container and detector merger.

        Args:
            T (int): The number of time steps to keep in history.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default='D:/mypythonfile/article/PIDRLRC/PIMADDPG/configurations/SUMO_SIM.conf', type=str,
                            help="configuration file path")
        args = parser.parse_args()
        config = configparser.ConfigParser()
        print('Read configuration file: %s' % (args.config))
        config.read(args.config)
        data_config = config['Data']
        training_config = config['Training']
        _mean =0
        _std = 0

        adj_filename = os.path.join(parent_path, data_config['adj_filename'])
        graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
        if config.has_option('Data', 'id_filename'):
            id_filename = data_config['id_filename']
        else:
            id_filename = None
        seed = 2024
        np.random.seed(seed)
        torch.manual_seed(seed)
        num_of_vertices = int(data_config['num_of_vertices'])
        num_for_predict = int(data_config['num_for_predict'])
        len_input = int(data_config['len_input'])
        dataset_name = data_config['dataset_name']
        batch_size = int(int(training_config['batch_size']) / 4)
        model_name = training_config['model_name']

        # ctx = training_config['ctx']
        # # os.environ["CUDA_VISIBLE_DEVICES"] = ctx
        # USE_CUDA = torch.cuda.is_available()
        DEVICE = torch.device('cpu')
        # print("CUDA:", USE_CUDA, DEVICE)

        learning_rate = float(training_config['learning_rate'])
        num_of_weeks = int(training_config['num_of_weeks'])
        num_of_days = int(training_config['num_of_days'])
        num_of_hours = int(training_config['num_of_hours'])
        num_heads = int(training_config['num_heads'])
        nb_chev_filter = int(training_config['nb_chev_filter'])
        kernel_size = ast.literal_eval(training_config['kernel_size'])
        in_channels = int(training_config['in_channels'])
        nb_block = int(training_config['nb_block'])
        K = int(training_config['K'])
        dropout = float(training_config['dropout'])
        folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (
        model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
        # print('folder_dir:', folder_dir)
        params_path = os.path.join('experiments', dataset_name, folder_dir)
        # print('params_path:', params_path)


        # adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
        # adj_mx = load_adj(adj_filename)

        # print(adj_filename)
        adj_mx = load_adjacent(adj_filename)  # 仅对 yinchuan 使用

        self.net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, num_heads, dropout, adj_mx,
                         num_for_predict, len_input, num_of_vertices, _mean, _std, kernel_size)
        params_filename = os.path.join(params_path, 'sumo_sim')  # 权重路径
        params_filename = os.path.join(parent_path, params_filename)
        self.net.load_state_dict(torch.load(params_filename, map_location=DEVICE))
        self.buffer = torch.zeros((batch_size+1, num_of_vertices, num_for_predict+len_input, 5), dtype=torch.float32)
        self.batch_size = batch_size
        self.count = -1
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def update(self, history, duration, speed_limit):
        self.net.train(False)
        with torch.no_grad():
            duration = 1 - np.array(duration) / 60
            current_state = np.zeros([1, 12, 5])
            current_state[0, :, -2] = np.repeat(duration, 3)
            current_state[0, :, -1] = np.repeat(speed_limit, 3)
            inputs = np.concatenate([history, current_state], axis=0)
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0)
            outputs, fd_q = self.net(inputs)
            self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
            self.buffer[:-1] = inputs
            self.count += 1
        if self.count % self.batch_size == 0 and self.count>0:  #取消更新
            self.backward()
        return outputs.squeeze(), fd_q.squeeze()

    def backward(self):
        self.net.train(True)
        self.optimizer.zero_grad()
        buffer = self.buffer.clone().detach()
        labels = buffer[1:, :, -2:-1, :2]
        # print(labels)
        outputs, fd_q = self.net(buffer[:-1])
        loss_q = masked_mae_loss(outputs[..., 0], labels[..., 0])
        loss_occ = masked_mae_loss(outputs[..., 1], labels[..., 1])
        loss_fd = masked_mae_loss(fd_q[..., 0], labels[..., 0])
        loss = 0.7 * loss_occ + 0.15 * (loss_q + loss_fd)
        loss.backward()
        self.optimizer.step()
        # print("training step: "+str(self.count // self.batch_size))