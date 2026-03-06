import os
import math
import scipy
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.mixture import GaussianMixture


class Interaction(object):
    def __init__(self, vrdata_dir, dtdata_dir, pid):

        # find the file 
        vr_file = [f for f in os.listdir(os.path.join(vrdata_dir, 'combined')) if f.endswith('.csv') and f.startswith(f'P{pid}_')]
        if len(vr_file) != 1: 
            print(f'Skipping pid={pid} as expected exactly one VR log file but found {len(vr_file)}: {vr_file}')
            return 
        vr_filename = vr_file[0]
        self.pid = pid

        self.log_path_from = os.path.join(vrdata_dir, f'combined/{vr_filename}')
        self.log_path_to = os.path.join(vrdata_dir, f'processed_basic/{vr_filename}')
        self.dt_path = os.path.join(dtdata_dir, f'DT_P{self.pid}.csv')

        self.load_data()
        self.preprocess_data_basic()
        self.save_processed_data()

        print(f'Logged formal data')
        print(f'    shape   = {self.df.shape}')
        print(f'    n_scenarios = {self.df.sid.unique()}\n')
        
    def load_data(self):
        print('\n#############################')
        print(f'Reading {self.log_path_from}')
        self.df = pd.read_csv(self.log_path_from)
        self.keep_only_formal_data()
        self.df['sid'] = self.df.Scenario.astype(int)
        self.df['pid'] = self.pid
        self.df.drop(columns=['Scenario'], inplace=True)

        # formal dt 
        print(f'Reading {self.dt_path}')
        self.dt = pd.read_csv(self.dt_path)
        self.dt['sid'] = self.dt.index + 1
        self.dt['pid'] = self.pid
        self.dt.drop(columns=['---'], inplace=True)
        self.dt = self.dt[['pid', 'sid', 'eHMI', 'yielding', 'angle', 'traffic flow']]        

    def preprocess_data_basic(self):
        self.df = self.df.drop_duplicates()
        self.recenter_locations()
        self.remove_no_eye_tracking()
        self.remove_collision()
        if self.df.shape[0] == 0:
            print('Skipping pid=', self.pid, 'as no data is left')
            return 
        
        self.get_time_elapsed_for_each_trial()
        self.calculate_distance_with_interaction_point()
        self.calculate_distance_between_agents()
        self.match_columns_for_all_data()

    def save_processed_data(self):
        if not os.path.exists(os.path.dirname(self.log_path_to)):
            os.makedirs(os.path.dirname(self.log_path_to), exist_ok=True)
        self.df.to_csv(self.log_path_to, index=False)
        print(f'Saved processed data to {self.log_path_to}')


    ################## Preprocessing data ###################
    def keep_only_formal_data(self):
        # only keep formal scenarios 
        self.df = self.df[~self.df.Scenario.str.startswith('F')]
        print(f'Keeping only formal scenarios, '
              f'shape = {self.df.shape}, scenarios = {self.df.Scenario.unique()}')

    def recenter_locations(self, center_x=-16800, center_y=-100):
        for col in self.df.columns:
            if 'Location_x' in col:
                self.df[col] = self.df[col] - center_x
            elif 'Location_y' in col:
                self.df[col] = self.df[col] - center_y

    def remove_no_eye_tracking(self):
        # find the scenario where ConfidenceValue is always 0, which means no eye tracking data at all
        sids_no_eyetracking = []
        for sid, group in self.df.groupby('sid'):
            if (group.ConfidenceValue == 0.0).all():
                sids_no_eyetracking.append(sid)
        print(f'Removing scenarios {sids_no_eyetracking} (no eye tracking data)')
        self.df = self.df[~self.df.sid.isin(sids_no_eyetracking)]
    
    def remove_collision(self, half_length=150, half_width=80):
        collision_scenarios = self.df[
            ((abs((self.df.Ped_Location_x - self.df.PodLeader_Location_x).to_numpy()) < half_length) &
             (abs((self.df.Ped_Location_y - self.df.PodLeader_Location_y).to_numpy()) < half_width)) | 
            ((abs((self.df.Ped_Location_x - self.df.PodFollower_Location_x).to_numpy()) < half_length) & 
             (abs((self.df.Ped_Location_y - self.df.PodFollower_Location_y).to_numpy()) < half_width))].sid.unique()
        if collision_scenarios.shape[0]:
            print(f'Removing scenarios {collision_scenarios} (collision)')
            self.df = self.df[self.df.sid.apply(lambda x: x not in collision_scenarios)]

    def get_time_elapsed_for_each_trial(self):    
        # create time elapsed for each trial 
        self.df['time'] = self.df.groupby('sid')['TimeElapsed'].transform(
            lambda x: x - x.iloc[0])
        # reorder
        column_order = self.df.columns.tolist()
        column_order.remove('time')
        column_order.remove('pid')
        column_order.remove('sid')
        column_order.insert(1, 'time')
        column_order.insert(2, 'pid')
        column_order.insert(3, 'sid')
        self.df = self.df[column_order]

    def match_columns_for_all_data(self):
        if 'PodLeader_EhmiStatus' not in self.df.columns:
            self.df['PodLeader_EhmiStatus'] = None
            self.df['PodFollower_EhmiStatus'] = None
            for sid in self.dt['sid'].unique():
                is_ehmi = self.dt[(self.dt.sid == sid) & (self.dt.pid == self.pid)].eHMI.iloc[0] == 'yes'
                is_yielding = self.dt[(self.dt.sid == sid) & (self.dt.pid == self.pid)].yielding.iloc[0]
                is_continuous = self.dt[(self.dt.sid == sid) & (self.dt.pid == self.pid)]['traffic flow'].iloc[0] != 0
                if is_ehmi: 
                    if is_yielding:
                        self.df.loc[(self.df['sid'] == sid) & (self.df['PodLeader_Ped_CenterDistance_x'] < 1330) & (self.df['PodLeader_Ped_CenterDistance_x'] >= 0), 'PodLeader_EhmiStatus'] = 'green'
                        if is_continuous:
                            self.df.loc[(self.df['sid'] == sid) & (self.df['PodFollower_Ped_CenterDistance_x'] < 1330) & (self.df['PodFollower_Ped_CenterDistance_x'] >= 0), 'PodFollower_EhmiStatus'] = 'green'
                    else:
                        self.df.loc[(self.df['sid'] == sid) & (self.df['PodLeader_Ped_CenterDistance_x'] < 1330) & (self.df['PodLeader_Ped_CenterDistance_x'] >= 0), 'PodLeader_EhmiStatus'] = 'red'
                        if is_continuous:
                            self.df.loc[(self.df['sid'] == sid) & (self.df['PodFollower_Ped_CenterDistance_x'] < 1330) & (self.df['PodFollower_Ped_CenterDistance_x'] >= 0), 'PodFollower_EhmiStatus'] = 'red'
        else:
            self.df.drop(columns=[
                'GazeOrigin_Location_x', 'GazeOrigin_Location_y', 'GazeOrigin_Location_z',
                'PodLeader_PedDistance_x', 'PodLeader_PedDistance_y', 
                'PodFollower_PedDistance_x', 'PodFollower_PedDistance_y'], inplace=True)

    def calculate_eye_angle(self, x, y):
        x, y = np.array(x), np.array(y)
        angle =  - np.arccos(x / np.sqrt(x**2 + y**2)) / np.pi * 180
        angle[np.where(y > 0)] = - 360 - angle[np.where(y > 0)]
        return angle
    
    def convert_rotation(self, angle, angle_min=-270):
        return np.floor(angle / angle_min) * 360 + angle
    
    def match_head_and_eyegaze(self):
        # make the range within -360~0 degree
        self.df.loc[:, ['Ped_Rotation_z']] = self.convert_rotation(self.df.groupby('sid')['Ped_Rotation_z'].transform(
            lambda x: x - np.ceil(x / 360)*360))
        self.df.loc[:, ['Ped_Rotation_z']] = self.df.groupby(['pid', 'sid'])['Ped_Rotation_z'].transform(
            lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous
        self.df.loc[:, ['Ped_Rotation_z_smoothed']] = self.df.groupby('sid')['Ped_Rotation_z'].transform(
            lambda x: self.smooth_array_gf(x, sigma=4))

        self.df.loc[:, ['EyeGaze_Rotation_z']] = self.convert_rotation(self.calculate_eye_angle(self.df['GazeDirection_x'], self.df['GazeDirection_y']))
        self.df.loc[self.df.ConfidenceValue == 0.0, ['EyeGaze_Rotation_z']] = np.nan  # set the eye gaze rotation to nan if confidence value is 0
        self.df.loc[:, ['EyeGaze_Rotation_z']] = self.df.groupby('sid')['EyeGaze_Rotation_z'].transform(
            lambda x: x.interpolate(method='linear').to_numpy())
        self.df.loc[:, ['EyeGaze_Rotation_z']] = self.df.groupby(['pid', 'sid'])['EyeGaze_Rotation_z'].transform(
            lambda x: np.unwrap(x.ffill().bfill(), period=360))  # unwrap the angles, making it continuous
        self.df.loc[:, ['EyeGaze_Rotation_z_smoothed']] = self.df.groupby('sid')['EyeGaze_Rotation_z'].transform(
            lambda x: self.smooth_array_gf(x, sigma=4))

    def calculate_distance_with_interaction_point(self):
        self.df['Ped_Distance'] = np.sqrt(
            self.df.Ped_Location_x**2 + self.df.Ped_Location_y**2)
        self.df['PodLeader_Distance'] = np.sqrt(
            self.df.PodLeader_Location_x**2 + self.df.PodLeader_Location_y**2)
        self.df['PodFollower_Distance'] = np.sqrt(
            self.df.PodFollower_Location_x**2 + self.df.PodFollower_Location_y**2)
    
    def calculate_distance_between_agents(self):
        half_pod_length = 150
        self.df['PodLeader_Ped_CenterDistance_x'] = self.df['Ped_Location_x'] - self.df['PodLeader_Location_x']
        self.df['PodLeader_Ped_CenterDistance_x_abs'] = self.df['PodLeader_Ped_CenterDistance_x'].abs()
        self.df['PodLeader_Ped_CenterDistance_y'] = self.df['Ped_Location_y'] - self.df['PodLeader_Location_y']
        self.df['PodLeader_Ped_CenterDistance_y_abs'] = self.df['PodLeader_Ped_CenterDistance_y'].abs()
        self.df['PodLeader_Ped_CenterDistance'] = (self.df['PodLeader_Ped_CenterDistance_x']**2 + self.df['PodLeader_Ped_CenterDistance_y']**2).apply(np.sqrt)

        self.df['PodFollower_Ped_CenterDistance_x'] = self.df['Ped_Location_x'] - self.df['PodFollower_Location_x']
        self.df['PodFollower_Ped_CenterDistance_x_abs'] = self.df['PodFollower_Ped_CenterDistance_x'].abs()
        self.df['PodFollower_Ped_CenterDistance_y'] = self.df['Ped_Location_y'] - self.df['PodFollower_Location_y']
        self.df['PodFollower_Ped_CenterDistance_y_abs'] = self.df['PodFollower_Ped_CenterDistance_y'].abs()
        self.df['PodFollower_Ped_CenterDistance'] = (self.df['PodFollower_Ped_CenterDistance_x']**2 + self.df['PodFollower_Ped_CenterDistance_y']**2).apply(np.sqrt)

    def preprocess_speed(self):
        self.df['DiffTime'] = self.df.groupby('sid')['TimeElapsed'].transform(
            lambda x: np.insert(np.diff(x), 0, np.nan))
        
        # pedestrian x axis, using smoothed trajectories! (instead of the raw one)
        self.df['DiffPedLocation_x'] = self.df.groupby('sid')['Ped_Location_x_smoothed'].transform(
            lambda x: np.insert(np.diff(x), 0, np.nan))
        self.df['Ped_Velocity_x'] = self.df['DiffPedLocation_x'] / self.df['DiffTime']

        # pedestrian y axis
        self.df['DiffPedLocation_y'] = self.df.groupby('sid')['Ped_Location_y_smoothed'].transform(
            lambda x: np.insert(np.diff(x), 0, np.nan))
        self.df['Ped_Velocity_y'] = self.df['DiffPedLocation_y'] / self.df['DiffTime']
        
        # pedestrian speed in absolute values
        self.df['Ped_Velocity_abs'] = np.sqrt(self.df.Ped_Velocity_x**2 + self.df.Ped_Velocity_y**2)

        # fill empty values with 0
        self.df.fillna({'Ped_Velocity_x': 0}, inplace=True)
        self.df.fillna({'Ped_Velocity_y': 0}, inplace=True)
        self.df.fillna({'Ped_Velocity_abs': 0}, inplace=True)

        # pedestrian forward speed direction
        self.df['Ped_Velocity_Rotation'] = self.convert_rotation(
            self.calculate_speed_direction(self.df.Ped_Velocity_x.to_numpy(), self.df.Ped_Velocity_y.to_numpy()))
        # self.df.fillna({'Ped_Velocity_Rotation': 0}, inplace=True) # replace nan as 0 may not be the right way
        self.df['Ped_Velocity_Rotation'] = self.df.groupby('sid')['Ped_Velocity_Rotation'].transform(
            lambda x: np.unwrap(x.ffill().bfill(), period=360))  # unwrap the angles, making it continuous
        self.df['Ped_Velocity_Rotation_smoothed'] = self.df.groupby('sid')['Ped_Velocity_Rotation'].transform(
            lambda x: self.smooth_array_gf(x, sigma=4))

        # get head velocity relative to the pedestrian velocity rotation 
        self.df['head_vel_relative_initial'] = self.convert_into_symmetric_angles(
            (self.df.Ped_Rotation_z_smoothed - self.df.Ped_Velocity_Rotation_smoothed).to_numpy(), angle_range=180)
        self.get_head_velocity_relative_threshold()

        # get final speed direction based on head-vel-relative and y-location
        direction = (np.logical_or(self.df['head_vel_relative_initial'].abs() < self.threshold, self.df.DiffPedLocation_y < 0)).astype(int)
        direction[direction == 0] = -1
        self.df['direction'] = self.fix_short_direction_segments(direction)
        self.df['Ped_Velocity'] = self.df.Ped_Velocity_abs * self.df.direction
        self.df['direction'] = (self.df.Ped_Velocity > 0).astype(int)
        self.df.loc[self.df.direction == 0, 'direction'] = -1  # 1 for forward, -1 for backward

        # smooth speed profile
        self.smooth_speed_gf()

        # get body rotation  
        self.df['body_rotation'] = self.convert_into_symmetric_angles(self.df.Ped_Velocity_Rotation_smoothed - 180*(self.df.direction == -1).to_numpy())
        self.df['body_rotation_mask'] = self.df['Ped_Velocity_smoothed'].abs() < 35  # mask the body rotation when speed is too low
        self.df.loc[self.df.body_rotation_mask, 'body_rotation'] = np.nan
        self.df['body_rotation_int'] = self.df.body_rotation.copy()
        self.df['body_rotation_int'] = self.df.groupby(['pid', 'sid'])['body_rotation_int'].transform(
            lambda x: x.interpolate(method='nearest', limit_direction='both').to_numpy()) # this is for direction where nearest makes more sense
        self.df['body_rotation_int'] = self.df.groupby(['pid', 'sid'])['body_rotation_int'].transform(
            lambda x: x.ffill().bfill())
        self.df['body_rotation_unwrap'] = self.df.groupby(['pid', 'sid'])['body_rotation_int'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous
        self.df.loc[self.df.body_rotation_mask, 'body_rotation_unwrap'] = np.nan
        self.df['body_rotation_unwrap'] = self.df.groupby(['pid', 'sid'])['body_rotation_unwrap'].transform(
            lambda x: x.interpolate(method='linear').to_numpy()) # this is reinterpolate by linear
        self.df['body_rotation_unwrap'] = self.df.groupby(['pid', 'sid'])['body_rotation_unwrap'].transform(lambda x: x.ffill().bfill())
        self.df['body_rotation_smoothed'] = self.df.groupby(['pid', 'sid'])['body_rotation_unwrap'].transform(lambda x: self.smooth_array_gf(x, sigma=4))

        self.df['head_vel_relative'] = self.convert_into_symmetric_angles((self.df.Ped_Rotation_z - self.df.body_rotation).to_numpy(), angle_range=180)
        self.df['head_vel_relative_smoothed'] = self.convert_into_symmetric_angles((self.df.Ped_Rotation_z_smoothed - self.df.body_rotation_smoothed).to_numpy(), angle_range=180)
        self.df['head_vel_relative_smoothed'] = self.df.groupby(['pid', 'sid'])['head_vel_relative_smoothed'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous

        # self.df['eye_vel_relative'] = self.convert_into_symmetric_angles((self.df.EyeGaze_Rotation_z - (self.df.Ped_Velocity_Rotation - 180*(self.df.direction == -1))).to_numpy(), angle_range=180)
        self.df['eye_vel_relative'] = self.convert_into_symmetric_angles((self.df.EyeGaze_Rotation_z - self.df.body_rotation).to_numpy(), angle_range=180)
        self.df['eye_vel_relative_smoothed'] = self.convert_into_symmetric_angles((self.df.EyeGaze_Rotation_z_smoothed - self.df.body_rotation_smoothed).to_numpy(), angle_range=180)
        self.df['eye_vel_relative_smoothed'] = self.df.groupby(['pid', 'sid'])['eye_vel_relative_smoothed'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous

        self.df['eye_head_relative'] = self.convert_into_symmetric_angles((self.df.EyeGaze_Rotation_z - self.df.Ped_Rotation_z).to_numpy(), angle_range=180)
        self.df['eye_head_relative_smoothed'] = self.convert_into_symmetric_angles((self.df.EyeGaze_Rotation_z_smoothed - self.df.Ped_Rotation_z_smoothed).to_numpy(), angle_range=180)
        self.df['eye_head_relative_smoothed'] = self.df.groupby(['pid', 'sid'])['eye_head_relative_smoothed'].transform(lambda x: np.unwrap(x, period=360))  # unwrap the angles, making it continuous

        # recenter all angles
        self.df['Ped_Velocity_Rotation_smoothed'] = self.df.groupby(['pid', 'sid'])['Ped_Velocity_Rotation_smoothed'].transform(self.recenter_angles)
        self.df['Ped_Rotation_z_smoothed'] = self.df.groupby(['pid', 'sid'])['Ped_Rotation_z_smoothed'].transform(self.recenter_angles)
        self.df['body_rotation_smoothed'] = self.df.groupby(['pid', 'sid'])['body_rotation_smoothed'].transform(self.recenter_angles)
        self.df['EyeGaze_Rotation_z_smoothed'] = self.df.groupby(['pid', 'sid'])['EyeGaze_Rotation_z_smoothed'].transform(self.recenter_angles)
        self.df['head_vel_relative_smoothed'] = self.df.groupby(['pid', 'sid'])['head_vel_relative_smoothed'].transform(self.recenter_angles)
        self.df['eye_vel_relative_smoothed'] = self.df.groupby(['pid', 'sid'])['eye_vel_relative_smoothed'].transform(self.recenter_angles)
        self.df['eye_head_relative_smoothed'] = self.df.groupby(['pid', 'sid'])['eye_head_relative_smoothed'].transform(self.recenter_angles)

        # convert relative angles into sysmetric angles
        self.df['head_vel_relative_smoothed'] = self.convert_into_symmetric_angles(self.df['EyeGaze_Rotation_z_smoothed'], angle_range=180)
        self.df['eye_vel_relative_smoothed'] = self.convert_into_symmetric_angles(self.df['eye_vel_relative_smoothed'], angle_range=180)
        self.df['eye_head_relative_smoothed'] = self.convert_into_symmetric_angles(self.df['eye_head_relative_smoothed'], angle_range=180)
        # TODO: body rotation is still not good,,, consider whether using relative angle to this... 

        self.df = self.df.drop(columns=['DiffTime', 'DiffPedLocation_x'])

    def recenter_angles(self, values):
        center = values.mean()
        n = (center + 180) // 360
        return values - 360*n

    def mask_near_direction_change(self, value, direction_series, window=2):
        change_mask = direction_series.shift() != direction_series
        change_mask.iloc[0] = False  # 第一个元素没有前一个元素，设为False

        # 生成布尔掩码
        mask = np.zeros(len(direction_series), dtype=bool)

        # 前后2行偏移量
        for offset in range(-window, window):  
            mask |= change_mask.shift(offset, fill_value=False)

        # 将value设为NaN
        value[mask] = np.nan
        return value 

    def fix_short_direction_segments(self, direction_series, min_length=15):
        direction = direction_series.to_numpy()
        # 找每个段的开始和结束
        change_idx = np.flatnonzero(np.diff(direction) != 0) + 1
        segment_bounds = np.r_[0, change_idx, len(direction)]
        
        # 逐段处理
        seg_lengths = np.diff(segment_bounds)
        seg_values = direction[segment_bounds[:-1]]
        
        # 找出短段
        short_mask = seg_lengths < min_length
        
        # 修正短段
        for i, short in enumerate(short_mask):
            if short:
                prev_val = seg_values[i - 1] if i > 0 else None
                next_val = seg_values[i + 1] if i < len(seg_values) - 1 else None
                if prev_val == next_val and prev_val is not None:
                    direction[segment_bounds[i]:segment_bounds[i+1]] = prev_val

        return direction

    def convert_into_symmetric_angles(self, angle, angle_range=180):
        return (angle + angle_range) % 360 - angle_range

    def smooth_trajectories_gf(self, sigma=4):
        self.df['Ped_Location_x_smoothed'] = self.df.groupby('sid')['Ped_Location_x'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
        self.df['Ped_Location_y_smoothed'] = self.df.groupby('sid')['Ped_Location_y'].transform(
            lambda x: self.smooth_array_gf(x, sigma))

    def smooth_speed_gf(self, sigma=4):
        self.df['Ped_Velocity_x_smoothed'] = self.df.groupby('sid')['Ped_Velocity_x'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
        self.df['Ped_Velocity_y_smoothed'] = self.df.groupby('sid')['Ped_Velocity_y'].transform(
            lambda x: self.smooth_array_gf(x, sigma)) 
        self.df['Ped_Velocity_abs_smoothed'] = self.df.groupby('sid')['Ped_Velocity_abs'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
        self.df['Ped_Velocity_smoothed'] = self.df.groupby('sid')['Ped_Velocity'].transform(
            lambda x: self.smooth_array_gf(x, sigma))
    
    def smooth_array_gf(self, array, sigma=4):
        return scipy.ndimage.gaussian_filter1d(array, sigma=sigma, axis=0, mode='nearest')
        
    def correct_velocity_rotation_range(self, vx, vy, alpha):
        if vx > 0 and vy < 0:
            return - alpha
        elif vx < 0 and vy < 0:
            return alpha - 180
        elif vx < 0 and vy > 0:
            return - 180 - alpha 
        else:
            return alpha - 360
    
    def calculate_speed_direction(self, arr_vx, arr_vy):
        res = []
        arr_alpha = np.arctan(np.abs(arr_vy) / np.abs(arr_vx)) * 180 / np.pi
        for vx, vy, alpha in zip(arr_vx, arr_vy, arr_alpha):
            res.append(self.correct_velocity_rotation_range(vx, vy, alpha))
        return np.array(res)

    def get_consecutive_trues(self, bool_array, verbose=False):
        # Identify consecutive True groups
        diff = np.diff(np.concatenate(([False], bool_array, [False])).astype(int))
        start_indices = np.where(diff == 1)[0]
        end_indices = np.where(diff == -1)[0]

        # Length of consecutive True sequences
        lengths = end_indices - start_indices
        # print("Consecutive True lengths:", lengths)  # Output: [2, 1, 3]
        if verbose:
            return start_indices, end_indices, lengths
        else:
            return lengths 

    def find_nearest_zero_velocity_before_pod_moving(self, df_j):
        df_j = df_j.reset_index()
        diff_dist = df_j.PodLeader_Distance.diff()
        idx = diff_dist[diff_dist != 0].index[1]

        bool_velocity_near_zero = (df_j.Ped_Velocity_smoothed.abs() < 3).to_numpy()
        _, end, lengths = self.get_consecutive_trues(bool_velocity_near_zero, verbose=True)
        idx_before = np.where(end < idx)[0]
        if len(idx_before) > 0:
            end_last_idx = idx_before[-1]
            last_idx = end[end_last_idx]
            return last_idx - int(lengths[end_last_idx] / 2)
        else:
            return None
    
    def get_pod_trigger_points(self):
        self.podleader_trigger_points = dict()
        self.podleader_trigger_index = dict()

        for j in self.df.sid.unique():
            df_j = self.df[self.df.sid == j]
            idx_base = df_j.index.to_numpy()[0]

            diff_dist = df_j.PodLeader_Location_x.diff().reset_index(drop=True)
            idx = diff_dist[diff_dist != 0].index[1]
            j = int(j)
            self.podleader_trigger_points[j] = int(idx)
            self.podleader_trigger_index[j] = int(idx_base + idx)

        print('Pods moving trigger points:')
        print(self.podleader_trigger_points)
        print(self.podleader_trigger_index)

    def get_head_velocity_relative_threshold(self):
        data_ped = self.df.head_vel_relative_initial.abs().to_numpy()

        x_vals, density, x_argmax = self.fit_gaussian_mixture(data_ped)
        x_diff_idx = density[x_argmax[0]: x_argmax[1]].argmin() + x_argmax[0]
        x_diff = int(x_vals[x_diff_idx])

        self.threshold = max(min(x_diff, 90), 45)
        print(f'\nSetting head_vel_relative threshold as {self.threshold}')
    
    def fit_gaussian_mixture(self, data):
        # Fit a GMM with 2 components
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data.reshape(-1, 1))  # Reshape data to be 2D for GMM

        # Extract parameters
        weights = gmm.weights_  # Weights of the two components
        means = gmm.means_.flatten()  # Means of the two components
        covariances = gmm.covariances_.flatten() 

        x_vals = np.linspace(min(data), max(data), 1000)
        density = np.zeros_like(x_vals)
        x_argmax = []

        # Plot each Gaussian component and sum them up for the total density
        for weight, mean, covariance in zip(weights, means, covariances):
            std_dev = np.sqrt(covariance)  # Standard deviation
            component_density = weight * norm.pdf(x_vals, mean, std_dev)
            # plt.plot(x_vals, component_density, label=f'Gaussian (mean={mean:.2f}, std={std_dev:.2f})')
            density += component_density
            x_argmax.append(component_density.argmax())
        x_argmax.sort()
        
        return x_vals, density, x_argmax
    

def main(args):
    if args.pids == [0]:
        args.pids = list(range(1, 52))
    for pid in args.pids:
        Interaction(
            vrdata_dir=args.vrdata_dir,
            dtdata_dir=args.dtdata_dir,
            pid=pid)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--vrdata_dir", type=str, default='data/vrdata')
    parser.add_argument("--dtdata_dir", type=str, default='data/dtdata')
    parser.add_argument("--pids", type=int, default=[0], nargs='+',
                        help='participant id, can be a list of ids or a single id')
    args = parser.parse_args()

    main(args)

## example
# python -m utils.preprocessing -f filenames
# python -m utils.preprocessing -f P1_20241217103615516 P2_202412171419360 P3_20241217152233801 P4_2024121911738427 P5_20241219132245990 P6_2024121915620772 P7_2024121915567133 P8_2024121916394796 > out1-8.txt
# python -m utils.preprocessing -f P9_20251610917686 P10_20251613298921 P11_202516151537305 P12_202516171753723 P13_202517101430120 P14_20251711546819 P15_20251714314880 P16_202517151737922 P17_202517162823137 P18_2025171745643 > out9-18.txt
# python -m utils.preprocessing -f P19_202519171112119 P20_202511010139902 P21_202511011150733 P22_202511013946226 P23_202511014555424 P24_202511015455563 P25_2025110161322537 P26_202511017318590 P27_202511194845442 P28_20251111384223 P29_20251111435533 P30_2025111151424898 P31_202511116142262 > out19-31.txt
# python -m utils.preprocessing -f P32_202511395855284 P33_2025113131228931 P34_2025113135340982 P35_202511315915944 P36_20251131675773 P37_202511317103643 P38_2025114101220373 P39_20251141112424 P40_2025114131911860 P41_202511414230295 P42_202511415655706 P43_2025114162926461 P44_2025114171410580 > out32-44.txt
# python -m utils.preprocessing -f P45_202511610513261 P46_202511611914394 P47_202511614223187 P48_2025116151043950 P49_2025116171511625 > out45-49.txt
# python -m utils.preprocessing -f P50_202511813834529 P51_2025118141418658 > out50-51.txt
# python -m utils.preprocessing -f P50_202511813834529 P51_2025118141418658 --suffix _noremoval > out50-51_noremoval.txt
    
# python -m utils.preprocessing -f P1_20241217103615516 P2_202412171419360 P3_20241217152233801 P4_2024121911738427 P5_20241219132245990 P6_2024121915620772 P7_2024121915567133 P8_2024121916394796 P9_20251610917686 P10_20251613298921 P11_202516151537305 P12_202516171753723 P13_202517101430120 P14_20251711546819 P15_20251714314880 P16_202517151737922 P17_202517162823137 P18_2025171745643 P19_202519171112119 P20_202511010139902 P21_202511011150733 P22_202511013946226 P23_202511014555424 P24_202511015455563 P25_2025110161322537 P26_202511017318590 P27_202511194845442 P28_20251111384223 P29_20251111435533 P30_2025111151424898 P31_202511116142262 P32_202511395855284 P33_2025113131228931 P34_2025113135340982 P35_202511315915944 P36_20251131675773 P37_202511317103643 P38_2025114101220373 P39_20251141112424 P40_2025114131911860 P41_202511414230295 P42_202511415655706 P43_2025114162926461 P44_2025114171410580 P45_202511610513261 P46_202511611914394 P47_202511614223187 P48_2025116151043950 P49_2025116171511625 P50_202511813834529 P51_2025118141418658 2>&1 | tee out_preprocess.txt
      