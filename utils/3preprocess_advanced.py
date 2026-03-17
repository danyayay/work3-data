import os
import re 
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
    def __init__(self, vrdata_dir, dtdata_dir, videodata_dir, psid, is_savefig=False, is_visualize=False):
        self.pid, self.sid = re.match(r'P(\d+)S(\d+)', psid).groups()
        self.pid, self.sid = int(self.pid), int(self.sid)
        self.log_path_from = os.path.join(vrdata_dir, f'aligned/{psid}.csv')
        self.log_path_to = os.path.join(vrdata_dir, f'processed_advanced/{psid}.csv')
        self.dt_path = os.path.join(dtdata_dir, f'DT_P{self.pid}.csv')
        self.video_path = os.path.join(videodata_dir, f'P{self.pid}', f'{psid}.mp4')
        
        self.is_savefig = is_savefig
        self.fig_dir = 'figures/indiv/{}'.format(psid)

        self.load_data()
        assert self.df.sid.nunique() == 1, f'Multiple sids found in the data for pid={self.pid}'
        self.preprocess_data_advanced()
        
        print(f'Preprocessed data')
        print(f'    shape   = {self.df.shape}')
        print(f'    n_sids = {self.df.sid.unique()}\n')

        if is_visualize: self.visualize()
        
        self.manage_columns()
        self.save_processed_data()

    def load_data(self):
        print('\n#############################')
        print(f'Reading {self.log_path_from}')
        self.df = pd.read_csv(self.log_path_from)

        # dt 
        self.dt = pd.read_csv(self.dt_path)
        self.dt['pid'] = self.pid
        self.dt['sid'] = self.dt.index + 1
        self.dt['psid'] = f'P{self.pid}S' + self.dt['sid'].astype(str)
        self.dt.drop(columns=['---'], inplace=True)
        self.dt = self.dt[['pid', 'sid', 'psid', 'eHMI', 'yielding', 'angle', 'traffic flow']]

    def preprocess_data_advanced(self):
        self.preprocess_eyegaze_data()
        self.get_pod_trigger_points()
        self.smooth_trajectories_gf()
        self.preprocess_speed()
        self.remove_bad_speed()
        self.mask_out_nonfixation_gaze()

    def save_processed_data(self):
        # save the filtered data 
        if not os.path.exists(os.path.dirname(self.log_path_to)):
            os.makedirs(os.path.dirname(self.log_path_to), exist_ok=True)
        self.df.to_csv(self.log_path_to, index=False)
        print(f'Saved filtered data to {self.log_path_to}')
    
    def manage_columns(self):
        self.df.rename(columns={
            'Ped_Location_x_smoothed': 'ped_loc_x',
            'Ped_Location_y_smoothed': 'ped_loc_y',
            'Ped_Rotation_z_smoothed': 'head_rot_z',
            'EyeGaze_Rotation_z_smoothed': 'eye_rot_z',
            'body_rotation_smoothed': 'body_rot_z',
            'Ped_Velocity_x_smoothed': 'ped_vel_x',
            'Ped_Velocity_y_smoothed': 'ped_vel_y',
            'Ped_Velocity_smoothed': 'ped_vel',
            'HitObject': 'hit_object',
            'head_vel_relative_smoothed': 'head_vel_rel',
            'eye_vel_relative_smoothed': 'eye_vel_rel',
            'eye_head_relative_smoothed': 'eye_head_rel',
            'PodLeader_Location_x': 'leader_loc_x',
            'PodLeader_Velocity_x': 'leader_vel_x',
            'PodLeader_EhmiStatus': 'leader_ehmi',
            'PodFollower_Location_x': 'follower_loc_x',
            'PodFollower_Velocity_x': 'follower_vel_x',
            'PodFollower_EhmiStatus': 'follower_ehmi',
            'Ped_Distance': 'dist_ped2center',
            'PodLeader_Distance': 'dist_leader2center',
            'PodFollower_Distance': 'dist_follower2center',
        }, inplace=True)
        self.df = self.df[[
            'time', 'pid', 'sid',
            'ped_loc_x', 'ped_loc_y', 'ped_vel_x', 'ped_vel_y', 'ped_vel',
            'head_rot_z', 'eye_rot_z', 'body_rot_z', 
            'head_vel_rel', 'eye_vel_rel', 'eye_head_rel',
            'mask_attention', 'px_u', 'px_v', 'hit_object', 
            'leader_loc_x', 'leader_vel_x', 'leader_ehmi',
            'follower_loc_x', 'follower_vel_x', 'follower_ehmi',
            'dist_ped2center', 'dist_leader2center', 'dist_follower2center'
        ]]


    ################# Visualization #####################
    def visualize(self):
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

        self.title = self.dt[self.dt.sid == self.sid][[
            'psid', 'eHMI', 'yielding', 'angle', 'traffic flow']].iloc[0].astype(str).to_list()
        self.plot_interaction_trajectories_together(self.df)
        # self.plot_gazehit_together(self.df)

        self.plot_when_passed_collision_point_per_trial(self.df)
        self.plot_ped_rotation_z_per_trial(self.df)
        self.plot_ped_rotation_z_rel_per_trial(self.df)

        self.plot_ped_velocity_per_trial(self.df)
        self.plot_pod_trajectory_velocity_per_trial(self.df, 'pod_leader')
        self.plot_pod_trajectory_velocity_per_trial(self.df, 'pod_follower')

        self.plot_gaze_event_together(self.df)
        self.plot_gaze_on_screen_overlay(self.df)
        # intermediate plots 
        # self.plot_ped_rotation_diff_per_trial(self.df)
        # self.plot_ped_y_diff_per_trial(self.df)
        # self.plot_trajectory_per_trial(self.df)

    def plot_interaction_trajectories_together(self, df, is_plot_pod=False):
        # define the boundary
        max_x, min_x = 400, -400
        max_y, min_y = -250, 450
        # print('Interaction area:')
        # print(f'x ({min_x}, {max_x})')
        # print(f'y ({min_y}, {max_y})')

        # plot
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(df.Ped_Location_x, df.Ped_Location_y, c=df['time'], s=1, cmap='grey')
        plt.scatter(df.Ped_Location_x_smoothed, df.Ped_Location_y_smoothed, c=df['time'], s=1, cmap='Blues')

        # if plot pod trajectory
        if is_plot_pod: plt.plot(df.PodLeader_Location_x, df.PodLeader_Location_y, label='pod')
        
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.colorbar(label='time')
        plt.title(self.title)
        fig.tight_layout()
        
        if self.is_savefig:
            plt.savefig(
                f'{self.fig_dir}/interaction.png', bbox_inches='tight')
        plt.close()

    def plot_when_passed_collision_point_per_trial(self, df):
        _ = plt.figure(figsize=(5, 4))

        df_j = df[df.sid == self.sid]
        # pedestrian
        plt.plot(df_j.time, df_j.Ped_Distance, label='Pedestrian')
        plt.scatter(df_j.time.iloc[df_j.Ped_Distance.argmin()], df_j.Ped_Distance.min())

        # pod leader
        plt.plot(df_j.time, df_j.PodLeader_Distance, label='Pod Leader')
        plt.scatter(df_j.time.iloc[df_j.PodLeader_Distance.argmin()], df_j.PodLeader_Distance.min())
        # plot the start moving point
        diff_dist = df_j.PodLeader_Distance.diff()
        plt.axvline(x=df_j.time.loc[diff_dist[diff_dist != 0].index[1]], color='gray')
        # pod follower
        if df_j.PodFollower_Velocity_x.isnull().sum() == 0:
            plt.plot(df_j.time, df_j.PodFollower_Distance, label='Pod Follower')
            plt.scatter(df_j.time.iloc[df_j.PodFollower_Distance.argmin()], df_j.PodFollower_Distance.min())

        plt.annotate(
            int(df_j.Ped_Distance.loc[diff_dist[diff_dist != 0].index[1]]),
            xy=(df_j.time.loc[diff_dist[diff_dist != 0].index[1]], 0),
                        xytext=(df_j.time.loc[diff_dist[diff_dist != 0].index[1]], 0.5), color='gray')

        # legend    
        plt.title(self.title)
        plt.ylim(0, 4400)
        # plot ehmi status 
        if len([s for s in df_j.columns if "EhmiStatus" in s]):
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1)
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1)
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1)
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1)
        plt.legend(loc='best')

        if self.is_savefig:
            plt.savefig(
                f'{self.fig_dir}/collision.png', bbox_inches='tight')       
        plt.close()
    
    def plot_ped_rotation_z_per_trial(self, df):
        _ = plt.figure(figsize=(5, 4))

        min_y = df[['Ped_Rotation_z', 'Ped_Rotation_z_smoothed', 'EyeGaze_Rotation_z', 'EyeGaze_Rotation_z_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].min().min()
        max_y = df[['Ped_Rotation_z', 'Ped_Rotation_z_smoothed', 'EyeGaze_Rotation_z', 'EyeGaze_Rotation_z_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].max().max()

        df_j = df[df.sid == self.sid]
        p = plt.plot(df_j.time, df_j.Ped_Rotation_z, alpha=0.3)
        plt.plot(df_j.time, df_j.Ped_Rotation_z_smoothed, c=p[0].get_color(), label='head orientation')

        p = plt.plot(df_j.time, np.ma.masked_where(df_j.ConfidenceValue == 0.0, df_j.EyeGaze_Rotation_z), alpha=0.3)
        plt.plot(df_j.time, df_j.EyeGaze_Rotation_z_smoothed, c=p[0].get_color(), label='eye gaze')

        p, = plt.plot(df_j.time, df_j.Ped_Velocity_Rotation, alpha=0.3)
        plt.plot(df_j.time, df_j.Ped_Velocity_Rotation_smoothed, c=p.get_color(), label='velocity rotation')

        p, = plt.plot(df_j.time, df_j.body_rotation_unwrap, alpha=0.3)
        # axes[k, l].plot(df_j.time, df_j.body_rotation_unwrap, c=p.get_color(), linestyle='--')
        plt.plot(df_j.time, df_j.body_rotation_smoothed, c=p.get_color(), label='body rotation')
        # axes[k, l].plot(df_j.time, df_j.body_rotation, c='pink')

        plt.title(self.title)
        # plot ehmi status 
        if len([s for s in df_j.columns if "EhmiStatus" in s]):
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1)
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1)
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1)
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1)
        plt.xlim(0, df.time.max())
        plt.ylim(min_y, max_y)
        plt.legend(loc='best')
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(f'{self.fig_dir}/headeye.png', bbox_inches='tight')
        plt.close()

    def plot_ped_rotation_z_rel_per_trial(self, df):
        _ = plt.figure(figsize=(5, 4))

        min_y = df[['head_vel_relative', 'head_vel_relative_smoothed', 'eye_vel_relative', 'eye_vel_relative_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].min().min()
        max_y = df[['head_vel_relative', 'head_vel_relative_smoothed', 'eye_vel_relative', 'eye_vel_relative_smoothed', 
                    'Ped_Velocity_Rotation', 'Ped_Velocity_Rotation_smoothed', 'body_rotation_unwrap', 'body_rotation_smoothed']].max().max()

        df_j = df[df.sid == self.sid]
        p, = plt.plot(df_j.time, df_j.head_vel_relative, alpha=0.3)
        plt.plot(df_j.time, df_j.head_vel_relative_smoothed, label='relative head orientation', c=p.get_color())

        p, = plt.plot(df_j.time, np.ma.masked_where(df_j.ConfidenceValue == 0.0, df_j.eye_vel_relative), alpha=0.3)
        plt.plot(df_j.time, df_j.eye_vel_relative_smoothed, c=p.get_color(), label='relative eye gaze')

        p, = plt.plot(df_j.time, df_j.Ped_Velocity_Rotation, alpha=0.3)
        plt.plot(df_j.time, df_j.Ped_Velocity_Rotation_smoothed, label='velocity rotation', c=p.get_color())

        # p, = axes[k, l].plot(df_j.time, df_j.eye_head_relative, alpha=0.3)
        # axes[k, l].plot(df_j.time, df_j.eye_head_relative_smoothed, c=p.get_color(), label='eye head relative')
        p, = plt.plot(df_j.time, df_j.body_rotation_unwrap, alpha=0.3)
        # axes[k, l].plot(df_j.time, df_j.body_rotation_unwrap, c=p.get_color(), linestyle='--')
        plt.plot(df_j.time, df_j.body_rotation_smoothed, c=p.get_color(), label='body rotation')
        # axes[k, l].plot(df_j.time, df_j.body_rotation_int, c='pink')

        plt.title(self.title)
        # plot ehmi status 
        if len([s for s in df_j.columns if "EhmiStatus" in s]):
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=plt.gca().get_xaxis_transform())
        plt.xlim(0, df.time.max())
        plt.ylim(min_y, max_y)
        plt.legend(loc='best')
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(f'{self.fig_dir}/headeye_rel.png', bbox_inches='tight')
        plt.close()

    def plot_ped_rotation_diff_per_trial(self, df):
        _ = plt.figure(figsize=(5, 4))

        # TODO: need to dynamically choose yaw range
        df_j = df[df.sid == self.sid]
        plt.plot(df_j.time, df_j.head_vel_relative, label='head vel diff')
        # plot ehmi status
        if len([s for s in df_j.columns if "EhmiStatus" in s]):
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=plt.gca().get_xaxis_transform())
        plt.legend(loc='best')
        plt.xlim(0, df.time.max())
        plt.ylim(-180, 180)
        plt.title(self.title)
        plt.axhline(y=-self.threshold, color='grey', linestyle='--')
        plt.axhline(y=self.threshold, color='grey', linestyle='--')
        plt.annotate(self.threshold, 
                    xy=(df.time.max()-2, self.threshold), 
                    xytext=(df.time.max()-2, self.threshold), color='gray')                
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(f'{self.fig_dir}/head_vel_diff.png', bbox_inches='tight')
        plt.close()

    def plot_ped_y_diff_per_trial(self, df):
        _ = plt.figure(figsize=(5, 4))

        df_j = df[df.sid == self.sid]
        plt.plot(df_j.time, df_j.DiffPedLocation_y, label='y diff')
        # plot ehmi status
        if len([s for s in df_j.columns if "EhmiStatus" in s]):
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=plt.gca().get_xaxis_transform())
        plt.legend(loc='best')
        plt.title(self.title)
        plt.xlim(0, df.time.max())
        plt.ylim(df.DiffPedLocation_y.min(), df.DiffPedLocation_y.max())
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(f'{self.fig_dir}/ped_y_diff.png', bbox_inches='tight')
        plt.close()

    def plot_ped_velocity_per_trial(self, df, is_smooth=True):
        _ = plt.figure(figsize=(5, 4))

        COLOR_VEL_x = 'red'
        COLOR_VEL_y = 'green'
        COLOR_VEL = 'blue'
        SPEED_MAX = max(df.Ped_Velocity_smoothed.abs().max(), 160)

        df_j = df[df.sid == self.sid]
        # original velocity
        plt.plot(df_j.time, df_j.Ped_Velocity_x, c=COLOR_VEL_x, alpha=0.3)
        plt.plot(df_j.time, df_j.Ped_Velocity_y, c=COLOR_VEL_y, alpha=0.3)
        plt.plot(df_j.time, df_j.Ped_Velocity, c=COLOR_VEL, alpha=0.3)
        
        # smoothed velocity 
        if is_smooth:
            plt.plot(df_j.time, df_j.Ped_Velocity_x_smoothed, c=COLOR_VEL_x, label='x')
            plt.plot(df_j.time, df_j.Ped_Velocity_y_smoothed, c=COLOR_VEL_y, label='y')
        plt.plot(df_j.time, df_j.Ped_Velocity_smoothed, c=COLOR_VEL)
        plt.ylabel('Velocity')
        plt.xlabel('Time (second)')
        plt.ylim(-SPEED_MAX, SPEED_MAX)
        plt.xlim(0, df.time.max())
        plt.title(self.title)
        plt.legend(loc='best')

        diff_dist = df_j.PodLeader_Distance.diff()
        # plot ehmi status 
        if len([s for s in df_j.columns if "EhmiStatus" in s]):
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=plt.gca().get_xaxis_transform())
            plt.fill_between(df_j.time, 0, 1, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=plt.gca().get_xaxis_transform())
        plt.axhline(y=0, color='gray', linestyle='--')
        # plot start point
        plt.axvline(x=df_j.time.loc[diff_dist[diff_dist != 0].index[1]], color='gray')
        plt.annotate(
            int(df_j.Ped_Distance.loc[diff_dist[diff_dist != 0].index[1]]), 
            xy=(df_j.time.loc[diff_dist[diff_dist != 0].index[1]], -150), 
            xytext=(df_j.time.loc[diff_dist[diff_dist != 0].index[1]], -150), color='gray')
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(
                    f'{self.fig_dir}/ped_speed.png', bbox_inches='tight')
        plt.close()

    def plot_pod_trajectory_velocity_per_trial(self, df, agent):
        # TODO: now we're in 2d space, think about how to analyze the results
        _ = plt.figure(figsize=(5, 4))

        if agent == 'pod_leader':
            col_name = 'PodLeader_Location_x'
        elif agent == 'pod_follower':
            col_name = 'PodFollower_Location_x'
        else:
            raise ValueError(f'No such {agent}')

        COLOR_TRAJ = "#69b3a2"
        COLOR_VEL = "#3399e6"

        df_j = df[df.sid == self.sid]
        # VR trajectory
        plt.plot(df_j.time, df_j[col_name], label='Trajectory', c=COLOR_TRAJ)
        plt.ylabel('Trajectory (cm)', c=COLOR_TRAJ)
        plt.tick_params(axis="y", labelcolor=COLOR_TRAJ, length=0)

        ax2 = plt.gca().twinx()
        # VR velocity
        if agent == 'pod_leader':
            ax2.plot(df_j.time, df_j.PodLeader_Velocity_x, label='Velocity', c=COLOR_VEL)
            if len([s for s in df_j.columns if "EhmiStatus" in s]):
                ax2.fill_between(df_j.time, -150, 150, where=df_j.PodLeader_EhmiStatus=='green', color='green', alpha=0.1, transform=ax2.get_xaxis_transform())
                ax2.fill_between(df_j.time, -150, 150, where=df_j.PodLeader_EhmiStatus=='red', color='red', alpha=0.1, transform=ax2.get_xaxis_transform())
        elif agent == 'pod_follower':
            ax2.plot(df_j.time, df_j.PodFollower_Velocity_x, label='Velocity', c=COLOR_VEL)
            if len([s for s in df_j.columns if "EhmiStatus" in s]):
                ax2.fill_between(df_j.time, -150, 150, where=df_j.PodFollower_EhmiStatus=='green', color='cyan', alpha=0.1, transform=ax2.get_xaxis_transform())
                ax2.fill_between(df_j.time, -150, 150, where=df_j.PodFollower_EhmiStatus=='red', color='violet', alpha=0.1, transform=ax2.get_xaxis_transform())

        # calculated velocity based on trajectory
        ax2.plot(df_j.time.iloc[:-1],
                ((df_j[col_name] - df_j[col_name].shift(-1))/(df_j.time - df_j.time.shift(-1))).iloc[:-1], alpha=0.2)

        ax2.set_ylabel('Velocity (cm/s)', c=COLOR_VEL)
        ax2.tick_params(axis="y", labelcolor=COLOR_VEL, length=0)
        plt.title(self.title)
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(
                f'{self.fig_dir}/{agent}_traj_speed.png', bbox_inches='tight')
        plt.close()

    def plot_trajectory_per_trial(self, df):
        _ = plt.figure(figsize=(5, 4))

        df_j = df[df.sid == self.sid]
        if df_j.Ped_Location_x.iloc[0] < -100:
            plt.plot(df_j.time, -df_j.Ped_Location_x, alpha=0.3, c='red')
            plt.plot(df_j.time, -df_j.Ped_Location_x_smoothed, label='x(reversed)', c='red')
        else:
            plt.plot(df_j.time, df_j.Ped_Location_x, alpha=0.3, c='red')
            plt.plot(df_j.time, df_j.Ped_Location_x_smoothed, label='x', c='red')
        plt.plot(df_j.time, df_j.Ped_Location_y, alpha=0.3, c='green')
        plt.plot(df_j.time, df_j.Ped_Location_y_smoothed, label='y', c='green')

        plt.title(self.title)
        plt.xlim(0, df.time.max())
        plt.legend(loc='best')
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(
                f'{self.fig_dir}/ped_traj.png', bbox_inches='tight')
        plt.close()
    
    def plot_gaze_event_together(self, df):
        _ = plt.figure(figsize=(5, 4))

        df_j = df[df.sid == self.sid]
        plt.scatter(df_j.time[df_j.mask_fixation], np.ones(df_j.time[df_j.mask_fixation].shape[0]), label='fixation', color='red', s=1)
        plt.scatter(df_j.time[df_j.mask_saccade], np.ones(df_j.mask_saccade[df_j.mask_saccade].shape[0]), label='saccade', color='pink', s=1)
        plt.scatter(df_j.time[df_j.mask_smooth_pursuit_follower], np.ones(df_j.mask_smooth_pursuit_follower[df_j.mask_smooth_pursuit_follower].shape[0]), label='smooth pursuit follower', color='orange', alpha=0.5, s=1)
        plt.scatter(df_j.time[df_j.mask_smooth_pursuit_leader], np.ones(df_j.mask_smooth_pursuit_leader[df_j.mask_smooth_pursuit_leader].shape[0]), label='smooth pursuit leader', color='green', alpha=0.5, s=1)
        plt.scatter(df_j.time[df_j.mask_noise], np.ones(df_j.mask_noise[df_j.mask_noise].shape[0]), label='noise', color='gray', s=1)
        plt.title(self.title)
        plt.xlim(0, df.time.max())
        plt.legend(loc='best')
        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(
                f'{self.fig_dir}/gaze_event.png', bbox_inches='tight')
        plt.close()

    def plot_gaze_on_screen_overlay(self, df, every_n_frames=3):
        # read video
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print('Cannot open', self.video_path)
            return 
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        # plot
        for i in range(0, frame_count, every_n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((int(i), rgb))
            # print(f"Read frame {i}, shape={frame.shape}")
        
        n_row = math.ceil(len(frames) / 10)
        n_col = min(len(frames), 10)

        fig, axs = plt.subplots(n_row, n_col, figsize=(1.5*n_col, 1*(n_row)))
        axs = axs.flatten() if len(frames) > 1 else [axs]

        for ax, (i, fr) in zip(axs, frames):
            ax.imshow(fr)
            ax.set_title(f'frame {i}')
            pt = df.iloc[i][['px_u', 'px_v']].values.tolist() if df.iloc[i]['mask_attention'] else [None, None]
            ax.scatter(pt[0], pt[1], c='red', s=50)
            ax.axis('off')

        for ax in axs[len(frames):]:
            ax.axis('off')

        plt.tight_layout()

        if self.is_savefig:
            plt.savefig(
                f'{self.fig_dir}/gaze_on_screen.png', bbox_inches='tight')
        plt.close()

    ################## Preprocessing data ##################    
    def calculate_eye_angle(self, x, y):
        x, y = np.array(x), np.array(y)
        angle =  - np.arccos(x / np.sqrt(x**2 + y**2)) / np.pi * 180
        angle[np.where(y > 0)] = - 360 - angle[np.where(y > 0)]
        return angle
    
    def convert_rotation(self, angle, angle_min=-270):
        return np.floor(angle / angle_min) * 360 + angle
    
    def preprocess_eyegaze_data(self):
        objects_envs = [
            'ground', 'build', 'tree', 'tarppost', 'recycle_bin', 'lightpost', 'bench', 'stairs', 
            'pot', 'move', 'lamp', 'drinkingfountain', 'railing', 'chair', 'landscape', 'sign']
        objects_widget = ['isready', 'isreadyfix', 'instruction', 'mainmenu', 'widget']
        objects_podleader = ['pod_leader']
        objects_podfollower = ['pod_follower']
        objects_goal = ['goal']
        objects_neighbor = ['neighbor']
        object_dict = {
            'env': objects_envs, 'widget': objects_widget, 'pod_leader': objects_podleader, 
            'pod_follower': objects_podfollower, 'goal': objects_goal, 'neighbor': objects_neighbor}
        
        def which_cat(name, object_dict):
            for object_cat, object_list in object_dict.items():
                for obj in object_list:
                    if obj in name:
                        return object_cat 
            return name

        # get actor and component 
        self.df['HitActor'] = self.df['HitActor'].astype(str).apply(lambda x: x.lower())
        self.df.loc[:, ['HitObject']] = self.df['HitActor'].apply(lambda x: which_cat(x, object_dict))
        
        # preprocess head and eye gaze data 
        self.match_head_and_eyegaze()
        self.df.loc[self.df.ConfidenceValue == 0.0, ['HitObject', 'EyeGaze_Rotation_z']] = None
        self.df.HitObject = self.df.HitObject.replace({'nan': None})

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
    
    def check_starting_points_range(self, x_init, y_init, sid):
        '''
        45 degree: x ~ (-350, -250), y ~ (250, 350)
        90 degree: x ~ (-50, 50), y ~ (350, 450)
        135 degree: x ~ (350, 250), y ~ (250, 350)
        '''
        # get angle
        angle = self.dt[self.dt.sid==sid].angle.iloc[0]
        # range by angles 
        if angle == 45: 
            if -350 < x_init < -250 and 250 < y_init < 350:
                return True
            else:
                return False
        elif angle == 90:
            if -50 < x_init < 50 and 350 < y_init < 450:
                return True
            else:
                return False
        elif angle == 135: 
            if 250 < x_init < 350 and 250 < y_init < 350:
                return True
            else:
                return False
    
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

    def remove_bad_speed(self, speed_threshold=300): 
        bad_scenarios = self.df[self.df.Ped_Velocity_smoothed > speed_threshold].sid.unique()
        if bad_scenarios.shape[0]:
            print(f'Removing scenarios {bad_scenarios} (bad speed)')
            self.df = self.df[self.df.sid.apply(lambda x: x not in bad_scenarios)]

    def mask_out_nonfixation_gaze(self):
        df = self.df.copy()
        df['leader_yaw'] = np.rad2deg(np.atan2((df['PodLeader_Location_y'] - df['Ped_Location_y_smoothed']).values, 
                                                (df['PodLeader_Location_x'] - df['Ped_Location_x_smoothed']).values))
        df['follower_yaw'] = np.rad2deg(np.atan2((df['PodFollower_Location_y'] - df['Ped_Location_y_smoothed']).values, 
                                                (df['PodFollower_Location_x'] - df['Ped_Location_x_smoothed']).values))
        df.rename(columns={
            'Ped_Location_x_smoothed': 'loc_x', 'Ped_Location_y_smoothed': 'loc_y', 
            'Ped_Velocity_x_smoothed': 'vel_x', 'Ped_Velocity_y_smoothed': 'vel_y', 
            'Ped_Velocity_smoothed': 'vel_r', 'Ped_Velocity_Rotation_smoothed': 'vel_yaw', 
            'body_rotation_smoothed': 'body_yaw', 'Ped_Rotation_z_smoothed': 'head_yaw', 'EyeGaze_Rotation_z_smoothed': 'eye_yaw', 
            'head_vel_relative_smoothed': 'head_vel_relyaw', 'eye_vel_relative_smoothed': 'eye_vel_relyaw', 'eye_head_relative_smoothed': 'eye_head_relyaw',
            'PodLeader_Location_x': 'leader_x', 'PodLeader_Location_y': 'leader_y',
            'PodFollower_Location_x': 'follower_x', 'PodFollower_Location_y': 'follower_y',
            'PodLeader_Ped_CenterDistance_x': 'dist_pedleader_x', 'PodLeader_Ped_CenterDistance_y': 'dist_pedleader_y',
            'PodFollower_Ped_CenterDistance_x': 'dist_pedfollower_x', 'PodFollower_Ped_CenterDistance_y': 'dist_pedfollower_y',
            'PodLeader_EhmiStatus': 'ehmileader', 'PodFollower_EhmiStatus': 'ehmifollower',
            'HitObject_env': 'hit_env', 'HitObject_goal': 'hit_goal', 'HitObject_neighbor': 'hit_neighbor', 
            'HitObject_pod_leader': 'hit_leader', 'HitObject_pod_follower': 'hit_follower'
            }, inplace=True)
        

        via = 'loc'
        thd_fix=30
        thd_sac=100
        max_dur_blink=400
        min_dur_event=100
        smooth_method="g"
        filter_sigma=4
        filter_win=9
        filter_poly=2
        from utils.util_eye import extract_fix_sac_sp
        fix, sac, sp_leader, sp_follower, att, noise, eye_vel, mask_keep = extract_fix_sac_sp(df[[
            'GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 
            'time', 'ConfidenceValue',
            'eye_yaw', 'loc_x', 'loc_y', 'leader_x', 'leader_y', 'follower_x', 'follower_y']].values, 
		via=via, thd_fix=thd_fix, thd_sac=thd_sac, max_dur_blink=max_dur_blink, min_dur_event=min_dur_event,)
        self.df['mask_attention'] = att.astype(bool)
        self.df['mask_fixation'] = fix.astype(bool)
        self.df['mask_saccade'] = sac.astype(bool)
        self.df['mask_smooth_pursuit_leader'] = sp_leader.astype(bool)
        self.df['mask_smooth_pursuit_follower'] = sp_follower.astype(bool)
        self.df['mask_noise'] = noise.astype(bool)


def main(args):
    if args.psids is None:
        args.psids = [f.replace('.csv', '') for f in os.listdir(os.path.join(args.vrdata_dir, 'aligned')) if f.endswith('.csv')]
        args.psids.sort()
    for psid in args.psids:
        Interaction(
            vrdata_dir=args.vrdata_dir,
            dtdata_dir=args.dtdata_dir,
            videodata_dir=args.videodata_dir,
            psid=psid, is_savefig=args.is_savefig,
            is_visualize=args.is_visualize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--vrdata_dir", type=str, default='data/vrdata')
    parser.add_argument("--dtdata_dir", type=str, default='data/dtdata')
    parser.add_argument("--videodata_dir", type=str, default='data/videodata/videoclips')
    parser.add_argument("--psids", nargs='+', 
                        # default=['P1S1']
                       )

    parser.add_argument("--is_visualize", action='store_true', default=False)
    parser.add_argument("--is_savefig", action='store_true', default=True)
    
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
      