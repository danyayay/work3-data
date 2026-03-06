import os
import re
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd


class Interaction(object):
    def __init__(self, vrdata_dir, dtdata_dir, videodata_dir, videodata_trimmed_dir, psid, is_trim_video=False):
        self.pid, self.sid = re.match(r'P(\d+)S(\d+)', psid).groups()
        self.pid, self.sid = int(self.pid), int(self.sid)
        self.log_path_from = os.path.join(vrdata_dir, f'processed_advanced/{psid}.csv')
        self.log_path_to = os.path.join(vrdata_dir, f'filtered/{psid}.csv')
        self.dt_path = os.path.join(dtdata_dir, f'DT_P{self.pid}.csv')
        self.video_path = os.path.join(videodata_dir, f'P{self.pid}', f'{psid}.mp4')
        
        self.load_data()
        self.filter_data()

        print(f'Filtered data')
        print(f'    shape   = {self.df.shape}')
        print(f'    n_sids = {self.df.sid.unique()}\n')
        
        # process video
        if not os.path.exists(videodata_trimmed_dir):
            os.makedirs(videodata_trimmed_dir, exist_ok=True)
        self.video_path_from = os.path.join(videodata_dir, f'P{self.pid}', f'{psid}.mp4')
        self.video_path_to = os.path.join(videodata_trimmed_dir, f'{psid}.mp4')

        if is_trim_video and (hasattr(self, 'until_frame') or hasattr(self, 'from_frame') or hasattr(self, 'to_frame')):
            self.trim_video_by_frame()
        else:
            shutil.copy(self.video_path_from, self.video_path_to)
            print(f'No trimming needed for video, copied to {self.video_path_to}')

        if hasattr(self, 'last_time'):
            self.df.time = self.df.time - self.last_time # reset time to start from 0 after trimming
        self.save_filtered_data()
        
    def load_data(self):
        print('\n#############################')
        print(f'Reading {self.log_path_from}')
        self.df = pd.read_csv(self.log_path_from)

        # formal dt 
        self.dt = pd.read_csv(self.dt_path)
        self.dt['sid'] = self.dt.index + 1
        self.dt['pid'] = self.pid
        self.dt.drop(columns=['---'], inplace=True)
        self.dt = self.dt[['pid', 'sid', 'eHMI', 'yielding', 'angle', 'traffic flow']]

    def filter_data(self):
        self.trim_white_to_green() 
        self.get_pod_trigger_points()
        self.remove_pod_already_left()
        
    def save_filtered_data(self):
        # save the filtered data 
        if not os.path.exists(os.path.dirname(self.log_path_to)):
            os.makedirs(os.path.dirname(self.log_path_to), exist_ok=True)
        self.df.to_csv(self.log_path_to, index=False)
        print(f'Saved filtered data to {self.log_path_to}')
    
    ################# Trim video #################
    # def get_video_new_video_time(self):
    #     command = [
    #         "ffprobe", 
    #         "-v", "error", 
    #         "-select_streams", "v:0", 
    #         "-show_frames", 
    #         "-print_format", "csv", 
    #         self.video_path_from
    #     ]
    #     import io
    #     result = subprocess.run(command, capture_output=True, text=True)
        
    #     # Check for errors before processing
    #     if result.returncode == 0:
    #         # Convert the string output into a DataFrame
    #         df = pd.read_csv(io.StringIO(result.stdout), header=None)
    #     else:
    #         print(f"Error: {result.stderr}")
    #     print(df)

    #     return df.iloc[self.new_start_frame, 5], df.iloc[-1, 5]

    def trim_video_by_frame(self):
        start_time = self.last_time if hasattr(self, 'last_time') else self.df.time.iloc[0]
        end_time = self.df.time.iloc[-1]

        command = [
            "ffmpeg",
            "-y",
            "-i", self.video_path_from,
            "-ss", str(start_time),
            "-to", str(end_time), 
            self.video_path_to
        ]

        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            print(f"ffmpeg failed for P{self.pid}S{self.sid}: returncode={completed.returncode}")
            print(completed.stderr.decode(errors='ignore'))
        else:
            print(f"Clipped frames {start_time}–{end_time} saved to {self.video_path_to}")

    ################## Filter data ##################
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
        diff_dist = df_j.dist_leader2center.diff()
        idx = diff_dist[diff_dist != 0].index[1]

        bool_velocity_near_zero = (df_j.ped_vel.abs() < 3).to_numpy()
        _, end, lengths = self.get_consecutive_trues(bool_velocity_near_zero, verbose=True)
        idx_before = np.where(end < idx)[0]
        if len(idx_before) > 0:
            end_last_idx = idx_before[-1]
            last_idx = end[end_last_idx]
            return last_idx - int(lengths[end_last_idx] / 2)
        else:
            return None
        
    def trim_white_to_green(self):
        self.trim_wrong_head_rotation()
        self.trim_wrong_starting_points()

    def trim_wrong_head_rotation(self):
        lengths = self.df.groupby('sid')['head_rot_z'].apply(lambda x: len(self.get_consecutive_trues(x > 0)))
        if (lengths > 0).sum() > 0:
            print(lengths[lengths > 0].to_dict())
        for i in lengths[lengths > 0].index.to_numpy():
            df_i = self.df[self.df.sid == i]
            idx_new = self.find_nearest_zero_velocity_before_pod_moving(df_i)
            if idx_new is not None and idx_new > 1:
                self.until_frame = int(idx_new)
                self.last_time = df_i.time.iloc[int(idx_new) - 1]
                print(f'\nTrimming sid {i} at index {idx_new} (white to green - rotation)')
                # bc of df.drop_duplicates(), there are missing index
                self.df = self.df.drop(index=[i for i in df_i.index.to_numpy() if df_i.index[0] <= i < df_i.index[idx_new]])

    def trim_wrong_starting_points(self):
        df_i = self.df.copy()
        # based on starting position 
        if not self.check_starting_points_range(
            df_i.ped_loc_x.iloc[0], df_i.ped_loc_y.iloc[0], self.sid):
            idx_new = self.find_nearest_zero_velocity_before_pod_moving(df_i)
            if idx_new is not None and idx_new > 1:
                self.until_frame = int(idx_new)
                self.last_time = df_i.time.iloc[int(idx_new) - 1]
                print(f'Trimming sid {self.sid} at index {idx_new} (white to green - starting positions)')
                # bc of df.drop_duplicates(), there are missing index
                self.df = self.df.drop(index=[i for i in df_i.index.to_numpy() if df_i.index[0] <= i < df_i.index[idx_new]])
        # based on last zero velocity
        else:
            idx_new = self.find_nearest_zero_velocity_before_pod_moving(df_i)
            if idx_new is not None and idx_new > 1:
                self.until_frame = int(idx_new)
                self.last_time = df_i.time.iloc[int(idx_new) - 1]
                print('Trimming sid', self.sid, 'at index', idx_new, '(white to green - last zero velocity)')
                self.df = self.df.drop(index=[i for i in df_i.index.to_numpy() if df_i.index[0] <= i < df_i.index[idx_new]])
        
        # based on eye gazing object
        start_name = None
        for c in df_i.hit_object.astype(str).unique():
            if c != 'nan':
                if 'start' in c:
                    start_name = c
                    break
        if start_name:
            idx_new3 = df_i[df_i.hit_object.astype(str) == start_name].index.to_numpy()
            idx_new3_start, idx_new3_end = idx_new3[0], idx_new3[-1]
            time_pasted = df_i.time.loc[idx_new3_end] - df_i.time.iloc[0]
            time_remained = df_i.time.iloc[-1] - df_i.time.loc[idx_new3_start]
            trial_start_idx = df_i.index[0]
            if time_pasted < time_remained: # delete forward
                idx_new3_start = None
                # delete forward 
                if idx_new is not None:
                    if df_i.index[idx_new] < idx_new3_end:
                        idx_new3_start = df_i.index[idx_new] + 1
                else:
                    idx_new3_start = df_i.index[0]
                if idx_new3_start is not None and idx_new3_start > 1: 
                    self.from_frame = int(idx_new3_start - trial_start_idx)
                    self.to_frame = int(idx_new3_end - trial_start_idx)
                    print('Trimming sid', self.sid, 'from', idx_new3_start-trial_start_idx, 'to', idx_new3_end-trial_start_idx, '(white to green - gazing object)')
                    self.df = self.df.drop(index=np.arange(idx_new3_start, idx_new3_end+1).tolist())
            else: # delete backward 
                idx_new3_end = df_i.index[-1]
                self.from_frame = int(idx_new3_start - trial_start_idx)
                self.to_frame = int(idx_new3_end)
                print('Trimming sid', self.sid, 'from', idx_new3_start-trial_start_idx, 'to the end (white to green - gazing object)')
                self.df = self.df.drop(index=np.arange(idx_new3_start, idx_new3_end+1).tolist())
    
    def remove_pod_already_left(self):
        is_there_podleader = self.df.groupby('sid')['dist_leader2center'].transform(lambda x: (x < 2000).sum() > 0)
        if (~is_there_podleader).sum() > 0:
            print(f'\nRemoving sids {self.df[~is_there_podleader].sid.unique()} (no pod)')
        self.df = self.df[is_there_podleader]

    def remove_mistakenly_triggered_sids(self):
        self.get_mistakenly_triggered_sids()
        self.df = self.df[self.df.sid.apply(lambda x: x not in self.list_mistakenly_triggered)]
        
    def get_mistakenly_triggered_sids(self):
        self.list_mistakenly_triggered = []
        for j, idx in self.podleader_trigger_points.items():
            if idx == 1:
                self.list_mistakenly_triggered.append(str(j))
        print(f'Removing sids {self.list_mistakenly_triggered} (mistakenly triggered)')

    def get_pod_trigger_points(self):
        self.podleader_trigger_points = dict()
        self.podleader_trigger_index = dict()

        for j in self.df.sid.unique():
            df_j = self.df[self.df.sid == j]
            idx_base = df_j.index.to_numpy()[0]

            diff_dist = df_j.leader_loc_x.diff().reset_index(drop=True)
            idx = diff_dist[diff_dist != 0].index[1]
            j = int(j)
            self.podleader_trigger_points[j] = idx
            self.podleader_trigger_index[j] = idx_base + idx

        print('\nFormal experiment:')
        print(self.podleader_trigger_points)
        print(self.podleader_trigger_index)

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
      

def main(args):
    if args.psids is None:
        args.psids = [f.replace('.csv', '') for f in os.listdir(os.path.join(args.vrdata_dir, 'aligned')) if f.endswith('.csv')]
        args.psids.sort()
    for psid in args.psids:
        Interaction(
            vrdata_dir=args.vrdata_dir,
            dtdata_dir=args.dtdata_dir,
            videodata_dir=args.videodata_dir,
            videodata_trimmed_dir=args.videodata_trimmed_dir,
            psid=psid, 
            is_trim_video=args.is_trim_video
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--vrdata_dir", type=str, default='data/vrdata')
    parser.add_argument("--dtdata_dir", type=str, default='data/dtdata')
    parser.add_argument("--psids", nargs='+', 
                        # default=['P23S11']
                       )
    
    parser.add_argument("--videodata_dir", type=str, default='data/videodata/videoclips')
    parser.add_argument("--videodata_trimmed_dir", type=str, default='data/videodata/videoclips_ready')
    parser.add_argument("--is_trim_video", type=bool, default=True)
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
      