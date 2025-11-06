import os
import glob
import cv2
import pandas as pd
import subprocess


def get_video_metadata(raw_video_dir):
    videofilenames = [f for f in os.listdir(raw_video_dir) if f.endswith('.mp4')]
    csvfilenames = [f.rstrip('.csv') for f in os.listdir(f'{raw_video_dir}/video_metadata/') if f.endswith('.csv')]
    for filename in videofilenames:
        filename = filename.strip('.mp4')
        source_video_filepath = f"{raw_video_dir}/{filename}.mp4"
        source_csv_filepath = f"{raw_video_dir}/video_metadata/{filename}.csv"
        if filename in csvfilenames:
            print(f"Video metadata for {source_video_filepath} already exists at {source_csv_filepath}, skipping extraction.")
        else:
            subprocess.run(f"ffprobe -select_streams v:0 -show_frames -print_format csv {source_video_filepath} > {source_csv_filepath}", shell=True)


def get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid, start, forward=None, start_sid=None):
    vr_file = glob.glob(f"{vr_data_dir}/P{pid}_*")
    vr_df = pd.read_csv(vr_file[0])

    vr_df_final = pd.read_csv(f"{vr_data_dir}/../dfs_combined.csv")
    vr_df_final = vr_df_final[vr_df_final['pid'] == pid]

    sids = [int(i) for i in vr_df['Scenario'].unique() if not i.startswith('F')]
    sids = [sid for sid in sids if sid in vr_df_final['sid'].unique()]
    sids.sort()
    if start_sid is not None:
        sids = sids[sids.index(start_sid):]
    print(sids)
    
    # find the timeelapsed point for each trial
    time_df = pd.DataFrame(columns=['sid', 'start', 'start_abs', 'end_abs', 'start_rel', 'end_rel', 'start_time', 'end_time'])
    for i, sid in enumerate(sids):
        df_sid = vr_df[vr_df['Scenario'] == str(sid)]
        start_sid = df_sid.iloc[0]['TimeElapsed']
        end_sid = df_sid.iloc[-1]['TimeElapsed']
        if forward and sid in forward.keys():
            start_sid += 0.03*forward[sid]
            end_sid += 0.03*forward[sid]
        if i == 0:
            line = {'sid': sid, 'file': source_video_name, 'start': start, 
                    'start_abs': start_sid, 'end_abs': end_sid,
                    'start_rel': start_sid - start_sid, 'end_rel': end_sid - start_sid}
            time_df = pd.DataFrame([line])
        else:
            line = {'sid': sid, 'file': source_video_name, 'start': start, 
                    'start_abs': start_sid, 'end_abs': end_sid, 
                    'start_rel': start_sid - time_df.iloc[0]['start_abs'], 
                    'end_rel': end_sid - time_df.iloc[0]['start_abs']}
            time_df = pd.concat([time_df, pd.DataFrame([line])], ignore_index=True)
    if not os.path.exists(f"{raw_video_dir}/trial_metadata"):
        os.makedirs(f"{raw_video_dir}/trial_metadata")
    time_df.to_csv(f"{raw_video_dir}/trial_metadata/{source_video_name}.csv", index=False)


def extract_video_clips(raw_video_dir, video_clip_dir, pid):
    df_tms = pd.read_csv(f'{raw_video_dir}/trial_metadata/P{pid}.csv')
    files = df_tms['file'].unique()

    for source_video_name in files:
        print('Processing video:', source_video_name)
        df_tm = df_tms[df_tms['file'] == source_video_name]
        start = df_tm['start'].values[0]
        sids = df_tm['sid'].unique()
        sids.sort()
        for sid in sids:
            start_time = df_tm[df_tm['sid'] == sid]['start_rel'].values[0] + start
            end_time = df_tm[df_tm['sid'] == sid]['end_rel'].values[0] + start

            # output video
            output_path = f"{video_clip_dir}/P{pid}/P{pid}S{sid}.mp4"
            if not os.path.exists(f"{video_clip_dir}/P{pid}"):
                os.makedirs(f"{video_clip_dir}/P{pid}")
            command = [
                "ffmpeg",
                "-y",
                "-i", f'{raw_video_dir}/videos/{source_video_name}.mp4',
                "-ss", str(start_time),
                "-to", str(end_time), 
                output_path
            ]

            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if completed.returncode != 0:
                print(f"ffmpeg failed for P{pid}S{sid}: returncode={completed.returncode}")
                print(completed.stderr.decode(errors='ignore'))
            else:
                print(f"Clipped frames {start_time}–{end_time} saved to {output_path}")


def extract_video_frames_from_clips(videoclip_dir, videoclip_name, videoframe_dir):
    filepath = f'{videoclip_dir}/{videoclip_name}.mp4'
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print('Cannot open', filepath)
        return []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        print('Video has 0 frames')
        return []
    
    frames = []
    videoframe_dir = os.path.join(videoframe_dir, videoclip_name)
    os.makedirs(videoframe_dir, exist_ok=True)

    for i in range(total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((int(i), rgb))
        if videoframe_dir:
            fname = os.path.join(videoframe_dir, f'frame_{i:06d}.jpg')
            cv2.imwrite(fname, frame)

    cap.release()
    return frames

    
# for pid in pids:
#     # get the frame file for time mapping
#     subprocess.run(f"ffprobe -select_streams v:0 -show_frames -print_format csv {raw_video_dir}P{pid}.mp4 > {raw_video_dir}frames/frames_{pid}.csv", shell=True)
#     frame_df = pd.read_csv(f"{video_clip_dir}frames/frames_{pid}.csv")
#     frame_df = frame_df.iloc[:, 5] # get only the pkt_pts_time column
#     frame_df = frame_df.reset_index()
#     frame_df.columns = ['frame_idx', 'time']

#     # get the vr files for trial selection
#     vr_files = glob.glob(f"{vr_data_dir}P{pid}_*.csv")
#     vr_df = pd.read_csv(f"{vr_data_dir}{vr_files[0]}.csv")
#     # find the timeelapsed point for each trial
#     time_df = pd.DataFrame(columns=['sid', 'time_start_abs', 'time_end_abs', 'time_start_rel', 'time_end_rel'])
#     for sid in range(4, 6):
#         sid = str(sid)
#         dff = vr_df[vr_df['Scenario'] == sid]
#         if sid == '1':
#             start_sid = dff.iloc[0]['TimeElapsed']
#             end_sid = dff.iloc[-1]['TimeElapsed']
#             line = {'sid': sid, 'time_start_abs': start_sid, 'time_end_abs': end_sid, 
#                     'time_start_rel': start_sid - start_sid, 'time_end_rel': end_sid - start_sid}
#             time_df = time_df.append(line, ignore_index=True)
#         else:
#             start_sid = dff.iloc[0]['TimeElapsed']
#             end_sid = dff.iloc[-1]['TimeElapsed']
#             line = {'sid': sid, 'time_start_abs': start_sid, 'time_end_abs': end_sid, 
#                     'time_start_rel': start_sid - time_df.iloc[0]['time_start_abs'], 
#                     'time_end_rel': end_sid - time_df.iloc[0]['time_start_abs']}
#             time_df = time_df.append(line, ignore_index=True)
#         start = start_time_dict[pid]
#         start_time = time_df[time_df['sid'] == sid]['time_start_rel'].values[0] + start
#         end_time = time_df[time_df['sid'] == sid]['time_end_rel'].values[0] + start

#         # output video
#         output_path = f"{video_clip_dir}P{pid}S{sid}.mp4"
#         command = [
#             "ffmpeg",
#             "-i", f'{raw_video_dir}P{pid}.mp4',
#             "-ss", str(start_time),
#             "-to", str(end_time), 
#             output_path
#         ]

#         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
#         print(f"Clipped frames {start_time}–{end_time} saved to {output_path}")
            

if __name__ == "__main__":

    # pids = [1, 2, 3, 6, 7, 8, 13, 19, 22, 28, 32, 35, 42, 43, 45]
    # pids = [1, 13, 22]

    raw_video_dir = 'data/rawvideos'
    video_clip_dir = 'data/videoclips'
    video_frame_dir = 'data/videoframes'
    vr_data_dir = 'data/vrdata/combined'

    # first extract video metadata if not already done
    # get_video_metadata(raw_video_dir)

    # second prepare the trial metadata files and finally extract the video clips
    # source_video_name = 'P1'
    # pid = 1
    # start = 299.124300
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, 
    #                    pid=pid, start=start, forward={3: 1, 12: 1})
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid)
    # extract_video_frames_from_clips(video_clip_dir, 'P22S1', video_frame_dir)
    
    # source_video_name = 'P2'
    # pid = 2
    # start = 201.790367
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # source_video_name = 'P3'
    # pid = 3
    # start = 2.646100
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, forward={10: 2, 11: 1}, start_sid=2)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # source_video_name = 'P6'
    # pid = 6
    # start = 197.575067
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid)
    
    # source_video_name = 'P7'
    # pid = 7
    # start = 344.320567
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid)
    
    # source_video_name = 'P8'
    # pid = 8
    # start = 249.372233
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, forward={10: 1}, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid)

    # source_video_name = 'P13'
    # pid = 13
    # start = 229.027867
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, forward={6: 1})
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid)

    # source_video_name = 'P19'
    # pid = 19
    # start = 336.181600
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # source_video_name = 'P22'
    # pid = 22
    # start = 196.238100
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
     
    # source_video_name = 'P28'
    # pid = 28
    # start = 390.974267
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, forward={11: 1})
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid)
    
    # source_video_name = 'P32'
    # pid = 32
    # start = 261.045433
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid)
    
    # source_video_name = 'P35'
    # pid = 35
    # start = 350.417700
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # source_video_name = 'P42'
    # pid = 42
    # start = 295.963100
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # source_video_name = 'P43'
    # pid = 43
    # start = 238.788700
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # source_video_name = 'P45'
    # pid = 45
    # start = 361.009833
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # pid = 4
    # source_video_name = 'P4-1'
    # start = 331.084133
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # source_video_name = 'P4-2'
    # start = 9.538800
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, start_sid=5)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 

    # pid = 5
    # source_video_name = 'P5-2'
    # start = 227.554433
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, start_sid=8)
    # source_video_name = 'P5-3'
    # start = 17.515378
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 
    
    # pid = 9
    # source_video_name = 'P9-1'
    # start = 106.079033
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 

    # pid = 10
    # source_video_name = 'P10-2'
    # start = 36.156933
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, forward={2: 1, 6: 1, 10: 1})
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 

    # pid = 11
    # source_video_name = 'P11-2'
    # start = 261.629567
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start)
    # source_video_name = 'P11-3'
    # start = 13.929200
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, start_sid=3)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 

    # pid = 12
    # source_video_name = 'P12-1'
    # start = 428.708967
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, start_sid=3)
    # source_video_name = 'P12-2'
    # start = 46.168967
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, start_sid=2)
    # source_video_name = 'P12-3'
    # start = 11.092767 
    # get_trial_metadata(raw_video_dir, source_video_name, vr_data_dir, pid=pid, start=start, start_sid=10)
    # extract_video_clips(raw_video_dir, video_clip_dir, pid=pid) 