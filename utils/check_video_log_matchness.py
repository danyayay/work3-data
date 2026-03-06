import os 
import cv2
import pandas as pd
import subprocess


def count_files(dir):
    dir = 'data/videodata/videoclips/'
    files = []
    for d in os.listdir(dir):
        if d.startswith('P'):
            par = []
            for f in os.listdir(os.path.join(dir, d)):
                if f.endswith('.mp4'):
                    files.append(f)
                    par.append(f)
            print(d, len(par))
    print(len(files))


def get_precise_video_duration(filename):
    # -select_streams v selects only the video stream
    # -show_entries stream=duration gets the microsecond-accurate time
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", filename
    ]
    
    output = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(output)


def plot_video_log_dur_diff(df, title='video_log_matchness'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.histplot(df['dur_diff'], bins=30, kde=True)
    plt.title('Distribution of Duration Differences (Video - Log)')
    plt.xlabel('Duration Difference (seconds)')
    plt.ylabel('Frequency')
    plt.savefig(f'results/{title}.png')


def video_log_matchness(video_dir='data/videodata/videoclips', log_dir='data/vrdata/combined'):
    df = pd.DataFrame(columns=['pid', 'sid', 'psid', 'video_dur', 'log_dur'])

    for pid in range(1, 52):
        for sid in range(1, 13):
            if video_dir == 'data/videodata/videoclips':
                video_path = os.path.join(video_dir, f'P{pid}', f'P{pid}S{sid}.mp4')
            elif video_dir == 'data/videodata/videoclips_ready':
                video_path = os.path.join(video_dir, f'P{pid}S{sid}.mp4')
            else:
                raise ValueError(f'Unknown video_dir: {video_dir}')

            if os.path.exists(video_path):
                # get video duration
                video_dur = get_precise_video_duration(video_path)

                if log_dir == 'data/vrdata/combined':
                    log_path = os.path.join(log_dir, f'P{pid}S{sid}.csv')
                    # get log duration
                    log_path = [f for f in os.listdir(log_dir) if f.startswith(f'P{pid}_') and f.endswith('.csv')]
                    assert len(log_path) == 1, f'Log file is not unique for P{pid}'
                    log_path = os.path.join(log_dir, log_path[0])
                elif log_dir == 'data/vrdata/filtered':
                    log_path = os.path.join(log_dir, f'P{pid}S{sid}.csv')
                if os.path.exists(log_path):
                    csv = pd.read_csv(log_path)
                    csv_sid = csv.loc[csv['sid'] == sid]
                    log_dur = csv_sid.time.max() - csv_sid.time.min()
                else:
                    print(f'Log file does not exist for P{pid}')
                    log_dur = None
                
                if df.shape[0] == 0:
                    df = pd.DataFrame({'pid': [pid], 'sid': [sid], 'psid': [f'P{pid}S{sid}'], 'video_dur': [video_dur], 'log_dur': [log_dur]})
                else: 
                    df = pd.concat([df, pd.DataFrame({'pid': pid, 'sid': sid, 'psid': f'P{pid}S{sid}', 'video_dur': video_dur, 'log_dur': log_dur}, index=[0])], ignore_index=True)
    df['dur_diff'] = df['video_dur'] - df['log_dur']
    title = 'video_log_matchness__' + video_dir.split('/')[-1] + '__' + log_dir.split('/')[-1]

    df.to_csv(f'results/{title}.csv', index=False)
    plot_video_log_dur_diff(df, title=title)


video_log_matchness(
    video_dir='data/videodata/videoclips_ready', 
    log_dir='data/vrdata/filtered')