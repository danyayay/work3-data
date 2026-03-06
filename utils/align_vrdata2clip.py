"""
Align logged trajectory/time-series data to video frames using decord.

This script reads a logged CSV (roughly 20 Hz) and a video file (e.g. 29.97fps),
interpolates numeric columns from the log onto the video frame timestamps, and
saves an aligned CSV where each row corresponds to a video frame.

Usage (from shell):
  python align_vrdata.py --video path/to/video.mp4 --log path/to/dfs_combined.csv

The script will try to auto-detect the timestamp column in the CSV. If your
timestamp column has a non-standard name, pass --time-col TIMECOL. Output CSV
is written next to the input log with suffix `_aligned.csv` by default.
"""

from __future__ import annotations

import os
import re
import argparse
import logging
from typing import Optional
import numpy as np
import pandas as pd
import subprocess

from utils.transform_eyegaze import cal_eyegaze_on_screen


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
	"""Try to infer which column is the timestamp/time column.

	Heuristics:
	- look for common names containing 'time' or 'timestamp'
	- otherwise choose the numeric column that is strictly increasing
	"""
	candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower() or c.lower() == "ts"]
	if candidates:
		return candidates[0]

	# find numeric columns that are monotonic increasing
	for c in df.select_dtypes(include=[np.number]).columns:
		series = pd.to_numeric(df[c], errors="coerce").dropna()
		if len(series) >= 2 and (series.diff().fillna(0) >= 0).all():
			return c

	return None


def _normalize_time_to_seconds(series: pd.Series) -> np.ndarray:
	"""Convert a numeric time series to seconds and make it relative (start at 0).

	Detects epoch in microseconds/milliseconds and converts to seconds.
	"""
	t = pd.to_numeric(series, errors="coerce").astype(float)
	if t.isna().all():
		raise ValueError("Time column could not be converted to numeric values")

	mean_val = t.dropna().mean()
	# heuristics for epoch scale
	if mean_val > 1e14:  # extremely large: probably microseconds
		t = t / 1e6
	elif mean_val > 1e11:  # very large: likely milliseconds
		t = t / 1e3
	# else: already seconds or relative seconds

	# make relative to first timestamp
	t = t - t.iloc[0]
	return t.values


def align_log_to_video(
	videodata_dir: str,
	vrdata_dir: str,
	psid: str, 
) -> str:
	"""Align the log CSV to the video frames and write an output CSV.

	Returns the path to the written CSV.
	"""
	## prepare video data
	pid, sid = re.match(r"P(\d+)S(\d+)", psid).groups()
	video_path = f"{videodata_dir}/P{pid}/P{pid}S{sid}.mp4"
	vrdata_dir_from = f"{vrdata_dir}/processed_basic/"
	
	def _get_frame_timestamps_ffprobe(path: str):
		# get per-frame timestamps (best_effort_timestamp_time)
		cmd_frames = [
			"ffprobe", "-v", "error", "-select_streams", "v:0",
			"-show_entries", "frame=best_effort_timestamp_time",
			"-of", "csv=p=0", path
		]
		p = subprocess.run(cmd_frames, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		if p.returncode != 0:
			raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
		lines = [ln.strip().rstrip(',') for ln in p.stdout.splitlines() if ln.strip()]
		if not lines:
			raise RuntimeError("ffprobe returned no frame timestamps")
		times = np.array([float(x) for x in lines], dtype=float)

		# get r_frame_rate for reference (may be fractional)
		cmd_fps = [
			"ffprobe", "-v", "error", "-select_streams", "v:0",
			"-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", path
		]
		p2 = subprocess.run(cmd_fps, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		fps = None
		if p2.returncode == 0:
			fps_str = p2.stdout.strip()
			if "/" in fps_str:
				num, den = fps_str.split("/")
				try:
					fps = float(num) / float(den)
				except Exception:
					fps = None
			else:
				try:
					fps = float(fps_str)
				except Exception:
					fps = None
		return times, fps

	target_times, video_fps = _get_frame_timestamps_ffprobe(video_path)
	frame_count = len(target_times)
	logging.info("Video: %s, frames=%d, fps=%.6g", video_path, frame_count, video_fps)
	# target_times = np.concatenate([vr.get_frame_timestamp([0])[:,0], 
	# 							vr.get_frame_timestamp(np.arange(frame_count))[:,1]])  
	
    ## find the logged csv
	vr_files = [f for f in os.listdir(vrdata_dir_from) if f.startswith(f"P{pid}_") and f.endswith('.csv')]
	assert len(vr_files) == 1, f"Expected exactly one log CSV for P{pid}S{sid} in {vrdata_dir_from}, but found {len(vr_files)}: {vr_files}"

	vr_path = os.path.join(vrdata_dir_from, vr_files[0])
	df = pd.read_csv(vr_path)
	df = df[(df['pid'] == int(pid)) & (df['sid'] == int(sid))]
	time_col = 'time'
	df = df.copy()
	df = df.sort_values(time_col).reset_index(drop=True)

	# first check if the length of video and vr log file roughly match
	video_dur = frame_count / video_fps
	log_dur = df.time.iloc[-1]
	# assert abs(video_dur - log_dur) < 1, f"P{pid}S{sid}: Video duration ({video_dur:.1f}s) and log duration ({log_dur:.1f}s) differ by more than 1 second. Please check if the correct video and log are being aligned."
	if abs(video_dur - log_dur) >= 0.5:
		# write it to a file
		with open("alignment_warnings.txt", "a") as f:
			f.write(f"Warning: P{pid}S{sid}: Video duration ({video_dur:.1f}s) and log duration ({log_dur:.1f}s) differ by more than 0.8 second. Please check if the correct video and log are being aligned.\n")
		return

	# get eyegaze data 
	df_gaze = cal_eyegaze_on_screen(df)
	df = pd.merge(df, df_gaze[['px_u', 'px_v']], left_index=True, right_index=True, how='left')

	## align logged csv to video data
	# target_cols = ['Ped_Location_x_smoothed', 'Ped_Location_y_smoothed', \
	# 			'Ped_Velocity_x_smoothed', 'Ped_Velocity_y_smoothed', 'Ped_Velocity_smoothed',\
	# 			'PodLeader_Location_x', 'PodFollower_Location_x', \
	# 			'Ped_Rotation_x', 'Ped_Rotation_y', 'Ped_Rotation_z',\
	# 			'GazeDirection_x', 'GazeDirection_y', 'GazeDirection_z', 'px_u', 'px_v']
	
	target_df = pd.DataFrame({"video_time": target_times, "frame": np.arange(frame_count)})

	# interpolate numeric columns using numpy.interp for speed and stability
	times_arr = df[time_col].values
	# Ensure times are strictly increasing for interp
	if not (np.all(np.diff(times_arr) >= 0)):
		# as a fallback, sort
		order = np.argsort(times_arr)
		times_arr = times_arr[order]

	for col in df.columns:
		## if column is numeric: interpolate with np.interp (fast and stable)
		if df[col].dtype in [int, float]:
			vals = pd.to_numeric(df[col], errors="coerce").values
			# if sorting was applied above, reorder vals accordingly
			if 'order' in locals():
				vals = vals[order]
			# np.interp will use first/last values for out-of-bounds (constant extrapolation)
			if col == 'px_u' or col == 'px_v':
				# interp the original gaze data first
				valid_mask = ~np.isnan(vals)
				vals = np.interp(times_arr, times_arr[valid_mask], vals[valid_mask])
			target_df[col] = np.interp(target_times, times_arr, vals)

		## if column is non-numeric, we can optionally do a nearest merge (merge_asof) after interpolation
		elif df[col].dtype in [str, object, bool]: 
			merge_src = df[[col, 'time']].rename(columns={'time': f"log_time"})
			target_df = pd.merge_asof(target_df, merge_src, left_on="video_time", right_on="log_time", direction="nearest").drop(columns=["log_time"])

		else:
			print(f"Warning: Column {col} has unsupported dtype {df[col].dtype}, skipping interpolation and merge.")

	target_df.drop(columns=['video_time', 'frame'], inplace=True)  # keep only the interpolated columns, can add back frame number if needed
	target_df['pid'] = int(pid)
	target_df['sid'] = int(sid)

	# add provenance columns
	# target_df["source_video"] = video_path
	# target_df.rename(columns={
	# 	'Ped_Rotation_x': 'rot_x', 'Ped_Rotation_y': 'rot_y', 'Ped_Rotation_z': 'rot_z',\
	# 	'GazeDirection_x': 'gaze_x', 'GazeDirection_y': 'gaze_y', 'GazeDirection_z': 'gaze_z',\
	# 	}, inplace=True)
	# TODO: fill the missing values if needed especially for gaze direction
	# TODO: check how to process the follower positions when missing in our previous work

	# write output
	vrdata_dir_to = vrdata_dir + '/aligned/'
	if not os.path.exists(vrdata_dir_to):
		os.makedirs(vrdata_dir_to)
	output_csv = vrdata_dir_to + psid + ".csv"

	target_df.to_csv(output_csv, index=False)
	logging.info("    Saved to %s", output_csv)
	return output_csv


def _cli():
	parser = argparse.ArgumentParser(description="Align a log CSV to video frames, saving files in combined/ to aligned/")
	parser.add_argument("--videodata_dir", default='data/videodata/videoclips', help="Path to the video directory")
	parser.add_argument("--vrdata_dir", default='data/vrdata', help="Path to the logged CSV (e.g., dfs_combined.csv)")
	parser.add_argument("--pid", type=int, help="Participant ID to align (e.g., 1 for P1)")
	parser.add_argument("--verbose", action="store_true", default=True)
	args = parser.parse_args()

	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

	psid_list = []
	for folder in os.listdir(args.videodata_dir):
		folder_path = os.path.join(args.videodata_dir, folder)
		if os.path.isdir(folder_path) and folder.startswith("P"):
			for f in os.listdir(folder_path):
				if f.endswith('.mp4'):
					psid_list.append(f.split('.')[0])
	psid_list.sort()  # sort to ensure consistent order
	psid_list = ['P10S10']  # Example participant and session ID
	for psid in psid_list:
		align_log_to_video(args.videodata_dir, args.vrdata_dir, psid)


if __name__ == "__main__":
	_cli()

