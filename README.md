# FMs for Pedestrian Behavior Prediction


### Installation
```
conda create -n dinov2 python=3.11
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet 
mim install mmcv-full==1.5.0
mim install mmsegmentation
conda install ftfy regex
```

```
pip install torch==2.1.2
pip install torchvision==0.16.2
pip install numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
mim install mmsegmentation
conda install ftfy regex
```

依旧是mmcv._ext找不到...
conda create -n test2 python=3.10
conda activate test2
pip install torch==2.8.0
pip install torchvision==0.23.0
pip install numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
mim install mmsegmentation
conda install ftfy regex



"""
依然是没有kernel image的问题...
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
mim install mmsegmentation
conda install ftfy regex
"""

pip install --upgrade mmcv-full==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.8.0/index.html


<!-- To test  -->
python -c "from mmseg.apis import init_model, inference_model"


optuna-dashboard sqlite:///outputs/optuna_20260120_234115/optuna_study.db

optuna-dashboard sqlite:///outputs/optuna_20260121_094048/optuna_study.db


### need to change 
problem = multi / binary 
visual_encoder = pixels / simple / clip 
features = use / not use additional numeric features

python -m training.train_fusion --annotations '../data/features/annotations.VRmulti_test_close.json' --visual-encoder pixels --dataset-type vr --hp-trials 150

python -m training.train_fusion --annotations '../data/features/annotations.VRmulti_test_close.json' --n-classes 5 --visual-encoder simple --dataset-type vr --hp-trials 100 --hp-dashboard-port 7070 

python -m training.train_fusion --annotations '../data/features/annotations.VRmulti_test_close.json' --visual-encoder clip --dataset-type vr --hp-trials 150 --hp-dashboard-port 6060



python -m training.train_fusion --annotations '../data/features/annotations.VRbin_test_close.json' --visual-encoder pixels --dataset-type vr --hp-trials 150 --hp-dashboard-port 1010

python -m training.train_fusion --annotations '../data/features/annotations.VRbin_test_close.json' --visual-encoder simple --dataset-type vr --hp-trials 150 --hp-dashboard-port 2020

python -m training.train_fusion --annotations '../data/features/annotations.VRbin_test_close.json' --visual-encoder clip --dataset-type vr --hp-trials 150 --hp-dashboard-port 3030


python -m training.train_fusion --annotations '../data/features/annotations.VRbin_test_close.json' --visual-encoder simple --dataset-type vr --hp-trials 150 --hp-dashboard-port 2020


### Prepare VR videoclips 

* Step 1: Extract video metadata for the raw videos, using `get_video_metadata` function in `extract_videolips.py`
* Step 2: Extract the first valid frame in each raw video, using visual inspection in `video.ipynb`
* Step 3: Extract the trial metadata for each video, using `get_trial_metadata` function in `extract_videoclips.py` 
* Step 4: Manually combine all trial metadata for each participant
* Step 5: Extract videclips based on trial metadata `extract_video_frames_from_clips` function in `extract_videoclips.py` 


### Prepare VR+video data in general

* `utils/preprocessing_basic.py`: filter out the familiarization phase, and filter out trials with basic errors (e.g., bad signals)
* `utils.align_vrdata2clip.py`: align the logged VR data to the video frequency
* `utils.preprocessing_advanced.py`: preprocessing all columns through interpolation, ...
* `utils/filtering.py`: check whether there are a lot of silence in each trial

### Prepare GroundVQA features