import os 
import pandas as pd


dts_qn_path_from = 'data/dts_qn_old.csv'
dts_qn_path_to = 'data/dts_qn.csv'

vrdata_dir = 'data/vrdata/processed_advanced'
dfs_path_ro = 'data/vrdata/dfs_combined.csv'

# keep dts_qn with only those in vrdata_dir through merge
dts_qn = pd.read_csv(dts_qn_path_from)
vrfiles = [f for f in os.listdir(vrdata_dir) if f.endswith('.csv')]
vrdata = pd.DataFrame([os.path.join(vrdata_dir, f) for f in vrfiles], columns=['filepath'])
vrdata[['pid', 'sid']] = vrdata['filepath'].str.extract(r'P(\d+)S(\d+)')
vrdata['pid'] = vrdata['pid'].astype(int)
vrdata['sid'] = vrdata['sid'].astype(int)

print('Before merging: \nvrdata.shape', vrdata.shape)
print('dts_qn.shape', dts_qn.shape)

dts_qn_keep = pd.merge(dts_qn, vrdata, on=['pid', 'sid'], how='inner')
dts_qn_keep = dts_qn_keep.drop(columns=['filepath'])
print('after merging: \ndts_qn_keep.shape', dts_qn_keep.shape)
dts_qn_keep.to_csv(dts_qn_path_to, index=False)


## Save dfs
dfs = pd.concat([pd.read_csv(os.path.join(vrdata_dir, f)) for f in vrfiles], ignore_index=True)
print('dfs.shape', dfs.shape)
print('Saving dfs to', dfs_path_ro)
dfs.to_csv(dfs_path_ro, index=False)