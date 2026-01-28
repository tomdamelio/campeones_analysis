import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# Configuration
SUBJECT = '27'
SESSION = 'vr'

# Combined Runs Config
RUNS_CONFIG = [
    {'id': '002', 'acq': 'a', 'block': 'block1', 'task': '01'},
    {'id': '003', 'acq': 'a', 'block': 'block2', 'task': '02'},
    {'id': '004', 'acq': 'a', 'block': 'block3', 'task': '03'},
    {'id': '006', 'acq': 'a', 'block': 'block4', 'task': '04'},
    {'id': '007', 'acq': 'b', 'block': 'block1', 'task': '01'},
    {'id': '009', 'acq': 'b', 'block': 'block3', 'task': '03'},
    {'id': '010', 'acq': 'b', 'block': 'block4', 'task': '04'},
]

BASE_PATH = r'c:\Users\tdamelio\Desktop\campeones_analysis'
DERIVATIVES_PATH = os.path.join(BASE_PATH, 'data', 'derivatives', 'campeones_preproc')
XDF_PATH = os.path.join(BASE_PATH, 'data', 'sourcedata', 'xdf')

EPOCH_DURATION = 1.0
EPOCH_OVERLAP = 0.9 # Reduced from 0.9 for speed (Test 4)

def run_pipeline():
    all_epochs_data = [] 
    
    print(f"Starting DELTA REGRESSION pipeline for Subject {SUBJECT}, Combined Acq A+B")

    for run_info in RUNS_CONFIG:
        run_id = run_info['id']
        acq = run_info['acq']
        block_name = run_info['block']
        task_id = run_info['task']
        
        print(f"\nProcessing Run {run_id} (Acq {acq}, {block_name})...")
        
        eeg_dir = os.path.join(DERIVATIVES_PATH, f'sub-{SUBJECT}', f'ses-{SESSION}', 'eeg')
        vhdr_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{acq}_run-{run_id}_desc-preproc_eeg.vhdr')
        events_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{acq}_run-{run_id}_desc-preproc_events.tsv')
        xlsx_file = os.path.join(XDF_PATH, f'sub-{SUBJECT}', f'order_matrix_{SUBJECT}_{acq.upper()}_{block_name}_VR.xlsx')

        if not os.path.exists(vhdr_file):
            print(f"Skipping {run_id}, file not found.")
            continue
            
        preproc_data = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        events_df = pd.read_csv(events_file, sep='\t')
        config_df = pd.read_excel(xlsx_file)
        
        video_events = events_df[events_df['trial_type'].isin(['video', 'video_luminance'])].reset_index(drop=True)
        target_config = config_df.dropna(subset=['dimension']).reset_index(drop=True)
        
        for i, row in target_config.iterrows():
            if i >= len(video_events): break
                
            original_dimension = row['dimension']
            polarity = row['order_emojis_slider'] 
            video_id = row['video_id']
            onset = video_events.loc[i, 'onset']
            duration = video_events.loc[i, 'duration']
            
            dimension = original_dimension
            print(f"  Video {i+1}: Dim={dimension}, Pol={polarity}")
            
            video_data = preproc_data.copy().crop(tmin=onset, tmax=onset + duration)
            joy_data = video_data.get_data(picks=['joystick_x'])[0] 
            
            if polarity == 'inverse':
                joy_data = -joy_data
                
            step = EPOCH_DURATION - EPOCH_OVERLAP # 0.1s
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            
            eeg_picks = mne.pick_types(preproc_data.info, eeg=True, eog=False)
            eeg_data = video_data.get_data(picks=eeg_picks) 
            
            current_idx = 0
            prev_y_val = None 
            
            video_epochs = []
            
            while current_idx + n_samples_epoch <= video_data.n_times:
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                joy_window = joy_data[current_idx : current_idx + n_samples_epoch]
                
                y_val = np.mean(joy_window)
                
                if prev_y_val is not None:
                    delta = y_val - prev_y_val
                    
                    # NO Epsilon Filter for Regression - we want to predict small/zero changes too
                    
                    if pd.notna(video_id):
                         vid_id_str = f"{video_id}_{acq}"
                    else:
                         vid_id_str = f"lum_{run_id}_{i}_{acq}"

                    video_epochs.append({
                        'X': eeg_window,
                        'y': delta,     # Target is continuous delta
                        'y_raw': y_val,
                        'dimension': dimension,
                        'video_identifier': vid_id_str
                    })
                
                prev_y_val = y_val
                current_idx += n_step_samples
            
            all_epochs_data.extend(video_epochs)

    print(f"\nTotal Epochs: {len(all_epochs_data)}")
    if not all_epochs_data: return

    unique_dims = set(d['dimension'] for d in all_epochs_data)
    results_list = []

    for dim in unique_dims:
        print(f"\n--- DELTA REGRESSION for Dimension: {dim} ---")
        
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        unique_videos = list(set(d['video_identifier'] for d in dim_data))
        
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        y_raw_all = np.array([d['y_raw'] for d in dim_data])
        groups = np.array([d['video_identifier'] for d in dim_data])
        
        print(f"Data Shape: X={X_all.shape}, y={y_all.shape}")
        
        scores_r2 = []
        scores_rmse = []
        scores_pearson = []
        
        for test_vid in unique_videos:
            train_mask = groups != test_vid
            test_mask = groups == test_vid
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            y_raw_test = y_raw_all[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0: continue

            pipeline = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=100),
                Ridge(alpha=1.0) # Regression
            )
            
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
            except Exception as e:
                 print(f"Error fitting {test_vid}: {e}")
                 continue
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            pearson_val, _ = pearsonr(y_test, y_pred)
            
            scores_r2.append(r2)
            scores_rmse.append(rmse)
            scores_pearson.append(pearson_val)
            
            print(f"  Test Video: {test_vid} | R2: {r2:.4f} | RMSE: {rmse:.4f} | Pearson: {pearson_val:.4f}")
            
            # Plotting
            plots_dir = os.path.join(BASE_PATH, 'results', 'modeling', 'delta_regression_timeseries', dim)
            os.makedirs(plots_dir, exist_ok=True)
            
            fig, (ax_raw, ax_delta) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            # Top: Raw Value
            ax_raw.plot(y_raw_test, label=f'Raw {dim.capitalize()}', color='blue', alpha=0.6)
            ax_raw.set_title(f"Dim: {dim} | Video: {test_vid} | R2: {r2:.2f} | Pearson: {pearson_val:.2f}")
            ax_raw.legend()
            
            # Bottom: Delta
            ax_delta.plot(y_test, label='True Delta', color='black', alpha=0.8, linewidth=1.5)
            ax_delta.plot(y_pred, label='Predicted Delta', color='orangered', alpha=0.6, linestyle='--')
            ax_delta.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax_delta.set_ylabel('Delta (Slope)')
            ax_delta.set_xlabel('Epochs')
            ax_delta.legend()
            
            safe_vid = str(test_vid).replace('.','_').replace('/','_')
            plt.savefig(os.path.join(plots_dir, f'delta_reg_{dim}_{safe_vid}.png'))
            plt.close()
            
            results_list.append({
                'Subject': SUBJECT,
                'Dimension': dim,
                'TestVideo': test_vid,
                'TrainSize': len(y_train),
                'TestSize': len(y_test),
                'R2': r2,
                'RMSE': rmse,
                'Pearson': pearson_val
            })
            
        print(f"Average R2: {np.mean(scores_r2):.4f}")
        print(f"Average Pearson: {np.mean(scores_pearson):.4f}")

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_dir = os.path.join(BASE_PATH, 'results', 'modeling')
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(os.path.join(results_dir, f'sub-{SUBJECT}_delta_regression_results.csv'), index=False)
        
        print("\n--- Summary by Dimension ---")
        print(results_df.groupby('Dimension')[['R2', 'RMSE', 'Pearson']].mean())

if __name__ == '__main__':
    run_pipeline()
