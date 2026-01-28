import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

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
    # '008' excluded due to technical issues
    {'id': '009', 'acq': 'b', 'block': 'block3', 'task': '03'},
    {'id': '010', 'acq': 'b', 'block': 'block4', 'task': '04'},
]

BASE_PATH = r'c:\Users\tdamelio\Desktop\campeones_analysis'
DERIVATIVES_PATH = os.path.join(BASE_PATH, 'data', 'derivatives', 'campeones_preproc')
XDF_PATH = os.path.join(BASE_PATH, 'data', 'sourcedata', 'xdf')

EPOCH_DURATION = 1.0
EPOCH_OVERLAP = 0.9

def run_pipeline():
    all_epochs_data = [] 
    
    print(f"Starting REGRESSION pipeline for Subject {SUBJECT}, Combined Acq A+B")

    for run_info in RUNS_CONFIG:
        run_id = run_info['id']
        acq = run_info['acq']
        block_name = run_info['block']
        task_id = run_info['task']
        
        print(f"\nProcessing Run {run_id} (Acq {acq}, {block_name})...")
        
        # Paths
        eeg_dir = os.path.join(DERIVATIVES_PATH, f'sub-{SUBJECT}', f'ses-{SESSION}', 'eeg')
        vhdr_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{acq}_run-{run_id}_desc-preproc_eeg.vhdr')
        events_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{acq}_run-{run_id}_desc-preproc_events.tsv')
        xlsx_file = os.path.join(XDF_PATH, f'sub-{SUBJECT}', f'order_matrix_{SUBJECT}_{acq.upper()}_{block_name}_VR.xlsx')

        if not os.path.exists(vhdr_file):
            print(f"Skipping {run_id}, EEG file not found: {vhdr_file}")
            continue
            
        # Load Data
        preproc_data = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        events_df = pd.read_csv(events_file, sep='\t')
        config_df = pd.read_excel(xlsx_file)
        
        video_events = events_df[events_df['trial_type'].isin(['video', 'video_luminance'])].reset_index(drop=True)
        target_config = config_df.dropna(subset=['dimension']).reset_index(drop=True)
        
        for i, row in target_config.iterrows():
            if i >= len(video_events):
                break
                
            original_dimension = row['dimension']
            polarity = row['order_emojis_slider'] 
            video_id = row['video_id']
            onset = video_events.loc[i, 'onset']
            duration = video_events.loc[i, 'duration']
            
            dimension = original_dimension
            # Use unique video identifier including acquisition
            if pd.notna(video_id):
                 vid_str = f"{video_id}_{acq}"
            else:
                 vid_str = f"lum_{run_id}_{i}_{acq}"

            # Crop Data
            t_start = onset
            t_stop = onset + duration
            video_data = preproc_data.copy().crop(tmin=t_start, tmax=t_stop)
            
            # Joystick data
            joy_data = video_data.get_data(picks=['joystick_x'])[0]
            if polarity == 'inverse':
                joy_data = -joy_data
                
            # Epoching
            step = EPOCH_DURATION - EPOCH_OVERLAP # 0.1s
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            n_samples = video_data.n_times
            
            eeg_picks = mne.pick_types(preproc_data.info, eeg=True, eog=False, stim=False, misc=False)
            eeg_data = video_data.get_data(picks=eeg_picks)
            
            current_idx = 0
            while current_idx + n_samples_epoch <= n_samples:
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                joy_window = joy_data[current_idx : current_idx + n_samples_epoch]
                
                y_val = np.mean(joy_window)
                
                all_epochs_data.append({
                    'X': eeg_window, 
                    'y': y_val,
                    'dimension': dimension,
                    'video_identifier': vid_str,
                    'acq': acq
                })
                current_idx += n_step_samples

    print(f"\nTotal Epochs Generated: {len(all_epochs_data)}")
    if not all_epochs_data:
        return

    # Process per Dimension
    results_list = []
    unique_dims = set(d['dimension'] for d in all_epochs_data)

    print("\n" + "="*50)
    print("DATASET DIMENSIONS REPORT")
    print("="*50)

    for dim in unique_dims:
        print(f"\n--- Dimension: {dim} ---")
        
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        unique_videos = list(set(d['video_identifier'] for d in dim_data))
        
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        groups = np.array([d['video_identifier'] for d in dim_data])
        
        # Calculate Flattened Feature Shape (Approximate for 100 components)
        # Vectorizer output size = n_channels * n_times
        n_channels = X_all.shape[1]
        n_times = X_all.shape[2]
        n_features_flat = n_channels * n_times
        
        print(f"Total Epochs: {len(X_all)}")
        print(f"Feature Space (Raw): {n_channels} channels x {n_times} timepoints = {n_features_flat} features")
        print(f"Unique Videos (Folds): {len(unique_videos)}")
        
        scores_rmse = []
        scores_corr = []
        
        for i, test_vid in enumerate(unique_videos):
            train_mask = groups != test_vid
            test_mask = groups == test_vid
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            if i == 0:
                print(f"Example Fold (Test Video: {test_vid}):")
                print(f"  Training Set: {X_train.shape[0]} rows x {n_features_flat} raw cols (Reduced to 100 PCs)")
                print(f"  Test Set:     {X_test.shape[0]} rows")
            
            # Pipeline
            pipeline = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=100),
                Ridge(alpha=1.0)
            )
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            corr = pearsonr(y_test, y_pred)[0] if np.std(y_pred) > 1e-9 else 0.0
            
            scores_rmse.append(rmse)
            scores_corr.append(corr)

            # --- Plotting Predictions ---
            plots_dir = os.path.join(BASE_PATH, 'results', 'modeling', 'predictions_timeseries', dim)
            os.makedirs(plots_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            plt.plot(y_test, label='True', alpha=0.7)
            plt.plot(y_pred, label='Predicted', alpha=0.7, linestyle='--')
            plt.title(f"Dimension: {dim} | Video: {test_vid}\nRMSE: {rmse:.3f}, Corr: {corr:.3f}")
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Joystick Value")
            
            # Sanitize filename
            safe_vid = str(test_vid).replace('.','_').replace('/','_')
            plot_filename = f"pred_ts_{dim}_{safe_vid}.png"
            plt.savefig(os.path.join(plots_dir, plot_filename))
            plt.close()
            # ----------------------------
            
            results_list.append({
                'Dimension': dim,
                'TestVideo': test_vid,
                'TrainSize': len(y_train),
                'TestSize': len(y_test),
                'RMSE': rmse,
                'Correlation': corr
            })
            
        print(f"Average RMSE: {np.mean(scores_rmse):.4f} | Avg Correlation: {np.mean(scores_corr):.4f}")

    # Save Results
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_dir = os.path.join(BASE_PATH, 'results', 'modeling')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f'sub-{SUBJECT}_combined_regression_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    run_pipeline()
