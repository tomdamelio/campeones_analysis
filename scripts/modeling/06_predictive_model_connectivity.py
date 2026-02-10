import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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
EPOCH_OVERLAP = 0.9 # High density

from config import EEG_CHANNELS

def run_pipeline():
    all_epochs_data = [] 
    
    print(f"Starting CONNECTIVITY REGRESSION pipeline (PCA-50) for Subject {SUBJECT}")

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
            print(f"Skipping {run_id}, file not found")
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
            
            print(f"  Video {i+1}: Dim={dimension}")
            
            video_data = preproc_data.copy().crop(tmin=onset, tmax=onset + duration)
            joy_data = video_data.get_data(picks=['joystick_x'])[0] 
            
            if polarity == 'inverse':
                joy_data = -joy_data
                
            step = EPOCH_DURATION - EPOCH_OVERLAP
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            
            eeg_channels_present = [ch for ch in EEG_CHANNELS if ch in preproc_data.ch_names]
            eeg_data = video_data.get_data(picks=eeg_channels_present) 
            
            current_idx = 0
            prev_y_val = None 
            
            # Pre-calculate indices for upper triangle (INCLUDING DIAGONAL for Variance/Power)
            n_channels = len(eeg_channels_present)
            triu_indices = np.triu_indices(n_channels, k=0)
            
            video_epochs = []
            
            while current_idx + n_samples_epoch <= video_data.n_times:
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                joy_window = joy_data[current_idx : current_idx + n_samples_epoch]
                
                # --- CONNECTIVITY FEATURE EXTRACTION ---
                # Covariance Matrix (Channels x Channels)
                # We use Covariance to preserve signal power (diagonal) and channel interaction magnitude.
                with np.errstate(invalid='ignore'):
                    cov_matrix = np.cov(eeg_window)
                
                # Handle NaNs (due to flat signals)
                if np.isnan(cov_matrix).any():
                    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)
                
                # Flatten Upper Triangle (including diagonal)
                connectivity_features = cov_matrix[triu_indices]
                # ---------------------------------------
                
                y_val = np.mean(joy_window)
                
                if prev_y_val is not None:
                    delta = y_val - prev_y_val
                    
                    if pd.notna(video_id):
                         vid_id_str = f"{video_id}_{acq}"
                    else:
                         vid_id_str = f"lum_{run_id}_{i}_{acq}"

                    video_epochs.append({
                        'X': connectivity_features, # Now using connectivity!
                        'y': delta,
                        'dimension': dimension,
                        'video_identifier': vid_id_str
                    })
                
                prev_y_val = y_val
                current_idx += n_step_samples
            
            all_epochs_data.extend(video_epochs)

    print(f"\nTotal Epochs: {len(all_epochs_data)}")
    if not all_epochs_data: return

    # --- VARIANCE ANALYSIS (CONTROL) ---
    print("\n--- Explained Variance Analysis of Connectivity Features ---")
    # Take a random subsample to check variance (to avoid memory issues if dataset is huge)
    sample_data = np.array([d['X'] for d in all_epochs_data])
    if len(sample_data) > 5000:
        indices = np.random.choice(len(sample_data), 5000, replace=False)
        sample_data = sample_data[indices]
    
    pca_check = PCA(n_components=100)
    scaler_check = StandardScaler()
    sample_scaled = scaler_check.fit_transform(sample_data)
    pca_check.fit(sample_scaled)
    
    cumulative_variance = np.cumsum(pca_check.explained_variance_ratio_)
    var_50 = cumulative_variance[49] if len(cumulative_variance) >= 50 else cumulative_variance[-1]
    
    print(f"Explained Variance @ 50 components: {var_50:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), cumulative_variance[:100], marker='.')
    plt.axvline(x=50, color='r', linestyle='--', label=f'50 Comps ({var_50:.2%})')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance (Connectivity Features)')
    plt.legend()
    plt.grid(True)
    plot_dir = os.path.join(BASE_PATH, 'results', 'exploration')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'pca_variance_connectivity.png'))
    print(f"Variance plot saved to {os.path.join(plot_dir, 'pca_variance_connectivity.png')}")
    # ------------------------------------

    unique_dims = set(d['dimension'] for d in all_epochs_data)
    results_list = []

    for dim in unique_dims:
        print(f"\n--- Connectivity Regression for Dimension: {dim} ---")
        
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        unique_videos = list(set(d['video_identifier'] for d in dim_data))
        
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        groups = np.array([d['video_identifier'] for d in dim_data])
        
        scores_r2 = []
        scores_pearson = []
        
        for test_vid in unique_videos:
            train_mask = groups != test_vid
            test_mask = groups == test_vid
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0: continue

            pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=50), # Use 50 components
                Ridge(alpha=1.0)
            )
            
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
            except Exception as e:
                 print(f"Error {test_vid}: {e}")
                 continue
            
            r2 = r2_score(y_test, y_pred)
            pearson_val, _ = pearsonr(y_test, y_pred)
            
            scores_r2.append(r2)
            scores_pearson.append(pearson_val)
            
            print(f"  Test Video: {test_vid} | R2: {r2:.4f} | Pearson: {pearson_val:.4f}")

            # --- Plotting ---
            plots_dir = os.path.join(BASE_PATH, 'results', 'modeling', 'connectivity_timeseries', dim)
            os.makedirs(plots_dir, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_test, label='True Delta', color='black', alpha=0.7)
            ax.plot(y_pred, label='Predicted Delta (Connectivity)', color='orange', alpha=0.7, linestyle='--')
            ax.set_title(f"Connectivity Regression | Dim: {dim} | Video: {test_vid} | R2: {r2:.2f} | Pearson: {pearson_val:.2f}")
            ax.legend()
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Delta')
            
            safe_vid = str(test_vid).replace('.','_').replace('/','_')
            plt.savefig(os.path.join(plots_dir, f'conn_reg_{dim}_{safe_vid}.png'))
            plt.close()
            # ----------------
            
            results_list.append({
                'Subject': SUBJECT,
                'Dimension': dim,
                'TestVideo': test_vid,
                'R2': r2,
                'Pearson': pearson_val
            })
            
        print(f"Average R2: {np.mean(scores_r2):.4f}")

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_dir = os.path.join(BASE_PATH, 'results', 'modeling')
        os.makedirs(results_dir, exist_ok=True)
        results_df.to_csv(os.path.join(results_dir, f'sub-{SUBJECT}_connectivity_regression_results.csv'), index=False)

if __name__ == '__main__':
    run_pipeline()
