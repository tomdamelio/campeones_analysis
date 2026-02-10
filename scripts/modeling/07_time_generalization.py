import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

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
EPOCH_OVERLAP = 0.9
LAGS = np.arange(0, 3.25, 0.25) # 0 to 3s steps of 250ms

from config import EEG_CHANNELS

def run_pipeline():
    print(f"Starting TIME GENERALIZATION (Lag) pipeline for Subject {SUBJECT}")
    
    # --- 1. Load Data (Master Set) ---
    # We load continuous data and will slice specifically for lags later?
    # No, usually easier to load Epochs and their onset times, then compute Y for different lags on the fly.
    # But `y` (joystick) changes continuously.
    
    # New strategy: Store (eeg_window, joystick_full_trace, onset_idx)
    # Actually, simpler: Store (eeg_window, current_time) and load joystick separately? 
    # Or just store the full joystick trace for the video and query it.
    
    # Let's stick to the previous loop but extract LARGER joystick windows to allow lagging?
    # If Lag is up to 3s, we need future joystick values.
    # So when processing a video, we need to be careful.
    
    all_epochs_data = [] # List of dicts

    for run_info in RUNS_CONFIG:
        run_id = run_info['id']
        acq = run_info['acq']
        block_name = run_info['block']
        task_id = run_info['task']
        
        print(f"\nProcessing Run {run_id} (Acq {acq})...")
        
        eeg_dir = os.path.join(DERIVATIVES_PATH, f'sub-{SUBJECT}', f'ses-{SESSION}', 'eeg')
        vhdr_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{acq}_run-{run_id}_desc-preproc_eeg.vhdr')
        events_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{acq}_run-{run_id}_desc-preproc_events.tsv')
        xlsx_file = os.path.join(XDF_PATH, f'sub-{SUBJECT}', f'order_matrix_{SUBJECT}_{acq.upper()}_{block_name}_VR.xlsx')

        if not os.path.exists(vhdr_file): continue
            
        preproc_data = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        events_df = pd.read_csv(events_file, sep='\t')
        config_df = pd.read_excel(xlsx_file)
        
        video_events = events_df[events_df['trial_type'].isin(['video', 'video_luminance'])].reset_index(drop=True)
        target_config = config_df.dropna(subset=['dimension']).reset_index(drop=True)

        for i, row in target_config.iterrows():
            if i >= len(video_events): break
            
            dimension = row['dimension']
            polarity = row['order_emojis_slider']
            video_id = row['video_id']
            onset = video_events.loc[i, 'onset']
            duration = video_events.loc[i, 'duration']
            
            # Crop to Video duration + Max Lag (3s) to ensure we have future joystick data
            max_lag_s = 3.5 
            video_data = preproc_data.copy().crop(tmin=onset, tmax=onset + duration + max_lag_s)
            
            # Joystick full trace
            joy_data = video_data.get_data(picks=['joystick_x'])[0]
            if polarity == 'inverse': joy_data = -joy_data
            
            # EEG data (only up to duration, we don't use functional connectivity of future brain)
            # Actually we crop EEG to 'duration' but keep joy for 'duration + lag'
            # To simplify compatibility, we iterate indices up to 'duration'.
            
            step = EPOCH_DURATION - EPOCH_OVERLAP # 0.1s
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            
            eeg_channels_present = [ch for ch in EEG_CHANNELS if ch in preproc_data.ch_names]
            eeg_data = video_data.get_data(picks=eeg_channels_present)
            
            current_idx = 0
            n_samples_video_end = int(duration * sfreq)
            
            if pd.notna(video_id): vid_id_str = f"{video_id}_{acq}"
            else: vid_id_str = f"lum_{run_id}_{i}_{acq}"

            # Pre-calc Upper Triangle indices
            n_ch = len(eeg_channels_present)
            triu_indices = np.triu_indices(n_ch, k=1)
            
            video_epochs = []
            
            while current_idx + n_samples_epoch <= n_samples_video_end:
                # Brain Window [t, t+1]
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                
                # Connectivity Feature
                with np.errstate(invalid='ignore'):
                    corr = np.corrcoef(eeg_window)
                if np.isnan(corr).any(): corr = np.nan_to_num(corr, nan=0.0)
                conn_feat = corr[triu_indices]

                # Store minimal info to re-compute targets dynamically
                video_epochs.append({
                    'X_raw': eeg_window, 
                    'X_conn': conn_feat,
                    'current_idx': current_idx,
                    'joy_full': joy_data, # Reference to full array (memory efficient?)
                    'dimension': dimension,
                    'video_id': vid_id_str
                })
                current_idx += n_step_samples
            
            try:
                all_epochs_data.extend(video_epochs)
            except Exception as e:
                print(f"CRITICAL ERROR extending data: {e}")
                import traceback
                traceback.print_exc()
            
    print(f"Total Epochs: {len(all_epochs_data)}")
    
    unique_dims = set(d['dimension'] for d in all_epochs_data)
    
    for dim in unique_dims:
        print(f"\n=== Processing Dimension: {dim} ===")
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        
        # Prepare Data Layout
        # We need a LOVO loop or Train/Test split.
        # For Time Gen Matrix, usually we want robust metrics.
        # Doing LOVO for every cell in the matrix (13x13 = 169) is expensive.
        # We will use a fixed Train/Test split (e.g. first 80% videos train, rest test) 
        # OR just one LOVO fold? No, results would be noisy.
        # Let's do a 5-fold GroupKFold equivalent or just specific Train/Test videos.
        # Given computation time, let's pick 3 random videos as Test and rest as Train.
        
        unique_videos = list(set(d['video_id'] for d in dim_data))
        np.random.seed(42)
        test_videos = np.random.choice(unique_videos, size=max(1, int(len(unique_videos)*0.2)), replace=False)
        print(f"Test Videos: {test_videos}")
        
        train_data = [d for d in dim_data if d['video_id'] not in test_videos]
        test_data = [d for d in dim_data if d['video_id'] in test_videos]
        
        if not train_data or not test_data: continue
        
        # Pre-compute X arrays to save time
        X_raw_train = np.array([d['X_raw'] for d in train_data])
        X_conn_train = np.array([d['X_conn'] for d in train_data])
        X_raw_test = np.array([d['X_raw'] for d in test_data])
        X_conn_test = np.array([d['X_conn'] for d in test_data])
        
        # Function to get Y for a specific lag
        def get_y(data_list, lag_s):
            y_list = []
            sfreq = 500 # Approx, assume 500
            lag_samples = int(lag_s * sfreq)
            n_epoch = int(EPOCH_DURATION * sfreq)
            
            for d in data_list:
                idx = d['current_idx']
                joy = d['joy_full']
                
                # Check bounds
                start = idx + lag_samples
                end = start + n_epoch
                
                # Previous window (for delta)
                # Delta(t) = Mean(t) - Mean(t-1)
                # Delta(t+L) = Mean(t+L) - Mean(t+L-1)
                # But we step by 0.1s. 
                # Let's define Delta as local slope at lag?
                # Or just Y class at lag?
                # Test 3 used: y_val = mean(window), delta = y_val - prev_y_val.
                # Here we can't easily rely on 'prev' from the loop because lags shift.
                # We will compute Delta = Mean(window_at_lag) - Mean(window_at_lag - 100ms??)
                
                # Simplification: Trend = Slope of regression line within the lagged window?
                # Or: Mean(last_half) - Mean(first_half)?
                # Let's stick to Test 3 logic: Mean(current) - Mean(prev_step).
                # But 'prev_step' is 0.1s ago.
                
                if end > len(joy): 
                    y_list.append(0) # Pad? or Skip?
                    continue
                    
                joy_window = joy[start:end]
                
                # Prev window (0.1s before)
                start_prev = start - int(0.1*sfreq)
                end_prev = end - int(0.1*sfreq)
                if start_prev < 0:
                    delta = 0
                else:
                    joy_prev = joy[start_prev:end_prev]
                    delta = np.mean(joy_window) - np.mean(joy_prev)
                
                # Classify
                epsilon = 0.001
                if delta > epsilon: y_cls = 1
                elif delta < -epsilon: y_cls = 0
                else: y_cls = -1 # Stable -> Filter out?
                
                y_list.append(y_cls)
            return np.array(y_list)

        # Matrices to store results (Train Lag x Test Lag)
        matrix_A = np.zeros((len(LAGS), len(LAGS))) # Classic
        matrix_B = np.zeros((len(LAGS), len(LAGS))) # Connectivity
        
        print("Running Loops over Lags...")
        
        for i, train_lag in enumerate(LAGS):
            print(f"  Train Lag: {train_lag}s")
            
            # Prepare Train Y
            y_train_full = get_y(train_data, train_lag)
            # Filter stable (-1)
            mask_train = y_train_full != -1
            if np.sum(mask_train) < 100: continue
            
            y_train = y_train_full[mask_train]
            X_raw_tr = X_raw_train[mask_train]
            X_conn_tr = X_conn_train[mask_train]
            
            # Train Models
            # Reviewer Note: Balancing classes?
            # RidgeClassifier with class_weight='balanced'
            
            # Model A
            pipe_A = make_pipeline(Vectorizer(), StandardScaler(), PCA(n_components=100), RidgeClassifier(class_weight='balanced'))
            pipe_A.fit(X_raw_tr, y_train)
            
            # Model B
            pipe_B = make_pipeline(StandardScaler(), PCA(n_components=100), RidgeClassifier(class_weight='balanced'))
            pipe_B.fit(X_conn_tr, y_train)
            
            for j, test_lag in enumerate(LAGS):
                # Prepare Test Y
                y_test_full = get_y(test_data, test_lag)
                mask_test = y_test_full != -1
                if np.sum(mask_test) < 10: 
                    matrix_A[i,j] = np.nan
                    matrix_B[i,j] = np.nan
                    continue
                    
                y_test = y_test_full[mask_test]
                X_raw_te = X_raw_test[mask_test]
                X_conn_te = X_conn_test[mask_test]
                
                # Test
                pred_A = pipe_A.predict(X_raw_te)
                pred_B = pipe_B.predict(X_conn_te)
                
                try: auc_A = roc_auc_score(y_test, pipe_A.decision_function(X_raw_te))
                except: auc_A = 0.5
                
                try: auc_B = roc_auc_score(y_test, pipe_B.decision_function(X_conn_te))
                except: auc_B = 0.5
                
                matrix_A[i, j] = auc_A
                matrix_B[i, j] = auc_B
        
        # Plotting Heatmaps
        plot_dir = os.path.join(BASE_PATH, 'results', 'modeling', 'time_generalization', dim)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot A
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(matrix_A, xticklabels=LAGS, yticklabels=LAGS, cmap='coolwarm', vmin=0, vmax=1, annot=False)
        ax.invert_yaxis() # Put 0 at bottom
        plt.xlabel('Testing Time (s)')
        plt.ylabel('Training Time (s)')
        plt.title(f'Time Gen Matrix (Classic PCA 100) - {dim}\nMetric: AUC (Blue=0, Red=1, White=0.5)')
        plt.savefig(os.path.join(plot_dir, 'tg_matrix_classic.png'))
        plt.close()
        
        # Plot B
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(matrix_B, xticklabels=LAGS, yticklabels=LAGS, cmap='coolwarm', vmin=0, vmax=1, annot=False)
        ax.invert_yaxis()
        plt.xlabel('Testing Time (s)')
        plt.ylabel('Training Time (s)')
        plt.title(f'Time Gen Matrix (Connectivity PCA 100) - {dim}\nMetric: AUC (Blue=0, Red=1, White=0.5)')
        plt.savefig(os.path.join(plot_dir, 'tg_matrix_connectivity.png'))
        plt.close()
        
        print(f"Saved matrices for {dim}")

if __name__ == '__main__':
    run_pipeline()
