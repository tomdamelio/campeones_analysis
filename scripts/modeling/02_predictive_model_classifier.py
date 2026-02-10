import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Configuration
SUBJECT = '27'
SESSION = 'vr'
BASE_PATH = r'c:\Users\tdamelio\Desktop\campeones_analysis'
DERIVATIVES_PATH = os.path.join(BASE_PATH, 'data', 'derivatives', 'campeones_preproc')
XDF_PATH = os.path.join(BASE_PATH, 'data', 'sourcedata', 'xdf')

# Combined Runs Setup (A + B)
RUNS_CONFIG = [
    {'id': '002', 'acq': 'a', 'block': 'block1', 'task': '01'},
    {'id': '003', 'acq': 'a', 'block': 'block2', 'task': '02'},
    {'id': '004', 'acq': 'a', 'block': 'block3', 'task': '03'},
    {'id': '006', 'acq': 'a', 'block': 'block4', 'task': '04'},
    {'id': '007', 'acq': 'b', 'block': 'block1', 'task': '01'},
    # '008' excluded
    {'id': '009', 'acq': 'b', 'block': 'block3', 'task': '03'},
    {'id': '010', 'acq': 'b', 'block': 'block4', 'task': '04'},
]

EPOCH_DURATION = 1.0
EPOCH_OVERLAP = 0.9

from config import EEG_CHANNELS

def run_pipeline():
    all_epochs_data = [] # List to hold (X, y, meta) per video
    
    print(f"Starting CLASSIFICATION pipeline for Subject {SUBJECT}, Combined Acq A+B")

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
        
        # Handle excel reading robustly
        try:
            config_df = pd.read_excel(xlsx_file)
        except Exception as e:
            print(f"Error reading xlsx: {xlsx_file} - {e}")
            continue
        
        # Match events
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
            
            # Use dimension directly from config
            dimension = original_dimension
            
            # Create a unique identifier for the fold (Video)
            if pd.isna(video_id):
                 video_identifier = f"lum_{run_id}_{i}_{acq}"
            else:
                 video_identifier = f"{video_id}_{acq}"

            print(f"  Video {i+1}: Dim={dimension}, Pol={polarity}, ID={video_identifier}")
            
            # Crop Data
            t_start = onset
            t_stop = onset + duration
            video_data = preproc_data.copy().crop(tmin=t_start, tmax=t_stop)
            
            # Get Joystick Data
            joy_data = video_data.get_data(picks=['joystick_x'])[0] 
            
            if polarity == 'inverse':
                joy_data = -joy_data
                
            # Epoching Logic
            step = EPOCH_DURATION - EPOCH_OVERLAP 
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            n_samples_total = video_data.n_times
            
            # Get EEG Data
            eeg_channels_present = [ch for ch in EEG_CHANNELS if ch in preproc_data.ch_names]
            eeg_data = video_data.get_data(picks=eeg_channels_present) 
            
            current_idx = 0
            while current_idx + n_samples_epoch <= n_samples_total:
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                joy_window = joy_data[current_idx : current_idx + n_samples_epoch]
                
                # Target: Binary Class (Mean > 0 -> 1, else 0)
                # Filter Neutral: |y| <= 0.01
                y_val_cont = np.mean(joy_window)
                
                if abs(y_val_cont) <= 0.01:
                    # Skip neutral epochs
                    current_idx += n_step_samples
                    continue

                y_class = 1 if y_val_cont > 0.01 else 0
                
                all_epochs_data.append({
                    'X': eeg_window,
                    'y': y_class,
                    'dimension': dimension,
                    'video_identifier': video_identifier
                })
                current_idx += n_step_samples

    print(f"\nTotal Epochs Generated: {len(all_epochs_data)}")
    if len(all_epochs_data) == 0:
        print("No epochs generated. Exiting.")
        return

    unique_dims = set(d['dimension'] for d in all_epochs_data)
    results_list = []

    for dim in unique_dims:
        print(f"\n--- Classifer for Dimension: {dim} ---")
        
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        unique_videos = list(set(d['video_identifier'] for d in dim_data))
        
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        groups = np.array([d['video_identifier'] for d in dim_data])
        
        print(f"Data Shape: X={X_all.shape}, y={y_all.shape} | Unique Videos: {len(unique_videos)}")
        
        # Check overall imbalance
        pos_count = np.sum(y_all == 1)
        neg_count = np.sum(y_all == 0)
        print(f"Overall Balance: Pos={pos_count}, Neg={neg_count} (Pos Rate: {pos_count/len(y_all):.2%})")
        
        scores_acc = []
        scores_auc = []
        
        for test_vid in unique_videos:
            train_mask = groups != test_vid
            test_mask = groups == test_vid
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            # --- Pipeline ---
            pipeline = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=100),
                RidgeClassifier(alpha=1.0) # Using Ridge Classifier
            )
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Decision function for AUC
            try:
                y_score = pipeline.decision_function(X_test)
            except:
                y_score = y_pred 
            
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            
            # Calculate AUC (handle single-class edge cases)
            auc = np.nan
            if len(np.unique(y_test)) > 1:
                try:
                    auc = roc_auc_score(y_test, y_score)
                except ValueError:
                    pass
            
            scores_acc.append(acc)
            scores_auc.append(auc)
            
            print(f"  Test Video: {test_vid} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | AUC: {auc:.4f}")
            
            # --- Plotting Predictions (Time Series) ---
            plots_dir = os.path.join(BASE_PATH, 'results', 'modeling', 'classification_timeseries', dim)
            os.makedirs(plots_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 5))
            # Plot True Labels (stepped)
            plt.plot(y_test, label='True Class (0/1)', color='black', alpha=0.6, linewidth=2)
            # Plot Decision Score (normalized vaguely or just raw score)
            # Standardize score for visualization overlap? Or just second axis?
            # Let's use twinx for score
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(y_score, label='Decision Score', color='blue', alpha=0.5, linestyle='--')
            
            # Plot binary prediction
            # ax1.plot(y_pred, label='Pred Class', color='orange', alpha=0.5, linestyle=':')
            
            ax1.set_ylabel('Class')
            ax2.set_ylabel('Ridge Score (Distance)')
            ax1.set_xlabel('Epochs')
            plt.title(f"Dim: {dim} | Video: {test_vid} | Acc: {acc:.2f} | AUC: {auc:.2f}")
            
            # Combined Legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            safe_vid = str(test_vid).replace('.','_').replace('/','_')
            plot_filename = f"cls_ts_{dim}_{safe_vid}.png"
            plt.savefig(os.path.join(plots_dir, plot_filename))
            plt.close()
            # ------------------------------------------

            results_list.append({
                'Dimension': dim,
                'TestVideo': test_vid,
                'TrainSize': len(y_train),
                'TestSize': len(y_test),
                'TrainPosRate': np.mean(y_train),
                'TestPosRate': np.mean(y_test),
                'Accuracy': acc,
                'BalancedAccuracy': bal_acc,
                'AUC': auc
            })
            
        print(f"Average Acc: {np.mean(scores_acc):.4f}")

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_dir = os.path.join(BASE_PATH, 'results', 'modeling')
        os.makedirs(results_dir, exist_ok=True)
        filename = f'sub-{SUBJECT}_combined_classification_results.csv'
        output_path = os.path.join(results_dir, filename)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    run_pipeline()
