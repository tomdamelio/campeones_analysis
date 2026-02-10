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
# Configuration
SUBJECT = '27'
SESSION = 'vr'
# ACQ is now dynamic per run

# Combined Runs Config (A + B, excluding A-Task04-Run005 (short) and B-Task02-Run008(error))
# Note: User specified excluding Run 008.
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

from config import EEG_CHANNELS

def run_pipeline():
    all_epochs_data = [] # List to hold (X, y, meta)
    
    print(f"Starting TREND CLASSIFICATION pipeline (Up vs Down) for Subject {SUBJECT}, Combined Acq A+B (Run 008 skipped)")

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
        
        if len(video_events) != len(target_config):
             pass

        for i, row in target_config.iterrows():
            if i >= len(video_events):
                break
                
            original_dimension = row['dimension']
            polarity = row['order_emojis_slider'] 
            video_id = row['video_id']
            
            onset = video_events.loc[i, 'onset']
            duration = video_events.loc[i, 'duration']
            
            dimension = original_dimension
            
            print(f"  Video {i+1}: Dim={dimension}, Pol={polarity}, ID={video_id}, Dur={duration:.1f}s")
            
            t_start = onset
            t_stop = onset + duration
            
            video_data = preproc_data.copy().crop(tmin=t_start, tmax=t_stop)
            
            joy_data = video_data.get_data(picks=['joystick_x'])[0] 
            
            if polarity == 'inverse':
                joy_data = -joy_data
                
            step = EPOCH_DURATION - EPOCH_OVERLAP # 0.1s
            
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            
            n_samples = video_data.n_times
            
            eeg_channels_present = [ch for ch in EEG_CHANNELS if ch in preproc_data.ch_names]
            eeg_data = video_data.get_data(picks=eeg_channels_present) 
            
            current_idx = 0
            prev_y_val = None # Track previous epoch value
            
            video_epochs = []
            
            while current_idx + n_samples_epoch <= n_samples:
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                joy_window = joy_data[current_idx : current_idx + n_samples_epoch]
                
                # Current Value
                y_val = np.mean(joy_window)
                
                # Compute Trend if not first
                if prev_y_val is not None:
                    delta = y_val - prev_y_val
                    
                    # Logic: Filter Stable Epochs
                    # Only keep data where change is significant (> epsilon)
                    epsilon = 0.001 # Threshold for stability (Lowered to 0.001)
                    
                    if abs(delta) <= epsilon:
                        # Skip stable epochs
                        prev_y_val = y_val
                        current_idx += n_step_samples
                        continue
                        
                    # Class 1: Up (> epsilon)
                    # Class 0: Down (< -epsilon)
                    y_class = 1 if delta > 0 else 0
                    
                    # Create granular unique ID for LOVO (VideoID + Acq)
                    # This treats "Video 1 Day A" and "Video 1 Day B" as separate folds
                    if pd.notna(video_id):
                         vid_id_str = f"{video_id}_{acq}"
                    else:
                         vid_id_str = f"lum_{run_id}_{i}_{acq}"

                    video_epochs.append({
                        'X': eeg_window,
                        'y': y_class,
                        'delta': delta,
                        'y_raw': y_val,
                        'prev_y': prev_y_val,
                        'dimension': dimension,
                        'acq': acq,
                        'video_identifier': vid_id_str
                    })
                
                prev_y_val = y_val
                current_idx += n_step_samples
            
            all_epochs_data.extend(video_epochs)

    print(f"\nTotal Epochs Generated: {len(all_epochs_data)}")
    if len(all_epochs_data) == 0:
        print("No epochs generated. Exiting.")
        return

    unique_dims = set(d['dimension'] for d in all_epochs_data)
    results_list = []

    for dim in unique_dims:
        print(f"\n--- TREND Classifer for Dimension: {dim} ---")
        
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        unique_videos = list(set(d['video_identifier'] for d in dim_data))
        print(f"Unique videos found: {len(unique_videos)} ({unique_videos})")
        
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        y_raw_all = np.array([d['y_raw'] for d in dim_data])  # Raw joystick values
        groups = np.array([d['video_identifier'] for d in dim_data])
        acq_map = {d['video_identifier']: d['acq'] for d in dim_data}
        
        print(f"Data Shape: X={X_all.shape}, y={y_all.shape}")
        
        # Check overall imbalance
        pos_count = np.sum(y_all == 1)
        neg_count = np.sum(y_all == 0)
        print(f"Overall Balance: Up={pos_count}, Down/Stable={neg_count} (Up Rate: {pos_count/len(y_all):.2%})")
        
        scores_acc = []
        scores_bal_acc = []
        scores_auc = []
        
        for test_vid in unique_videos:
            train_mask = groups != test_vid
            test_mask = groups == test_vid
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            y_raw_test = y_raw_all[test_mask]  # Raw values for plotting
            
            if len(np.unique(y_test)) < 2:
                # Handle single class case
                pass
            
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"  Test Video: {test_vid} | SKIPPED (Insufficient data: Train={len(X_train)}, Test={len(X_test)})")
                continue

            pipeline = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=100),
                RidgeClassifier(alpha=1.0, class_weight='balanced') # Fix for 0.5 BalAcc issue
            )
            
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
            except Exception as e:
                 print(f"  Test Video: {test_vid} | ERROR fitting: {e}")
                 continue
            
            try:
                y_score = pipeline.decision_function(X_test)
            except:
                y_score = y_pred
            
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            
            try:
                if len(np.unique(y_test)) > 1:
                    auc = roc_auc_score(y_test, y_score)
                else:
                    auc = np.nan
            except ValueError:
                auc = np.nan
            
            scores_acc.append(acc)
            scores_bal_acc.append(bal_acc)
            scores_auc.append(auc)
            
            print(f"  Test Video: {test_vid} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | AUC: {auc:.4f}")

            # --- Plotting Predictions (Time Series) ---
            plots_dir = os.path.join(BASE_PATH, 'results', 'modeling', 'trend_timeseries', dim)
            os.makedirs(plots_dir, exist_ok=True)
            
            fig, (ax_raw, ax_trend) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            # Top plot: Raw Dimension Value
            ax_raw.plot(y_raw_test, label=f'Raw {dim.capitalize()} Value', color='blue', alpha=0.7, linewidth=1.5)
            ax_raw.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax_raw.set_ylabel(f'{dim.capitalize()} (Joystick)')
            ax_raw.legend(loc='upper right')
            ax_raw.set_title(f"Dim: {dim} | Video: {test_vid} | Acc: {acc:.2f} | AUC: {auc:.2f}")
            
            # Bottom plot: True Trend + Decision Score
            ax_trend.plot(y_test, label='True Trend (Up/Down)', color='black', alpha=0.6, linewidth=2)
            ax_trend.set_ylabel('Class (1=Up, 0=Down)')
            ax_trend.set_xlabel('Epochs')
            
            ax_score = ax_trend.twinx()
            ax_score.plot(y_score, label='Decision Score', color='purple', alpha=0.5, linestyle='--')
            ax_score.set_ylabel('Ridge Score')
            
            # Combined Legend for bottom plot
            lines1, labels1 = ax_trend.get_legend_handles_labels()
            lines2, labels2 = ax_score.get_legend_handles_labels()
            ax_trend.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            safe_vid = str(test_vid).replace('.','_').replace('/','_')
            plot_filename = f"trend_ts_{dim}_{safe_vid}.png"
            plt.savefig(os.path.join(plots_dir, plot_filename))
            plt.close()
            # ------------------------------------------
            
            results_list.append({
                'Subject': SUBJECT,
                'Acq': acq_map.get(test_vid, 'unknown'),
                'Dimension': dim,
                'TestVideo': test_vid,
                'TrainSize': len(y_train),
                'TestSize': len(y_test),
                'UpRate_Train': np.mean(y_train),
                'UpRate_Test': np.mean(y_test),
                'Accuracy': acc,
                'BalancedAccuracy': bal_acc,
                'AUC': auc
            })
            
        print(f"Average Acc: {np.mean(scores_acc):.4f}")
        print(f"Average Bal Acc: {np.mean(scores_bal_acc):.4f}")

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_dir = os.path.join(BASE_PATH, 'results', 'modeling')
        os.makedirs(results_dir, exist_ok=True)
        filename = f'sub-{SUBJECT}_combined_filtered_trend_classification_results.csv'
        output_path = os.path.join(results_dir, filename)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        print("\n--- Summary by Dimension ---")
        summary = results_df.groupby('Dimension')[['Accuracy', 'BalancedAccuracy', 'AUC']].mean()
        print(summary)

if __name__ == '__main__':
    run_pipeline()
