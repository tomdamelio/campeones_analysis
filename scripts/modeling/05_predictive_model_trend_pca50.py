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
EPOCH_OVERLAP = 0.9 # Keeps same overlap as Test 3

def run_pipeline():
    all_epochs_data = [] 
    
    print(f"Starting TREND CLASSIFICATION (PCA 50) pipeline for Subject {SUBJECT}")

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
            
            print(f"  Video {i+1}: Dim={dimension}, Pol={polarity}")
            
            video_data = preproc_data.copy().crop(tmin=onset, tmax=onset + duration)
            joy_data = video_data.get_data(picks=['joystick_x'])[0] 
            
            if polarity == 'inverse':
                joy_data = -joy_data
                
            step = EPOCH_DURATION - EPOCH_OVERLAP
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
                    epsilon = 0.001 
                    
                    if abs(delta) <= epsilon:
                        prev_y_val = y_val
                        current_idx += n_step_samples
                        continue
                        
                    y_class = 1 if delta > 0 else 0
                    
                    if pd.notna(video_id):
                         vid_id_str = f"{video_id}_{acq}"
                    else:
                         vid_id_str = f"lum_{run_id}_{i}_{acq}"

                    video_epochs.append({
                        'X': eeg_window,
                        'y': y_class,
                        'delta': delta,
                        'y_raw': y_val,
                        'dimension': dimension,
                        'video_identifier': vid_id_str,
                        'acq': acq
                    })
                
                prev_y_val = y_val
                current_idx += n_step_samples
            
            all_epochs_data.extend(video_epochs)

    print(f"\nTotal Epochs Generated: {len(all_epochs_data)}")
    if not all_epochs_data: return

    unique_dims = set(d['dimension'] for d in all_epochs_data)
    results_list = []

    for dim in unique_dims:
        print(f"\n--- TREND PCA-50 Classifer for Dimension: {dim} ---")
        
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        unique_videos = list(set(d['video_identifier'] for d in dim_data))
        
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        y_raw_all = np.array([d['y_raw'] for d in dim_data])
        groups = np.array([d['video_identifier'] for d in dim_data])
        acq_map = {d['video_identifier']: d['acq'] for d in dim_data}
        
        print(f"Data Shape: X={X_all.shape}, y={y_all.shape}")
        
        scores_acc = []
        scores_bal_acc = []
        scores_auc = []
        
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
                PCA(n_components=50), # Changed to 50
                RidgeClassifier(alpha=1.0, class_weight='balanced')
            )
            
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                try:
                    y_score = pipeline.decision_function(X_test)
                except:
                    y_score = y_pred
            except Exception as e:
                 print(f"Error fitting {test_vid}: {e}")
                 continue
            
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else np.nan
            except: auc = np.nan
            
            scores_acc.append(acc)
            scores_bal_acc.append(bal_acc)
            scores_auc.append(auc)
            
            print(f"  Test Video: {test_vid} | Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | AUC: {auc:.4f}")
            
            # --- Plotting ---
            plots_dir = os.path.join(BASE_PATH, 'results', 'modeling', 'trend_timeseries_pca50', dim)
            os.makedirs(plots_dir, exist_ok=True)
            
            fig, (ax_raw, ax_trend) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            ax_raw.plot(y_raw_test, label=f'Raw {dim.capitalize()}', color='blue', alpha=0.7)
            ax_raw.set_title(f"Dim: {dim} | Video: {test_vid} | BalAcc: {bal_acc:.2f} | AUC: {auc:.2f} (PCA-50)")
            
            ax_trend.plot(y_test, label='True Trend', color='black', alpha=0.6)
            ax_score = ax_trend.twinx()
            ax_score.plot(y_score, label='Score', color='purple', alpha=0.5, linestyle='--')
            
            safe_vid = str(test_vid).replace('.','_').replace('/','_')
            plt.savefig(os.path.join(plots_dir, f'trend_pca50_{dim}_{safe_vid}.png'))
            plt.close()
            
            results_list.append({
                'Subject': SUBJECT,
                'Acq': acq_map.get(test_vid, 'unknown'),
                'Dimension': dim,
                'TestVideo': test_vid,
                'TrainSize': len(y_train),
                'TestSize': len(y_test),
                'Accuracy': acc,
                'BalancedAccuracy': bal_acc,
                'AUC': auc
            })

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_dir = os.path.join(BASE_PATH, 'results', 'modeling')
        os.makedirs(results_dir, exist_ok=True)
        filename = f'sub-{SUBJECT}_combined_filtered_trend_classification_results_pca50.csv'
        results_df.to_csv(os.path.join(results_dir, filename), index=False)
        
        print("\n--- Summary by Dimension (PCA 50) ---")
        print(results_df.groupby('Dimension')[['Accuracy', 'BalancedAccuracy', 'AUC']].mean())

if __name__ == '__main__':
    run_pipeline()
