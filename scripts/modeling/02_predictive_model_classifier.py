import mne
import pandas as pd
import numpy as np
import os
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Configuration
SUBJECT = '27'
SESSION = 'vr'
ACQ = 'a'
TASK_RUNS = {
    '002': 'block1',
    '003': 'block2',
    '004': 'block3',
    '006': 'block4'
}
BASE_PATH = r'c:\Users\tdamelio\Desktop\campeones_analysis'
DERIVATIVES_PATH = os.path.join(BASE_PATH, 'data', 'derivatives', 'campeones_preproc')
XDF_PATH = os.path.join(BASE_PATH, 'data', 'sourcedata', 'xdf')

EPOCH_DURATION = 1.0
EPOCH_OVERLAP = 0.9

def run_pipeline():
    all_epochs_data = [] # List to hold (X, y, meta) per video
    
    print(f"Starting CLASSIFICATION pipeline for Subject {SUBJECT}, Acq {ACQ}")

    for run_id, block_name in TASK_RUNS.items():
        print(f"\nProcessing Run {run_id} ({block_name})...")
        
        # Paths
        eeg_dir = os.path.join(DERIVATIVES_PATH, f'sub-{SUBJECT}', f'ses-{SESSION}', 'eeg')
        
        task_map = {'002': '01', '003': '02', '004': '03', '006': '04'}
        task_id = task_map.get(run_id)
        
        vhdr_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{ACQ}_run-{run_id}_desc-preproc_eeg.vhdr')
        events_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{ACQ}_run-{run_id}_desc-preproc_events.tsv')
        xlsx_file = os.path.join(XDF_PATH, f'sub-{SUBJECT}', f'order_matrix_{SUBJECT}_{ACQ.upper()}_{block_name}_VR.xlsx')

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
                
            step = EPOCH_DURATION - EPOCH_OVERLAP 
            
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
                
                # Target: Binary Class (Mean > 0 -> 1, else 0)
                y_val_cont = np.mean(joy_window)
                y_class = 1 if y_val_cont > 0 else 0
                
                all_epochs_data.append({
                    'X': eeg_window,
                    'y': y_class,
                    'dimension': dimension,
                    'video_identifier': video_id if pd.notna(video_id) else f'lum_{run_id}_{i}' 
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
        print(f"Unique videos found: {len(unique_videos)} ({unique_videos})")
        
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        groups = np.array([d['video_identifier'] for d in dim_data])
        
        print(f"Data Shape: X={X_all.shape}, y={y_all.shape}")
        
        # Check overall imbalance
        pos_count = np.sum(y_all == 1)
        neg_count = np.sum(y_all == 0)
        print(f"Overall Balance: Pos={pos_count}, Neg={neg_count} (Pos Rate: {pos_count/len(y_all):.2%})")
        
        scores_acc = []
        scores_bal_acc = []
        scores_auc = []
        
        for test_vid in unique_videos:
            train_mask = groups != test_vid
            test_mask = groups == test_vid
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            # Check test balance
            if len(np.unique(y_test)) < 2:
                print(f"  Warning: Test fold {test_vid} has only one class ({np.unique(y_test)}). AUC undefined.")
            
            pipeline = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=100),
                RidgeClassifier(alpha=1.0)
            )
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Decision function for AUC
            try:
                y_score = pipeline.decision_function(X_test)
            except:
                y_score = y_pred # Fallback if needed
            
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
            
            results_list.append({
                'Subject': SUBJECT,
                'Acq': ACQ,
                'Dimension': dim,
                'TestVideo': test_vid,
                'TrainSize': len(y_train),
                'TestSize': len(y_test),
                'PosRate_Train': np.mean(y_train),
                'PosRate_Test': np.mean(y_test),
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
        filename = f'sub-{SUBJECT}_acq-{ACQ}_classification_results.csv'
        output_path = os.path.join(results_dir, filename)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        print("\n--- Summary by Dimension ---")
        summary = results_df.groupby('Dimension')[['Accuracy', 'BalancedAccuracy', 'AUC']].mean()
        print(summary)

if __name__ == '__main__':
    run_pipeline()
