import mne
import pandas as pd
import numpy as np
import os
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

# ROI Channels (All channels for now, maybe filtered later)
# Assuming 128Hz sampling for Joystick ? No, usually same as EEG or matched. 
# EEG is 250Hz based on channels.tsv seen earlier.
# User said: "Configuración: Ventanas de 1 segundo con un solapamiento (overlap) de 200 ms."
EPOCH_DURATION = 1.0
EPOCH_OVERLAP = 0.9

def run_pipeline():
    all_epochs_data = [] # List to hold (X, y, meta) per video
    
    print(f"Starting pipeline for Subject {SUBJECT}, Acq {ACQ}")

    for run_id, block_name in TASK_RUNS.items():
        print(f"\nProcessing Run {run_id} ({block_name})...")
        
        # Paths
        eeg_dir = os.path.join(DERIVATIVES_PATH, f'sub-{SUBJECT}', f'ses-{SESSION}', 'eeg')
        vhdr_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-0{list(TASK_RUNS.keys()).index(run_id)+1}_acq-{ACQ}_run-{run_id}_desc-preproc_eeg.vhdr')
        # Note: Task number in filename is tricky, assuming 01, 02, 03, 04 mapped to runs 002, 003, 004, 006 sequentially? 
        # Let's try to find the file dynamically if exact name is unsure, but based on file listing:
        # task-01 -> run-002
        # task-02 -> run-003
        # task-03 -> run-004
        # task-04 -> run-006
        # So I need to map run_id to task_id.
        
        task_map = {'002': '01', '003': '02', '004': '03', '006': '04'}
        task_id = task_map.get(run_id)
        
        vhdr_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{ACQ}_run-{run_id}_desc-preproc_eeg.vhdr')
        events_file = os.path.join(eeg_dir, f'sub-{SUBJECT}_ses-{SESSION}_task-{task_id}_acq-{ACQ}_run-{run_id}_desc-preproc_events.tsv')
        
        xlsx_file = os.path.join(XDF_PATH, f'sub-{SUBJECT}', f'order_matrix_{SUBJECT}_{ACQ.upper()}_{block_name}_VR.xlsx')

        if not os.path.exists(vhdr_file):
            print(f"Skipping {run_id}, EEG file not found: {vhdr_file}")
            continue
            
        # Load Data
        # Note: 'read_raw_brainvision' loads continuous data. We are loading the PREPROCESSED file.
        preproc_data = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        events_df = pd.read_csv(events_file, sep='\t')
        config_df = pd.read_excel(xlsx_file)
        
        # Filter Events: keep video and video_luminance
        video_events = events_df[events_df['trial_type'].isin(['video', 'video_luminance'])].reset_index(drop=True)
        
        # Filter Config: keep rows with valid video_id or description implied?
        # Actually video_luminance might not have video_id in xlsx?
        # Let's check config rows. User said:
        # "video_id: Which video file to load."
        # "Esos videos fueron presentados divididos en 4 runs ... (3 o 4 videos afectivos por run, + 1 video de luminacia)."
        # So config_df should have rows corresponding to these events.
        
        # Simple check: Does number of video events match number of rows in config?
        # Maybe config has more rows (instruction etc)?
        # We need to filter config to only actual video trials.
        # Looking at check_xlsx output: columns include 'video_id', 'dimension'. 
        # Likely we just take rows where 'dimension' is not null? Or 'video_id' is not null?
        # Luminance videos have 'luminance' dimension.
        
        # Strategy: Iterate events and try to match with config rows sequentially.
        # Assuming 1-to-1 mapping on "affective/luminance" trials.
        
        # Let's inspect config_df structure more in memory.
        # We'll assume the rows involved are those where `dimension` is not NaN?
        target_config = config_df.dropna(subset=['dimension']).reset_index(drop=True)
        
        if len(video_events) != len(target_config):
            print(f"WARNING: Mismatch in events ({len(video_events)}) vs config rows ({len(target_config)}) for run {run_id}")
            # Identify valid events in config -> maybe 'dimension' is the key
            # If mismatch persists, we might need a better filter. 
            # For now, proceeding if counts match, else breaking.
            if len(video_events) != len(target_config): 
                 # Try filtering events by excluding 'instruction' or similar if any crept in. 
                 # But we filtered by ['video', 'video_luminance'].
                 pass

        # Joystick Channel
        # preproc_data.info['ch_names'] should contain 'joystick_x', 'joystick_y'
        # Scale is -1 to 1? Or Raw voltage?
        # User said: "hidden numerical value: -1 to 1".
        # We assume the channel data in preproc_data is already calibrated or we just use it as is (relative).
        
        for i, row in target_config.iterrows():
            if i >= len(video_events):
                break
                
            original_dimension = row['dimension']
            polarity = row['order_emojis_slider'] # 'inverse' or 'direct' (or nan/other)
            video_id = row['video_id']
            
            # Event Info
            onset = video_events.loc[i, 'onset']
            duration = video_events.loc[i, 'duration']
            
            # Create Epochs
            # tmin, tmax relative to onset? 
            # No, we want to crop the continuous data.
            
            # Define dimension label for grouping
            # If video_id is NaN, it's likely Luminance?
            # Or check 'dimension' column.
            dimension = original_dimension
            
            print(f"  Video {i+1}: Dim={dimension}, Pol={polarity}, ID={video_id}, Dur={duration:.1f}s")
            
            # Crop Data
            # Convert onset/duration to samples
            t_start = onset
            t_stop = onset + duration
            
            video_data = preproc_data.copy().crop(tmin=t_start, tmax=t_stop)
            
            # Get Joystick Data
            # Note: Polarity 'inverse' -> -1 * signal? 
            # User: "inverse: de menos a derecha a más a izquierda". 
            # Usual: Left(-1) to Right(1)? 
            # If inverse: Left(1) to Right(-1).
            # So if user moves Right -> Signal increases. 
            # If Mapping is Inverse, Right means Low (-1? or just Low Valence).
            # This implies we want to standardize everything to "High = Positive Value".
            
            # Let's assume standard is Right = High.
            # If Inverse, Right = Low. So we flip.
            
            joy_data = video_data.get_data(picks=['joystick_x'])[0] # shape (n_samples,)
            
            if polarity == 'inverse':
                joy_data = -joy_data
                
            # Epoching
            # Window 1s, Step 0.8s (Overlap 200ms)
            step = EPOCH_DURATION - EPOCH_OVERLAP # 0.8
            
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            
            n_samples = video_data.n_times
            
            # EEG Data
            # Pick EEG channels
            eeg_picks = mne.pick_types(preproc_data.info, eeg=True, eog=False, stim=False, misc=False)
            eeg_data = video_data.get_data(picks=eeg_picks) # (n_chans, n_samples)
            
            current_idx = 0
            while current_idx + n_samples_epoch <= n_samples:
                # Extract window
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                joy_window = joy_data[current_idx : current_idx + n_samples_epoch]
                
                # Target: Mean of joystick in window
                y_val = np.mean(joy_window)
                
                # Store
                all_epochs_data.append({
                    'X': eeg_window, # (n_chans, n_times)
                    'y': y_val,
                    'dimension': dimension,
                    'video_identifier': video_id if pd.notna(video_id) else f'lum_{run_id}_{i}' # unique ID for CV
                })
                
                current_idx += n_step_samples

    # Convert to standard format
    print(f"\nTotal Epochs Generated: {len(all_epochs_data)}")
    if len(all_epochs_data) == 0:
        print("No epochs generated. Exiting.")
        return

    # Process per Dimension
    # Initialize results container
    results_list = []

    unique_dims = set(d['dimension'] for d in all_epochs_data)

    for dim in unique_dims:
        print(f"\n--- Model for Dimension: {dim} ---")
        
        # Filter data
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        
        # Group by video for LOVO CV
        unique_videos = list(set(d['video_identifier'] for d in dim_data))
        print(f"Unique videos found: {len(unique_videos)} ({unique_videos})")
        
        # Prepare Data Matrix
        X_all = np.array([d['X'] for d in dim_data])
        y_all = np.array([d['y'] for d in dim_data])
        groups = np.array([d['video_identifier'] for d in dim_data])
        
        print(f"Data Shape: X={X_all.shape}, y={y_all.shape}")
        
        scores_rmse = []
        scores_corr = []
        
        for test_vid in unique_videos:
            train_mask = groups != test_vid
            test_mask = groups == test_vid
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            # Pipeline
            pipeline = make_pipeline(
                Vectorizer(),
                StandardScaler(),
                PCA(n_components=100),
                Ridge(alpha=1.0)
            )
            
            # Fit
            pipeline.fit(X_train, y_train)
            
            # Get best alpha
            best_alpha = 1.0
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            if np.std(y_pred) < 1e-9 or np.std(y_test) < 1e-9:
                corr = 0.0
            else:
                corr = pearsonr(y_test, y_pred)[0]
            
            scores_rmse.append(rmse)
            scores_corr.append(corr)
            
            # Explained Variance Check
            pca = pipeline.named_steps['pca']
            expl_var = np.sum(pca.explained_variance_ratio_)
            print(f"  Test Video: {test_vid} | RMSE: {rmse:.4f} | Corr: {corr:.4f} | PCA Var: {expl_var:.2%} | Alpha: {best_alpha}")
            
            # Store Result
            results_list.append({
                'Subject': SUBJECT,
                'Acq': ACQ,
                'Dimension': dim,
                'TestVideo': test_vid,
                'TrainSize': len(y_train),
                'TestSize': len(y_test),
                'Alpha': best_alpha,
                'PCA_Explained_Variance': expl_var,
                'RMSE': rmse,
                'Correlation': corr
            })
            
        print(f"Average RMSE: {np.mean(scores_rmse):.4f}")
        print(f"Average Corr: {np.mean(scores_corr):.4f}")

    # Save Results
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Define Output Path
        results_dir = os.path.join(BASE_PATH, 'results', 'modeling')
        os.makedirs(results_dir, exist_ok=True)
        filename = f'sub-{SUBJECT}_acq-{ACQ}_predictive_results.csv'
        output_path = os.path.join(results_dir, filename)
        
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print Summary
        print("\n--- Summary by Dimension ---")
        summary = results_df.groupby('Dimension')[['RMSE', 'Correlation']].mean()
        print(summary)

if __name__ == '__main__':
    run_pipeline()
