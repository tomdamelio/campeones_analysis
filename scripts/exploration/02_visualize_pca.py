
import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
RESULTS_DIR = os.path.join(BASE_PATH, 'results', 'exploration')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCH_DURATION = 1.0
EPOCH_OVERLAP = 0.9

def run_pca_visualization():
    all_epochs_data = [] 
    
    print(f"Starting PCA Visualization for Subject {SUBJECT}, Combined Acq A+B")

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
            if pd.notna(video_id):
                 vid_id_str = f"{video_id}_{acq}"
            else:
                 vid_id_str = f"lum_{run_id}_{i}_{acq}"

            # Crop to video
            t_start = onset
            t_stop = onset + duration
            video_data = preproc_data.copy().crop(tmin=t_start, tmax=t_stop)
            
            # Joystick data (Continuous Target)
            joy_data = video_data.get_data(picks=['joystick_x'])[0] 
            if polarity == 'inverse':
                joy_data = -joy_data
                
            # EEG Data (Features)
            eeg_picks = mne.pick_types(preproc_data.info, eeg=True, eog=False, stim=False, misc=False)
            eeg_data = video_data.get_data(picks=eeg_picks)
            
            # Epoching
            step = EPOCH_DURATION - EPOCH_OVERLAP # 0.1s
            sfreq = preproc_data.info['sfreq']
            n_samples_epoch = int(EPOCH_DURATION * sfreq)
            n_step_samples = int(step * sfreq)
            n_samples = video_data.n_times
            
            current_idx = 0
            prev_y_val = None
            
            while current_idx + n_samples_epoch <= n_samples:
                eeg_window = eeg_data[:, current_idx : current_idx + n_samples_epoch]
                joy_window = joy_data[current_idx : current_idx + n_samples_epoch]
                
                # Targets
                y_val = np.mean(joy_window) # Continuous
                
                # Trend Class
                y_class_trend = 0 # Default (or 'Stable' placeholder for now)
                trend_label = 'First/Stable'
                if prev_y_val is not None:
                    delta = y_val - prev_y_val
                    epsilon = 0.01
                    if delta > epsilon:
                        y_class_trend = 1
                        trend_label = 'Up'
                    elif delta < -epsilon:
                        y_class_trend = -1
                        trend_label = 'Down'
                    else:
                        y_class_trend = 0
                        trend_label = 'Stable'
                
                all_epochs_data.append({
                    'X': eeg_window, # (n_channels, n_times)
                    'y_cont': y_val,
                    'y_trend': trend_label,
                    'dimension': dimension,
                    'video_id': vid_id_str,
                    'acq': acq
                })
                
                prev_y_val = y_val
                current_idx += n_step_samples

    print(f"\nTotal Epochs Generated: {len(all_epochs_data)}")
    if not all_epochs_data:
        print("No epochs generated.")
        return

    # Process per Dimension separately
    unique_dims = set(d['dimension'] for d in all_epochs_data)
    
    for dim in unique_dims:
        print(f"\nComputing PCA for Dimension: {dim}")
        
        dim_data = [d for d in all_epochs_data if d['dimension'] == dim]
        X_raw = np.array([d['X'] for d in dim_data])
        y_cont = [d['y_cont'] for d in dim_data]
        y_trend = [d['y_trend'] for d in dim_data]
        
        # Pipeline: Vectorize -> Scale -> PCA
        # We need to extract features first to fit PCA
        # 1. Vectorize
        vectorizer = Vectorizer()
        X_vect = vectorizer.fit_transform(X_raw)
        
        # 2. Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_vect)
        
        # 3. PCA
        pca = PCA(n_components=3) # Get top 3
        X_pca = pca.fit_transform(X_scaled)
        
        explained_var = pca.explained_variance_ratio_
        print(f"Explained Variance (PC1, PC2, PC3): {explained_var}")
        
        # Create DataFrame for plotting
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Joystick Value'] = y_cont
        df_pca['Trend'] = y_trend
        
        # Plot 1: Color by Continuous Value
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Joystick Value', palette='viridis', alpha=0.6, s=15)
        plt.title(f'PCA Space (PC1 vs PC2) - {dim} - Continuous Value\nExpVar: {explained_var[:2].sum():.2%}')
        output_file_cont = os.path.join(RESULTS_DIR, f'pca_space_{dim}_continuous.png')
        plt.savefig(output_file_cont, dpi=300)
        plt.close()
        
        # Plot 2: Color by Trend Class
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Trend', palette='Set1', alpha=0.6, s=15)
        plt.title(f'PCA Space (PC1 vs PC2) - {dim} - Trend Class\nExpVar: {explained_var[:2].sum():.2%}')
        output_file_trend = os.path.join(RESULTS_DIR, f'pca_space_{dim}_trend.png')
        plt.savefig(output_file_trend, dpi=300)
        plt.close()
        
        print(f"Saved PCA plots for {dim}")
        
    # --- New: Global Cumulative Explained Variance Plot ---
    print("\nGenerating Global Cumulative Explained Variance Plot...")
    
    # 1. Fit Global PCA on ALL data
    X_all_raw = np.array([d['X'] for d in all_epochs_data])
    
    vectorizer_global = Vectorizer()
    X_all_vect = vectorizer_global.fit_transform(X_all_raw)
    
    scaler_global = StandardScaler()
    X_all_scaled = scaler_global.fit_transform(X_all_vect)
    
    # Fit 100 components on global data
    n_global_comp = 100
    pca_global = PCA(n_components=n_global_comp)
    pca_global.fit(X_all_scaled)
    
    global_variance_ratio = pca_global.explained_variance_ratio_
    global_cumulative_variance = np.cumsum(global_variance_ratio)
    
    print(f"Global PCA Variance (first 5 cumulative): {global_cumulative_variance[:5]}")
    
    # Create DataFrame for plotting
    df_cumulative = pd.DataFrame({
        'Component': range(1, n_global_comp + 1),
        'Cumulative Explained Variance': global_cumulative_variance
    })
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_cumulative, x='Component', y='Cumulative Explained Variance', marker='o', markersize=4)
    
    plt.xscale('log')
    plt.title('Global Cumulative Explained Variance (First 100 Components)')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.xlabel('Principal Component (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Add labels for specific points (e.g. 50, 100)
    final_var = global_cumulative_variance[-1]
    var_50 = global_cumulative_variance[49] # index 49 is 50th comp
    
    plt.axvline(x=50, color='r', linestyle='--', alpha=0.5)
    plt.text(50, var_50, f' 50 PCs: {var_50:.2%}', va='bottom')
    
    plt.text(100, final_var, f' 100 PCs: {final_var:.2%}', va='bottom')

    output_file = os.path.join(RESULTS_DIR, 'pca_global_cumulative_variance_logx.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

if __name__ == '__main__':
    run_pca_visualization()
