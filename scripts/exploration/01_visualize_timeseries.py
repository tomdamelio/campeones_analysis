
import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_visualization():
    all_traces = [] 
    
    print(f"Starting Time Series Visualization for Subject {SUBJECT}, Combined Acq A+B")

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
        # We only need joystick channel for this script, but reading raw usually reads everything or headers
        # We'll load raw but valid picks
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
            
            # Format Video ID
            if pd.notna(video_id):
                 vid_id_str = f"{video_id}_{acq}"
            else:
                 vid_id_str = f"lum_{run_id}_{i}_{acq}"
            
            print(f"  Video {vid_id_str}: Dim={dimension}, Pol={polarity}, Dur={duration:.1f}s")
            
            t_start = onset
            t_stop = onset + duration
            
            # Crop to video duration
            video_data = preproc_data.copy().crop(tmin=t_start, tmax=t_stop)
            
            # Get Joystick data
            joy_data = video_data.get_data(picks=['joystick_x'])[0] 
            times = video_data.times # Time relative to crop start (0 to duration)
            
            # Apply Polarity Correction
            if polarity == 'inverse':
                joy_data = -joy_data
                
            # Downsample for plotting if too dense (optional, but good for speed/viz)
            # Assuming sfreq is high (e.g. 500Hz), let's decimate or just plot all
            # For 1-3 mins, 500Hz is ~90k points. A bit heavy for seaborn lineplot with many hues.
            # Let's resample to 10Hz (100ms) which is enough for trend visualization
            target_sfreq = 10
            decim = int(preproc_data.info['sfreq'] / target_sfreq)
            if decim > 1:
                joy_data = joy_data[::decim]
                times = times[::decim]

            # Store trace
            # Create a dataframe for this trace
            df_trace = pd.DataFrame({
                'Time': times,
                'Value': joy_data,
                'VideoID': vid_id_str,
                'Dimension': dimension,
                'Acquisition': acq
            })
            
            all_traces.append(df_trace)

    print(f"\nTotal Traces Extracted: {len(all_traces)}")
    if len(all_traces) == 0:
        print("No traces extracted. Exiting.")
        return

    full_df = pd.concat(all_traces, ignore_index=True)
    
    # Visualization
    sns.set_theme(style="whitegrid")
    unique_dims = full_df['Dimension'].unique()
    
    for dim in unique_dims:
        print(f"Generating plot for {dim}...")
        dim_data = full_df[full_df['Dimension'] == dim]
        
        plt.figure(figsize=(12, 6))
        
        # Plot each video as a separate line
        sns.lineplot(data=dim_data, x='Time', y='Value', hue='VideoID', style='Acquisition', 
                     palette='tab20', linewidth=1.5)
        
        plt.title(f'Joystick Time Series - {dim} (Subject {SUBJECT})')
        plt.xlabel('Time (s)')
        plt.ylabel('Joystick Value (Corrected)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        output_file = os.path.join(RESULTS_DIR, f'timeseries_{dim}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved: {output_file}")

if __name__ == '__main__':
    run_visualization()
