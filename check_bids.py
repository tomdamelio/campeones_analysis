import os
import json
import pandas as pd
from pathlib import Path

def check_bids_files():
    """Verificar los archivos BIDS clave"""
    print("Verificando estructura BIDS...")
    
    # Verificar dataset_description.json
    try:
        with open('./data/raw/dataset_description.json', 'r') as f:
            dataset_desc = json.load(f)
            print("\nDataset Description:")
            print(f"  Nombre: {dataset_desc.get('Name')}")
            print(f"  Versión BIDS: {dataset_desc.get('BIDSVersion')}")
    except Exception as e:
        print(f"Error al leer dataset_description.json: {e}")
    
    # Verificar participants.tsv
    try:
        participants_path = './data/raw/participants.tsv'
        if os.path.exists(participants_path) and os.path.getsize(participants_path) > 0:
            participants = pd.read_csv(participants_path, sep='\t')
            print("\nParticipants.tsv:")
            print(participants)
        else:
            print("\nParticipants.tsv está vacío o no existe")
    except Exception as e:
        print(f"Error al leer participants.tsv: {e}")
    
    # Verificar archivo de eventos
    try:
        events_path = './data/raw/sub-18/ses-vr/eeg/sub-18_ses-vr_task-01_acq-a_run-001_events.tsv'
        if os.path.exists(events_path):
            events = pd.read_csv(events_path, sep='\t')
            print("\nEvents.tsv:")
            print(events.head())
            print(f"\nTotal de eventos: {len(events)}")
            
            # Verificar tipos de eventos
            if 'trial_type' in events.columns:
                print("\nTipos de eventos:")
                print(events['trial_type'].value_counts())
        else:
            print("\nArchivo de eventos no encontrado")
    except Exception as e:
        print(f"Error al leer archivo de eventos: {e}")
    
    # Verificar archivo de canales
    try:
        channels_path = './data/raw/sub-18/ses-vr/eeg/sub-18_ses-vr_task-01_acq-a_run-001_channels.tsv'
        if os.path.exists(channels_path):
            channels = pd.read_csv(channels_path, sep='\t')
            print("\nChannels.tsv:")
            print(f"Total de canales: {len(channels)}")
            
            # Verificar tipos de canales
            if 'type' in channels.columns:
                print("\nTipos de canales:")
                print(channels['type'].value_counts())
        else:
            print("\nArchivo de canales no encontrado")
    except Exception as e:
        print(f"Error al leer archivo de canales: {e}")
    
    # Verificar scans.tsv
    try:
        scans_path = './data/raw/sub-18/ses-vr/sub-18_ses-vr_scans.tsv'
        if os.path.exists(scans_path):
            scans = pd.read_csv(scans_path, sep='\t')
            print("\nScans.tsv:")
            print(scans)
        else:
            print("\nArchivo scans.tsv no encontrado")
    except Exception as e:
        print(f"Error al leer archivo scans.tsv: {e}")

if __name__ == "__main__":
    check_bids_files() 