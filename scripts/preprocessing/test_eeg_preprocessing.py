"""
EEG preprocessing pipeline for a single participant.

- Modular, reproducible, and BIDS-compliant.
- Designed for interactive and script-based execution (VSCode/Jupyter #%% blocks).
- All steps are logged and outputs are saved in data/derivatives/.

Follows project rules: modularity, docstrings, reproducible paths, no magics.
"""

#%% 

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject, get_rejection_threshold
from git import Repo
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
from mne_icalabel import label_components
from pyprep import NoisyChannels

from campeones_analysis.utils import bids_compliance
from campeones_analysis.utils.log_preprocessing import LogPreprocessingDetails
from campeones_analysis.utils.preprocessing_helpers import (
    correct_channel_types,
    set_chs_montage,
)

# %%
# --------- Configuración de sujeto y paths ---------
repo = Repo(os.getcwd(), search_parent_directories=True)
repo_root = repo.git.rev_parse("--show-toplevel")

# %%
# Parámetros editables
subject = "17"
session = "vr"
task = "01"
run = "002"
acq = "a"  # Antes era 'day', ahora es 'acq' para BIDS
data = "eeg"

# Paths para datos BIDS
bids_root = os.path.join(repo_root, "data", "raw")
derivatives_folder = os.path.join(repo_root, "data", "derivatives")
bids_dir = os.path.join(derivatives_folder, f"sub-{subject}", f"ses-{session}", "eeg")
os.makedirs(bids_dir, exist_ok=True)

# %%
# --------- 1. Cargar datos raw desde BIDS ---------
# Crear BIDSPath para los datos raw
bids_path = BIDSPath(
    subject=subject,
    session=session,
    task=task,
    run=run,
    acquisition=acq,
    datatype=data,
    root=bids_root,
    extension=".vhdr",
)

# Cargar datos raw desde BIDS
print(f"Cargando datos desde: {bids_path.fpath}")
raw = read_raw_bids(bids_path, verbose=False)
print(f"Datos cargados: {raw.info}")

# %%

# %%
# --------- 2. Cargar eventos desde merged_events ---------
# Ruta al archivo de eventos fusionados
events_path = os.path.join(
    derivatives_folder, 
    "merged_events",
    f"sub-{subject}",
    f"ses-{session}",
    "eeg",
    f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}_run-{run}_desc-merged_events.tsv"
)

# Cargar eventos
print(f"Cargando eventos desde: {events_path}")
events_df = pd.read_csv(events_path, sep='\t')
print(f"Eventos cargados: {len(events_df)} eventos")
print("Tipos de eventos:", events_df['trial_type'].unique())

# Verificar que los onset times sean razonables
print(f"Onset times: min={events_df['onset'].min():.1f}s, max={events_df['onset'].max():.1f}s")
print(f"Duraciones: min={events_df['duration'].min():.1f}s, max={events_df['duration'].max():.1f}s")

# NOTA: Asegúrate de usar merged_events (con onset times reales) 
# no events/ (que tiene todos los onset en 0.0)

# Convertir eventos a anotaciones MNE
annotations = mne.Annotations(
    onset=events_df['onset'].values,
    duration=events_df['duration'].values,
    description=events_df['trial_type'].values
)

# Agregar las anotaciones al raw
raw.set_annotations(annotations)
print(f"Anotaciones agregadas: {len(raw.annotations)}")

# %%
# --------- 3. Configurar reporte y logging ---------
report = mne.Report(
    title=f"Preprocessing sub-{subject} for session {session}, task {task}, run {run}"
)
json_path = os.path.join(
    repo_root, derivatives_folder, "logs_preprocessing_details_all_subjects_eeg.json"
)
log_preprocessing = LogPreprocessingDetails(json_path, subject, session, task)

# Cargar datos en memoria si no están ya cargados
raw.load_data()
raw.info["bads"] = log_preprocessing.import_bad_channels_another_task()

print(raw.info)

# Plot sensor location in the scalp
# raw.plot_sensors(show_names=True)
# plt.show()

# Add the raw data info to the report
report.add_raw(raw=raw, title="Raw", psd=True)
log_preprocessing.log_detail("info", str(raw.info))

# %%
# --------- 4. Filtrado ---------
hpass = 1.0
lpass = 64
raw_filtered = (
    raw.copy()
    .notch_filter(np.arange(50, 250, 50))
    .filter(l_freq=hpass, h_freq=lpass)
)


# Guardar raw filtrado como derivado BIDS
def save_filtered_bids(raw_filtered, bids_path, derivatives_folder):
    """Guarda el raw filtrado en formato BIDS en la carpeta de derivados."""
    filtered_bids_path = bids_path.copy().update(
        root=derivatives_folder, description="filtered"
    )
    write_raw_bids(
        raw_filtered,
        filtered_bids_path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
    )


save_filtered_bids(raw_filtered, bids_path, derivatives_folder)

# Log the filter settings
log_preprocessing.log_detail("hpass_filter", hpass)
log_preprocessing.log_detail("lpass_filter", lpass)
log_preprocessing.log_detail("filter_type", "bandpass")

# %%
# --------- 3. Inspección visual y automática de canales ---------
raw_filtered.compute_psd().plot()
nd = NoisyChannels(raw_filtered, do_detrend=False, random_state=42)
nd.find_all_bads(ransac=True, channel_wise=True)
bads = nd.get_bads()
print(f"Bad channels detected: {bads}")
if bads:
    raw_filtered.info["bads"] = bads
raw_filtered.plot(n_channels=32)
plt.show(block=True)
report.add_raw(raw=raw_filtered, title="Filtered Raw", psd=True)
log_preprocessing.log_detail("bad_channels", raw_filtered.info["bads"])

# %%
# --------- 5. Crear epochs con duración variable basada en eventos ---------
# En lugar de usar ventanas fijas, vamos a crear epochs que cubran toda la duración
# de cada evento, ya que cada uno puede tener duración diferente

# Crear una lista de epochs individuales con duración variable
epochs_list = []
epochs_metadata = []

print(f"Creando epochs con duración variable para {len(events_df)} eventos...")

for idx, row in events_df.iterrows():
    onset_time = row['onset']
    duration = row['duration'] 
    trial_type = row['trial_type']
    
    # Crear epoch con período pre-estímulo para baseline correction
    # tmin = -0.3s antes del evento, tmax = duration después del onset
    tmin = -0.3  # 300ms antes del evento para baseline
    tmax = duration  # Duración completa del evento
    
    # Encontrar el onset en samples
    onset_sample = int(onset_time * raw_filtered.info['sfreq'])
    
    # Crear un evento temporal para este epoch
    temp_event = np.array([[onset_sample, 0, 1]])  # [sample, prev_id, event_id]
    temp_event_id = {trial_type: 1}
    
    try:
        # Crear epoch para este evento individual
        temp_epochs = mne.Epochs(
            raw_filtered,
            events=temp_event,
            event_id=temp_event_id,
            tmin=tmin,
            tmax=tmax,
            preload=False,
            verbose=False,
            baseline=None  # No aplicar baseline por ahora
        )
        
        if len(temp_epochs) > 0:  # Verificar que el epoch sea válido
            epochs_list.append(temp_epochs)
            epochs_metadata.append({
                'trial_type': trial_type,
                'onset': onset_time,
                'duration': duration,
                'epoch_idx': idx
            })
            
    except Exception as e:
        print(f"Error creando epoch para evento {idx} ({trial_type}): {e}")
        continue

print(f"Epochs creados exitosamente: {len(epochs_list)}")
print(f"Tipos de eventos procesados: {set([meta['trial_type'] for meta in epochs_metadata])}")

# Para compatibilidad con el resto del pipeline, usamos el primer epoch como referencia
# pero guardamos toda la información de epochs variables
if epochs_list:
    epochs = epochs_list[0]  # Para mantener compatibilidad con código existente
    
    # Log de información sobre epochs variables
    durations = [meta['duration'] for meta in epochs_metadata]
    log_preprocessing.log_detail("n_epochs_variable_duration", len(epochs_list))
    log_preprocessing.log_detail("min_duration", min(durations))
    log_preprocessing.log_detail("max_duration", max(durations))
    log_preprocessing.log_detail("mean_duration", np.mean(durations))
    log_preprocessing.log_detail("epochs_metadata", epochs_metadata)
    
    # Crear un epoch concatenado para análisis general (opcional)
    # Nota: Esto será útil para algunos análisis pero no todos
    print(f"Duración mínima: {min(durations):.2f}s, máxima: {max(durations):.2f}s")
    
else:
    print("No se pudieron crear epochs válidos")
    epochs = None

# %%
# --------- 6. Rechazo automático y manual de epochs (adaptado para duración variable) ---------
# Con epochs de duración variable, aplicamos el rechazo a cada epoch individualmente
if epochs_list and len(epochs_list) > 0:
    print("Aplicando rechazo automático a epochs de duración variable...")
    
    # Procesar cada epoch individualmente
    epochs_clean_list = []
    all_reject_logs = []
    
    for i, temp_epochs in enumerate(epochs_list):
        trial_type = epochs_metadata[i]['trial_type']
        duration = epochs_metadata[i]['duration']
        
        try:
            # Aplicar AutoReject solo si el epoch tiene suficiente duración
            if duration > 0.5:  # Solo aplicar si la duración es mayor a 0.5s
                # Usar parámetros más conservadores para epochs individuales
                ar = AutoReject(
                    thresh_method="random_search",  # Más rápido para epochs individuales
                    cv=3,  # Menos folds para epochs individuales
                    random_state=42,
                    n_jobs=1
                )
                temp_epochs_clean = ar.fit_transform(temp_epochs)
            else:
                # Para eventos muy cortos, solo aplicar threshold básico
                reject_criteria = get_rejection_threshold(temp_epochs)
                temp_epochs_clean = temp_epochs.copy()
                temp_epochs_clean.drop_bad(reject=reject_criteria)
            
            epochs_clean_list.append(temp_epochs_clean)
            all_reject_logs.append(temp_epochs_clean.drop_log)
            
            print(f"Epoch {i} ({trial_type}, {duration:.2f}s): {len(temp_epochs_clean)} epochs válidos")
            
        except Exception as e:
            print(f"Error procesando epoch {i} ({trial_type}): {e}")
            # Si hay error, mantener el epoch original
            epochs_clean_list.append(temp_epochs)
            all_reject_logs.append(temp_epochs.drop_log)
    
    # Para compatibilidad con el resto del código, usar el primer epoch limpio
    if epochs_clean_list:
        epochs_clean = epochs_clean_list[0]
        
        # Calcular estadísticas de rechazo
        total_original = len(epochs_list)
        total_clean = len(epochs_clean_list)
        
        print(f"Epochs procesados: {total_original}")
        print(f"Epochs válidos después de limpieza: {total_clean}")
        
        # Log estadísticas
        log_preprocessing.log_detail("epochs_clean_variable_duration", len(epochs_clean_list))
        log_preprocessing.log_detail("epochs_original_count", total_original)
        log_preprocessing.log_detail("epochs_processing_success_rate", total_clean/total_original if total_original > 0 else 0)
        
        # Plotear el primer epoch limpio para inspección
        if len(epochs_clean) > 0:
            epochs_clean.plot(n_channels=32)
            plt.show(block=True)
            report.add_epochs(epochs=epochs_clean, title="Epochs clean (first example)", psd=False)
        
    else:
        print("Error: No se pudieron procesar epochs")
        epochs_clean = None
        
else:
    print("Error: No hay epochs para procesar")
    epochs_clean = None

# %%
# --------- 7. ICA y clasificación automática (adaptado para epochs variables) ---------
if epochs_clean is not None:
    print("Aplicando ICA a epochs de duración variable...")
    
    n_components = 15
    method = "picard"
    max_iter = "auto"
    random_state = 42
    
    # Entrenar ICA con el conjunto de epochs disponibles
    # Para epochs de duración variable, usamos todos los epochs válidos para entrenar
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=random_state,
    )
    
    # Usar raw_filtered para entrenar ICA (más estable que epochs variables)
    print("Entrenando ICA en datos raw filtrados...")
    ica.fit(raw_filtered)
    
    # Detectar componentes de artefactos usando epochs_clean como referencia
    try:
        eog_components, _ = ica.find_bads_eog(inst=epochs_clean, ch_name="R_EYE")
    except:
        eog_components = []
        print("No se pudo detectar EOG components")
        
    try:
        ecq_components, _ = ica.find_bads_ecg(inst=epochs_clean, ch_name="ECG")
    except:
        ecq_components = []
        print("No se pudo detectar ECG components")
        
    try:
        muscle_components, _ = ica.find_bads_muscle(epochs_clean, threshold=0.7)
    except:
        muscle_components = []
        print("No se pudo detectar muscle components")
        
    print(f"EOG components detected: {eog_components}")
    print(f"ECG components detected: {ecq_components}")
    print(f"Muscle components detected: {muscle_components}")
    
    # Clasificación automática con ICLabel
    try:
        ic_labels = label_components(epochs_clean, ica, method="iclabel")
        print("Classification of all ICA components. Results:")
        print(ic_labels["labels"])
        label_names = ic_labels["labels"]
    except Exception as e:
        print(f"Error en ICLabel classification: {e}")
        label_names = ["unknown"] * n_components
    
    # Determinar componentes a excluir
    pattern_matching_artifacts = np.unique(
        ecq_components + eog_components + muscle_components
    )
    
    channel_artifact_indices = [
        i for i, label in enumerate(label_names) if label == "channel noise"
    ]
    
    # Find components that coincide between pattern matching and ICLabel output for exclusion
    to_exclude = []
    for idx in pattern_matching_artifacts:
        if idx < len(label_names) and label_names[idx] in [
            "muscle artifact",
            "eye blink", 
            "heart beat",
            "channel noise",
        ]:
            to_exclude.append(idx)
            
    if len(eog_components) > 0 and eog_components[0] < 3:
        to_exclude.append(eog_components[0])
        
    to_exclude = np.unique(to_exclude + channel_artifact_indices)
    ica.exclude = to_exclude.tolist()
    
    print(f"Componentes ICA a excluir: {ica.exclude}")
    
    # Plot the sources identified by ICA
    ica.plot_sources(epochs_clean, block=True, show=True)
    plt.show(block=True)
    report.add_ica(ica, title="ICA", inst=epochs_clean)
    
    # Aplicar ICA a todos los epochs individuales
    epochs_ica_list = []
    for i, temp_epochs_clean in enumerate(epochs_clean_list):
        try:
            temp_epochs_ica = ica.apply(inst=temp_epochs_clean)
            epochs_ica_list.append(temp_epochs_ica)
        except Exception as e:
            print(f"Error aplicando ICA a epoch {i}: {e}")
            epochs_ica_list.append(temp_epochs_clean)  # Mantener sin ICA si hay error
    
    # Para compatibilidad, usar el primer epoch con ICA aplicado
    epochs_ica = epochs_ica_list[0] if epochs_ica_list else epochs_clean
    
    # Log detalles de ICA
    log_preprocessing.log_detail("ica_components", ica.exclude)
    log_preprocessing.log_detail("ica_method", method)
    log_preprocessing.log_detail("ica_max_iter", max_iter)
    log_preprocessing.log_detail("ica_random_state", random_state)
    log_preprocessing.log_detail("ica_applied_to_variable_epochs", len(epochs_ica_list))
    
else:
    print("Error: No hay epochs limpios para aplicar ICA")
    epochs_ica = None
    epochs_ica_list = []

# %%
# --------- 8. Limpieza final de epochs (adaptado para duración variable) ---------
if epochs_ica is not None and epochs_ica_list:
    print("Aplicando limpieza final a epochs de duración variable...")
    
    # Para epochs de duración variable, aplicamos baseline usando el período pre-estímulo
    # Cada epoch incluye 300ms antes del onset para baseline correction
    print("NOTA: Aplicando baseline correction usando período pre-estímulo (-0.3s a 0s)")
    print("Cada epoch incluye 300ms antes del onset del evento")
    
    # Procesar cada epoch individualmente para limpieza final
    epochs_final_list = []
    
    for i, temp_epochs_ica in enumerate(epochs_ica_list):
        trial_type = epochs_metadata[i]['trial_type']
        duration = epochs_metadata[i]['duration']
        
        # Clonar epochs para limpieza final
        temp_epochs_final = temp_epochs_ica.copy()
        
        # Aplicar baseline usando el período pre-estímulo
        try:
            # Usar el período pre-estímulo (-0.3 a 0) como baseline
            baseline = (-0.3, 0)
            temp_epochs_final.apply_baseline(baseline)
            print(f"Baseline aplicado a epoch {i} ({trial_type}): {baseline}")
        except Exception as e:
            print(f"No se pudo aplicar baseline a epoch {i}: {e}")
            # Si falla, intentar con baseline más corto
            try:
                baseline = (-0.1, 0)
                temp_epochs_final.apply_baseline(baseline)
                print(f"Baseline alternativo aplicado a epoch {i}: {baseline}")
            except:
                print(f"No se pudo aplicar ningún baseline a epoch {i}")
        
        epochs_final_list.append(temp_epochs_final)
        print(f"Epoch final {i}: {trial_type}, duración: {duration:.2f}s, samples: {len(temp_epochs_final)}")
    
    # Para compatibilidad, usar el primer epoch final
    epochs_ica = epochs_final_list[0] if epochs_final_list else epochs_ica
    
    # Plotear el primer epoch para inspección
    if len(epochs_ica) > 0:
        epochs_ica.plot(n_channels=32)
        plt.show(block=True)
    
    # Calcular estadísticas finales
    total_processed = len(epochs_final_list)
    total_original = len(epochs_list) if epochs_list else 0
    
    success_rate = total_processed / total_original * 100 if total_original > 0 else 0
    
    print(f"Epochs procesados exitosamente: {total_processed}/{total_original} ({success_rate:.1f}%)")
    
    # Log estadísticas finales
    log_preprocessing.log_detail("epochs_final_count", total_processed)
    log_preprocessing.log_detail("epochs_success_rate", success_rate)
    log_preprocessing.log_detail("baseline_period", "(-0.3, 0)")
    log_preprocessing.log_detail("baseline_applied_to_variable_epochs", True)
    log_preprocessing.log_detail("epochs_final_metadata", epochs_metadata)
    
else:
    print("Error: No hay epochs con ICA para limpieza final")
    epochs_final_list = []

# Save the epochs after ICA application and drop epochs
# bids_compliance.save_epoched_bids(epochs_ica, derivatives_folder, subject, session,
#                                   task, data, desc = 'epochedICA', events = events, event_id =event_id)


# %%
# --------- 9. Rereferencia e interpolación (adaptado para epochs variables) ---------
if epochs_final_list:
    print("Aplicando rereferencia e interpolación a epochs de duración variable...")
    
    # Aplicar rereferencia e interpolación a todos los epochs individuales
    epochs_interpolated_list = []
    
    for i, temp_epochs_final in enumerate(epochs_final_list):
        trial_type = epochs_metadata[i]['trial_type']
        duration = epochs_metadata[i]['duration']
        
        try:
            # Cargar datos en memoria si no están cargados
            temp_epochs_copy = temp_epochs_final.copy().load_data()
            
            # Agregar canal de referencia FCz
            temp_epochs_ref = mne.add_reference_channels(temp_epochs_copy, ref_channels=["FCz"])
            
            # Aplicar montaje
            bvef_file_path = os.path.join(repo_root, "BC-32_FCz_modified.bvef")
            montage = mne.channels.read_custom_montage(bvef_file_path)
            temp_epochs_ref.set_montage(montage)
            
            # Rereferencia a promedio
            temp_epochs_rereferenced, _ = mne.set_eeg_reference(
                inst=temp_epochs_ref, ref_channels="average", copy=True
            )
            
            # Interpolación de canales malos
            temp_epochs_interpolated = temp_epochs_rereferenced.copy().interpolate_bads()
            
            epochs_interpolated_list.append(temp_epochs_interpolated)
            
            print(f"Epoch {i} ({trial_type}, {duration:.2f}s): rereferencia e interpolación exitosa")
            
        except Exception as e:
            print(f"Error en rereferencia/interpolación para epoch {i}: {e}")
            # Mantener epoch original si hay error
            epochs_interpolated_list.append(temp_epochs_final)
    
    # Para compatibilidad con código existente
    if epochs_interpolated_list:
        epochs_interpolate = epochs_interpolated_list[0]
        
        # Agregar al reporte el primer ejemplo
        report.add_epochs(
            epochs=epochs_interpolate, 
            title="Epochs interpolated and rereferenced (first example)", 
            psd=True
        )
        
        print(f"Procesamiento completo: {len(epochs_interpolated_list)} epochs finales")
        
        # Log detalles de rereferencia
        log_preprocessing.log_detail("rereferenced_channels", "grand_average")
        log_preprocessing.log_detail("interpolated_channels", epochs_interpolate.info["bads"])
        log_preprocessing.log_detail("final_epochs_count", len(epochs_interpolated_list))
        
    else:
        print("Error: No se pudieron procesar epochs para rereferencia")
        epochs_interpolate = None
        epochs_interpolated_list = []
        
else:
    print("Error: No hay epochs finales para rereferencia e interpolación")
    epochs_interpolate = None
    epochs_interpolated_list = []

# %%
# --------- 10. Guardar archivos finales (adaptado para epochs variables) ---------
if epochs_interpolated_list and epochs_interpolate is not None:
    print("Guardando epochs de duración variable procesados...")
    
    # Guardar cada epoch individualmente con metadata específica
    for i, temp_epochs_interpolated in enumerate(epochs_interpolated_list):
        trial_type = epochs_metadata[i]['trial_type']
        duration = epochs_metadata[i]['duration']
        
        try:
            # Crear descripción específica para este epoch
            desc = f"preproc-{trial_type}-dur{duration:.1f}s"
            
            # Crear eventos específicos para este epoch
            temp_events = np.array([[0, 0, 1]])  # Evento al inicio
            temp_event_id = {trial_type: 1}
            
            # Guardar epoch individual
            bids_compliance.save_epoched_bids(
                temp_epochs_interpolated,
                derivatives_folder,
                subject,
                session,
                task,
                data,
                desc=desc,
                events=temp_events,
                event_id=temp_event_id,
            )
            
            print(f"Epoch {i} guardado: {trial_type} (duración: {duration:.2f}s)")
            
        except Exception as e:
            print(f"Error guardando epoch {i}: {e}")
    
    # También guardar un archivo conjunto con el primer epoch como referencia
    try:
        # Crear eventos básicos para compatibilidad
        basic_events = np.array([[0, 0, 1]])
        basic_event_id = {"variable_duration_epochs": 1}
        
        bids_compliance.save_epoched_bids(
            epochs_interpolate,
            derivatives_folder,
            subject,
            session,
            task,
            data,
            desc="preproc-variableDuration",
            events=basic_events,
            event_id=basic_event_id,
        )
        print("Archivo de referencia guardado: preproc-variableDuration")
        
    except Exception as e:
        print(f"Error guardando archivo de referencia: {e}")
    
    # Crear evoked responses para epochs que lo permitan
    evoked_list = []
    evoked_titles = []
    
    print("Creando evoked responses para epochs de duración variable...")
    
    for i, temp_epochs_interpolated in enumerate(epochs_interpolated_list[:5]):  # Limitar a 5
        trial_type = epochs_metadata[i]['trial_type']
        duration = epochs_metadata[i]['duration']
        
        try:
            if len(temp_epochs_interpolated) > 0:
                evoked = temp_epochs_interpolated.average()
                evoked_list.append(evoked)
                evoked_titles.append(f"Evoked {trial_type} ({duration:.1f}s)")
                print(f"Evoked creado para {trial_type}")
        except Exception as e:
            print(f"No se pudo crear evoked para {trial_type}: {e}")
    
    # Agregar evoked responses al reporte
    if evoked_list:
        report.add_evokeds(evokeds=evoked_list, titles=evoked_titles)
        print(f"Agregados {len(evoked_list)} evoked responses al reporte")
    
    # Guardar reporte HTML
    html_report_fname = bids_compliance.make_bids_basename(
        subject=subject,
        session=session,
        task=task,
        suffix=data,
        extension=".html",
        desc="preprocReport-variableDuration",
    )
    report.save(os.path.join(bids_dir, html_report_fname), overwrite=True)
    print(f"Reporte guardado: {html_report_fname}")
    
    # Guardar detalles de preprocessing
    log_preprocessing.save_preprocessing_details()
    print("Preprocessing completado exitosamente para epochs de duración variable")
    
else:
    print("Error: No hay epochs interpolados para guardar")

# Resumen final
if epochs_interpolated_list:
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Total de epochs procesados: {len(epochs_interpolated_list)}")
    print(f"Tipos de eventos: {set([meta['trial_type'] for meta in epochs_metadata])}")
    durations = [meta['duration'] for meta in epochs_metadata]
    print(f"Duración promedio: {np.mean(durations):.2f}s (min: {min(durations):.2f}s, max: {max(durations):.2f}s)")
    print(f"Archivos guardados en: {bids_dir}")
    print("=====================")

# %%
# =============================================================================
# CAMBIOS REALIZADOS PARA TRABAJAR CON DATOS BIDS
# =============================================================================
# 
# Este script ha sido modificado para trabajar con:
# 1. Datos raw en formato BIDS (data/raw/) en lugar de archivos XDF
# 2. Eventos procesados desde merged_events en lugar de triggers del raw
# 3. Estructura de directorios BIDS estándar
# 
# Parámetros de ejemplo:
# - subject: "17"  
# - session: "vr"
# - task: "01"
# - run: "002"
# - acq: "a"
#
# Los datos se cargan desde:
# - Raw: data/raw/sub-17/ses-vr/eeg/sub-17_ses-vr_task-01_acq-a_run-002_eeg.vhdr
# - Eventos: data/derivatives/merged_events/sub-17/ses-vr/eeg/sub-17_ses-vr_task-01_acq-a_run-002_desc-merged_events.tsv
# =============================================================================

