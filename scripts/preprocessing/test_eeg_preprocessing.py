"""
EEG preprocessing pipeline for a single participant.

- Modular, reproducible, and BIDS-compliant.
- Designed for interactive and script-based execution (VSCode/Jupyter #%% blocks).
- All steps are logged and outputs are saved in data/derivatives/.

Follows project rules: modularity, docstrings, reproducible paths, no magics.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pyxdf
from autoreject import AutoReject, get_rejection_threshold
from git import Repo
from mne_bids import BIDSPath, write_raw_bids
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
subject = "12"
session = "vr"
task = "01"
run = "002"
day = "a"
data = "eeg"

results_folder = "data"
derivatives_folder = os.path.join(repo_root, "data", "derivatives")
bids_dir = os.path.join(derivatives_folder, f"sub-{subject}", f"ses-{session}", "eeg")
os.makedirs(bids_dir, exist_ok=True)

# Ruta al archivo .xdf
df_xdf = (
    Path(repo_root)
    / f"data/sub-{subject}/ses-{session}/physio/sub-{subject}_ses-{session}_day-{day}_task-{task}_run-{run}_eeg.xdf"
)

# 1. Leer el archivo .xdf usando pyxdf y convertir todos los streams a MNE RawArray
streams, header = pyxdf.load_xdf(str(df_xdf))

print("Streams encontrados en el archivo XDF:")
for s in streams:
    print(s["info"]["name"][0])

# %%
# SEGUIR DESDE ACA
# Por algun motivo, esto no me corre bien. Parece que guardarlo como .xdf no fue una buena idea
# Considerar guardarlo como .fif para el futuro
# Para resolver ahora, hacer prubeas minimas leyendo un .xdf en otro script de testo
# Solamente leer un bloque (3 o 4 estimulos) y quedarnos con toda la data junta (todos los streams)
# Una vez que tenga eso, escalar con ese script para seguir con el siguiente bloque del pariticpante,
# y asi sucesivamente.
# Partir de pruebas chiquitas en `./test` e ir escalando con eso.
# Ir documentando todo en un nuevo documento dentro de `.docs` que sea tipo diario de investigacion
# para que no se nos olvide todo lo que hicimos.
# Avanzar lento pero entendiendo que estoy haciendo
# Chequear cuantos y cuales son los canales dentor de mi data, para estar seguro de que esoty entendiendo
# la naturaleza de mis datos.


# Función para convertir un stream a RawArray y guardar como .fif
def save_stream_as_fif(stream, out_dir, base_name):
    name = stream["info"]["name"][0]
    sfreq = float(stream["info"]["nominal_srate"][0])
    data = np.array(stream["time_series"]).T  # <--- CORREGIDO
    ch_names = [
        ch["label"][0] for ch in stream["info"]["desc"][0]["channels"][0]["channel"]
    ]
    # Heurística: si es EEG, todos 'eeg', si es markers, 'stim', si es joystick, 'misc'
    if "eeg" in name.lower() or "brainamp" in name.lower():
        ch_types = ["eeg"] * len(ch_names)
    elif "marker" in name.lower():
        ch_types = ["stim"] * len(ch_names)
    else:
        ch_types = ["misc"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    fif_path = Path(out_dir) / f"{base_name}_{name}_raw.fif"
    raw.save(fif_path, overwrite=True)
    print(f"Guardado: {fif_path}")
    return fif_path


# Guardar todos los streams como fif
base_name = f"sub-{subject}_ses-{session}_day-{day}_task-{task}_run-{run}"
for s in streams:
    save_stream_as_fif(s, bids_dir, base_name)

# %%

# Buscar el stream de EEG (ajusta el criterio según tu archivo)
eeg_stream = None
for s in streams:
    if "eeg" in s["info"]["name"][0].lower():
        eeg_stream = s
        break
if eeg_stream is None:
    raise RuntimeError("No se encontró un stream de EEG en el archivo XDF.")

# Extraer datos y metadatos
sfreq = float(eeg_stream["info"]["nominal_srate"][0])
data = eeg_stream["time_series"].T  # shape: (n_channels, n_times)
ch_names = [
    ch["label"][0] for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
]
ch_types = ["eeg"] * len(ch_names)

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)
print(raw.info)

# %%

# 2. Guardar como .fif en derivatives
fif_path = (
    Path(bids_dir)
    / f"sub-{subject}_ses-{session}_day-{day}_task-{task}_run-{run}_eeg_raw.fif"
)
raw.save(fif_path, overwrite=True)

# %%

# 3. (Opcional) Crear BIDSPath para el .fif si quieres usar herramientas BIDS
bids_path = BIDSPath(
    subject=subject,
    session=session,
    task=task,
    run=run,
    datatype=data,
    suffix="eeg",
    extension=".fif",
    root=derivatives_folder,
)

report = mne.Report(
    title=f"Preprocessing sub-{subject} for session {session}, task {task}, run {run}"
)
json_path = os.path.join(
    repo_root, derivatives_folder, "logs_preprocessing_details_all_subjects_eeg.json"
)
log_preprocessing = LogPreprocessingDetails(json_path, subject, session, task)


# %%
# --------- 1. Leer datos brutos ---------
# Si quieres seguir el pipeline con el raw .fif, puedes cargarlo así:
raw = mne.io.read_raw_fif(fif_path, preload=True)
raw.info["bads"] = log_preprocessing.import_bad_channels_another_task()

print(raw.info)

# Plot sensor location in the scalp
# raw.plot_sensors(show_names=True)
# plt.show()

# Add the raw data info to the report
report.add_raw(raw=raw, title="Raw", psd=True)
log_preprocessing.log_detail("info", str(raw.info))

# %%
# --------- 2. Filtrado ---------
hpass = 0.5
lpass = 48
raw_filtered = (
    raw.load_data()
    .copy()
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
# --------- 4. Cargar triggers ---------
filtered_annotations = mne.Annotations(onset=[], duration=[], description=[])
for ann in raw_filtered.annotations:
    if "go/" in ann["description"] or "nogo/" in ann["description"]:
        filtered_annotations.append(ann["onset"], ann["duration"], ann["description"])
raw_filtered.set_annotations(filtered_annotations)
events, event_id = mne.events_from_annotations(raw_filtered)

# %%
# --------- 5. Epoching ---------
tmin = -0.3
tmax = 1.2
# baseline correction should be done after ICA
epochs = mne.Epochs(
    raw_filtered,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    preload=False,
    verbose=False,
)
report.add_epochs(epochs=epochs, title="Epochs")
log_preprocessing.log_detail("n_epochs", len(epochs))
log_preprocessing.log_detail("tmin", tmin)
log_preprocessing.log_detail("tmax", tmax)

# %%
# --------- 6. Rechazo automático y manual de epochs ---------
# TODO: add rejection for acceloremeter (available from sub 14 onwards)
folds = 5
ar = AutoReject(
    thresh_method="bayesian_optimization", cv=folds, random_state=42, n_jobs=-1
)
epochs_clean = ar.fit_transform(epochs)
reject = get_rejection_threshold(epochs)
ar.get_reject_log(epochs).plot("horizontal")
ar_reject_epochs = [
    n_epoch
    for n_epoch, log in enumerate(epochs_clean.drop_log)
    if log == ("AUTOREJECT",)
]
log_preprocessing.log_detail("autoreject_epochs", ar_reject_epochs)
log_preprocessing.log_detail("autoreject_threshold", reject)
log_preprocessing.log_detail("len_autoreject_epochs", len(ar_reject_epochs))
epochs_clean.plot(n_channels=32)
plt.show(block=True)
manual_reject_epochs = [
    n_epoch for n_epoch, log in enumerate(epochs_clean.drop_log) if log == ("USER",)
]
print(f"Manually rejected epochs: {manual_reject_epochs}")
total_epochs_rejected = (
    (len(ar_reject_epochs) + len(manual_reject_epochs)) / len(epochs) * 100
)
print(f"Total epochs rejected: {total_epochs_rejected}%")
log_preprocessing.log_detail("manual_reject_epochs", manual_reject_epochs)
log_preprocessing.log_detail("len_manual_reject_epochs", len(manual_reject_epochs))
epochs_clean.plot_drop_log()
report.add_epochs(epochs=epochs_clean, title="Epochs clean", psd=False)
epochs_clean.drop_bad()

# %%
# --------- 7. ICA y clasificación automática ---------
n_components = 15
method = "picard"
max_iter = "auto"
random_state = 42
ica = mne.preprocessing.ICA(
    n_components=n_components,
    method=method,
    max_iter=max_iter,
    random_state=random_state,
)
ica.fit(epochs_clean)
eog_components, _ = ica.find_bads_eog(inst=epochs_clean, ch_name="R_EYE")
ecq_components, _ = ica.find_bads_ecg(inst=epochs_clean, ch_name="ECG")
muscle_components, _ = ica.find_bads_muscle(epochs_clean, threshold=0.7)
print(f"EOG components detected: {eog_components}")
print(f"ECG components detected: {ecq_components}")
print(f"Muscle components detected: {muscle_components}")
ic_labels = label_components(epochs_clean, ica, method="iclabel")
print("Classification of all ICA components. Results:")
print(ic_labels["labels"])
label_names = ic_labels["labels"]
pattern_matching_artifacts = np.unique(
    ecq_components + eog_components + muscle_components
)
channel_artifact_indices = [
    i for i, label in enumerate(label_names) if label == "channel noise"
]

# Find components that coincide between pattern matching and ICLabel output for exclusion
# We'll only exclude components that match the artifacts found via pattern matching
# and are classified as 'muscle artifact', 'eye blink', 'heart beat', or 'channel noise'
to_exclude = []
for idx in pattern_matching_artifacts:
    if label_names[idx] in [
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


# (Optional) Plot the ICA components for visual inspection
# ica.plot_components(inst=epochs_clean, picks=range(15))

# Plot the sources identified by ICA
ica.plot_sources(epochs_clean, block=True, show=True)
plt.show(block=True)
report.add_ica(ica, title="ICA", inst=epochs_clean)
epochs_ica = ica.apply(inst=epochs_clean)
log_preprocessing.log_detail("ica_components", ica.exclude)
log_preprocessing.log_detail("ica_method", method)
log_preprocessing.log_detail("ica_max_iter", max_iter)
log_preprocessing.log_detail("ica_random_state", random_state)

# %%
# --------- 8. Limpieza final de epochs ---------
baseline = (-0.3, 0)
epochs_ica.apply_baseline(baseline)
log_preprocessing.log_detail("baseline", baseline)
epochs_ica.plot(n_channels=32)
plt.show(block=True)
all_manual_epochs = [
    n_epoch for n_epoch, log in enumerate(epochs_ica.drop_log) if log == ("USER",)
]
manual_reject_epochs_after_ica = [
    n_epoch for n_epoch in all_manual_epochs if n_epoch not in manual_reject_epochs
]
print(f"Manually rejected epochs after ICA: {manual_reject_epochs_after_ica}")
total_epochs_rejected = (
    (
        len(ar_reject_epochs)
        + len(manual_reject_epochs)
        + len(manual_reject_epochs_after_ica)
    )
    / len(epochs)
    * 100
)
print(f"Total epochs rejected: {total_epochs_rejected}%")
log_preprocessing.log_detail(
    "manual_reject_epochs_after_ica", manual_reject_epochs_after_ica
)
log_preprocessing.log_detail(
    "len_manual_reject_epochs_after_ica", len(manual_reject_epochs_after_ica)
)
log_preprocessing.log_detail("total_epochs_rejected", total_epochs_rejected)
log_preprocessing.log_detail("epochs_drop_log", epochs_ica.drop_log)
log_preprocessing.log_detail("epochs_drop_log_description", epochs_ica.drop_log)

# Save the epochs after ICA application and drop epochs
# bids_compliance.save_epoched_bids(epochs_ica, derivatives_folder, subject, session,
#                                   task, data, desc = 'epochedICA', events = events, event_id =event_id)


# %%
# --------- 9. Rereferencia e interpolación ---------
epochs_ica = mne.add_reference_channels(epochs_ica.load_data(), ref_channels=["FCz"])
bvef_file_path = os.path.join(repo_root, "BC-32_FCz_modified.bvef")
montage = mne.channels.read_custom_montage(bvef_file_path)
epochs_ica.set_montage(montage)
epochs_rereferenced, _ = mne.set_eeg_reference(
    inst=epochs_ica, ref_channels="average", copy=True
)
report.add_epochs(
    epochs=epochs_rereferenced, title="Epochs interpolated and rereferenced", psd=True
)
log_preprocessing.log_detail("rereferenced_channels", "grand_average")
epochs_interpolate = epochs_rereferenced.copy().interpolate_bads()
log_preprocessing.log_detail("interpolated_channels", epochs_ica.info["bads"])

# %%
# --------- 10. Guardar archivos finales ---------
bids_compliance.save_epoched_bids(
    epochs_interpolate,
    derivatives_folder,
    subject,
    session,
    task,
    data,
    desc="preproc",
    events=events,
    event_id=event_id,
)
p300_evoked = mne.combine_evoked(
    [
        epochs_interpolate["go/correct"].average(),
        epochs_interpolate["nogo/correct"].average(),
    ],
    weights=[1, -1],
)
onoff_evoked = mne.combine_evoked(
    [
        epochs_interpolate["go/correct/on-task"].average(),
        epochs_interpolate["nogo/correct/spontaneous"].average(),
    ],
    weights=[1, -1],
)
report.add_evokeds(
    evokeds=[p300_evoked, onoff_evoked],
    titles=["Evoked P300 Go/Nogo", "Evoked On-tas vs Spontenous MW for go trials"],
)
html_report_fname = bids_compliance.make_bids_basename(
    subject=subject,
    session=session,
    task=task,
    suffix=data,
    extension=".html",
    desc="preprocReport",
)
report.save(os.path.join(bids_dir, html_report_fname), overwrite=True)
log_preprocessing.save_preprocessing_details()

# %%


def test_eeg_preprocessing():
    # Crear datos de prueba
    n_channels = 32
    n_times = 1000
    sfreq = 250

    # Crear datos aleatorios
    data = np.random.randn(n_channels, n_times)

    # Crear info
    ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Crear objeto Raw
    raw = mne.io.RawArray(data, info)

    # Aplicar preprocesamiento
    raw = correct_channel_types(raw)
    raw = set_chs_montage(raw)

    # Verificar que los canales están correctamente configurados
    assert len(raw.ch_names) == n_channels
    assert all(ch_type == "eeg" for ch_type in raw.get_channel_types())

    # Verificar que el montaje se aplicó correctamente
    assert raw.info["dig"] is not None
