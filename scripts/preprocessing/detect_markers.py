#!/usr/bin/env python
"""
Script para la Fase C del proceso de análisis:
- Detecta automáticamente marcadores en los canales AUDIO y PHOTO
- Para AUDIO: Utiliza detección basada en amplitud o filtrado en banda (según configuración)
- Para PHOTO: Utiliza análisis de frecuencia para detectar parpadeos visuales de 2 Hz
- Busca picos coincidentes entre ambos canales (marcadores audiovisuales)
- Crea anotaciones a partir de los picos detectados, priorizando coincidencias
- Visualiza las señales con las anotaciones automáticas
- Permite editar manualmente las anotaciones (habilitado por defecto)
- Fusiona las anotaciones finales con los eventos originales y guarda en derivatives/merged_events

Los marcadores audiovisuales son señales que aparecen simultáneamente en los videos
del experimento: un sonido de silbato (whistle) de 500 Hz y un parpadeo visual (flicker) de 2 Hz.

El marcador alrededor de cada vídeo experimental se compone de una pantalla negra silenciosa
de 2 s, seguida de un parpadeo audiovisual de 1 s que alterna el blanco y el negro
a 60 fps mientras suena un silbido de 500 Hz, y termina con otra pantalla
negra silenciosa de 2 s. Esta estructura de 2 + 1 + 2 s se inserta tanto antes
como después de cada clip de estímulo.

La detección utiliza técnicas específicas para cada tipo de señal:
- Para el canal AUDIO: detección basada en amplitud (por defecto) o filtrado en banda alrededor de 500 Hz
- Para el canal PHOTO: filtrado en banda alrededor de 2 Hz + detección de envolvente

Uso:
    # Uso básico (solo crea merged_events):
    python detect_markers.py --subject 16 --session vr --task 02 --run 003 --acq a
    
    # Para guardar también archivos intermedios (opcional):
    python detect_markers.py --subject 16 --session vr --task 02 --run 003 --acq a --save-auto-events --save-edited-events

Parámetros importantes:
    --photo-distance: Distancia mínima entre picos en segundos (default: 25)
    --audio-threshold: Factor para el umbral de detección en el canal AUDIO (default: 2.0)
    --photo-threshold: Factor para el umbral de detección en el canal PHOTO (default: 1.5)
    --use-amplitude-detection: Usar detección basada en amplitud para el canal AUDIO (default: True)
    --whistle-freq: Frecuencia del silbido a detectar en Hz (default: 500)
    --whistle-bandwidth: Ancho de banda del filtro para detectar silbidos en Hz (default: 50)
    --whistle-duration: Duración mínima del silbido en segundos (default: 0.05)
    --flicker-freq: Frecuencia del parpadeo visual a detectar en Hz (default: 2)
    --flicker-bandwidth: Ancho de banda del filtro para detectar parpadeos en Hz (default: 0.5)
    --flicker-duration: Duración mínima del parpadeo en segundos (default: 0.8)
    --visualize-detection: Visualizar el proceso de detección de silbidos y parpadeos
    --compare-manual-auto: Comparar anotaciones automáticas con manuales (si existen)
    --enable-manual-edit: Habilitar edición manual de anotaciones (activado por defecto)
    --no-manual-edit: Deshabilitar edición manual de anotaciones
    --force-save: Forzar guardado de anotaciones editadas manualmente sin preguntar
    --manual-save-dir: Directorio donde guardar eventos editados manualmente (default: edited_events)
    --load-events: Cargar eventos existentes desde derivatives/events (activado por defecto)
    --no-load-events: No cargar eventos existentes
    --events-dir: Directorio dentro de derivatives donde buscar los eventos existentes (default: events)
    --events-desc: Descripción de los eventos a cargar (default: None)
    --merge-events: Fusionar los eventos existentes con las nuevas anotaciones, actualizando onsets y duraciones (activado por defecto)
    --no-merge-events: No fusionar los eventos existentes con las nuevas anotaciones
    --merged-save-dir: Directorio dentro de derivatives donde guardar los eventos fusionados (default: merged_events)
    --merged-desc: Descripción para los eventos fusionados (default: merged)
    --save-auto-events: Guardar también las anotaciones automáticas en auto_events (opcional)
    --save-edited-events: Guardar también las anotaciones editadas en edited_events (opcional)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.signal import find_peaks, butter, filtfilt, spectrogram

# Buscar la raíz del repositorio
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

# Asegurarse de que podemos importar módulos del proyecto
sys.path.insert(0, str(repo_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detecta automáticamente marcadores en los canales AUDIO/PHOTO"
    )
    parser.add_argument("--subject", type=str, required=True, 
                        help="ID del sujeto (e.g., '16')")
    parser.add_argument("--session", type=str, default="vr",
                        help="ID de la sesión (default: 'vr')")
    parser.add_argument("--task", type=str, required=True,
                        help="ID de la tarea (e.g., '02')")
    parser.add_argument("--run", type=str, required=True,
                        help="ID del run (e.g., '003')")
    parser.add_argument("--acq", type=str, default=None,
                        help="Parámetro de adquisición (e.g., 'a')")
    parser.add_argument("--audio-threshold", type=float, default=2.0,
                        help="Factor para el umbral de detección en el canal AUDIO (default: 2.0)")
    parser.add_argument("--photo-threshold", type=float, default=1.5,
                        help="Factor para el umbral de detección en el canal PHOTO (default: 1.5)")
    parser.add_argument("--photo-distance", type=float, default=25,
                        help="Distancia mínima entre picos en segundos (aplicada a ambos canales) (default: 25)")
    parser.add_argument("--max-time-diff", type=float, default=2.0,
                        help="Diferencia máxima de tiempo (en segundos) para considerar dos picos como coincidentes (default: 2.0)")
    parser.add_argument("--whistle-freq", type=float, default=500,
                        help="Frecuencia del silbido a detectar en Hz (default: 500)")
    parser.add_argument("--whistle-bandwidth", type=float, default=50,
                        help="Ancho de banda del filtro para detectar silbidos en Hz (default: 50)")
    parser.add_argument("--whistle-duration", type=float, default=0.05,
                        help="Duración mínima del silbido en segundos (default: 0.05)")
    parser.add_argument("--flicker-freq", type=float, default=2,
                        help="Frecuencia del parpadeo visual a detectar en Hz (default: 2)")
    parser.add_argument("--flicker-bandwidth", type=float, default=0.5,
                        help="Ancho de banda del filtro para detectar parpadeos en Hz (default: 0.5)")
    parser.add_argument("--flicker-duration", type=float, default=0.8,
                        help="Duración mínima del parpadeo en segundos (default: 0.8)")
    parser.add_argument("--use-amplitude-detection", action="store_true",
                        help="Usar detección basada en amplitud para el canal AUDIO en lugar de filtrado por frecuencia")
    parser.add_argument("--no-amplitude-detection", dest="use_amplitude_detection", action="store_false",
                        help="Usar filtrado por frecuencia para el canal AUDIO en lugar de detección basada en amplitud")
    parser.add_argument("--no-zscore", action="store_true",
                        help="No aplicar z-score a las señales")
    parser.add_argument("--save-dir", type=str, default="auto_events",
                        help="Directorio dentro de derivatives donde guardar los eventos (default: auto_events)")
    parser.add_argument("--visualize-detection", action="store_true",
                        help="Visualizar el proceso de detección de silbidos y parpadeos")
    parser.add_argument("--compare-manual-auto", action="store_true",
                        help="Comparar anotaciones automáticas con manuales (si existen)")
    parser.add_argument("--enable-manual-edit", action="store_true", default=True,
                        help="Habilitar edición manual de anotaciones en la visualización (activado por defecto)")
    parser.add_argument("--no-manual-edit", dest="enable_manual_edit", action="store_false",
                        help="Deshabilitar edición manual de anotaciones en la visualización")
    parser.add_argument("--force-save", action="store_true",
                        help="Forzar guardado de anotaciones editadas manualmente sin preguntar")
    parser.add_argument("--manual-save-dir", type=str, default="edited_events",
                        help="Directorio dentro de derivatives donde guardar eventos editados manualmente (default: edited_events)")
    parser.add_argument("--load-events", action="store_true", default=True,
                        help="Cargar eventos existentes desde derivatives/events (activado por defecto)")
    parser.add_argument("--no-load-events", dest="load_events", action="store_false",
                        help="No cargar eventos existentes")
    parser.add_argument("--events-dir", type=str, default="events",
                        help="Directorio dentro de derivatives donde buscar los eventos existentes (default: events)")
    parser.add_argument("--events-desc", type=str, default=None,
                        help="Descripción de los eventos a cargar (default: None)")
    parser.add_argument("--merge-events", action="store_true", default=True,
                        help="Fusionar los eventos existentes con las nuevas anotaciones, actualizando onsets y duraciones (activado por defecto)")
    parser.add_argument("--no-merge-events", dest="merge_events", action="store_false",
                        help="No fusionar los eventos existentes con las nuevas anotaciones")
    parser.add_argument("--merged-save-dir", type=str, default="merged_events",
                        help="Directorio dentro de derivatives donde guardar los eventos fusionados (default: merged_events)")
    parser.add_argument("--merged-desc", type=str, default="merged",
                        help="Descripción para los eventos fusionados (default: merged)")
    parser.add_argument("--save-auto-events", action="store_true",
                        help="Guardar también las anotaciones automáticas en auto_events (opcional)")
    parser.add_argument("--save-edited-events", action="store_true",
                        help="Guardar también las anotaciones editadas en edited_events (opcional)")
    
    # Establecer valores predeterminados
    parser.set_defaults(use_amplitude_detection=True, enable_manual_edit=True, load_events=True, merge_events=True)
    
    return parser.parse_args()


def load_raw_data(subject, session, task, run, acq=None):
    """
    Carga los datos raw originales.
    
    Parameters
    ----------
    subject : str
        ID del sujeto
    session : str
        ID de la sesión
    task : str
        ID de la tarea
    run : str
        ID del run
    acq : str, optional
        Parámetro de adquisición
    
    Returns
    -------
    tuple
        (raw, bids_path) con los datos raw y la ruta BIDS
    """
    print(f"\n=== Cargando datos para sub-{subject} ses-{session} task-{task} run-{run} ===\n")
    
    # Definir rutas
    bids_root = repo_root / 'data' / 'raw'
    
    # Asegurar formato correcto de parámetros
    if task and task.isdigit():
        task = task.zfill(2)
    if run and run.isdigit():
        run = run.zfill(3)
    if acq:
        acq = acq.lower()
    
    # Crear BIDSPath para datos raw
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        root=bids_root,
        extension='.vhdr'
    )
    
    print(f"Ruta de datos raw: {bids_path.fpath}")
    
    # Verificar que el archivo existe
    if not bids_path.fpath.exists():
        raise FileNotFoundError(f"Archivo raw no encontrado: {bids_path.fpath}")
    
    # Cargar datos raw
    raw = read_raw_bids(bids_path, verbose=False)
    print(f"Datos raw cargados: {raw}")
    
    return raw, bids_path


def apply_zscore_to_raw(raw):
    """
    Aplica z-score a los datos raw para normalizar las señales.
    Maneja casos problemáticos como señales constantes, NaN o varianza cero.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
        
    Returns
    -------
    mne.io.Raw
        Objeto Raw con los datos normalizados
    """
    import warnings
    
    # Crear una copia del raw para no modificar el original
    raw_zscore = raw.copy()
    
    # Cargar los datos en memoria
    data = raw_zscore.get_data()
    
    # Lista para almacenar canales problemáticos
    problematic_channels = []
    
    # Aplicar z-score a cada canal
    for i in range(data.shape[0]):
        channel_name = raw.ch_names[i]
        channel_data = data[i]
        
        # Verificar si hay NaN en el canal
        if np.isnan(channel_data).any():
            print(f"¡ADVERTENCIA! Canal {channel_name} contiene valores NaN. Usando nan_policy='omit'.")
            problematic_channels.append(f"{channel_name} (NaN)")
        
        # Verificar si el canal es constante (varianza cero)
        if np.var(channel_data) == 0:
            print(f"¡ADVERTENCIA! Canal {channel_name} tiene varianza cero (señal constante). Manteniendo valores originales.")
            problematic_channels.append(f"{channel_name} (varianza cero)")
            # Mantener los valores originales para canales constantes
            continue
        
        # Aplicar z-score con manejo de NaN
        try:
            zscore_data = stats.zscore(channel_data, nan_policy='omit')
            
            # Verificar si el resultado contiene NaN o infinitos
            if np.isnan(zscore_data).any() or np.isinf(zscore_data).any():
                print(f"¡ADVERTENCIA! Canal {channel_name} produjo NaN/Inf después del z-score. Manteniendo valores originales.")
                problematic_channels.append(f"{channel_name} (NaN/Inf post-zscore)")
                # Mantener los valores originales
                continue
            
            # Si todo está bien, usar los datos z-scoreados
            data[i] = zscore_data
            
        except Exception as e:
            print(f"¡ERROR! No se pudo aplicar z-score al canal {channel_name}: {e}")
            print(f"Manteniendo valores originales para el canal {channel_name}.")
            problematic_channels.append(f"{channel_name} (error: {str(e)})")
            # Mantener los valores originales
            continue
    
    # Mostrar resumen de canales problemáticos
    if problematic_channels:
        print(f"\nCanales con problemas en z-score ({len(problematic_channels)} de {data.shape[0]}):")
        for ch in problematic_channels:
            print(f"  - {ch}")
        print("Estos canales mantuvieron sus valores originales.\n")
    
    # Crear un nuevo objeto Raw con los datos procesados
    info = raw.info
    raw_zscore = mne.io.RawArray(data, info)
    
    # Copiar las anotaciones del raw original
    raw_zscore.set_annotations(raw.annotations)
    
    return raw_zscore


def detect_whistle_in_audio(raw, channel='AUDIO', target_freq=500, bandwidth=50, min_duration=0.05, threshold_factor=5, min_distance_sec=25, visualize_detection=False):
    """
    Detecta silbidos (whistles) en el canal de audio usando filtrado en banda
    y detección de envolvente para encontrar tonos sinusoidales de frecuencia específica.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    channel : str
        Nombre del canal de audio
    target_freq : float
        Frecuencia objetivo del silbido en Hz (default: 500 Hz)
    bandwidth : float
        Ancho de banda del filtro en Hz (default: 50 Hz)
    min_duration : float
        Duración mínima del silbido en segundos (default: 0.8 segundos)
    threshold_factor : float
        Factor para calcular el umbral como múltiplo de la desviación estándar (default: 5)
    min_distance_sec : float
        Distancia mínima entre picos en segundos
    visualize_detection : bool
        Si es True, visualiza el proceso de detección de silbidos
        
    Returns
    -------
    numpy.ndarray
        Array con los tiempos (en segundos) donde se detectaron silbidos
    """
    print(f"\n=== Detectando silbidos de {target_freq} Hz en el canal {channel} ===\n")
    
    # Verificar que el canal existe
    if channel not in raw.ch_names:
        print(f"¡ADVERTENCIA! Canal {channel} no encontrado. Buscando alternativas...")
        similar_channels = [ch for ch in raw.ch_names if channel.lower() in ch.lower()]
        if similar_channels:
            channel = similar_channels[0]
            print(f"Usando canal alternativo: {channel}")
        else:
            print(f"No se encontraron canales similares a {channel}. No se pueden detectar silbidos.")
            return np.array([])
    
    # Obtener los datos del canal y la frecuencia de muestreo
    audio_data = raw.get_data(picks=channel)[0]
    sfreq = raw.info['sfreq']
    
    # Verificar si la frecuencia de muestreo es suficiente para la frecuencia objetivo
    nyquist = sfreq / 2
    if target_freq > nyquist:
        print(f"¡ADVERTENCIA! La frecuencia objetivo ({target_freq} Hz) está por encima de la frecuencia de Nyquist ({nyquist} Hz).")
        print(f"Ajustando a la máxima frecuencia posible: {nyquist * 0.9} Hz")
        target_freq = nyquist * 0.9
        bandwidth = min(bandwidth, nyquist * 0.2)
    
    # 1. Filtrar la señal en la banda de frecuencia del silbido
    low_freq = max(0.1, target_freq - bandwidth/2)  # Asegurar que no sea menor que 0.1 Hz
    high_freq = min(nyquist * 0.95, target_freq + bandwidth/2)  # Asegurar que no se acerque demasiado a Nyquist
    
    print(f"Aplicando filtro pasabanda: {low_freq:.2f} - {high_freq:.2f} Hz (Nyquist: {nyquist:.2f} Hz)")
    
    # Diseñar filtro pasabanda
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Verificar que las frecuencias estén dentro del rango válido
    if low <= 0 or high >= 1:
        print("¡ADVERTENCIA! Frecuencias de filtro fuera de rango. Ajustando...")
        low = max(0.01, low)
        high = min(0.99, high)
        print(f"Frecuencias ajustadas: {low:.2f} - {high:.2f} (normalizado a Nyquist)")
    
    b, a = butter(4, [low, high], btype='band')
    
    # Aplicar filtro
    filtered_signal = filtfilt(b, a, audio_data)
    
    # 2. Calcular la envolvente de la señal filtrada
    analytic_signal = signal.hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    
    # 3. Suavizar la envolvente
    window_size = int(0.1 * sfreq)  # Ventana de 100 ms
    if window_size % 2 == 0:
        window_size += 1  # Asegurar que sea impar
    smoothed_envelope = signal.savgol_filter(envelope, window_size, 2)
    
    # 4. Calcular umbral adaptativo
    threshold = np.std(smoothed_envelope) * threshold_factor
    
    # 5. Detectar picos en la envolvente suavizada
    min_distance_samples = int(min_distance_sec * sfreq)
    min_width_samples = int(min_duration * sfreq)
    
    peaks, properties = find_peaks(
        smoothed_envelope, 
        height=threshold,
        distance=min_distance_samples,
        width=min_width_samples
    )
    
    # Convertir índices de picos a tiempos en segundos
    peak_times = peaks / sfreq
    
    print(f"Se detectaron {len(peak_times)} silbidos en el canal {channel}")
    
    # Visualizar la detección si se solicita
    if visualize_detection and len(peak_times) > 0:
        plt.figure(figsize=(15, 10))
        
        # Tiempo en segundos
        time = np.arange(len(audio_data)) / sfreq
        
        # Señal original
        plt.subplot(4, 1, 1)
        plt.plot(time, audio_data)
        plt.title(f'Señal original del canal {channel}')
        plt.xlabel('Tiempo (s)')
        
        # Señal filtrada
        plt.subplot(4, 1, 2)
        plt.plot(time, filtered_signal)
        plt.title(f'Señal filtrada en banda {low_freq:.2f}-{high_freq:.2f} Hz')
        plt.xlabel('Tiempo (s)')
        
        # Envolvente y picos
        plt.subplot(4, 1, 3)
        plt.plot(time, smoothed_envelope)
        plt.axhline(y=threshold, color='r', linestyle='--', label='Umbral')
        plt.plot(peak_times, smoothed_envelope[peaks], 'ro', label='Picos detectados')
        plt.title('Envolvente suavizada con picos detectados')
        plt.xlabel('Tiempo (s)')
        plt.legend()
        
        # Espectrograma
        plt.subplot(4, 1, 4)
        f, t, Sxx = spectrogram(audio_data, sfreq)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.ylabel('Frecuencia [Hz]')
        plt.xlabel('Tiempo [s]')
        plt.title('Espectrograma')
        plt.colorbar(label='Potencia/dB')
        
        # Marcar los picos en el espectrograma
        for peak_time in peak_times:
            plt.axvline(x=peak_time, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        print("Visualización del proceso de detección completada.")
    
    return peak_times


def detect_flicker_in_photo(raw, channel='PHOTO', flicker_freq=2, bandwidth=0.5, min_duration=0.8, threshold_factor=1.5, min_distance_sec=25, visualize_detection=False):
    """
    Detecta parpadeos visuales (flicker) en el canal PHOTO usando análisis de frecuencia
    para encontrar señales cuadradas de frecuencia específica.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    channel : str
        Nombre del canal de foto
    flicker_freq : float
        Frecuencia del parpadeo visual en Hz (default: 2 Hz)
    bandwidth : float
        Ancho de banda del filtro en Hz (default: 0.5 Hz)
    min_duration : float
        Duración mínima del parpadeo en segundos (default: 0.8 segundos)
    threshold_factor : float
        Factor para calcular el umbral como múltiplo de la desviación estándar (default: 2.0)
    min_distance_sec : float
        Distancia mínima entre picos en segundos
    visualize_detection : bool
        Si es True, visualiza el proceso de detección de parpadeos
        
    Returns
    -------
    numpy.ndarray
        Array con los tiempos (en segundos) donde se detectaron parpadeos
    """
    print(f"\n=== Detectando parpadeos visuales de {flicker_freq} Hz en el canal {channel} ===\n")
    
    # Verificar que el canal existe
    if channel not in raw.ch_names:
        print(f"¡ADVERTENCIA! Canal {channel} no encontrado. Buscando alternativas...")
        similar_channels = [ch for ch in raw.ch_names if channel.lower() in ch.lower()]
        if similar_channels:
            channel = similar_channels[0]
            print(f"Usando canal alternativo: {channel}")
        else:
            print(f"No se encontraron canales similares a {channel}. No se pueden detectar parpadeos.")
            return np.array([])
    
    # Obtener los datos del canal y la frecuencia de muestreo
    photo_data = raw.get_data(picks=channel)[0]
    sfreq = raw.info['sfreq']
    
    # Verificar si la frecuencia de muestreo es suficiente para la frecuencia objetivo
    nyquist = sfreq / 2
    if flicker_freq > nyquist:
        print(f"¡ADVERTENCIA! La frecuencia del parpadeo ({flicker_freq} Hz) está por encima de la frecuencia de Nyquist ({nyquist} Hz).")
        print(f"Ajustando a la máxima frecuencia posible: {nyquist * 0.9} Hz")
        flicker_freq = nyquist * 0.9
        bandwidth = min(bandwidth, nyquist * 0.2)
    
    # 1. Aplicar un filtro pasabanda centrado en la frecuencia del parpadeo
    low_freq = max(0.1, flicker_freq - bandwidth/2)  # Asegurar que no sea menor que 0.1 Hz
    high_freq = min(nyquist * 0.95, flicker_freq + bandwidth/2)  # Asegurar que no se acerque demasiado a Nyquist
    
    print(f"Aplicando filtro pasabanda: {low_freq:.2f} - {high_freq:.2f} Hz (Nyquist: {nyquist:.2f} Hz)")
    
    # Diseñar filtro pasabanda
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Verificar que las frecuencias estén dentro del rango válido
    if low <= 0 or high >= 1:
        print("¡ADVERTENCIA! Frecuencias de filtro fuera de rango. Ajustando...")
        low = max(0.01, low)
        high = min(0.99, high)
        print(f"Frecuencias ajustadas: {low:.2f} - {high:.2f} (normalizado a Nyquist)")
    
    b, a = butter(4, [low, high], btype='band')
    
    # Aplicar filtro
    filtered_signal = filtfilt(b, a, photo_data)
    
    # 2. Rectificar la señal (valor absoluto) para detectar la energía en la banda de frecuencia
    rectified_signal = np.abs(filtered_signal)
    
    # 3. Aplicar un filtro paso bajo para suavizar la señal rectificada
    cutoff = min(1.0, nyquist * 0.9)  # Hz, asegurar que no se acerque demasiado a Nyquist
    cutoff_norm = cutoff / nyquist
    
    if cutoff_norm >= 1:
        print("¡ADVERTENCIA! Frecuencia de corte del filtro paso bajo fuera de rango. Ajustando...")
        cutoff_norm = 0.99
        print(f"Frecuencia de corte ajustada: {cutoff_norm:.2f} (normalizado a Nyquist)")
    
    b, a = butter(4, cutoff_norm, btype='low')
    envelope = filtfilt(b, a, rectified_signal)
    
    # 4. Calcular umbral adaptativo
    threshold = np.std(envelope) * threshold_factor
    
    # 5. Detectar picos en la envolvente
    min_distance_samples = int(min_distance_sec * sfreq)
    min_width_samples = int(min_duration * sfreq)
    
    peaks, properties = find_peaks(
        envelope, 
        height=threshold,
        distance=min_distance_samples,
        width=min_width_samples,
        prominence=threshold * 0.1  # Criterio de prominencia reducido al 10%
    )
    
    # Convertir índices de picos a tiempos en segundos
    peak_times = peaks / sfreq
    
    print(f"Se detectaron {len(peak_times)} parpadeos visuales en el canal {channel}")
    
    # Visualizar la detección si se solicita
    if visualize_detection and len(peak_times) > 0:
        plt.figure(figsize=(15, 12))
        
        # Tiempo en segundos
        time = np.arange(len(photo_data)) / sfreq
        
        # Señal original
        plt.subplot(5, 1, 1)
        plt.plot(time, photo_data)
        plt.title(f'Señal original del canal {channel}')
        plt.xlabel('Tiempo (s)')
        
        # Señal filtrada
        plt.subplot(5, 1, 2)
        plt.plot(time, filtered_signal)
        plt.title(f'Señal filtrada en banda {low_freq:.2f}-{high_freq:.2f} Hz')
        plt.xlabel('Tiempo (s)')
        
        # Señal rectificada
        plt.subplot(5, 1, 3)
        plt.plot(time, rectified_signal)
        plt.title('Señal rectificada (valor absoluto)')
        plt.xlabel('Tiempo (s)')
        
        # Envolvente y picos
        plt.subplot(5, 1, 4)
        plt.plot(time, envelope)
        plt.axhline(y=threshold, color='r', linestyle='--', label='Umbral')
        plt.plot(peak_times, envelope[peaks], 'ro', label='Picos detectados')
        plt.title('Envolvente con picos detectados')
        plt.xlabel('Tiempo (s)')
        plt.legend()
        
        # Espectrograma
        plt.subplot(5, 1, 5)
        f, t, Sxx = spectrogram(photo_data, sfreq, nperseg=int(sfreq * 2))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.ylabel('Frecuencia [Hz]')
        plt.xlabel('Tiempo [s]')
        plt.title('Espectrograma')
        plt.colorbar(label='Potencia/dB')
        plt.ylim(0, min(10, nyquist))  # Limitar el rango de frecuencias para mejor visualización
        
        # Marcar los picos en el espectrograma
        for peak_time in peak_times:
            plt.axvline(x=peak_time, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Visualización adicional: FFT de un segmento con parpadeo
        if len(peak_times) > 0:
            # Tomar un segmento alrededor del primer pico detectado
            peak_idx = int(peak_times[0] * sfreq)
            segment_start = max(0, peak_idx - int(2 * sfreq))
            segment_end = min(len(photo_data), peak_idx + int(2 * sfreq))
            segment = photo_data[segment_start:segment_end]
            
            # Calcular FFT
            segment_time = np.arange(len(segment)) / sfreq
            segment_fft = np.abs(np.fft.rfft(segment))
            segment_freqs = np.fft.rfftfreq(len(segment), 1/sfreq)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(segment_time, segment)
            plt.title(f'Segmento con parpadeo detectado (alrededor de {peak_times[0]:.2f}s)')
            plt.xlabel('Tiempo [s]')
            
            plt.subplot(2, 1, 2)
            plt.plot(segment_freqs, segment_fft)
            plt.title('Espectro de frecuencia del segmento')
            plt.xlabel('Frecuencia [Hz]')
            plt.ylabel('Amplitud')
            plt.xlim(0, min(10, nyquist))  # Limitar para mejor visualización
            plt.axvline(x=flicker_freq, color='r', linestyle='--', label=f'{flicker_freq} Hz')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        print("Visualización del proceso de detección completada.")
    
    return peak_times


def detect_whistle_by_amplitude(raw, channel='AUDIO', window_size_sec=0.1, threshold_factor=2.0, min_duration=0.15, min_distance_sec=25, visualize_detection=False):
    """
    Detecta silbidos (whistles) en el canal de audio usando un enfoque basado en amplitud.
    Este método es útil cuando la frecuencia de muestreo es demasiado baja para detectar
    la frecuencia específica del silbido.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    channel : str
        Nombre del canal de audio
    window_size_sec : float
        Tamaño de la ventana para suavizado en segundos (default: 0.1)
    threshold_factor : float
        Factor para calcular el umbral como múltiplo de la desviación estándar (default: 2.0)
    min_duration : float
        Duración mínima del silbido en segundos (default: 0.3 segundos)
    min_distance_sec : float
        Distancia mínima entre picos en segundos
    visualize_detection : bool
        Si es True, visualiza el proceso de detección de silbidos
        
    Returns
    -------
    numpy.ndarray
        Array con los tiempos (en segundos) donde se detectaron silbidos
    """
    print(f"\n=== Detectando silbidos por amplitud en el canal {channel} ===\n")
    
    # Verificar que el canal existe
    if channel not in raw.ch_names:
        print(f"¡ADVERTENCIA! Canal {channel} no encontrado. Buscando alternativas...")
        similar_channels = [ch for ch in raw.ch_names if channel.lower() in ch.lower()]
        if similar_channels:
            channel = similar_channels[0]
            print(f"Usando canal alternativo: {channel}")
        else:
            print(f"No se encontraron canales similares a {channel}. No se pueden detectar silbidos.")
            return np.array([])
    
    # Obtener los datos del canal y la frecuencia de muestreo
    audio_data = raw.get_data(picks=channel)[0]
    sfreq = raw.info['sfreq']
    
    print(f"Frecuencia de muestreo del canal {channel}: {sfreq} Hz")
    print(f"Frecuencia de Nyquist: {sfreq/2} Hz")
    
    # 1. Calcular la envolvente de amplitud de la señal
    # Usar valor absoluto como aproximación simple
    amplitude = np.abs(audio_data)
    
    # 2. Suavizar la envolvente con un filtro de media móvil
    window_size = int(window_size_sec * sfreq)
    if window_size % 2 == 0:
        window_size += 1  # Asegurar que sea impar
    
    # Suavizar usando savgol_filter o convolución simple
    try:
        smoothed_amplitude = signal.savgol_filter(amplitude, window_size, 2)
    except:
        # Alternativa: filtro de media móvil
        window = np.ones(window_size) / window_size
        smoothed_amplitude = np.convolve(amplitude, window, mode='same')
    
    # 3. Calcular umbral adaptativo
    # Usar percentil alto (99%) para ser más robusto frente a valores atípicos
    percentile_99 = np.percentile(smoothed_amplitude, 99)
    median_amplitude = np.median(smoothed_amplitude)
    threshold = median_amplitude + threshold_factor * (percentile_99 - median_amplitude)
    
    print(f"Umbral de detección: {threshold:.2f} (mediana: {median_amplitude:.2f}, factor: {threshold_factor})")
    
    # 4. Detectar picos en la envolvente suavizada
    min_distance_samples = int(min_distance_sec * sfreq)
    min_width_samples = int(min_duration * sfreq)
    
    # Detectar todos los picos que superen el umbral
    peaks, properties = find_peaks(
        smoothed_amplitude, 
        height=threshold,
        distance=min_distance_samples,
        width=min_width_samples,
        prominence=threshold * 0.1  # Criterio de prominencia reducido al 10%
    )
    
    # Convertir índices de picos a tiempos en segundos
    peak_times = peaks / sfreq
    
    print(f"Se detectaron {len(peak_times)} silbidos en el canal {channel}")
    
    # Visualizar la detección si se solicita
    if visualize_detection and len(audio_data) > 0:
        plt.figure(figsize=(15, 10))
        
        # Tiempo en segundos
        time = np.arange(len(audio_data)) / sfreq
        
        # Señal original
        plt.subplot(4, 1, 1)
        plt.plot(time, audio_data)
        plt.title(f'Señal original del canal {channel}')
        plt.xlabel('Tiempo (s)')
        
        # Envolvente de amplitud
        plt.subplot(4, 1, 2)
        plt.plot(time, amplitude)
        plt.title('Envolvente de amplitud (valor absoluto)')
        plt.xlabel('Tiempo (s)')
        
        # Envolvente suavizada y picos
        plt.subplot(4, 1, 3)
        plt.plot(time, smoothed_amplitude)
        plt.axhline(y=threshold, color='r', linestyle='--', label='Umbral')
        if len(peak_times) > 0:
            plt.plot(peak_times, smoothed_amplitude[peaks], 'ro', label='Picos detectados')
        plt.title('Envolvente suavizada con picos detectados')
        plt.xlabel('Tiempo (s)')
        plt.legend()
        
        # Espectrograma
        plt.subplot(4, 1, 4)
        f, t, Sxx = spectrogram(audio_data, sfreq, nperseg=int(sfreq * 2))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.ylabel('Frecuencia [Hz]')
        plt.xlabel('Tiempo [s]')
        plt.title('Espectrograma')
        plt.colorbar(label='Potencia/dB')
        
        # Marcar los picos en el espectrograma
        for peak_time in peak_times:
            plt.axvline(x=peak_time, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        print("Visualización del proceso de detección completada.")
    
    return peak_times


def detect_markers(raw, audio_threshold=2.0, photo_threshold=1.5, photo_distance=25, 
                  whistle_freq=500, whistle_bandwidth=50, whistle_duration=0.05,
                  flicker_freq=2, flicker_bandwidth=0.5, flicker_duration=0.8,
                  use_amplitude_detection=True, visualize_detection=False):
    """
    Detecta marcadores en los canales AUDIO y PHOTO.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    audio_threshold : float, optional
        Factor para el umbral de detección en el canal AUDIO
    photo_threshold : float, optional
        Factor para el umbral de detección en el canal PHOTO
    photo_distance : float, optional
        Distancia mínima entre picos en segundos (aplicada a ambos canales)
    whistle_freq : float, optional
        Frecuencia del silbido a detectar en Hz
    whistle_bandwidth : float, optional
        Ancho de banda del filtro para detectar silbidos en Hz
    whistle_duration : float, optional
        Duración mínima del silbido en segundos
    flicker_freq : float, optional
        Frecuencia del parpadeo visual a detectar en Hz
    flicker_bandwidth : float, optional
        Ancho de banda del filtro para detectar parpadeos en Hz
    flicker_duration : float, optional
        Duración mínima del parpadeo en segundos
    use_amplitude_detection : bool, optional
        Si es True, usa detección basada en amplitud para el canal AUDIO
    visualize_detection : bool, optional
        Si es True, visualiza la detección de silbidos y parpadeos
        
    Returns
    -------
    dict
        Diccionario con los picos detectados en cada canal
    """
    print("\n=== Detectando marcadores en los canales AUDIO/PHOTO ===\n")
    
    # Verificar que los canales existen
    available_channels = raw.ch_names
    required_channels = ['AUDIO', 'PHOTO']
    
    # Diccionario para almacenar los picos detectados
    peaks = {}
    
    # Detectar picos en cada canal
    for channel in required_channels:
        if channel in available_channels:
            print(f"Detectando picos en el canal {channel}...")
            
            # Detectar picos según el canal
            if channel == 'AUDIO':
                # Para AUDIO: elegir método de detección
                if use_amplitude_detection:
                    # Usar el enfoque basado en amplitud
                    peak_times = detect_whistle_by_amplitude(
                        raw, 
                        channel=channel,
                        window_size_sec=0.1,
                        threshold_factor=audio_threshold,
                        min_duration=whistle_duration,
                        min_distance_sec=photo_distance,
                        visualize_detection=visualize_detection
                    )
                else:
                    # Usar el enfoque basado en frecuencia
                    peak_times = detect_whistle_in_audio(
                        raw, 
                        channel=channel,
                        target_freq=whistle_freq,
                        bandwidth=whistle_bandwidth,
                        min_duration=whistle_duration,
                        threshold_factor=audio_threshold,
                        min_distance_sec=photo_distance,
                        visualize_detection=visualize_detection
                    )
            elif channel == 'PHOTO':
                # Para PHOTO: usar la función especializada para detectar parpadeos visuales
                peak_times = detect_flicker_in_photo(
                    raw, 
                    channel=channel,
                    flicker_freq=flicker_freq,
                    bandwidth=flicker_bandwidth,
                    min_duration=flicker_duration,
                    threshold_factor=photo_threshold,
                    min_distance_sec=photo_distance,
                    visualize_detection=visualize_detection
                )
            
            # Almacenar los tiempos de picos
            peaks[channel] = peak_times
            
            print(f"Se detectaron {len(peak_times)} picos en el canal {channel}")
        else:
            print(f"¡ADVERTENCIA! Canal {channel} no encontrado en los datos.")
            
            # Intentar encontrar canales similares
            similar_channels = [ch for ch in available_channels if channel.lower() in ch.lower()]
            if similar_channels:
                print(f"Canales similares encontrados para {channel}: {similar_channels}")
                alt_channel = similar_channels[0]
                
                print(f"Usando canal alternativo: {alt_channel}")
                
                # Obtener los datos del canal alternativo
                if channel == 'AUDIO':
                    # Para AUDIO: elegir método de detección
                    if use_amplitude_detection:
                        # Usar el enfoque basado en amplitud
                        peak_times = detect_whistle_by_amplitude(
                            raw, 
                            channel=alt_channel,
                            window_size_sec=0.1,
                            threshold_factor=audio_threshold,
                            min_duration=whistle_duration,
                            min_distance_sec=photo_distance,
                            visualize_detection=visualize_detection
                        )
                    else:
                        # Usar el enfoque basado en frecuencia
                        peak_times = detect_whistle_in_audio(
                            raw, 
                            channel=alt_channel,
                            target_freq=whistle_freq,
                            bandwidth=whistle_bandwidth,
                            min_duration=whistle_duration,
                            threshold_factor=audio_threshold,
                            min_distance_sec=photo_distance,
                            visualize_detection=visualize_detection
                        )
                elif channel == 'PHOTO':
                    # Para PHOTO: usar la función especializada para detectar parpadeos visuales
                    peak_times = detect_flicker_in_photo(
                        raw, 
                        channel=alt_channel,
                        flicker_freq=flicker_freq,
                        bandwidth=flicker_bandwidth,
                        min_duration=flicker_duration,
                        threshold_factor=photo_threshold,
                        min_distance_sec=photo_distance,
                        visualize_detection=visualize_detection
                    )
                
                # Almacenar los tiempos de picos
                peaks[channel] = peak_times
                
                print(f"Se detectaron {len(peak_times)} picos en el canal alternativo {alt_channel}")
            else:
                print(f"No se encontraron canales similares para {channel}. No se detectarán picos en este canal.")
                peaks[channel] = np.array([])
    
    return peaks


def find_coincident_peaks(peaks, max_time_diff=2.0):
    """
    Encuentra picos coincidentes entre los canales AUDIO y PHOTO.
    
    Parameters
    ----------
    peaks : dict
        Diccionario con los picos detectados en cada canal
    max_time_diff : float, optional
        Diferencia máxima de tiempo (en segundos) para considerar dos picos como coincidentes
        
    Returns
    -------
    dict
        Diccionario con los picos coincidentes y no coincidentes
    """
    print("\n=== Buscando picos coincidentes entre AUDIO y PHOTO ===\n")
    
    # Obtener los picos de cada canal
    audio_peaks = peaks.get('AUDIO', np.array([]))
    photo_peaks = peaks.get('PHOTO', np.array([]))
    
    if len(audio_peaks) == 0 or len(photo_peaks) == 0:
        print("No hay suficientes picos para buscar coincidencias.")
        return {
            'coincident': [],
            'audio_only': audio_peaks,
            'photo_only': photo_peaks
        }
    
    # Listas para almacenar los resultados
    coincident_peaks = []
    audio_only = []
    photo_only = []
    
    # Marcar los picos ya emparejados
    audio_matched = np.zeros(len(audio_peaks), dtype=bool)
    photo_matched = np.zeros(len(photo_peaks), dtype=bool)
    
    # Buscar coincidencias
    for i, audio_time in enumerate(audio_peaks):
        # Calcular diferencias de tiempo con todos los picos de PHOTO
        time_diffs = np.abs(photo_peaks - audio_time)
        
        # Encontrar el pico de PHOTO más cercano
        if len(time_diffs) > 0:
            min_diff_idx = np.argmin(time_diffs)
            min_diff = time_diffs[min_diff_idx]
            
            # Si la diferencia es menor que el umbral y el pico de PHOTO no ha sido emparejado
            if min_diff <= max_time_diff and not photo_matched[min_diff_idx]:
                # Registrar la coincidencia usando el tiempo del canal AUDIO (en lugar del promedio)
                coincident_time = audio_time
                coincident_peaks.append(coincident_time)
                
                # Marcar ambos picos como emparejados
                audio_matched[i] = True
                photo_matched[min_diff_idx] = True
    
    # Recopilar picos no emparejados
    for i, matched in enumerate(audio_matched):
        if not matched:
            audio_only.append(audio_peaks[i])
    
    for i, matched in enumerate(photo_matched):
        if not matched:
            photo_only.append(photo_peaks[i])
    
    # Convertir a arrays de NumPy
    coincident_peaks = np.array(coincident_peaks)
    audio_only = np.array(audio_only)
    photo_only = np.array(photo_only)
    
    print(f"Se encontraron {len(coincident_peaks)} picos coincidentes entre AUDIO y PHOTO")
    print(f"Picos solo en AUDIO: {len(audio_only)}")
    print(f"Picos solo en PHOTO: {len(photo_only)}")
    
    return {
        'coincident': coincident_peaks,
        'audio_only': audio_only,
        'photo_only': photo_only
    }


def create_annotations_from_coincident_peaks(coincident_peaks):
    """
    Crea anotaciones a partir de los picos coincidentes.
    
    Parameters
    ----------
    coincident_peaks : dict
        Diccionario con los picos coincidentes y no coincidentes
        
    Returns
    -------
    mne.Annotations
        Objeto Annotations con las anotaciones creadas
    """
    print("\n=== Creando anotaciones a partir de los picos coincidentes ===\n")
    
    # Inicializar listas para las anotaciones
    onsets = []
    durations = []
    descriptions = []
    
    # Añadir picos coincidentes (mayor confianza)
    for peak_time in coincident_peaks['coincident']:
        onsets.append(peak_time)
        durations.append(1.0)  # Duración de 1 segundo
        descriptions.append("auto_AUDIOVISUAL")  # Marcador audiovisual (mayor confianza)
    
    # Añadir picos solo de audio (menor confianza)
    for peak_time in coincident_peaks['audio_only']:
        onsets.append(peak_time)
        durations.append(1.0)
        descriptions.append("auto_AUDIO_only")
    
    # Añadir picos solo de foto (menor confianza)
    for peak_time in coincident_peaks['photo_only']:
        onsets.append(peak_time)
        durations.append(1.0)
        descriptions.append("auto_PHOTO_only")
    
    # Si no hay picos, devolver anotaciones vacías
    if len(onsets) == 0:
        print("No se detectaron picos. Se devolverán anotaciones vacías.")
        return mne.Annotations([], [], [])
    
    # Ordenar las anotaciones por tiempo de inicio
    sorted_indices = np.argsort(onsets)
    onsets = np.array(onsets)[sorted_indices]
    durations = np.array(durations)[sorted_indices]
    descriptions = np.array(descriptions)[sorted_indices]
    
    # Crear objeto Annotations
    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions
    )
    
    print(f"Se crearon {len(annotations)} anotaciones automáticas")
    print(f"  - {len(coincident_peaks['coincident'])} marcadores audiovisuales")
    print(f"  - {len(coincident_peaks['audio_only'])} marcadores solo de audio")
    print(f"  - {len(coincident_peaks['photo_only'])} marcadores solo de foto")
    
    return annotations


def save_annotations_auto(raw, annotations, bids_path, save_dir="auto_events", 
                      audio_threshold=5, photo_threshold=1.5, photo_distance=25, 
                      whistle_freq=500, whistle_bandwidth=50, whistle_duration=0.05,
                      flicker_freq=2, flicker_bandwidth=0.5, flicker_duration=0.8,
                      use_amplitude_detection=True):
    """
    Guarda las anotaciones automáticas en un archivo TSV en derivatives/auto_events.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    annotations : mne.Annotations
        Objeto Annotations con las anotaciones automáticas
    bids_path : mne_bids.BIDSPath
        Ruta BIDS del archivo raw
    save_dir : str, optional
        Nombre del directorio dentro de derivatives donde guardar los eventos
    audio_threshold : float, optional
        Factor para el umbral de detección en el canal AUDIO
    photo_threshold : float, optional
        Factor para el umbral de detección en el canal PHOTO
    photo_distance : float, optional
        Distancia mínima entre picos en segundos
    whistle_freq : float, optional
        Frecuencia del silbido a detectar en Hz
    whistle_bandwidth : float, optional
        Ancho de banda del filtro para detectar silbidos en Hz
    whistle_duration : float, optional
        Duración mínima del silbido en segundos
    flicker_freq : float, optional
        Frecuencia del parpadeo visual a detectar en Hz
    flicker_bandwidth : float, optional
        Ancho de banda del filtro para detectar parpadeos en Hz
    flicker_duration : float, optional
        Duración mínima del parpadeo en segundos
    use_amplitude_detection : bool, optional
        Si es True, se usó detección basada en amplitud para el canal AUDIO
        
    Returns
    -------
    str
        Ruta del archivo guardado
    """
    # Extraer información del bids_path
    subject = bids_path.subject
    session = bids_path.session
    task = bids_path.task
    run = bids_path.run
    acq = bids_path.acquisition
    
    # Crear BIDSPath para guardar las anotaciones automáticas
    auto_root = repo_root / 'data' / 'derivatives' / save_dir
    
    output_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        suffix='events',
        description='autoann',  # Entidad desc-autoann para distinguir esta versión
        extension='.tsv',
        root=auto_root,
        check=False
    )
    
    # Crear el directorio si no existe
    os.makedirs(output_path.directory, exist_ok=True)
    
    # Convertir anotaciones a DataFrame
    annotations_df = pd.DataFrame({
        'onset': annotations.onset,
        'duration': annotations.duration,
        'trial_type': annotations.description
    })
    
    # Guardar el DataFrame como TSV
    annotations_df.to_csv(output_path.fpath, sep='\t', index=False)
    
    print(f"Anotaciones guardadas en: {output_path.fpath}")
    
    # Crear el archivo JSON asociado
    json_path = output_path.fpath.with_suffix('.json')
    
    # Información sobre el método de detección de audio
    audio_detection_method = "Detección basada en amplitud" if use_amplitude_detection else "Filtrado en banda de frecuencia"
    
    # Crear el contenido del JSON con los campos requeridos
    json_content = {
        # Campos estándar de columnas
        "onset": {"Description": "Event onset in seconds"},
        "duration": {"Description": "Event duration in seconds"},
        "trial_type": {"Description": "Event description/type"},
        
        # Añadir los metadatos como objetos para cumplir con el esquema BIDS
        "SidecarDescription": {
            "Description": "Eventos detectados automáticamente",
            "EventTypes": {
                "auto_AUDIOVISUAL": "Marcador audiovisual detectado simultáneamente en canales AUDIO y PHOTO",
                "auto_AUDIO_only": "Marcador detectado solo en canal AUDIO",
                "auto_PHOTO_only": "Marcador detectado solo en canal PHOTO"
            }
        },
        "ProcessingMethod": {
            "Algorithm": f"Canal AUDIO: {audio_detection_method}; Canal PHOTO: Filtrado en banda para parpadeos visuales",
            "Parameters": {
                "MinimumDistance": f"{photo_distance} segundos",
                "Audio": {
                    "Method": audio_detection_method,
                    "ThresholdFactor": audio_threshold,
                    "WhistleFrequency": f"{whistle_freq} Hz" if not use_amplitude_detection else "N/A",
                    "WhistleBandwidth": f"{whistle_bandwidth} Hz" if not use_amplitude_detection else "N/A",
                    "WhistleMinDuration": f"{whistle_duration} segundos"
                },
                "Photo": {
                    "ThresholdFactor": photo_threshold,
                    "FlickerFrequency": f"{flicker_freq} Hz",
                    "FlickerBandwidth": f"{flicker_bandwidth} Hz",
                    "FlickerMinDuration": f"{flicker_duration} segundos"
                }
            }
        },
        "GeneratedBy": {
            "Name": "detect_markers.py",
            "Version": "1.4",
            "Description": f"Detección automática de marcadores audiovisuales con {audio_detection_method} para silbidos y análisis de frecuencia para parpadeos visuales de 2 Hz"
        },
        "MetadataDate": {"DateCreated": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")}
    }
    
    # Guardar el JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4)
    
    # Crear dataset_description.json si no existe
    create_dataset_description(auto_root)
    
    return str(output_path.fpath)


def create_dataset_description(auto_root):
    """
    Crea el archivo dataset_description.json para la carpeta de derivados.
    
    Parameters
    ----------
    auto_root : pathlib.Path
        Ruta a la carpeta de derivados
    """
    dataset_desc_path = auto_root / 'dataset_description.json'
    
    if not dataset_desc_path.exists():
        print(f"Creando dataset_description.json en {auto_root}")
        
        # Crear el README si no existe (requerido por BIDS)
        readme_path = auto_root / 'README'
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("# Auto Events\n\n")
                f.write("Este directorio contiene eventos detectados automáticamente en los canales AUDIO/PHOTO.\n")
                f.write("Los eventos fueron detectados usando el script detect_markers.py.\n")
                f.write("\nEl proceso de detección busca marcadores audiovisuales (coincidencias entre picos de AUDIO y PHOTO)\n")
                f.write("que corresponden a estímulos de silbato (whistle) y parpadeo visual (flicker) presentados simultáneamente\n")
                f.write("en los videos del experimento.\n")
        
        # Crear manualmente el dataset_description.json para asegurar el formato correcto
        dataset_desc = {
            "Name": "auto_events",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "Authors": [
                "D'Amelio, Tomás Ariel",
                "COCUCO"
            ],
            "GeneratedBy": [{
                "Name": "detect_markers.py",
                "Version": "1.4",
                "Description": f"Detección automática de marcadores audiovisuales con detección basada en amplitud para silbidos y análisis de frecuencia para parpadeos visuales de 2 Hz"
            }],
            "SourceDatasets": [{
                "URL": "file:///../../raw"  # URL en formato URI válido
            }]
        }
        
        # Guardar el JSON
        with open(dataset_desc_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_desc, f, indent=4)
        
        print(f"Archivo dataset_description.json creado en: {dataset_desc_path}")


def visualize_signals_with_annotations(raw, annotations, apply_zscore=True):
    """
    Visualiza las señales con las anotaciones automáticas.
    Permite agregar, editar o eliminar anotaciones manualmente.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    annotations : mne.Annotations
        Objeto Annotations con las anotaciones automáticas
    apply_zscore : bool, optional
        Si es True, aplica z-score a las señales para normalizarlas
        
    Returns
    -------
    tuple
        (updated_annotations, has_changes) con las anotaciones actualizadas y un flag que indica si hubo cambios
    """
    print("\n=== Visualizando señales con anotaciones automáticas ===\n")
    
    # Verificar que los canales existen
    available_channels = raw.ch_names
    required_channels = ['AUDIO', 'PHOTO', 'joystick_x']  # Mantener joystick_x para visualización
    
    # Lista para almacenar los canales que realmente existen
    channels_to_pick = []
    
    for channel in required_channels:
        if channel in available_channels:
            channels_to_pick.append(channel)
        else:
            print(f"¡ADVERTENCIA! Canal {channel} no encontrado en los datos.")
            
            # Intentar encontrar canales similares
            similar_channels = [ch for ch in available_channels if channel.lower() in ch.lower()]
            if similar_channels:
                print(f"Canales similares encontrados para {channel}: {similar_channels}")
                channels_to_pick.extend(similar_channels[:1])  # Usar el primer canal similar
    
    if not channels_to_pick:
        print("No se encontraron los canales requeridos ni similares. Se usarán los primeros tres canales disponibles.")
        channels_to_pick = available_channels[:3]
    
    print(f"Se usarán los siguientes canales: {channels_to_pick}")
    
    # Crear una copia del raw para no modificar el original
    raw_plot = raw.copy()
    
    # Guardar el número original de anotaciones para comparar después
    original_annotations = annotations.copy()
    
    # Añadir anotaciones al raw_plot
    raw_plot.set_annotations(annotations)
    
    # Seleccionar solo los canales requeridos
    raw_plot = raw_plot.pick_channels(channels_to_pick)
    
    # Aplicar z-score si se solicita
    if apply_zscore:
        print("Aplicando z-score a las señales para normalización...")
        raw_plot = apply_zscore_to_raw(raw_plot)
        print("Z-score aplicado. Las señales están ahora en unidades de desviación estándar.")
    
    # Instrucciones para el usuario
    print("\n--- INSTRUCCIONES PARA AÑADIR ANOTACIONES MANUALMENTE ---")
    print("1. Presiona 'a' y arrastra para seleccionar una región")
    print("2. Ingresa un nombre para la anotación en la ventana emergente")
    print("   - Para coincidencias audiovisuales: 'auto_AUDIOVISUAL'")
    print("   - Para eventos solo de audio: 'auto_AUDIO_only'")
    print("   - Para eventos solo de foto: 'auto_PHOTO_only'")
    print("   - Para otros eventos, usa nombres descriptivos")
    print("3. Para eliminar una anotación: Haz clic derecho sobre ella")
    print("4. Para ajustar los límites: Arrastra los bordes de una anotación")
    print("5. Presiona 'j'/'k' para ajustar la escala vertical")
    print("6. Cierra la ventana para finalizar y guardar los cambios\n")
    
    # Visualizar con escalados explícitos para evitar problemas de división por cero
    print("Abriendo visualizador de MNE. Cierra la ventana para continuar.")
    
    # Definir escalados específicos para evitar problemas con canales misc
    scalings = {}
    for ch_name in raw_plot.ch_names:
        ch_type = raw_plot.get_channel_types(picks=ch_name)[0]
        if ch_type == 'misc':
            # Para canales misc, usar un escalado fijo que funcione bien con datos z-scoreados
            scalings[ch_type] = 2.0  # Escalado conservador para datos normalizados
        elif ch_type == 'eeg':
            scalings[ch_type] = 20e-6
        elif ch_type == 'eog':
            scalings[ch_type] = 150e-6
        elif ch_type == 'ecg':
            scalings[ch_type] = 5e-4
        else:
            # Para otros tipos, usar un escalado genérico
            scalings[ch_type] = 1.0
    
    fig = raw_plot.plot(
        title=f"Canales {', '.join(channels_to_pick)} con anotaciones automáticas",
        scalings=scalings,
        duration=540,
        start=0,
        show=True,
        block=True
    )
    
    # Obtener las anotaciones actualizadas después de cerrar el visualizador
    updated_annotations = raw_plot.annotations
    
    # Mostrar información sobre las anotaciones
    print("\nVisualizador cerrado.")
    print(f"Anotaciones originales: {len(original_annotations)}")
    print(f"Anotaciones actualizadas: {len(updated_annotations)}")
    
    # Verificar si realmente hubo cambios en las anotaciones
    has_changes = False
    
    # Si el número de anotaciones es diferente, definitivamente hay cambios
    if len(original_annotations) != len(updated_annotations):
        has_changes = True
    else:
        # Comparar cada anotación para ver si hay cambios
        for i in range(len(original_annotations)):
            if (original_annotations.onset[i] != updated_annotations.onset[i] or
                original_annotations.duration[i] != updated_annotations.duration[i] or
                original_annotations.description[i] != updated_annotations.description[i]):
                has_changes = True
                break
    
    return updated_annotations, has_changes


def compare_manual_and_auto_annotations(raw, manual_path, auto_path):
    """
    Compara las anotaciones manuales y automáticas.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    manual_path : str
        Ruta al archivo de anotaciones manuales
    auto_path : str
        Ruta al archivo de anotaciones automáticas
    """
    print("\n=== Comparando anotaciones manuales y automáticas ===\n")
    
    # Cargar anotaciones manuales
    manual_df = pd.read_csv(manual_path, sep='\t')
    manual_annot = mne.Annotations(
        onset=manual_df['onset'].values,
        duration=manual_df['duration'].values,
        description=manual_df['trial_type'].values
    )
    
    # Cargar anotaciones automáticas
    auto_df = pd.read_csv(auto_path, sep='\t')
    auto_annot = mne.Annotations(
        onset=auto_df['onset'].values,
        duration=auto_df['duration'].values,
        description=auto_df['trial_type'].values
    )
    
    # Verificar que los canales existen
    available_channels = raw.ch_names
    required_channels = ['AUDIO', 'PHOTO', 'joystick_x']  # Mantener joystick_x para visualización
    
    # Lista para almacenar los canales que realmente existen
    channels_to_pick = []
    
    for channel in required_channels:
        if channel in available_channels:
            channels_to_pick.append(channel)
        else:
            print(f"¡ADVERTENCIA! Canal {channel} no encontrado en los datos.")
            
            # Intentar encontrar canales similares
            similar_channels = [ch for ch in available_channels if channel.lower() in ch.lower()]
            if similar_channels:
                print(f"Canales similares encontrados para {channel}: {similar_channels}")
                channels_to_pick.extend(similar_channels[:1])  # Usar el primer canal similar
    
    if not channels_to_pick:
        print("No se encontraron los canales requeridos ni similares. Se usarán los primeros tres canales disponibles.")
        channels_to_pick = available_channels[:3]
    
    print(f"Se usarán los siguientes canales: {channels_to_pick}")
    
    # Crear una copia del raw para no modificar el original
    raw_man = raw.copy().pick_channels(channels_to_pick)
    raw_man.set_annotations(manual_annot)
    
    raw_auto = raw.copy().pick_channels(channels_to_pick)
    raw_auto.set_annotations(auto_annot)
    
    # Crear un raw combinado con ambas anotaciones
    raw_combined = raw.copy().pick_channels(channels_to_pick)
    
    # Combinar anotaciones
    combined_annot = mne.Annotations(
        onset=np.concatenate([manual_annot.onset, auto_annot.onset]),
        duration=np.concatenate([manual_annot.duration, auto_annot.duration]),
        description=np.concatenate([
            [f"manual_{desc}" for desc in manual_annot.description],
            [f"auto_{desc}" for desc in auto_annot.description]
        ])
    )
    
    raw_combined.set_annotations(combined_annot)
    
    # Aplicar z-score
    raw_combined = apply_zscore_to_raw(raw_combined)
    
    # Definir escalados específicos para evitar problemas con canales misc
    scalings = {}
    for ch_name in raw_combined.ch_names:
        ch_type = raw_combined.get_channel_types(picks=ch_name)[0]
        if ch_type == 'misc':
            # Para canales misc, usar un escalado fijo que funcione bien con datos z-scoreados
            scalings[ch_type] = 2.0  # Escalado conservador para datos normalizados
        elif ch_type == 'eeg':
            scalings[ch_type] = 20e-6
        elif ch_type == 'eog':
            scalings[ch_type] = 150e-6
        elif ch_type == 'ecg':
            scalings[ch_type] = 5e-4
        else:
            # Para otros tipos, usar un escalado genérico
            scalings[ch_type] = 1.0
    
    # Visualizar
    print("Abriendo visualizador de MNE con anotaciones manuales y automáticas. Cierra la ventana para continuar.")
    fig = raw_combined.plot(
        title=f"Comparación de anotaciones manuales y automáticas",
        scalings=scalings,
        duration=540,
        start=0,
        show=True,
        block=True
    )
    
    print("\nVisualizador cerrado.")


def save_annotations_edited(raw, updated_annotations, bids_path, auto_path, save_dir="edited_events"):
    """
    Guarda las anotaciones editadas manualmente en un archivo TSV en derivatives/edited_events.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw de MNE con los datos
    updated_annotations : mne.Annotations
        Anotaciones editadas manualmente
    bids_path : mne_bids.BIDSPath
        Ruta BIDS del archivo raw
    auto_path : str
        Ruta al archivo de anotaciones automáticas original
    save_dir : str, optional
        Nombre del directorio dentro de derivatives donde guardar los eventos
        
    Returns
    -------
    str
        Ruta del archivo guardado
    """
    # Extraer información del bids_path
    subject = bids_path.subject
    session = bids_path.session
    task = bids_path.task
    run = bids_path.run
    acq = bids_path.acquisition
    
    # Crear BIDSPath para guardar las anotaciones editadas manualmente
    edited_root = repo_root / 'data' / 'derivatives' / save_dir
    
    output_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        suffix='events',
        description='edited',  # Entidad desc-edited para distinguir esta versión
        extension='.tsv',
        root=edited_root,
        check=False
    )
    
    # Crear el directorio si no existe
    os.makedirs(output_path.directory, exist_ok=True)
    
    # Convertir anotaciones a DataFrame
    annotations_df = pd.DataFrame({
        'onset': updated_annotations.onset,
        'duration': updated_annotations.duration,
        'trial_type': updated_annotations.description
    })
    
    # Guardar el DataFrame como TSV
    annotations_df.to_csv(output_path.fpath, sep='\t', index=False)
    
    print(f"Anotaciones editadas manualmente guardadas en: {output_path.fpath}")
    
    # Crear el archivo JSON asociado
    json_path = output_path.fpath.with_suffix('.json')
    
    # Intentar cargar el JSON original si existe
    original_json_path = Path(auto_path).with_suffix('.json')
    original_json = {}
    if original_json_path.exists():
        try:
            with open(original_json_path, 'r') as f:
                original_json = json.load(f)
        except json.JSONDecodeError:
            print(f"Error al leer el archivo JSON original: {original_json_path}")
    
    # Crear el contenido del JSON con los campos requeridos
    json_content = {
        # Campos estándar de columnas
        "onset": original_json.get("onset", {"Description": "Event onset in seconds"}),
        "duration": original_json.get("duration", {"Description": "Event duration in seconds"}),
        "trial_type": original_json.get("trial_type", {"Description": "Event description/type"}),
        
        # Añadir los metadatos como objetos para cumplir con el esquema BIDS
        "SidecarDescription": {
            "Description": "Eventos detectados automáticamente y editados manualmente",
            "EventTypes": {
                "auto_AUDIOVISUAL": "Marcador audiovisual detectado simultáneamente en canales AUDIO y PHOTO",
                "auto_AUDIO_only": "Marcador detectado solo en canal AUDIO",
                "auto_PHOTO_only": "Marcador detectado solo en canal PHOTO",
                "manual_annotation": "Anotación añadida o editada manualmente"
            }
        },
        "SourceData": {"Sources": [auto_path]},
        "ProcessingMethod": {
            "Algorithm": "Detección automática con edición manual",
            "EditingMethod": "Edición manual en visualizador MNE"
        },
        "GeneratedBy": {
            "Name": "detect_markers.py",
            "Version": "1.5",
            "Description": "Detección automática de marcadores audiovisuales con edición manual"
        },
        "MetadataDate": {"DateCreated": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")}
    }
    
    # Guardar el JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4)
    
    # Crear dataset_description.json si no existe
    create_dataset_description_edited(edited_root)
    
    return str(output_path.fpath)


def create_dataset_description_edited(edited_root):
    """
    Crea el archivo dataset_description.json para la carpeta de derivados de eventos editados.
    
    Parameters
    ----------
    edited_root : pathlib.Path
        Ruta a la carpeta de derivados
    """
    dataset_desc_path = edited_root / 'dataset_description.json'
    
    if not dataset_desc_path.exists():
        print(f"Creando dataset_description.json en {edited_root}")
        
        # Crear el README si no existe (requerido por BIDS)
        readme_path = edited_root / 'README'
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("# Edited Events\n\n")
                f.write("Este directorio contiene eventos detectados automáticamente y editados manualmente.\n")
                f.write("Los eventos fueron detectados usando el script detect_markers.py y posteriormente editados manualmente.\n")
                f.write("\nEl proceso de detección automática busca marcadores audiovisuales (coincidencias entre picos de AUDIO y PHOTO)\n")
                f.write("que corresponden a estímulos de silbato (whistle) y parpadeo visual (flicker) presentados simultáneamente\n")
                f.write("en los videos del experimento. Luego, estos eventos pueden ser editados manualmente para corregir errores\n")
                f.write("o añadir anotaciones adicionales.\n")
        
        # Crear manualmente el dataset_description.json para asegurar el formato correcto
        dataset_desc = {
            "Name": "edited_events",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "Authors": [
                "D'Amelio, Tomás Ariel",
                "COCUCO"
            ],
            "GeneratedBy": [{
                "Name": "detect_markers.py",
                "Version": "1.5",
                "Description": "Detección automática de marcadores audiovisuales con edición manual"
            }],
            "SourceDatasets": [{
                "URL": "file:///../../raw"  # URL en formato URI válido
            }]
        }
        
        # Guardar el JSON
        with open(dataset_desc_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_desc, f, indent=4)
        
        print(f"Archivo dataset_description.json creado en: {dataset_desc_path}")


def load_events_file(subject, session, task, run, acq=None, events_dir="events", desc=None):
    """
    Carga un archivo de eventos existente desde derivatives/events.
    
    Parameters
    ----------
    subject : str
        ID del sujeto
    session : str
        ID de la sesión
    task : str
        ID de la tarea
    run : str
        ID del run
    acq : str, optional
        Parámetro de adquisición
    events_dir : str, optional
        Nombre del directorio dentro de derivatives donde buscar los eventos
    desc : str, optional
        Descripción de los eventos a cargar
    
    Returns
    -------
    pd.DataFrame
        DataFrame con los eventos cargados o None si no se encuentra el archivo
    """
    print(f"\n=== Cargando eventos existentes para sub-{subject} ses-{session} task-{task} run-{run} ===\n")
    
    # Definir rutas
    events_root = repo_root / 'data' / 'derivatives' / events_dir
    
    # Asegurar formato correcto de parámetros
    if task and task.isdigit():
        task = task.zfill(2)
    if run and run.isdigit():
        run = run.zfill(3)
    if acq:
        acq = acq.lower()
    
    # Crear BIDSPath para eventos
    events_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        suffix='events',
        description=desc,
        extension='.tsv',
        root=events_root,
        check=False
    )
    
    print(f"Buscando archivo de eventos en: {events_path.fpath}")
    
    # Verificar que el archivo existe
    if not events_path.fpath.exists():
        print(f"¡ADVERTENCIA! Archivo de eventos no encontrado: {events_path.fpath}")
        
        # Si no se encuentra con la descripción especificada, intentar sin descripción
        if desc:
            print(f"Intentando buscar sin especificar descripción...")
            alt_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                run=run,
                acquisition=acq,
                datatype='eeg',
                suffix='events',
                extension='.tsv',
                root=events_root,
                check=False
            )
            if alt_path.fpath.exists():
                print(f"Archivo encontrado sin descripción: {alt_path.fpath}")
                events_path = alt_path
            else:
                # Listar todos los archivos de eventos disponibles
                print("Archivos de eventos disponibles:")
                for file in events_root.glob(f"sub-{subject}_ses-{session}_task-{task}_*_events.tsv"):
                    print(f"  - {file.name}")
                return None
        else:
            # Listar todos los archivos de eventos disponibles
            print("Archivos de eventos disponibles:")
            for file in events_root.glob(f"sub-{subject}_ses-{session}_task-{task}_*_events.tsv"):
                print(f"  - {file.name}")
            return None
    
    # Cargar eventos
    events_df = pd.read_csv(events_path.fpath, sep='\t')
    print(f"Eventos cargados: {len(events_df)} filas")
    
    return events_df

def display_events_in_order(events_df):
    """
    Muestra los eventos en su orden original, línea por línea.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame con los eventos a mostrar
    """
    if events_df is None or len(events_df) == 0:
        print("No hay eventos para mostrar.")
        return
    
    print("\n=== EVENTOS EN ORDEN ORIGINAL ===\n")
    print("Mostrando eventos en orden temporal:\n")
    
    # Usar el DataFrame en su orden original
    events_ordered = events_df.copy()
    
    # Mostrar cada fila
    for idx, row in events_ordered.iterrows():
        print(f"Evento #{idx + 1}:")
        print(f"  Onset: {row.get('onset', 'N/A'):.2f} s")
        print(f"  Duración: {row.get('duration', 'N/A'):.2f} s")
        print(f"  Tipo: {row.get('trial_type', 'N/A')}")
        
        # Mostrar campos adicionales si existen
        for col in row.index:
            if col not in ['onset', 'duration', 'trial_type']:
                print(f"  {col}: {row.get(col, 'N/A')}")
        
        print("-" * 40)
    
    print("\nFin de los eventos en orden original.")

def merge_events_with_annotations(original_events_df, annotations, max_duration_diff=1.0):
    """
    Fusiona los eventos originales con las nuevas anotaciones, actualizando los onsets y duraciones.
    Los eventos originales se mantienen en su orden original.
    
    Parameters
    ----------
    original_events_df : pd.DataFrame
        DataFrame con los eventos originales (de derivatives/events)
    annotations : mne.Annotations
        Anotaciones generadas o editadas manualmente
    max_duration_diff : float, optional
        Diferencia máxima permitida en duraciones para generar un warning
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los eventos fusionados
    """
    print("\n=== Fusionando eventos originales con nuevas anotaciones ===\n")
    
    # Verificar que hay eventos para fusionar
    if original_events_df is None or len(original_events_df) == 0:
        print("¡ERROR! No hay eventos originales para fusionar.")
        return None
    
    if annotations is None or len(annotations) == 0:
        print("¡ERROR! No hay anotaciones nuevas para fusionar.")
        return None
    
    # Mantener el orden original de los eventos (NO invertir)
    # Los eventos ya están en el orden correcto temporal
    events_df = original_events_df.copy().reset_index(drop=True)
    
    # Verificar que el número de eventos coincide
    if len(events_df) != len(annotations):
        print(f"¡ADVERTENCIA! El número de eventos originales ({len(events_df)}) no coincide con el número de anotaciones ({len(annotations)}).")
        print("La fusión puede generar resultados incorrectos.")
        
        # Si hay menos anotaciones que eventos, truncar eventos
        if len(annotations) < len(events_df):
            print(f"Truncando eventos originales para coincidir con el número de anotaciones ({len(annotations)}).")
            events_df = events_df.iloc[:len(annotations)].copy()
        else:
            # Si hay más anotaciones que eventos, truncar anotaciones
            print(f"Truncando anotaciones para coincidir con el número de eventos originales ({len(events_df)}).")
            # Crear nuevas anotaciones con solo los primeros elementos
            annotations = mne.Annotations(
                onset=annotations.onset[:len(events_df)],
                duration=annotations.duration[:len(events_df)],
                description=annotations.description[:len(events_df)]
            )
    
    # Crear una copia del DataFrame para no modificar el original
    merged_df = events_df.copy()
    
    # Actualizar onsets y duraciones
    merged_df['onset'] = annotations.onset
    merged_df['duration'] = annotations.duration
    
    # Verificar diferencias significativas en duración
    for i, (orig_dur, new_dur) in enumerate(zip(events_df['duration'], merged_df['duration'])):
        if abs(orig_dur - new_dur) > max_duration_diff:
            print(f"¡ADVERTENCIA! Diferencia significativa en la duración del evento {i+1}:")
            print(f"  Original: {orig_dur:.2f}s")
            print(f"  Nueva: {new_dur:.2f}s")
    
    # Mostrar resumen de la fusión
    print("\nResumen de la fusión:")
    print(f"  Eventos originales: {len(original_events_df)}")
    print(f"  Eventos fusionados: {len(merged_df)}")
    print(f"  Columnas preservadas: {', '.join(merged_df.columns)}")
    
    return merged_df

def save_merged_events(merged_df, bids_path, original_events_path, merged_save_dir="merged_events", merged_desc="merged"):
    """
    Guarda los eventos fusionados en un archivo TSV en derivatives/merged_events.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame con los eventos fusionados
    bids_path : mne_bids.BIDSPath
        Ruta BIDS del archivo raw
    original_events_path : str or Path
        Ruta al archivo de eventos originales
    merged_save_dir : str, optional
        Nombre del directorio dentro de derivatives donde guardar los eventos
    merged_desc : str, optional
        Descripción para los eventos fusionados
        
    Returns
    -------
    str
        Ruta del archivo guardado
    """
    if merged_df is None or len(merged_df) == 0:
        print("No hay eventos fusionados para guardar.")
        return None
    
    # Extraer información del bids_path
    subject = bids_path.subject
    session = bids_path.session
    task = bids_path.task
    run = bids_path.run
    acq = bids_path.acquisition
    
    # Crear BIDSPath para guardar los eventos fusionados
    merged_root = repo_root / 'data' / 'derivatives' / merged_save_dir
    
    output_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        acquisition=acq,
        datatype='eeg',
        suffix='events',
        description=merged_desc,
        extension='.tsv',
        root=merged_root,
        check=False
    )
    
    # Crear el directorio si no existe
    os.makedirs(output_path.directory, exist_ok=True)
    
    # Guardar el DataFrame como TSV
    merged_df.to_csv(output_path.fpath, sep='\t', index=False)
    
    print(f"Eventos fusionados guardados en: {output_path.fpath}")
    
    # Crear el archivo JSON asociado
    json_path = output_path.fpath.with_suffix('.json')
    
    # Intentar cargar el JSON original si existe
    original_json_path = Path(original_events_path).with_suffix('.json')
    original_json = {}
    if original_json_path.exists():
        try:
            with open(original_json_path, 'r') as f:
                original_json = json.load(f)
        except json.JSONDecodeError:
            print(f"Error al leer el archivo JSON original: {original_json_path}")
    
    # Crear el contenido del JSON con los campos del original y metadatos adicionales
    json_content = original_json.copy()
    
    # Añadir información sobre el proceso de fusión
    json_content["ProcessingMethod"] = {
        "Description": "Fusión de eventos originales con anotaciones detectadas automáticamente y editadas manualmente",
        "OriginalEventsSource": str(original_events_path),
        "MergeDate": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
    }
    
    # Añadir información sobre el generador
    if "GeneratedBy" not in json_content:
        json_content["GeneratedBy"] = {}
    
    json_content["GeneratedBy"]["Name"] = "detect_markers.py"
    json_content["GeneratedBy"]["Version"] = "1.5"
    json_content["GeneratedBy"]["Description"] = "Fusión de eventos originales con anotaciones detectadas/editadas"
    
    # Guardar el JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4)
    
    # Crear dataset_description.json si no existe
    create_dataset_description_merged(merged_root)
    
    return str(output_path.fpath)

def create_dataset_description_merged(merged_root):
    """
    Crea el archivo dataset_description.json para la carpeta de derivados de eventos fusionados.
    
    Parameters
    ----------
    merged_root : pathlib.Path
        Ruta a la carpeta de derivados
    """
    dataset_desc_path = merged_root / 'dataset_description.json'
    
    if not dataset_desc_path.exists():
        print(f"Creando dataset_description.json en {merged_root}")
        
        # Crear el README si no existe (requerido por BIDS)
        readme_path = merged_root / 'README'
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("# Merged Events\n\n")
                f.write("Este directorio contiene eventos fusionados a partir de eventos originales y anotaciones detectadas/editadas.\n")
                f.write("Los eventos provienen de la fusión de los eventos de derivatives/events con las anotaciones generadas por detect_markers.py.\n")
                f.write("Los eventos originales mantienen toda su metadata y estructura, pero con onsets y duraciones actualizados.\n")
        
        # Crear manualmente el dataset_description.json para asegurar el formato correcto
        dataset_desc = {
            "Name": "merged_events",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "Authors": [
                "D'Amelio, Tomás Ariel",
                "COCUCO"
            ],
            "GeneratedBy": [{
                "Name": "detect_markers.py",
                "Version": "1.5",
                "Description": "Fusión de eventos originales con anotaciones detectadas/editadas"
            }],
            "SourceDatasets": [
                {
                    "URL": "file:///../../raw"  # URL en formato URI válido
                },
                {
                    "URL": "file:///../events"  # URL en formato URI válido
                }
            ]
        }
        
        # Guardar el JSON
        with open(dataset_desc_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_desc, f, indent=4)
        
        print(f"Archivo dataset_description.json creado en: {dataset_desc_path}")

def main():
    """Función principal del script."""
    args = parse_args()
    
    try:
        # Cargar datos raw
        raw, bids_path = load_raw_data(
            args.subject, args.session, args.task, args.run, args.acq
        )
        
        # Cargar eventos existentes si se solicita
        original_events_df = None
        original_events_path = None
        if args.load_events or args.merge_events:
            original_events_df = load_events_file(
                args.subject, args.session, args.task, args.run, args.acq,
                events_dir=args.events_dir, desc=args.events_desc
            )
            
            if original_events_df is not None:
                # Guardar la ruta del archivo original
                events_root = repo_root / 'data' / 'derivatives' / args.events_dir
                events_path = BIDSPath(
                    subject=args.subject,
                    session=args.session,
                    task=args.task,
                    run=args.run,
                    acquisition=args.acq,
                    datatype='eeg',
                    suffix='events',
                    description=args.events_desc,
                    extension='.tsv',
                    root=events_root,
                    check=False
                )
                original_events_path = events_path.fpath
                
                # Mostrar eventos en orden original
                display_events_in_order(original_events_df)
            else:
                print("No se pudieron cargar eventos originales. No se realizará la fusión.")
                if args.merge_events:
                    print("Se continuará con la detección de marcadores, pero sin fusión.")
        
        # Detectar marcadores
        peaks = detect_markers(
            raw, 
            audio_threshold=args.audio_threshold,
            photo_threshold=args.photo_threshold,
            photo_distance=args.photo_distance,
            whistle_freq=args.whistle_freq,
            whistle_bandwidth=args.whistle_bandwidth,
            whistle_duration=args.whistle_duration,
            flicker_freq=args.flicker_freq,
            flicker_bandwidth=args.flicker_bandwidth,
            flicker_duration=args.flicker_duration,
            use_amplitude_detection=args.use_amplitude_detection,
            visualize_detection=args.visualize_detection
        )
        
        # Encontrar picos coincidentes entre AUDIO y PHOTO
        coincident_peaks = find_coincident_peaks(
            peaks,
            max_time_diff=args.max_time_diff
        )
        
        # Crear anotaciones a partir de los picos coincidentes
        annotations = create_annotations_from_coincident_peaks(coincident_peaks)
        
        # Guardar anotaciones automáticas solo si se especifica
        auto_path = None
        if args.save_auto_events:
            auto_path = save_annotations_auto(
                raw, annotations, bids_path, save_dir=args.save_dir,
                audio_threshold=args.audio_threshold,
                photo_threshold=args.photo_threshold,
                photo_distance=args.photo_distance,
                whistle_freq=args.whistle_freq,
                whistle_bandwidth=args.whistle_bandwidth,
                whistle_duration=args.whistle_duration,
                flicker_freq=args.flicker_freq,
                flicker_bandwidth=args.flicker_bandwidth,
                flicker_duration=args.flicker_duration,
                use_amplitude_detection=args.use_amplitude_detection
            )
            print(f"Anotaciones automáticas guardadas en: {auto_path}")
        else:
            print("No se guardan anotaciones automáticas (usar --save-auto-events si se necesitan)")
            # Crear una ruta temporal para compatibilidad con funciones que esperan auto_path
            auto_path = f"temp_auto_events_{args.subject}_{args.session}_{args.task}_{args.run}.tsv"
        
        # Visualizar señales con anotaciones y permitir edición manual si está habilitada
        updated_annotations, has_changes = visualize_signals_with_annotations(
            raw, annotations, apply_zscore=not args.no_zscore
        )
        
        # Variable para almacenar la ruta del archivo de anotaciones finales
        final_annotations_path = auto_path
        
        # Si está habilitada la edición manual y hubo cambios, guardar las anotaciones editadas solo si se especifica
        if args.enable_manual_edit and has_changes and args.save_edited_events:
            print("\n¡Se detectaron cambios en las anotaciones!")
            
            # Si se especificó --force-save, guardar sin preguntar
            if args.force_save:
                edited_path = save_annotations_edited(
                    raw, updated_annotations, bids_path, auto_path,
                    save_dir=args.manual_save_dir
                )
                print(f"\nAnotaciones editadas guardadas exitosamente en: {edited_path}")
                final_annotations_path = edited_path
            else:
                # Preguntar al usuario si desea guardar los cambios
                while True:
                    response = input("\n¿Deseas guardar las anotaciones editadas en edited_events? (yes/no): ").strip().lower()
                    if response in ['yes', 'y', 'si', 's']:
                        edited_path = save_annotations_edited(
                            raw, updated_annotations, bids_path, auto_path,
                            save_dir=args.manual_save_dir
                        )
                        print(f"\nAnotaciones editadas guardadas exitosamente en: {edited_path}")
                        final_annotations_path = edited_path
                        break
                    elif response in ['no', 'n']:
                        print("\nLos cambios en las anotaciones NO han sido guardados en edited_events.")
                        break
                    else:
                        print("Por favor, responde 'yes' o 'no'.")
        elif args.enable_manual_edit and not has_changes and args.save_edited_events:
            print("\nNo se detectaron cambios en las anotaciones.")
            
            # Preguntar si se quiere forzar el guardado aunque no haya cambios
            if args.force_save:
                edited_path = save_annotations_edited(
                    raw, updated_annotations, bids_path, auto_path,
                    save_dir=args.manual_save_dir
                )
                print(f"\nAnotaciones guardadas sin cambios en: {edited_path}")
                final_annotations_path = edited_path
            elif input("\n¿Deseas guardar las anotaciones en edited_events de todos modos? (yes/no): ").strip().lower() in ['yes', 'y', 'si', 's']:
                edited_path = save_annotations_edited(
                    raw, updated_annotations, bids_path, auto_path,
                    save_dir=args.manual_save_dir
                )
                print(f"\nAnotaciones guardadas sin cambios en: {edited_path}")
                final_annotations_path = edited_path
        elif args.enable_manual_edit and has_changes and not args.save_edited_events:
            print("\n¡Se detectaron cambios en las anotaciones!")
            print("Las anotaciones editadas se usarán para merged_events pero no se guardarán por separado.")
            print("(Usar --save-edited-events si necesitas guardar también en edited_events)")
        elif args.enable_manual_edit and not has_changes:
            print("\nNo se detectaron cambios en las anotaciones.")
        
        # Si se solicitó fusionar los eventos, hacerlo ahora
        if args.merge_events and original_events_df is not None:
            # Determinar qué anotaciones usar para la fusión
            annotations_to_merge = updated_annotations if has_changes else annotations
            
            # Fusionar eventos originales con las nuevas anotaciones
            merged_df = merge_events_with_annotations(
                original_events_df, annotations_to_merge
            )
            
            if merged_df is not None:
                # Guardar eventos fusionados
                merged_path = save_merged_events(
                    merged_df, bids_path, original_events_path,
                    merged_save_dir=args.merged_save_dir,
                    merged_desc=args.merged_desc
                )
                
                if merged_path:
                    print(f"\nEventos fusionados guardados exitosamente en: {merged_path}")
                else:
                    print("\nError al guardar los eventos fusionados.")
            else:
                print("\nError en la fusión de eventos. No se guardaron eventos fusionados.")
        
        # Comparar con anotaciones manuales solo si se especifica el flag
        if args.compare_manual_auto:
            try:
                # Construir la ruta a las anotaciones manuales
                aligned_root = repo_root / 'data' / 'derivatives' / 'aligned_events'
                manual_path = BIDSPath(
                    subject=args.subject,
                    session=args.session,
                    task=args.task,
                    run=args.run,
                    acquisition=args.acq,
                    datatype='eeg',
                    suffix='events',
                    description='withann',
                    extension='.tsv',
                    root=aligned_root,
                    check=False
                ).fpath
                
                if manual_path.exists():
                    print(f"\nSe encontraron anotaciones manuales en: {manual_path}")
                    compare_manual_and_auto_annotations(raw, manual_path, auto_path)
                else:
                    print("\nNo se encontraron anotaciones manuales para comparar.")
            except Exception as e:
                print(f"\nError al intentar comparar anotaciones manuales y automáticas: {e}")
        
        print("\n=== Proceso completado exitosamente ===")
        print("\nPróximos pasos:")
        print("1. Revisar las anotaciones automáticas generadas")
        if not args.compare_manual_auto:
            print("2. Usar --compare-manual-auto para comparar con anotaciones manuales")
        if not args.enable_manual_edit:
            print("3. Usar --enable-manual-edit para volver a habilitar la edición manual de anotaciones")
        else:
            print("3. Si deshabilitaste la edición manual, puedes usar --no-manual-edit")
        if args.merge_events:
            print(f"4. Validar la estructura BIDS con 'bids-validator data/derivatives/{args.merged_save_dir}'")
        else:
            print(f"4. Validar la estructura BIDS con 'bids-validator data/derivatives/{args.save_dir}'")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())