#!/usr/bin/env python
"""
Script para explorar y visualizar la señal tónica y fásica de EDA para un participante.

Uso:
    python scripts/sanity_check/explore_cvx_eda.py --sub 14

Carga:
- data/derivatives/physio/sub-14/sub-14_desc-physio_features_concatenated.npy
- data/derivatives/physio/sub-14/idx_data_sub-14_concatenated.npy

Genera un plot con dos subplots (tónica arriba, fásica abajo) y líneas verticales en los inicios/fines de segmentos.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Explora y grafica señal tónica y fásica de EDA para un participante.")
parser.add_argument('--sub', type=str, required=True, help='ID del participante (ej: 14)')
args = parser.parse_args()

sub = args.sub
physio_dir = Path(__file__).resolve()
while not (physio_dir / ".git").exists() and physio_dir != physio_dir.parent:
    physio_dir = physio_dir.parent
physio_dir = physio_dir / "data" / "derivatives" / "physio" / f"sub-{sub}"

features_file = physio_dir / f"sub-{sub}_desc-physio_features_concatenated.npz"
indices_file = physio_dir / f"idx_data_sub-{sub}_concatenated.npz"

if not features_file.exists():
    raise FileNotFoundError(f"No se encontró el archivo de features: {features_file}")
if not indices_file.exists():
    raise FileNotFoundError(f"No se encontró el archivo de índices: {indices_file}")

features = np.load(features_file)['arr_0']
indices = np.load(indices_file)['arr_0']

# Asumimos: features.shape = (n_timepoints, n_features), tónica = col 0, fásica = col 1, joystick_y = col 2
signal_tonica = features[:, 0]
signal_fasica = features[:, 1]
signal_joystick = features[:, 4] if features.shape[1] > 4 else None

n_subplots = 3 if signal_joystick is not None else 4
fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 10 if n_subplots == 3 else 8), sharex=True)

ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2] if n_subplots == 3 else None

# Plot señal tónica
ax1.plot(signal_tonica, color='tab:blue')
ax1.set_title(f'Sujeto {sub} - Señal Tónica (EDA)')
ax1.set_ylabel('Tónica (a.u.)')

# Plot señal fásica
ax2.plot(signal_fasica, color='tab:orange')
ax2.set_title(f'Sujeto {sub} - Señal Fásica (EDA)')
ax2.set_ylabel('Fásica (a.u.)')

# Plot joystick_y si existe
if ax3 is not None:
    ax3.plot(signal_joystick, color='tab:green')
    ax3.set_title(f'Sujeto {sub} - Joystick Y')
    ax3.set_ylabel('Joystick Y')
    ax3.set_xlabel('Timepoint')
else:
    ax2.set_xlabel('Timepoint')

# Dibujar líneas verticales en los inicios y fines de segmentos
for start, end in indices:
    ax1.axvline(start, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(end, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(start, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(end, color='gray', linestyle='--', alpha=0.5)
    if ax3 is not None:
        ax3.axvline(start, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(end, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
output_file = physio_dir / f"sub-{sub}_eda_tonica_fasica_joystick.png"
plt.savefig(output_file)
print(f"Plot guardado en: {output_file}")
plt.show() 