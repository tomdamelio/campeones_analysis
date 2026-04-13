import json
import os

# --- 1. CONFIGURACIÓN ---
SUBJECT_IDS = ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
               "29", "30", "31", "32", "33", "34", "35", "36", "37", "38",
               "39", "40", "42", "43", "46"]

VALIDATION_LOG_PATH = r"results/physio_validation_log.json" 
MARKERS_VALIDATION_LOG_PATH = r"results/markers_validation_log.json"

# --- 2. LÓGICA DE AUDITORÍA ---

def load_jsons():
    """Carga ambos JSONs."""
    physio_data, markers_data = {}, {}
    
    if os.path.exists(VALIDATION_LOG_PATH):
        with open(VALIDATION_LOG_PATH, 'r', encoding='utf-8') as f:
            physio_data = json.load(f).get("subjects", {})
    else:
        print(f"⚠️ No se encontró {VALIDATION_LOG_PATH}")

    if os.path.exists(MARKERS_VALIDATION_LOG_PATH):
        with open(MARKERS_VALIDATION_LOG_PATH, 'r', encoding='utf-8') as f:
            markers_data = json.load(f).get("subjects", {})
    else:
        print(f"⚠️ No se encontró {MARKERS_VALIDATION_LOG_PATH}")
        
    return physio_data, markers_data

def evaluate_block(block_key, physio_info, marker_note):
    """
    Evalúa si un bloque es MALO cruzando las dos fuentes.
    Devuelve (True, "Razón") si es malo, o (False, "") si está OK.
    """
    # 1. Chequeo de Comportamiento (Markers)
    if marker_note:
        note_upper = marker_note.strip().upper()
        # Un estímulo malo NO invalida el bloque completo
        is_bad_stimulus = note_upper.startswith("ESTIMULO BAD") or note_upper.startswith("BAD ESTIMULO")
        
        # Invalida si dice explícitamente BAD (y no es solo un estímulo) o No Tomado
        is_bad_block = (note_upper.startswith("BAD") and not is_bad_stimulus) or \
                       note_upper.startswith("NO SE TOMO") or \
                       note_upper.startswith("NO TOMADO")
                       
        if is_bad_block:
            motivo = marker_note if len(marker_note) < 40 else marker_note[:40] + "..."
            return True, f"Markers: {motivo}"

    # 2. Chequeo de Fisiología (EDA)
    if physio_info:
        eda_info = physio_info.get("eda", {}) or physio_info.get("gsr", {})
        eda_category = eda_info.get("category", "good")
        
        if eda_category in ["bad", "maybe"]:
            return True, f"EDA: {eda_category}"
    else:
        # Si no hay info de fisio del todo para este bloque, lo marcamos como faltante
        return True, "Falta registro de Fisio"

    return False, "OK"

def generate_report():
    physio_data, markers_data = load_jsons()
    
    # Contadores Globales
    sujetos_perfectos = 0
    conteo_faltantes = {} # Diccionario para agrupar {cantidad_bloques_malos: cantidad_sujetos}
    sujetos_sin_sesion = 0
    
    detalles_sujetos = []

    print("\n🔍 Analizando JSONs y cruzando validaciones...\n")

    for sub_id in SUBJECT_IDS:
        # Extraemos todas las llaves (bloques) únicas para este sujeto desde AMBOS jsons
        claves_marcadores = set(markers_data.get(sub_id, {}).keys())
        claves_fisio = set(physio_data.get(sub_id, {}).keys())
        todas_las_claves = claves_marcadores.union(claves_fisio)
        
        # Contadores por sesión para este sujeto (Se esperan 4 de acq-a y 4 de acq-b)
        bloques_ok_acq_a = 0
        bloques_ok_acq_b = 0
        bloques_malos_detalle = []

        for key in todas_las_claves:
            marker_note = markers_data.get(sub_id, {}).get(key, "")
            physio_info = physio_data.get(sub_id, {}).get(key, {})
            
            es_malo, razon = evaluate_block(key, physio_info, marker_note)
            
            if es_malo:
                bloques_malos_detalle.append(f"  ❌ {key} -> {razon}")
            else:
                if "acq-a" in key:
                    bloques_ok_acq_a += 1
                elif "acq-b" in key:
                    bloques_ok_acq_b += 1
        
        # Calcular cuántos bloques faltan/están mal respecto a los 8 esperados (4 por sesión)
        # Si un bloque ni siquiera existía en los JSON, se restará acá matemáticamente
        malos_o_ausentes_a = 4 - bloques_ok_acq_a
        malos_o_ausentes_b = 4 - bloques_ok_acq_b
        
        total_malos_o_ausentes = malos_o_ausentes_a + malos_o_ausentes_b
        
        # Chequeo de Sesión Completa Faltante
        le_falta_sesion = (bloques_ok_acq_a == 0) or (bloques_ok_acq_b == 0)
        
        # --- Actualizar Contadores Globales ---
        if total_malos_o_ausentes == 0:
            sujetos_perfectos += 1
        else:
            conteo_faltantes[total_malos_o_ausentes] = conteo_faltantes.get(total_malos_o_ausentes, 0) + 1
            
        if le_falta_sesion:
            sujetos_sin_sesion += 1
            
        # --- Guardar detalle para imprimir después ---
        if total_malos_o_ausentes > 0:
            alerta_sesion = " 🚨 [FALTA SESIÓN COMPLETA]" if le_falta_sesion else ""
            detalles_sujetos.append(
                f"\nSujeto {sub_id} (Faltan/Malos: {total_malos_o_ausentes}/8){alerta_sesion}\n" +
                "\n".join(bloques_malos_detalle)
            )

    # --- IMPRESIÓN DEL REPORTE FINAL ---
    print("="*60)
    print("📊 REPORTE DE CALIDAD DE DATOS (MARKERS + PHYSIO)")
    print("="*60)
    print(f"Total de sujetos en la lista: {len(SUBJECT_IDS)}\n")
    
    print(f"✅ Sujetos con TODOS los bloques OK (8/8): {sujetos_perfectos}")
    
    print("\n📉 Distribución de sujetos con bloques descartados:")
    for cantidad_malos in sorted(conteo_faltantes.keys()):
        cantidad_sujetos = conteo_faltantes[cantidad_malos]
        plural = "s" if cantidad_sujetos > 1 else ""
        print(f"   ➤ {cantidad_sujetos} sujeto{plural} con {cantidad_malos} bloque(s) malo(s)/faltante(s)")
        
    print(f"\n🗑️ Sujetos a los que les falta al menos UNA SESIÓN COMPLETA: {sujetos_sin_sesion}")
    print("="*60)
    
    print("\n📋 REPORTE DETALLADO DE DESCARTES POR SUJETO:")
    print("-" * 60)
    for detalle in detalles_sujetos:
        print(detalle)

if __name__ == "__main__":
    generate_report()