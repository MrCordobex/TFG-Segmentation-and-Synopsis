#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import subprocess

from utils.grouping    import grouping
from utils.inference   import yolo_prediction
from utils.functions   import clean_noise
from utils.sinopsis    import sinopsis

CLUST_PKL = 'clust.pkl'
PRED_PKL  = 'pred.pkl'

def run_clustering():
    """Ejecuta el clustering y salva el DataFrame en disk."""
    print("[CLUST] Iniciando clustering…", flush=True)
    df = grouping(
        'Fotos_video_personas3',
        'rbf',
        'FastMDS',
        'HDBSCAN',
        n_clusters=2,
        plot_similitud=False,
        plot_reduction=False,
        plot_clustering=False,
        plot_images=False
    )
    df.to_pickle(CLUST_PKL)
    print(f"[CLUST] Terminado. Salvado en {CLUST_PKL}", flush=True)

def run_prediction():
    """Ejecuta la predicción YOLO y salva el DataFrame en disk."""
    print("[PRED ] Iniciando predicción YOLO…", flush=True)
    model_path     = "yolo11n-seg.pt"
    input_folder   = "Fotos_video_personas3"
    output_folder  = "Fotos_video_personas3_yolo"
    filter_classes = [1,3,4,8]
    threshold      = 0.2

    # Listar SOLO imágenes
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    nombres = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in extensiones
    ]
    df_frames = pd.DataFrame({"frame_id": nombres})

    df_pred = yolo_prediction(
        df_frames,
        model_path,
        input_folder,
        output_folder,
        filter_classes,
        conf_threshold=threshold
    )
    df_pred.to_pickle(PRED_PKL)
    print(f"[PRED ] Terminado. Salvado en {PRED_PKL}", flush=True)

def orchestrator():
    """Orquesta el pipeline: lanza clust & pred, espera, mergea y postprocesa."""
    python_exe = sys.executable
    script     = os.path.abspath(__file__)

    # 1) Lanza dos procesos separados
    p1 = subprocess.Popen([python_exe, script, "clust"])
    p2 = subprocess.Popen([python_exe, script, "pred"])

    # 2) Espera a ambos
    p1.wait()
    p2.wait()

    # 3) Carga resultados
    df_clust     = pd.read_pickle(CLUST_PKL)
    df_inferenced= pd.read_pickle(PRED_PKL)

    # 4) Merge final
    print("[MAIN ] Ejecutando merge...", flush=True)
    df_total = df_inferenced.merge(
        df_clust[['frame_id','label']],
        on='frame_id',
        how='left'
    )
    df_total['label'] = df_total['label'].fillna(-1).astype(int)
    print("[MAIN ] Merge completado", flush=True)

    # 5) clean_noise
    print("[MAIN ] Aplicando clean_noise...", flush=True)
    df_total = clean_noise(df_total)
    print("[MAIN ] clean_noise completado", flush=True)

    # 6) Sinopsis final
    print("[MAIN ] Ejecutando sinopsis...", flush=True)
    frames = sinopsis(
        df_total,
        "Fotos_video_personas3_yolo",
        "Fotos_video_personas3_sinopsis",
        plot_series=True,
        plot_frames=True
    )
    print("[MAIN ] sinopsis completado", flush=True)

if __name__ == "__main__":
    # Si se pasa "clust" o "pred", ejecuta solo esa parte
    if len(sys.argv) == 2 and sys.argv[1] == "clust":
        run_clustering()
        sys.exit(0)
    if len(sys.argv) == 2 and sys.argv[1] == "pred":
        run_prediction()
        sys.exit(0)

    # Sin args: ejecuta el orquestador
    orchestrator()
