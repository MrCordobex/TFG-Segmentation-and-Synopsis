#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import pandas as pd
import shutil

from utils.video      import video_to_frames, frames_to_video
from utils.grouping   import grouping
from utils.inference  import yolo_prediction
from utils.functions  import clean_noise
from utils.sinopsis   import sinopsis


def run_clustering(frames_folder, run_dir):
    clust_pkl = os.path.join(run_dir, "clust.pkl")
    print("[CLUST] Iniciando clustering...", flush=True)
    df_clust = grouping(
        frames_folder,
        'rbf',
        'FastMDS',
        'HDBSCAN',
        n_clusters=2,
        plot_similitud=False,
        plot_reduction=False,
        plot_clustering=False,
        plot_images=False
    )
    df_clust.to_pickle(clust_pkl)
    print(f"[CLUST] Terminado. Salvado en {clust_pkl}", flush=True)


def run_prediction(frames_folder, model_path, run_dir, threshold, classes):
    pred_pkl = os.path.join(run_dir, "pred.pkl")
    yolo_output = os.path.join(run_dir, "Yolo_output")
    os.makedirs(yolo_output, exist_ok=True)
    print("[PRED] Iniciando predicción YOLO...", flush=True)

    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    nombres = [f for f in os.listdir(frames_folder)
               if os.path.splitext(f)[1].lower() in extensiones]
    df_frames = pd.DataFrame({"frame_id": nombres})

    df_pred = yolo_prediction(
        df_frames,
        model_path,
        frames_folder,
        yolo_output,
        classes,
        conf_threshold=threshold
    )
    df_pred.to_pickle(pred_pkl)
    print(f"[PRED] Terminado. Salvado en {pred_pkl}", flush=True)


def orchestrator(args):
    # Setup folders
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    # 1) Frames extraction
    frames_folder = os.path.join(run_dir, "Frames_video")
    os.makedirs(frames_folder, exist_ok=True)
    print("[VIDEO] Extrayendo frames...", flush=True)
    video_to_frames(
        args.video_path,
        frames_folder,
        prefix="frame",
        skip=args.skip
    )
    print(f"[VIDEO] Frames guardados en {frames_folder}", flush=True)

    # 2) Launch clustering and prediction as subprocesses
    python_exe = sys.executable
    script     = os.path.abspath(__file__)
    print("[MAIN] Lanzando clustering y predicción en paralelo...", flush=True)
    p1 = subprocess.Popen([
        python_exe, script,
        "clust",
        "--frames_folder", frames_folder,
        "--run_dir", run_dir
    ])
    # Pasar clases como argumentos separados
    class_args = []
    for c in args.classes:
        class_args = ["--classes"] + [str(c) for c in args.classes]
    p2 = subprocess.Popen([
        python_exe, script,
        "pred",
        "--frames_folder", frames_folder,
        "--run_dir", run_dir,
        "--model_path", args.model_path,
        "--threshold", str(args.threshold)
    ] + class_args)
    p1.wait()
    p2.wait()

    # 3) Merge and post-process
    print("[MAIN] Cargando y mergeando resultados...", flush=True)
    df_clust = pd.read_pickle(os.path.join(run_dir, "clust.pkl"))
    df_pred  = pd.read_pickle(os.path.join(run_dir, "pred.pkl"))
    df_total = df_pred.merge(
        df_clust[['frame_id','label']],
        on='frame_id',
        how='left'
    )
    df_total['label'] = df_total['label'].fillna(-1).astype(int)
    print("[MAIN] Merge completado", flush=True)

    print("[MAIN] Aplicando clean_noise...", flush=True)
    df_total = clean_noise(df_total)
    print("[MAIN] clean_noise completado", flush=True)

    # 4) Sinopsis y guardado de frames seleccionados
    sinopsis_folder = os.path.join(run_dir, "Frames_video_sinopsis")
    os.makedirs(sinopsis_folder, exist_ok=True)
    print("[MAIN] Ejecutando sinopsis...", flush=True)
    selected_frames = sinopsis(
        df_total,
        os.path.join(run_dir, "Yolo_output"),
        sinopsis_folder,
        plot_series=False,
        plot_frames=False
    )
    print(f"[MAIN] Sinopsis guardada en {sinopsis_folder}", flush=True)

    # 5) Copiar frames seleccionados a carpeta Output_sinopsis
    output_sinopsis_dir = os.path.join(run_dir, "Output_sinopsis")
    os.makedirs(output_sinopsis_dir, exist_ok=True)
    yolo_output = os.path.join(run_dir, "Yolo_output")
    for fname in selected_frames:
        src = os.path.join(yolo_output, fname)
        dst = os.path.join(output_sinopsis_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
    print(f"[MAIN] Copiados {len(selected_frames)} frames a {output_sinopsis_dir}", flush=True)

    # 6) Frames to video
    output_video = os.path.join(run_dir, "Video_Sinopsis.mp4")
    print("[VIDEO] Generando vídeo de sinopsis...", flush=True)
    frames_to_video(
        sinopsis_folder,
        output_video,
        fps=args.fps,
        prefix="frame"
    )
    print(f"[VIDEO] Video resultado en {output_video}", flush=True)

    print("[DONE] Pipeline completado.", flush=True)

    # 7) Cleanup: eliminar todo excepto Output_sinopsis y Video_Sinopsis.mp4
    print("[CLEANUP] Eliminando carpetas y archivos temporales...", flush=True)
    for entry in os.listdir(run_dir):
        path = os.path.join(run_dir, entry)
        # Saltamos Output_sinopsis y el fichero Video_Sinopsis.mp4
        if entry == "Output_sinopsis" or entry == os.path.basename(output_video):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    print("[CLEANUP] Finalizado. Solo permanecen Output_sinopsis y Video_Sinopsis.mp4", flush=True)


def parse_orchestrator(argv):
    parser = argparse.ArgumentParser(
        description="Orquestador: extrae frames, clustering, predicción y vídeo de sinopsis"
    )
    parser.add_argument('video_path', help="Ruta al archivo de vídeo MP4")
    parser.add_argument('model_path', help="Ruta al modelo YOLO (.pt)")
    parser.add_argument('--run_dir', default='RUN', help="Directorio base para outputs")
    parser.add_argument('--skip', type=int, default=1, help="Número de frames a saltar al extraer imágenes")
    parser.add_argument('--threshold', type=float, default=0.2, help="Umbral de confianza para YOLO")
    parser.add_argument('--fps', type=int, default=30, help="Frames por segundo para vídeo final")
    parser.add_argument('-c', '--classes', nargs='+', type=int, default=[0],
                        help="Clases a filtrar para YOLO (lista de enteros)")
    return parser.parse_args(argv)


def parse_clust(argv):
    parser = argparse.ArgumentParser(description="Ejecuta solo clustering")
    parser.add_argument('--frames_folder', required=True, help="Carpeta con frames")
    parser.add_argument('--run_dir', required=True, help="Directorio base para outputs")
    return parser.parse_args(argv)


def parse_pred(argv):
    parser = argparse.ArgumentParser(description="Ejecuta solo predicción YOLO")
    parser.add_argument('--frames_folder', required=True, help="Carpeta con frames")
    parser.add_argument('--run_dir', required=True, help="Directorio base para outputs")
    parser.add_argument('--model_path', required=True, help="Ruta al modelo YOLO (.pt)")
    parser.add_argument('--threshold', type=float, default=0.2, help="Umbral de confianza para YOLO")
    parser.add_argument('-c', '--classes', nargs='+', type=int, default=[0],
                        help="Clases a filtrar para YOLO (lista de enteros)")
    return parser.parse_args(argv)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'clust':
        args = parse_clust(sys.argv[2:])
        run_clustering(args.frames_folder, args.run_dir)
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == 'pred':
        args = parse_pred(sys.argv[2:])
        run_prediction(
            args.frames_folder,
            args.model_path,
            args.run_dir,
            args.threshold,
            args.classes
        )
        sys.exit(0)

    # Orchestrator
    args = parse_orchestrator(sys.argv[1:])
    orchestrator(args)
