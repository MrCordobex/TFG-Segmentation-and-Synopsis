import os
import cv2
from tqdm import tqdm


def video_to_frames(video_path, output_folder, prefix='frame', skip=1):
    print(f"📂 Creando carpeta de salida: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    print(f"🎥 Abriendo video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ ERROR: No se pudo abrir el video {video_path}")
        exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 Total de frames: {total_frames}")

    frame_index = 0
    saved_count = 0

    # Inicializamos tqdm
    pbar = tqdm(total=total_frames, desc='Procesando frames', unit='frame')

    while True:
        ret, frame = cap.read()
        if not ret:
            pbar.close()
            print("✅ Fin del video.")
            break

        if frame_index % skip == 0:
            filename = f"{prefix}_{frame_index:06d}.jpg"
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_index += 1
        pbar.update(1)

    cap.release()
    print(f"✅ Se guardaron {saved_count} frames.")

def frames_to_video(input_folder, output_video, fps=30, prefix='frame'):
    """
    Crea un video a partir de una secuencia de imágenes (frames) almacenadas en una carpeta.

    :param input_folder: Carpeta donde se encuentran los fotogramas (imágenes).
    :param output_video: Nombre (ruta) del archivo de video de salida, ej. 'salida.mp4'.
    :param fps: Cuadros por segundo para el video resultante.
    :param prefix: Prefijo de los nombres de archivo que identifiquen a los frames.
    """
    # 1. Listar todos los archivos de la carpeta que coincidan con el prefijo y terminen en .jpg (o .png)
    frames_list = sorted([f for f in os.listdir(input_folder)
                          if f.startswith(prefix) and f.lower().endswith(('.jpg', '.png'))])

    if not frames_list:
        print("No se encontraron fotogramas en la carpeta especificada.")
        return

    # 2. Leer el primer frame para obtener dimensiones
    first_frame_path = os.path.join(input_folder, frames_list[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"No se pudo leer el primer fotograma: {first_frame_path}")
        return

    height, width, channels = first_frame.shape

    # 3. Configurar el VideoWriter
    # - fourcc puede ser 'mp4v' para .mp4 o 'XVID' para .avi, etc.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 4. Recorrer todos los frames y escribirlos en el video
    for frame_name in frames_list:
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Advertencia: no se pudo leer {frame_path}. Se omitirá.")
            continue
        out.write(frame)

    # 5. Liberar el VideoWriter
    out.release()
    print(f"Video guardado en: {output_video}")