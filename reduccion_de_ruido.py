import os
import shutil
import subprocess
import tempfile

import torchaudio
import noisereduce as nr
import soundfile as sf
import numpy as np

# Torchaudio usa por defecto el backend "soundfile", que con frecuencia no incluye soporte
# para MP3 (libsndfile normalmente no tiene codecs MP3). Intentamos usar el backend "sox_io" si
# tu instalación lo soporta (usualmente cuando torchaudio se construye con sox/ffmpeg).
# Nota: con el nuevo dispatcher esta llamada puede ser un no-op, por lo que aún puede fallar.
try:
    torchaudio.set_audio_backend("sox_io")
except Exception:
    pass

def limpiar_mp3(path_mp3, salida_mp3="salida_limpia.mp3"):
    # 0. Resoluciones de ruta (para poder ejecutar el script desde cualquier carpeta).
    if not os.path.isabs(path_mp3):
        path_mp3 = os.path.join(os.path.dirname(__file__), path_mp3)
    if not os.path.isabs(salida_mp3):
        salida_mp3 = os.path.join(os.path.dirname(__file__), salida_mp3)

    # 1. Cargar el MP3.
    #    Intentamos usar torchaudio; si falla, convertimos a WAV usando ffmpeg.
    try:
        y, sr = torchaudio.load(path_mp3)  # y es un tensor [channels, samples]
        y_np = y.numpy()
    except Exception as e:
        # Fallback: usar ffmpeg para convertir a WAV temporal y luego leerlo.
        ffmpeg_exe = None
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = shutil.which("ffmpeg")

        if not ffmpeg_exe:
            raise RuntimeError(
                f"Fallo al abrir '{path_mp3}'. No se puede leer MP3 con el backend actual. "
                "Instala ffmpeg (o imageio-ffmpeg) y asegúrate de que esté en el PATH, o usa un WAV."
            ) from e

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", path_mp3, tmp_wav],
                check=True,
                capture_output=True,
            )
            y, sr = torchaudio.load(tmp_wav)
            y_np = y.numpy()
        finally:
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    # Convertir a numpy para noisereduce (solo canal 0 si es estéreo)
    y_np = y_np[0]

    # 2. Seleccionar un segmento de ruido (ej: primeros 0.5 segundos)
    noise_clip = y_np[0:int(sr*0.5)]

    # 3. Reducir ruido
    y_reducido = nr.reduce_noise(y=y_np, sr=sr, y_noise=noise_clip)

    # 4. Guardar como WAV temporal (intermedio para poder convertir a MP3).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        sf.write(tmp_wav, y_reducido, sr)

        # 5. Convertir a la extensión solicitada por el usuario
        salida_ext = os.path.splitext(salida_mp3)[1].lower()
        if salida_ext == "" or salida_ext == ".wav":
            # Solo renombrar/mover el WAV intermedio
            if not salida_ext:
                salida_mp3 = salida_mp3 + ".wav"
            os.replace(tmp_wav, salida_mp3)
        else:
            # Convertir usando ffmpeg (imageio-ffmpeg si está disponible)
            ffmpeg_exe = None
            try:
                from imageio_ffmpeg import get_ffmpeg_exe

                ffmpeg_exe = get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = shutil.which("ffmpeg")

            if not ffmpeg_exe:
                raise RuntimeError(
                    "No se encontró ffmpeg para convertir el archivo. "
                    "Instala ffmpeg o imageio-ffmpeg para guardar en MP3 u otro formato."
                )

            try:
                subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-i",
                        tmp_wav,
                        salida_mp3,
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as cp:
                raise RuntimeError(
                    "Error al convertir el resultado a MP3/otro formato:\n"
                    + (cp.stderr.decode(errors="ignore") if cp.stderr else str(cp))
                ) from cp
        print(f"Archivo limpio guardado en {salida_mp3}")
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

if __name__ == "__main__":
    limpiar_mp3("reduccion_de_ruido.mp3", "salida_limpia.mp3")