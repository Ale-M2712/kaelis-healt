import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Cargar audio
y, sr = librosa.load("C:\\repo\\kaelis-healt\\audio de prueba.wav", sr=16000)

# Espectrograma Mel
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
)

# Convertir a decibelios para visualizar
mel_db = librosa.power_to_db(mel_spec, ref=np.max)

# Mostrar
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Espectrograma Mel")
plt.tight_layout()
plt.show()