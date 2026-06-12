import librosa

# Ruta de tu archivo MP3
ruta_mp3 = './grupo a/A Night In Tunisia.mp3'

# Cargar el archivo de audio
y, sr = librosa.load(ruta_mp3, sr=22050)

print(f"Frecuencia de muestreo (sr): {sr} Hz")
print(f"Forma de la matriz de audio (y): {y.shape}")