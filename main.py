import librosa

# Ruta de tu archivo MP3
ruta_mp3 = './grupo a/A Night In Tunisia.mp3'

# Cargar el archivo de audio, lo hago durar un minuto y medio 
y1, sr1 = librosa.load(ruta_mp3, sr=None, duration=5400.0) 

print(f"Frecuencia de muestreo (sr): {sr1} Hz")
print(f"Forma de la matriz de audio (y): {y1.shape}")