"""
funciones.py

Todas las funciones necesarias para el TP. La funcion principal es
procesar_audio(), que toma un archivo de audio y devuelve:
  - h: exponente de Hurst (correlaciones temporales)
  - c: parametro de la Gaussiana estirada (forma de la distribucion)

Estas dos funciones se basan en Mendes et al. (2010).
"""

import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize_scalar
import soundfile as sf


# ----------------------------------------------------------------------
# FUNCION 1: Exponente de Hurst via DFA
# ----------------------------------------------------------------------
def dfa(signal, scales=None, order=1):
    """
    Calcula el exponente de Hurst (h) de una serie temporal usando DFA
    (Detrended Fluctuation Analysis).

    Idea del algoritmo:
      1. Se integra la señal (suma acumulada) -> queda como un "camino
         aleatorio".
      2. Se la corta en ventanas de distintos tamaños (n).
      3. En cada ventana se ajusta una recta y se mide cuanto se aleja
         la señal de esa recta (fluctuacion F(n)).
      4. Si F(n) crece como una potencia de n (F(n) ~ n^h), el exponente
         h es la pendiente en escala log-log.

    Interpretacion de h:
      h ~ 0.5 -> sin correlaciones (ruido blanco)
      h ~ 1.0 -> correlaciones de largo alcance (ruido rosa / 1/f)
      h ~ 1.5 -> muy correlacionado (camino aleatorio / ruido browniano)
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)

    # Paso 1: integrar la señal centrada
    y = np.cumsum(signal - np.mean(signal))

    # Paso 2: definir las escalas (tamaños de ventana) en escala log
    if scales is None:
        scales = np.unique(
            np.logspace(np.log10(10), np.log10(max(N // 4, 11)), 20).astype(int)
        )

    # Paso 3: calcular F(n) para cada escala
    fluct = np.zeros(len(scales))
    for i, n in enumerate(scales):
        n_ventanas = N // n
        if n_ventanas == 0:
            fluct[i] = np.nan
            continue

        t = np.arange(n)
        rms_por_ventana = np.zeros(n_ventanas)
        for w in range(n_ventanas):
            tramo = y[w * n:(w + 1) * n]
            coef = np.polyfit(t, tramo, order)        # ajuste lineal local
            tendencia = np.polyval(coef, t)
            rms_por_ventana[w] = np.sqrt(np.mean((tramo - tendencia) ** 2))

        fluct[i] = np.sqrt(np.mean(rms_por_ventana ** 2))

    # Paso 4: pendiente del ajuste log(F) vs log(n) = h
    validos = ~np.isnan(fluct) & (fluct > 0)
    h, _ = np.polyfit(np.log10(scales[validos]), np.log10(fluct[validos]), 1)

    return h


# ----------------------------------------------------------------------
# FUNCION 2: Parametro c de la Gaussiana estirada (ecuacion 1 del paper)
# ----------------------------------------------------------------------
def ajustar_c(z):
    """
    Estima el parametro c de la Gaussiana estirada:

        p(z) = (c/2) * (Gamma(3/c)/Gamma(1/c)^3)^(1/2)
               * exp( -(Gamma(3/c)/Gamma(1/c))^(c/2) * |z|^c )

    Recordar: c=1 -> distribucion Laplace (cola pesada)
              c=2 -> distribucion Gaussiana normal

    z debe tener media 0 y desvio estandar 1 (ya normalizado).

    El metodo es maxima verosimilitud: se busca el valor de c que
    hace mas "probable" haber observado los datos z.
    """
    def menos_log_verosimilitud(c):
        A = (gamma(3 / c) / gamma(1 / c) ** 3) ** 0.5
        B = (gamma(3 / c) / gamma(1 / c)) ** (c / 2)
        log_p = np.log(c / 2) + np.log(A) - B * np.abs(z) ** c
        return -np.sum(log_p)

    resultado = minimize_scalar(menos_log_verosimilitud, bounds=(0.2, 4), method="bounded")
    return resultado.x


# ----------------------------------------------------------------------
# FUNCION 3: Generar ruido sintetico (linea de base)
# ----------------------------------------------------------------------
def generar_ruido(n_muestras, beta, seed=None):
    """
    Genera una señal sintetica con espectro de potencia S(f) ~ 1/f^beta,
    via filtrado en el dominio de Fourier de ruido blanco gaussiano.

    beta = 0 -> ruido blanco    (h esperado ~ 0.5, sin memoria)
    beta = 1 -> ruido rosa      (h esperado ~ 1.0, "1/f", criticidad)
    beta = 2 -> ruido marron    (h esperado ~ 1.5, camino aleatorio)

    Devuelve la señal ya normalizada (media 0, desvio estandar 1),
    lista para usarse como si fuera un z_t.
    """
    rng = np.random.default_rng(seed)

    blanco = rng.normal(size=n_muestras)

    f = np.fft.rfft(blanco)
    freqs = np.fft.rfftfreq(n_muestras)
    freqs[0] = freqs[1]  # evitar division por cero en f=0

    # La potencia |F|^2 ~ 1/f^beta, por eso la amplitud escala 1/f^(beta/2)
    f_filtrado = f / (freqs ** (beta / 2))
    ruido = np.fft.irfft(f_filtrado, n=n_muestras)

    return (ruido - np.mean(ruido)) / np.std(ruido)


# ----------------------------------------------------------------------
# FUNCION 4: Procesar una serie ya normalizada (z) -> h y c
# ----------------------------------------------------------------------
def procesar_serie(z, sr, window_ms=20):
    """
    A partir de una señal ya normalizada z (media 0, desvio estandar 1),
    calcula:

      Paso 2: la serie de "intensidad" (envolvente RMS de z^2 en
              ventanas cortas, mas liviano que usar cada muestra)
      Paso 3: h (DFA) sobre la serie de intensidad
      Paso 6: c (Gaussiana estirada) sobre z

    'sr' es la cantidad de muestras por segundo de z, se usa para
    definir el tamaño de ventana en muestras a partir de window_ms.

    Devuelve un diccionario {"h": ..., "c": ...}
    """
    # --- Paso 2: envolvente de intensidad (z^2 promediado en ventanas) ---
    hop = int(sr * window_ms / 1000)
    n_ventanas = len(z) // hop
    intensidad = np.array([
        np.mean(z[i * hop:(i + 1) * hop] ** 2)
        for i in range(n_ventanas)
    ])

    # --- Paso 3: exponente de Hurst ---
    h = dfa(intensidad)

    # --- Paso 6: parametro c de la Gaussiana estirada ---
    c = ajustar_c(z)

    return {"h": h, "c": c}


# ----------------------------------------------------------------------
# FUNCION 5: Procesar un archivo de audio completo
# ----------------------------------------------------------------------
def procesar_audio(filepath, duracion_seg=120, window_ms=20):
    """
    Pipeline completo para un archivo de audio:

      Paso 1: Cargar el audio en mono y normalizar
              z_t = (u_t - media) / desvio_estandar
      Paso 2-3-6: ver procesar_serie()

    Devuelve un diccionario {"h": ..., "c": ...}
    """
    # --- Paso 1: cargar y normalizar ---
    y, sr = sf.read(filepath, always_2d=True)  # y tiene forma (muestras, canales)
    y = y.mean(axis=1)  # promediar canales -> mono

    # Tomar un segmento central representativo (mas rapido y comparable)
    n_total = len(y)
    n_seg = duracion_seg * sr
    if n_total > n_seg:
        inicio = (n_total - n_seg) // 2
        y = y[inicio:inicio + n_seg]

    z = (y - np.mean(y)) / np.std(y)

    return procesar_serie(z, sr, window_ms)
