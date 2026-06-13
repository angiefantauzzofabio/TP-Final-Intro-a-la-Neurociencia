"""
analisis.py

Script principal del TP.

Que hace:
  1. Recorre las carpetas songs/grupo_a (jazz) y songs/grupo_b (techno).
  2. Para cada archivo de audio, calcula h (exponente de Hurst) y
     c (parametro de la Gaussiana estirada) usando funciones.py.
  3. Genera ademas ruido blanco, rosa y marron directamente en Python
     (linea de base) y les calcula h y c de la misma forma.
  4. Guarda todo en una tabla (resultados.csv).
  5. Genera dos graficos:
       - grafico_h.png: "recta numerica" con el valor de h de cada
         audio, para ver donde caen jazz y techno respecto a los
         ruidos de referencia.
       - grafico_c_vs_h.png: scatter de c vs h, para comparar con
         la correlacion negativa (~-0.7) que reporta el paper.

Como usarlo:
  Tener esta estructura de carpetas:

    songs/
      grupo_a/   <- mp3 de jazz
      grupo_b/   <- mp3 de techno

  Y correr: python analisis.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from funciones import procesar_audio, procesar_serie, generar_ruido


# Mapeo carpeta -> nombre que va a aparecer en la tabla y graficos
CARPETAS = {
    "songs/grupo_a": "jazz",
    "songs/grupo_b": "techno",
}

EXTENSIONES_VALIDAS = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

# Parametros para generar el ruido sintetico (grupo C / linea de base)
SR_RUIDO = 44100          # muestras por segundo (igual que un audio tipico)
DURACION_RUIDO_SEG = 120  # mismo largo que el segmento de audio analizado
RUIDOS = {
    "ruido_blanco": 0,  # beta=0 -> h esperado ~0.5
    "ruido_rosa": 1,    # beta=1 -> h esperado ~1.0
    "ruido_marron": 2,  # beta=2 -> h esperado ~1.5
}


# ----------------------------------------------------------------------
# PASO 1b: generar ruidos sinteticos (linea de base / grupo C)
# ----------------------------------------------------------------------
def analizar_ruidos_sinteticos():
    """
    Genera ruido blanco, rosa y marron con generar_ruido() y les
    calcula h y c con procesar_serie(), igual que a las canciones.

    Sirve como linea de base / validacion: si todo esta bien,
    deberian salir h~0.5, h~1.0 y h~1.5 respectivamente.
    """
    print(f"\nGenerando ruidos sinteticos (linea de base)")
    print("-" * 50)

    filas = []
    n_muestras = SR_RUIDO * DURACION_RUIDO_SEG

    for nombre, beta in RUIDOS.items():
        print(f"  {nombre} (beta={beta}) ...", end=" ")
        z = generar_ruido(n_muestras, beta, seed=42)
        resultado = procesar_serie(z, SR_RUIDO)
        filas.append({
            "archivo": nombre,
            "grupo": "ruido",
            "h": resultado["h"],
            "c": resultado["c"],
        })
        print(f"h={resultado['h']:.3f}  c={resultado['c']:.3f}")

    return filas


# ----------------------------------------------------------------------
# PASO 1: recorrer las carpetas y calcular h y c para cada archivo
# ----------------------------------------------------------------------
def analizar_todo():
    filas = []

    for carpeta, etiqueta in CARPETAS.items():
        if not os.path.isdir(carpeta):
            print(f"[Aviso] No existe la carpeta: {carpeta} (se omite)")
            continue

        print(f"\nProcesando carpeta: {carpeta}  ({etiqueta})")
        print("-" * 50)

        for archivo in sorted(os.listdir(carpeta)):
            if not archivo.lower().endswith(EXTENSIONES_VALIDAS):
                continue

            ruta = os.path.join(carpeta, archivo)
            print(f"  {archivo} ...", end=" ")

            try:
                resultado = procesar_audio(ruta)
                filas.append({
                    "archivo": archivo,
                    "grupo": etiqueta,
                    "h": resultado["h"],
                    "c": resultado["c"],
                })
                print(f"h={resultado['h']:.3f}  c={resultado['c']:.3f}")
            except Exception as e:
                print(f"ERROR: {e}")

    # Agregar los ruidos sinteticos como tercer grupo
    filas += analizar_ruidos_sinteticos()

    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# PASO 2: grafico tipo "recta numerica" para h
# ----------------------------------------------------------------------
def graficar_h(df):
    """
    Eje horizontal = valor de h. Cada punto es un audio.
    Sirve para ver visualmente si el jazz cae cerca del ruido rosa
    (h~1.0) generado como linea de base, comparado con el techno.
    """
    colores = {"jazz": "tab:blue", "techno": "tab:red", "ruido": "tab:gray"}

    fig, ax = plt.subplots(figsize=(9, 4))
    for grupo in df["grupo"].unique():
        subset = df[df["grupo"] == grupo]
        # un poco de dispersion vertical solo para que no se solapen
        y_jitter = pd.Series(range(len(subset))) * 0.08 + 0.1
        ax.scatter(subset["h"], y_jitter, label=grupo,
                   color=colores.get(grupo, "black"), s=70, edgecolor="k")

    ax.set_xlabel("Exponente de Hurst (h)")
    ax.set_yticks([])
    ax.set_title("Exponente de Hurst por grupo")
    ax.legend()
    plt.tight_layout()
    plt.savefig("grafico_h.png", dpi=150)
    plt.close(fig)
    print("\nGrafico guardado: grafico_h.png")


# ----------------------------------------------------------------------
# PASO 3: grafico c vs h
# ----------------------------------------------------------------------
def graficar_c_vs_h(df):
    """
    Scatter de c (forma de la distribucion) vs h (correlaciones).
    El paper reporta una correlacion negativa de aprox -0.7 entre
    estas dos variables para su dataset de 8000+ canciones.
    """
    colores = {"jazz": "tab:blue", "techno": "tab:red", "ruido": "tab:gray"}

    fig, ax = plt.subplots(figsize=(6, 5))
    for grupo in df["grupo"].unique():
        subset = df[df["grupo"] == grupo]
        ax.scatter(subset["c"], subset["h"], label=grupo,
                   color=colores.get(grupo, "black"), s=70, edgecolor="k")

    ax.set_xlabel("c (parametro de la Gaussiana estirada)")
    ax.set_ylabel("h (exponente de Hurst)")
    ax.set_title("Relacion entre c y h")
    ax.legend()
    plt.tight_layout()
    plt.savefig("grafico_c_vs_h.png", dpi=150)
    plt.close(fig)
    print("Grafico guardado: grafico_c_vs_h.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = analizar_todo()

    print("\n" + "=" * 50)
    print("TABLA DE RESULTADOS")
    print("=" * 50)
    print(df.to_string(index=False))

    df.to_csv("resultados.csv", index=False)
    print("\nTabla guardada: resultados.csv")

    if not df.empty:
        print("\nPromedios por grupo:")
        print(df.groupby("grupo")[["h", "c"]].mean().to_string())

        # Correlacion entre c y h (para comparar con el -0.7 del paper)
        correlacion = df["c"].corr(df["h"])
        print(f"\nCorrelacion c vs h en tus datos: {correlacion:.3f}")
        print("(el paper reporta aprox -0.7 para su dataset)")

        graficar_h(df)
        graficar_c_vs_h(df)

    print("\nListo!")
