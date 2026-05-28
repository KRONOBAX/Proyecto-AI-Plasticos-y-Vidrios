"""
Limpieza del dataset antes de entrenar.
Elimina imagenes corruptas, truncadas, muy pequeñas y duplicadas.
Ejecutar entre 1_descargar_dataset.py y 2_entrenar.py
"""

import os
import hashlib
from pathlib import Path
from PIL import Image, ImageFile

# NO toleramos truncadas aqui, queremos detectarlas y eliminarlas
ImageFile.LOAD_TRUNCATED_IMAGES = False

DATASET_DIR  = Path("dataset")
CLASES       = ["plastico", "vidrio"]
MIN_ANCHO    = 50   # pixeles minimos de ancho
MIN_ALTO     = 50   # pixeles minimos de alto
EXTENSIONES  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def hash_imagen(ruta):
    """Calcula el hash MD5 del contenido de la imagen para detectar duplicados."""
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def verificar_imagen(ruta):
    """
    Intenta abrir y verificar la imagen.
    Retorna (True, motivo) si es valida, (False, motivo) si no lo es.
    """
    # Paso 1: verificacion basica de estructura
    try:
        with Image.open(ruta) as img:
            img.verify()
    except Exception as e:
        return False, f"corrupta: {e}"

    # Paso 2: cargar todos los pixeles (detecta truncadas)
    try:
        with Image.open(ruta) as img:
            img.load()  # fuerza lectura completa de todos los bytes
            img = img.convert("RGB")
            ancho, alto = img.size
            if ancho < MIN_ANCHO or alto < MIN_ALTO:
                return False, f"muy pequeña ({ancho}x{alto}px)"
    except Exception as e:
        return False, f"truncada o ilegible: {e}"

    return True, "ok"


def limpiar_clase(clase):
    carpeta = DATASET_DIR / clase
    if not carpeta.exists():
        print(f"  Carpeta no encontrada: {carpeta}")
        return

    archivos = [f for f in carpeta.iterdir() if f.suffix.lower() in EXTENSIONES]
    total    = len(archivos)
    print(f"\n  {clase.upper()}: {total} imagenes encontradas")

    eliminadas    = 0
    hashes_vistos = {}

    for ruta in archivos:
        # ── Verificar integridad ──
        valida, motivo = verificar_imagen(ruta)
        if not valida:
            print(f"    ELIMINADA [{motivo}]: {ruta.name}")
            ruta.unlink()
            eliminadas += 1
            continue

        # ── Verificar duplicados ──
        h = hash_imagen(ruta)
        if h in hashes_vistos:
            print(f"    ELIMINADA [duplicado de {hashes_vistos[h]}]: {ruta.name}")
            ruta.unlink()
            eliminadas += 1
        else:
            hashes_vistos[h] = ruta.name

    validas = total - eliminadas
    print(f"  Eliminadas: {eliminadas} | Validas restantes: {validas}")

    if validas < 10:
        print(f"  ADVERTENCIA: Solo quedan {validas} imagenes de {clase}.")
        print(f"  Se recomienda tener al menos 50. Agrega mas imagenes manualmente.")

    return validas


def main():
    print("=" * 50)
    print("   LIMPIEZA DEL DATASET")
    print("=" * 50)

    resultados = {}
    for clase in CLASES:
        validas = limpiar_clase(clase)
        resultados[clase] = validas

    print("\n" + "=" * 50)
    print("RESUMEN FINAL:")
    for clase, validas in resultados.items():
        estado = "OK" if validas and validas >= 50 else "POCAS IMAGENES"
        print(f"  {clase:<12}: {validas} imagenes  [{estado}]")

    print("\nLimpieza completada. Ahora ejecuta: python 2_entrenar.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
