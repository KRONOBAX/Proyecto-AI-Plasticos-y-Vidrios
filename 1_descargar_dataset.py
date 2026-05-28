"""
Script para descargar imagenes de entrenamiento usando DuckDuckGo.
Descarga imagenes de plastico y vidrio para entrenar la red neuronal.
"""

import os
import time
import urllib.request
import urllib.parse
import json
import re
from pathlib import Path

# Directorio base del dataset
BASE_DIR = Path("dataset")
CLASES = {
    "plastico": [
        "plastic bottle",
        "plastic container",
        "plastic bag",
        "plastic packaging",
        "plastic waste"
    ],
    "vidrio": [
        "glass bottle",
        "glass jar",
        "glass container",
        "glass window",
        "broken glass",
    ],
}

IMAGENES_POR_QUERY = 30  # imagenes a intentar descargar por busqueda


def descargar_imagen(url, ruta_destino):
    """Descarga una imagen desde una URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()
        # Verificar que sea una imagen valida (minimo 5KB)
        if len(data) < 5000:
            return False
        with open(ruta_destino, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def buscar_imagenes_duckduckgo(query, max_results=30):
    """Obtiene URLs de imagenes usando DuckDuckGo."""
    urls = []
    try:
        # Obtener token vqd de DuckDuckGo
        search_url = "https://duckduckgo.com/"
        params = urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            f"{search_url}?{params}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8")

        vqd_match = re.search(r'vqd=([^&"]+)', html)
        if not vqd_match:
            return urls
        vqd = vqd_match.group(1)

        # Buscar imagenes
        img_url = "https://duckduckgo.com/i.js"
        params = urllib.parse.urlencode({
            "l": "es-es",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,",
            "p": "1",
        })
        req = urllib.request.Request(
            f"{img_url}?{params}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))

        for item in data.get("results", [])[:max_results]:
            if "image" in item:
                urls.append(item["image"])

    except Exception as e:
        print(f"  Error buscando '{query}': {e}")

    return urls


def main():
    print("=== Descarga de Dataset: Plastico vs Vidrio ===\n")

    for clase, queries in CLASES.items():
        carpeta = BASE_DIR / clase
        carpeta.mkdir(parents=True, exist_ok=True)
        existentes = len(list(carpeta.glob("*.jpg"))) + len(list(carpeta.glob("*.png")))
        print(f"\nClase: {clase.upper()} ({existentes} imagenes existentes)")

        contador = existentes
        for query in queries:
            print(f"  Buscando: '{query}'...")
            urls = buscar_imagenes_duckduckgo(query, IMAGENES_POR_QUERY)
            print(f"  URLs encontradas: {len(urls)}")

            for i, url in enumerate(urls):
                ext = ".jpg"
                if ".png" in url.lower():
                    ext = ".png"
                nombre = f"{clase}_{contador:04d}{ext}"
                ruta = carpeta / nombre
                if ruta.exists():
                    contador += 1
                    continue
                if descargar_imagen(url, str(ruta)):
                    contador += 1
                    print(f"    [{contador}] Descargada: {nombre}")
                time.sleep(0.3)

        total = len(list(carpeta.glob("*.jpg"))) + len(list(carpeta.glob("*.png")))
        print(f"  Total {clase}: {total} imagenes")

    print("\n=== Descarga completada ===")
    print("\nResumen:")
    for clase in CLASES:
        carpeta = BASE_DIR / clase
        total = len(list(carpeta.glob("*.jpg"))) + len(list(carpeta.glob("*.png")))
        print(f"  {clase}: {total} imagenes")


if __name__ == "__main__":
    main()
