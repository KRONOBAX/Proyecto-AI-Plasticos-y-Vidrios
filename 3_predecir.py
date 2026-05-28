# Prediccion sobre una imagen nueva usando el modelo entrenado
# Uso: python 3_predecir.py <ruta_imagen>
#      python 3_predecir.py  (sin argumentos abre selector de archivo)

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Rutas y dispositivo ---
MODELOS_DIR = Path("modelos")
MODELO_PATH = MODELOS_DIR / "plastico_vidrio.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Carga del modelo entrenado ---
def cargar_modelo():
    checkpoint   = torch.load(MODELO_PATH, map_location=DEVICE, weights_only=False)
    clases       = checkpoint["clases"]
    img_size     = checkpoint.get("img_size", 224)
    arquitectura = checkpoint.get("arquitectura", "resnet50")

    if arquitectura == "resnet50":
        modelo = models.resnet50(weights=None)
        in_features = modelo.fc.in_features
        modelo.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(clases)),
        )
    else:
        modelo = models.mobilenet_v2(weights=None)
        in_features = modelo.classifier[1].in_features
        modelo.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(clases)),
        )

    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(DEVICE).eval()
    return modelo, clases, img_size


# --- Clasificacion de la imagen ---
def predecir(ruta_imagen, modelo, clases, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img    = Image.open(ruta_imagen).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs     = torch.softmax(modelo(tensor), dim=1)[0]
        pred_idx  = probs.argmax().item()
        confianza = probs[pred_idx].item()
        clase_pred = clases[pred_idx]

    print("\n" + "="*40)
    print(f"Imagen:    {Path(ruta_imagen).name}")
    print(f"Resultado: {clase_pred.upper()}")
    print(f"Confianza: {confianza*100:.1f}%")
    print("-"*40)
    for i, clase in enumerate(clases):
        barra = "#" * int(probs[i].item() * 30)
        print(f"  {clase:<12} {probs[i]*100:>5.1f}%  {barra}")
    print("="*40 + "\n")

    return clase_pred, confianza


# --- Funcion principal ---
def main():
    if not MODELO_PATH.exists():
        print(f"ERROR: No se encontro el modelo en {MODELO_PATH}")
        print("Ejecuta primero: python 2_entrenar.py")
        return

    # Si se pasa la ruta como argumento la usamos, si no abrimos el selector
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            ruta = filedialog.askopenfilename(
                title="Selecciona una imagen",
                filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")]
            )
            if not ruta:
                print("No se selecciono ninguna imagen.")
                return
        except Exception:
            print("Uso: python 3_predecir.py <ruta_imagen>")
            return

    if not Path(ruta).exists():
        print(f"ERROR: Archivo no encontrado: {ruta}")
        return

    print("Cargando modelo...")
    modelo, clases, img_size = cargar_modelo()
    predecir(ruta, modelo, clases, img_size)


if __name__ == "__main__":
    main()
