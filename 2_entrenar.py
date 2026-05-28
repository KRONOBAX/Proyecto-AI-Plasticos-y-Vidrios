# Entrenamiento de la red neuronal para clasificar Plastico vs Vidrio
# Ejecutar: python 2_entrenar.py

import os
import time
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from PIL import ImageFile

import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- Parametros del entrenamiento ---
DATASET_DIR    = Path("dataset")
MODELOS_DIR    = Path("modelos")
RESULTADOS_DIR = Path("resultados")

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 0.001
VAL_SPLIT   = 0.2
SEED        = 42
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transformaciones de imagen ---
# Para entrenamiento se aplica data augmentation (variaciones artificiales)
# Para validacion solo se redimensiona y normaliza
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --- Clase auxiliar del dataset ---
# Necesaria para aplicar distintas transformaciones a train y val
# Debe estar fuera de funciones para que Windows pueda serializarla
class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# --- Carga y division del dataset ---
def cargar_datos():
    dataset_completo = datasets.ImageFolder(str(DATASET_DIR))
    clases = dataset_completo.classes
    print(f"Clases detectadas: {clases}")

    n_total = len(dataset_completo)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    torch.manual_seed(SEED)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val])

    base_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=None)
    train_set = SubsetWithTransform(base_dataset, train_idx.indices, train_transforms)
    val_set   = SubsetWithTransform(base_dataset, val_idx.indices,   val_transforms)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=NUM_WORKERS > 0)

    print(f"Train: {len(train_set)} imagenes | Val: {len(val_set)} imagenes")
    return train_loader, val_loader, clases, base_dataset, val_idx.indices


# --- Construccion del modelo con Transfer Learning ---
# Usamos ResNet-50 ya entrenada y solo ajustamos las ultimas capas
def construir_modelo(n_clases=2):
    modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for name, param in modelo.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    in_features = modelo.fc.in_features
    modelo.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, n_clases),
    )
    return modelo.to(DEVICE)


# --- Detalle de validacion imagen por imagen ---
def mostrar_validacion(modelo, base_dataset, val_indices, clases):
    modelo.eval()
    correctas = incorrectas = 0

    print("\n--- Detalle de Validacion ---")
    print(f"  {'Imagen':<35} {'Real':<12} {'Prediccion':<12} {'Resultado'}")
    print("  " + "-"*75)

    with torch.no_grad():
        for idx in val_indices:
            ruta, label_real = base_dataset.samples[idx]
            nombre = Path(ruta).name
            img    = base_dataset.loader(ruta)
            tensor = val_transforms(img).unsqueeze(0).to(DEVICE)

            probs     = torch.softmax(modelo(tensor), dim=1)[0]
            pred_idx  = probs.argmax().item()
            confianza = probs[pred_idx].item()

            clase_real = clases[label_real]
            clase_pred = clases[pred_idx]
            resultado  = "OK" if pred_idx == label_real else "ERROR"

            if pred_idx == label_real:
                correctas += 1
            else:
                incorrectas += 1

            print(f"  {nombre:<35} {clase_real:<12} {clase_pred:<12} {resultado} ({confianza*100:.1f}%)")

    total = correctas + incorrectas
    print("  " + "-"*75)
    print(f"  Correctas: {correctas}/{total} | Incorrectas: {incorrectas}/{total}\n")


# --- Bucle de entrenamiento ---
def entrenar(modelo, train_loader, val_loader, base_dataset=None, val_indices=None, clases=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    use_amp = DEVICE.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    mejor_val_acc    = 0.0
    mejor_pesos      = copy.deepcopy(modelo.state_dict())
    historial        = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    paciencia_actual = 0
    EARLY_STOP       = 7

    print("\n" + "="*65)
    print(f"{'Ronda':>5} {'Error Entreno':>14} {'Precision Entreno':>18} {'Error Val':>10} {'Precision Val':>14}")
    print("="*65)

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Fase de entrenamiento
        modelo.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                salidas = modelo(imgs)
                loss    = criterion(salidas, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (salidas.argmax(dim=1) == labels).sum().item()
            train_total   += imgs.size(0)

        # Fase de validacion
        modelo.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    salidas = modelo(imgs)
                    loss    = criterion(salidas, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (salidas.argmax(dim=1) == labels).sum().item()
                val_total   += imgs.size(0)

        t_loss = train_loss / train_total
        v_loss = val_loss   / val_total
        t_acc  = train_correct / train_total
        v_acc  = val_correct   / val_total

        historial["train_loss"].append(t_loss)
        historial["val_loss"].append(v_loss)
        historial["train_acc"].append(t_acc)
        historial["val_acc"].append(v_acc)

        scheduler.step(v_loss)
        print(f"{epoch+1:>5} {t_loss:>14.4f} {t_acc*100:>17.2f}% {v_loss:>10.4f} {v_acc*100:>13.2f}%  ({time.time()-t0:.1f}s)")

        if v_acc > mejor_val_acc:
            mejor_val_acc    = v_acc
            mejor_pesos      = copy.deepcopy(modelo.state_dict())
            paciencia_actual = 0
        else:
            paciencia_actual += 1
            if paciencia_actual >= EARLY_STOP:
                print(f"\nEarly stopping en epoch {epoch+1}.")
                break

    modelo.load_state_dict(mejor_pesos)
    print(f"\nMejor Precision en Validacion: {mejor_val_acc*100:.2f}%")

    if base_dataset is not None and val_indices is not None and clases is not None:
        mostrar_validacion(modelo, base_dataset, val_indices, clases)

    return modelo, historial


# --- Reporte final de clasificacion ---
def evaluar(modelo, val_loader, clases):
    modelo.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs    = imgs.to(DEVICE)
            salidas = modelo(imgs)
            all_preds.extend(salidas.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== Reporte de Clasificacion ===")
    reporte = classification_report(all_labels, all_preds, target_names=clases)
    reporte = reporte.replace("precision", "precision ").replace("recall", "recobrado  ")
    reporte = reporte.replace("f1-score", "puntuacion ").replace("support", "muestras   ")
    reporte = reporte.replace("accuracy", "exactitud  ").replace("macro avg", "promedio   ")
    reporte = reporte.replace("weighted avg", "ponderado  ")
    print(reporte)

    cm = confusion_matrix(all_labels, all_preds)
    print("Matriz de Confusion:")
    print(f"  {'':>12} {'Pred: plastico':>15} {'Pred: vidrio':>13}")
    print(f"  {'Real: plastico':<14} {cm[0][0]:>15} {cm[0][1]:>13}")
    print(f"  {'Real: vidrio':<14} {cm[1][0]:>15} {cm[1][1]:>13}")
    return all_preds, all_labels


# --- Graficas de perdida y precision por epoch ---
def graficar(historial):
    RESULTADOS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(historial["train_loss"]) + 1)

    axes[0].plot(epochs, historial["train_loss"], label="Train")
    axes[0].plot(epochs, historial["val_loss"],   label="Val")
    axes[0].set_title("Error (Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [a*100 for a in historial["train_acc"]], label="Train")
    axes[1].plot(epochs, [a*100 for a in historial["val_acc"]],   label="Val")
    axes[1].set_title("Precision (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    ruta = RESULTADOS_DIR / "entrenamiento.png"
    plt.savefig(ruta)
    print(f"Grafica guardada en: {ruta}")
    plt.close()


# --- Funcion principal ---
def main():
    print(f"Usando dispositivo: {DEVICE}")
    MODELOS_DIR.mkdir(exist_ok=True)
    RESULTADOS_DIR.mkdir(exist_ok=True)

    for clase in ["plastico", "vidrio"]:
        carpeta = DATASET_DIR / clase
        if not carpeta.exists():
            print(f"ERROR: No existe la carpeta {carpeta}")
            return
        n = sum(len(list(carpeta.glob(f"*.{ext}"))) for ext in ["jpg", "jpeg", "png"])
        print(f"  {clase}: {n} imagenes")
        if n < 10:
            print(f"  ADVERTENCIA: muy pocas imagenes de {clase}. Se recomienda al menos 50.")

    print("\nCargando datos...")
    train_loader, val_loader, clases, base_dataset, val_indices = cargar_datos()

    print("\nConstruyendo modelo ResNet-50...")
    modelo = construir_modelo(n_clases=len(clases))

    # Si ya hay un modelo guardado lo cargamos para seguir mejorando
    ruta_modelo = MODELOS_DIR / "plastico_vidrio.pth"
    if ruta_modelo.exists():
        checkpoint = torch.load(ruta_modelo, map_location=DEVICE, weights_only=False)
        if checkpoint.get("arquitectura", "") == "resnet50":
            print("Modelo anterior encontrado. Continuando desde donde quedo...")
            modelo.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("Arquitectura diferente. Entrenando desde cero.")
    else:
        print("No hay modelo previo. Entrenando desde cero.")

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Parametros entrenables: {n_params:,}")

    print("\nIniciando entrenamiento...")
    modelo, historial = entrenar(modelo, train_loader, val_loader,
                                 base_dataset, val_indices, clases)

    evaluar(modelo, val_loader, clases)
    graficar(historial)

    torch.save({
        "model_state_dict": modelo.state_dict(),
        "clases":           clases,
        "img_size":         IMG_SIZE,
        "arquitectura":     "resnet50",
    }, ruta_modelo)
    print(f"\nModelo guardado en: {ruta_modelo}")

    with open(RESULTADOS_DIR / "historial.json", "w") as f:
        json.dump(historial, f, indent=2)

    print("\n=== Entrenamiento completado ===")
    print("Siguiente paso: python 4_interfaz.py")


if __name__ == "__main__":
    main()
