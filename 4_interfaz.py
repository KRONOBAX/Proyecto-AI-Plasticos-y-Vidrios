# Interfaz grafica para clasificar Plastico vs Vidrio
# Ejecutar: python 4_interfaz.py

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk

# --- Rutas y dispositivo ---
BASE_DIR    = Path(__file__).parent
MODELO_PATH = BASE_DIR / "modelos" / "plastico_vidrio.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Colores de la interfaz ---
COLOR_BG      = "#1a1a2e"
COLOR_PANEL   = "#16213e"
COLOR_ACCENT  = "#0f3460"
COLOR_BTN     = "#e94560"
COLOR_TEXT    = "#eaeaea"
COLOR_SUBTEXT = "#a0a0b0"
COLOR_PLASTICO = "#e94560"
COLOR_VIDRIO   = "#3498db"

# Colores segun nivel de confianza del modelo
COLOR_MEDIA = "#f39c12"   # amarillo: el modelo no esta muy seguro
COLOR_BAJA  = "#e67e22"   # naranja: el modelo casi adivino

UMBRAL_ALTO  = 0.75   # mayor a 75% = confianza alta
UMBRAL_MEDIO = 0.55   # entre 55% y 75% = confianza media

# --- Fuentes ---
FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_SMALL  = ("Segoe UI", 10)
FONT_RESULT = ("Segoe UI", 32, "bold")
FONT_CONF   = ("Segoe UI", 13)
FONT_BTN    = ("Segoe UI", 13, "bold")
FONT_LABEL  = ("Segoe UI", 11)

PANEL_W = 360


# --- Carga del modelo entrenado ---
def cargar_modelo():
    checkpoint   = torch.load(MODELO_PATH, map_location=DEVICE, weights_only=False)
    clases       = checkpoint["clases"]
    img_size     = checkpoint.get("img_size", 224)
    arquitectura = checkpoint.get("arquitectura", "resnet50")

    if arquitectura == "resnet50":
        modelo = models.resnet50(weights=None)
        in_f   = modelo.fc.in_features
        modelo.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(clases)),
        )
    else:
        modelo = models.mobilenet_v2(weights=None)
        in_f   = modelo.classifier[1].in_features
        modelo.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, len(clases)),
        )

    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(DEVICE).eval()
    return modelo, clases, img_size


# --- Clasificacion de una imagen ---
def predecir_imagen(pil_img, modelo, clases, img_size):
    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = t(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs    = torch.softmax(modelo(tensor), dim=1)[0]
        pred_idx = probs.argmax().item()
    return (
        clases[pred_idx],
        probs[pred_idx].item(),
        [(clases[i], probs[i].item()) for i in range(len(clases))],
    )


# --- Ventana principal ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador Plastico / Vidrio")
        self.configure(bg=COLOR_BG)
        self.resizable(True, True)
        self.state("zoomed")

        self.modelo      = None
        self.clases      = None
        self.img_size    = 224
        self._pil_actual = None

        self._construir_ui()
        self._cargar_modelo_async()
        self.bind("<Configure>", self._on_resize)

    # --- Construccion de la interfaz ---
    def _construir_ui(self):
        cabecera = tk.Frame(self, bg=COLOR_BG)
        cabecera.pack(fill=tk.X, pady=(16, 0), padx=24)

        tk.Label(
            cabecera, text="Clasificador  Plastico / Vidrio",
            font=FONT_TITLE, fg=COLOR_BTN, bg=COLOR_BG,
        ).pack(side=tk.LEFT)

        self.lbl_estado = tk.Label(
            cabecera, text="Cargando modelo...",
            font=FONT_SMALL, fg=COLOR_SUBTEXT, bg=COLOR_BG,
        )
        self.lbl_estado.pack(side=tk.RIGHT, pady=6)

        tk.Frame(self, bg=COLOR_ACCENT, height=1).pack(fill=tk.X, pady=6)

        cuerpo = tk.Frame(self, bg=COLOR_BG)
        cuerpo.pack(fill=tk.BOTH, expand=True, padx=24)

        # Panel izquierdo: imagen
        self.frame_img = tk.Frame(
            cuerpo, bg=COLOR_PANEL,
            highlightthickness=2, highlightbackground=COLOR_ACCENT
        )
        self.frame_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        self.canvas = tk.Canvas(self.frame_img, bg=COLOR_PANEL, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._mostrar_placeholder()

        # Panel derecho: resultado
        panel = tk.Frame(cuerpo, bg=COLOR_PANEL, width=PANEL_W)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)
        self._construir_panel_resultado(panel)

        pie = tk.Frame(self, bg=COLOR_BG)
        pie.pack(fill=tk.X, pady=20)

        tk.Button(
            pie, text="  Subir Imagen",
            font=FONT_BTN, fg="white", bg=COLOR_BTN,
            activebackground="#c0392b", activeforeground="white",
            relief=tk.FLAT, padx=32, pady=12, cursor="hand2",
            command=self._subir_imagen,
        ).pack()

    def _construir_panel_resultado(self, parent):
        tk.Label(
            parent, text="RESULTADO",
            font=("Segoe UI", 10, "bold"), fg=COLOR_SUBTEXT, bg=COLOR_PANEL,
        ).pack(pady=(28, 6))

        self.lbl_clase = tk.Label(
            parent, text="---",
            font=FONT_RESULT, fg=COLOR_TEXT, bg=COLOR_PANEL,
            wraplength=PANEL_W - 20,
        )
        self.lbl_clase.pack(pady=(0, 4))

        self.lbl_confianza = tk.Label(
            parent, text="", font=FONT_CONF, fg=COLOR_SUBTEXT, bg=COLOR_PANEL,
        )
        self.lbl_confianza.pack()

        # Mensaje de alerta cuando la confianza es baja
        self.lbl_alerta = tk.Label(
            parent, text="",
            font=("Segoe UI", 11, "italic"), fg=COLOR_MEDIA, bg=COLOR_PANEL,
            wraplength=PANEL_W - 30,
        )
        self.lbl_alerta.pack(pady=(4, 0))

        tk.Frame(parent, bg=COLOR_ACCENT, height=1).pack(fill=tk.X, padx=20, pady=16)

        tk.Label(
            parent, text="Probabilidades",
            font=("Segoe UI", 10, "bold"), fg=COLOR_SUBTEXT, bg=COLOR_PANEL,
        ).pack(anchor=tk.W, padx=20)

        self.frame_barras = tk.Frame(parent, bg=COLOR_PANEL)
        self.frame_barras.pack(fill=tk.X, padx=20, pady=12)

    # --- Placeholder cuando no hay imagen ---
    def _mostrar_placeholder(self):
        self.canvas.delete("all")
        self.after(50, self._dibujar_placeholder_centrado)

    def _dibujar_placeholder_centrado(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2:
            return
        self.canvas.create_text(
            w // 2, h // 2 - 20,
            text="imagen", font=("Segoe UI", 64), fill=COLOR_ACCENT,
        )
        self.canvas.create_text(
            w // 2, h // 2 + 50,
            text="Sube una imagen para clasificarla",
            font=FONT_LABEL, fill=COLOR_SUBTEXT,
        )

    def _on_resize(self, _event=None):
        if self._pil_actual:
            self._dibujar_imagen(self._pil_actual)
        else:
            self._dibujar_placeholder_centrado()

    # --- Carga del modelo en segundo plano ---
    def _cargar_modelo_async(self):
        if not MODELO_PATH.exists():
            self._set_estado(f"ERROR: modelo no encontrado en {MODELO_PATH}", error=True)
            return

        def _carga():
            try:
                m, c, s = cargar_modelo()
                self.modelo, self.clases, self.img_size = m, c, s
                self.after(0, lambda: self._set_estado(
                    f"Modelo listo  |  clases: {', '.join(c)}  |  {DEVICE}"
                ))
            except Exception as e:
                self.after(0, lambda: self._set_estado(f"Error: {e}", error=True))

        threading.Thread(target=_carga, daemon=True).start()

    def _set_estado(self, msg, error=False):
        self.lbl_estado.config(text=msg, fg=COLOR_BTN if error else COLOR_SUBTEXT)

    # --- Subir y mostrar imagen ---
    def _subir_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[
                ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not ruta:
            return
        try:
            pil = Image.open(ruta).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return

        self._pil_actual = pil
        self._dibujar_imagen(pil)
        self._clasificar(pil)

    def _dibujar_imagen(self, pil_img):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2:
            return
        preview = pil_img.copy()
        preview.thumbnail((w, h), Image.LANCZOS)
        panel_color = tuple(int(COLOR_PANEL.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        bg = Image.new("RGB", (w, h), panel_color)
        bg.paste(preview, ((w - preview.width) // 2, (h - preview.height) // 2))
        imgtk = ImageTk.PhotoImage(bg)
        self.canvas.delete("all")
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    # --- Clasificacion en hilo aparte para no congelar la ventana ---
    def _clasificar(self, pil_img):
        if self.modelo is None:
            messagebox.showwarning("Aviso", "El modelo aun se esta cargando.")
            return
        self._set_estado("Clasificando...")
        self.lbl_clase.config(text="...", fg=COLOR_TEXT)
        self.lbl_confianza.config(text="")
        self.lbl_alerta.config(text="")

        def _run():
            try:
                clase, conf, all_probs = predecir_imagen(
                    pil_img, self.modelo, self.clases, self.img_size
                )
                self.after(0, lambda: self._mostrar_resultado(clase, conf, all_probs))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=_run, daemon=True).start()

    # --- Mostrar resultado y nivel de confianza ---
    def _mostrar_resultado(self, clase, confianza, all_probs):
        if confianza >= UMBRAL_ALTO:
            color_res = COLOR_PLASTICO if "plastico" in clase.lower() else COLOR_VIDRIO
            alerta, color_alerta = "", COLOR_SUBTEXT
        elif confianza >= UMBRAL_MEDIO:
            color_res    = COLOR_MEDIA
            alerta       = "Confianza media, el resultado puede no ser exacto"
            color_alerta = COLOR_MEDIA
        else:
            color_res    = COLOR_BAJA
            alerta       = "Confianza baja, intenta con una foto mas clara"
            color_alerta = COLOR_BAJA

        self.lbl_clase.config(text=clase.upper(), fg=color_res)
        self.lbl_confianza.config(text=f"{confianza * 100:.1f}% de confianza")
        self.lbl_alerta.config(text=alerta, fg=color_alerta)

        # --- Barras de probabilidad ---
        for w in self.frame_barras.winfo_children():
            w.destroy()

        BAR_W = 150

        for nombre, prob in all_probs:
            color_barra = COLOR_PLASTICO if "plastico" in nombre.lower() else COLOR_VIDRIO
            fila = tk.Frame(self.frame_barras, bg=COLOR_PANEL)
            fila.pack(fill=tk.X, pady=6)

            tk.Label(
                fila, text=nombre.capitalize(),
                font=FONT_LABEL, fg=COLOR_TEXT, bg=COLOR_PANEL,
                width=9, anchor=tk.W,
            ).pack(side=tk.LEFT)

            contenedor = tk.Canvas(
                fila, width=BAR_W, height=18,
                bg=COLOR_ACCENT, highlightthickness=0,
            )
            contenedor.pack(side=tk.LEFT, padx=(4, 6))
            contenedor.create_rectangle(
                0, 0, max(2, int(BAR_W * prob)), 18,
                fill=color_barra, outline=""
            )

            tk.Label(
                fila, text=f"{prob * 100:.1f}%",
                font=FONT_LABEL, fg=COLOR_TEXT, bg=COLOR_PANEL,
                width=6, anchor=tk.W,
            ).pack(side=tk.LEFT)

        self._set_estado(f"Resultado: {clase.upper()}  |  {confianza * 100:.1f}% de confianza")


if __name__ == "__main__":
    App().mainloop()
