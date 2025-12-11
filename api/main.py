from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn
import io
import numpy as np
from PIL import Image
from tensorflow.keras import models

# --- Constantes ---
NB_CLASSES = 8
TAILLE_IMAGE = (256, 512)  # (H, W)

# --- Télécharger le modèle depuis GitHub si nécessaire ---
MODEL_URL = (
    "https://github.com/essanas/projet8_futurevision_segmentation/"
    "raw/main/models/unet_mobilenetv2_8c_fast.keras"
)
MODEL_PATH = "/tmp/unet_mobilenetv2_8c_fast.keras"  # Emplacement temporaire (ex : Render)


def download_model():
    """Télécharge le modèle depuis GitHub si nécessaire."""
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
    else:
        raise Exception("Erreur lors du téléchargement du modèle depuis GitHub")


# Télécharger le modèle au démarrage de l'application
download_model()

# --- Application FastAPI ---
app = FastAPI(
    title="FutureVision - API de segmentation d'images",
    version="1.0",
    description=(
        "Interface et API pour segmenter des images avec le modèle "
        "UNet-MobileNetV2 (8 classes)."
    ),
)

# --- CORS (pour autoriser les appels depuis le navigateur) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # en prod : à restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Middleware pour désactiver le cache navigateur ---
@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# --- Chargement du modèle ---
modele = models.load_model(
    MODEL_PATH,
    # stub pour la métrique custom utilisée à l'entraînement
    custom_objects={"dice_soft_macro": lambda y_true, y_pred: 0.0},
    compile=False,  # pas besoin de loss/metrics pour l'inférence
)

# --- Palette couleur pour les 8 classes ---
PALETTE = np.array([
    [128,  64, 128],  # route
    [244,  35, 232],  # trottoir
    [ 70,  70,  70],  # bâtiment
    [102, 102, 156],  # mur
    [190, 153, 153],  # clôture
    [153, 153, 153],  # poteau
    [250, 170,  30],  # feu
    [220, 220,   0],  # végétation
], dtype=np.uint8)

# --- HTML (interface utilisateur intégrée) ---
PAGE_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FutureVision - Segmentation d'images</title>
    <style>
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            font-family: "Segoe UI", Roboto, sans-serif;
            color: #f1f1f1;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        header {
            text-align: center;
            padding: 60px 20px 30px;
        }
        header h1 {
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 10px;
            color: #00e6ac;
        }
        header p {
            font-size: 1.1em;
            opacity: 0.8;
        }
        .card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            width: 90%;
            max-width: 900px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        input[type="file"] {
            display: none;
        }
        label.upload-btn {
            background-color: #00e6ac;
            color: #0f2027;
            padding: 12px 25px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease;
            margin-top: 10px;
            display: inline-block;
        }
        label.upload-btn:hover {
            background-color: #00b386;
        }
        .images {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-top: 30px;
        }
        .image-box {
            flex: 1 1 45%;
            margin: 10px;
            text-align: center;
        }
        .image-box img {
            width: 100%;
            border-radius: 10px;
            border: 2px solid #00e6ac;
            box-shadow: 0 0 15px rgba(0, 230, 172, 0.3);
        }
        .image-box h3 {
            margin-top: 12px;
            color: #00e6ac;
        }
        .loader {
            border: 4px solid rgba(255, 255, 255, 0.2);
            border-top: 4px solid #00e6ac;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        footer {
            margin-top: auto;
            padding: 30px;
            font-size: 0.9em;
            color: #ccc;
        }
        footer a {
            color: #00e6ac;
            text-decoration: none;
        }
        footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header>
        <h1>Future Vision Transport</h1>
        <p>Segmentation sémantique des scènes urbaines avec UNet-MobileNetV2 (8 classes)</p>
    </header>
    <div class="card">
        <label for="file-upload" class="upload-btn">📂 Choisir une image</label>
        <input id="file-upload" type="file" accept="image/png, image/jpeg">
        <div class="loader" id="loader"></div>
        <div class="images" id="images-display"></div>
    </div>
    <footer>
        <p>Développé avec 💚 par <strong>FutureVision</strong></p>
    </footer>
    <script>
        const fileInput = document.getElementById("file-upload");
        const loader = document.getElementById("loader");
        const imagesDisplay = document.getElementById("images-display");

        fileInput.addEventListener("change", async () => {
            const file = fileInput.files[0];
            if (!file) return;

            loader.style.display = "block";
            imagesDisplay.innerHTML = "";

            const formData = new FormData();
            formData.append("fichier", file);

            try {
                const response = await fetch("/predire", {
                    method: "POST",
                    body: formData
                });
                if (!response.ok) throw new Error("Erreur lors de la prédiction");

                const blob = await response.blob();
                const resultUrl = URL.createObjectURL(blob);
                const inputUrl = URL.createObjectURL(file);

                imagesDisplay.innerHTML = `
                    <div class="image-box">
                        <h3>Image originale</h3>
                        <img src="${inputUrl}" alt="Image d'origine">
                    </div>
                    <div class="image-box">
                        <h3>Résultat segmenté</h3>
                        <img src="${resultUrl}" alt="Masque segmenté">
                    </div>
                `;
            } catch (err) {
                alert("Erreur : " + err.message);
            } finally {
                loader.style.display = "none";
            }
        });
    </script>
</body>
</html>
"""

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
def page_principale():
    return PAGE_HTML


@app.get("/etat")
def etat():
    return {
        "statut": "opérationnel",
        "modele": MODEL_PATH,
        "taille_image": TAILLE_IMAGE,
        "nombre_de_classes": NB_CLASSES,
    }


def pretraiter_image(pil_img: Image.Image) -> np.ndarray:
    """Prétraitement de l'image d'entrée pour le modèle."""
    pil_img = pil_img.convert("RGB")
    # PIL attend (width, height)
    pil_img = pil_img.resize((TAILLE_IMAGE[1], TAILLE_IMAGE[0]), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    # ajout de la dimension batch : (1, H, W, 3)
    return arr[None, ...]


def masque_vers_png_couleur(masque_2d: np.ndarray) -> bytes:
    """Convertit un masque 2D d'indices de classes en PNG couleur."""
    masque_2d = masque_2d.astype(np.int32)
    masque_2d = np.clip(masque_2d, 0, len(PALETTE) - 1)
    masque_rgb = PALETTE[masque_2d]
    image_sortie = Image.fromarray(masque_rgb, mode="RGB")
    tampon = io.BytesIO()
    image_sortie.save(tampon, format="PNG", optimize=True)
    return tampon.getvalue()


@app.post("/predire")
async def predire(fichier: UploadFile = File(...)):
    try:
        contenu = await fichier.read()
        pil = Image.open(io.BytesIO(contenu))
        x = pretraiter_image(pil)
        pred = modele.predict(x, verbose=0)
        masque = np.argmax(pred, axis=-1)[0]
        png_bytes = masque_vers_png_couleur(masque)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return JSONResponse({"erreur": str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
