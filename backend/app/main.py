# /backend/app/main.py
import os
from pathlib import Path
from io import BytesIO
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from . import ml_model as modelmod
from .utils import validate_image   # make sure there's no space after the dot

app = FastAPI()

# Paths
BASE_DIR    = Path(__file__).resolve().parent   # .../backend/app
BACKEND_DIR = BASE_DIR.parent                   # .../backend

TEMPLATES_DIR = BACKEND_DIR / "templates"       # backend/templates
STATIC_DIR    = BACKEND_DIR / "static"          # backend/static
UPLOAD_DIR    = BACKEND_DIR / "uploads"         # backend/uploads

# Ensure dirs exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static + uploads
app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Jinja templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Load all crop models once
MODELS = modelmod.load_crop_models()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "crop_types": list(modelmod.CLASS_NAMES.keys()),
        "error": None
    })

def _save_image_normalized(file: UploadFile) -> Path:
    # Validate and normalize upload to JPEG, strip EXIF, unique filename
    validate_image(file)
    raw = file.file.read()
    try:
        im = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    fname = f"{uuid4().hex}.jpg"
    fpath = UPLOAD_DIR / fname
    im.save(fpath, format="JPEG", quality=90, optimize=True)
    return fpath

@app.post("/predict")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    crop_type: str = Form(...),
    explain: int = Form(1)
):
    try:
        saved_path = _save_image_normalized(file)

        pred = modelmod.predict(MODELS, str(saved_path), crop_type.lower())
        crop_name = pred["crop"]

        stats = modelmod.image_stats(str(saved_path))

        # Always compute Top-3; OOD path removed
        top3 = modelmod.topk_predictions(MODELS, str(saved_path), crop_type.lower(), k=3)

        # Optional explainability images
        lime_file = None
        occ_file = None
        ig_file = None
        if int(explain) == 1:
            try:
                lime_file = modelmod.lime_explain(
                    MODELS, str(saved_path), crop_type.lower(),
                    str(UPLOAD_DIR / f"{saved_path.stem}_lime.jpg")
                )
            except Exception as e:
                print(f"[warn] LIME failed: {e}")
            try:
                occ_file = modelmod.occlusion_heatmap(
                    MODELS, str(saved_path), crop_type.lower(),
                    str(UPLOAD_DIR / f"{saved_path.stem}_occ.jpg")
                )
            except Exception as e:
                print(f"[warn] Occlusion failed: {e}")
            try:
                ig_file = modelmod.integrated_gradients_explain(
                    MODELS, str(saved_path), crop_type.lower(),
                    str(UPLOAD_DIR / f"{saved_path.stem}_ig.jpg")
                )
            except Exception as e:
                print(f"[warn] IG failed: {e}")

        return templates.TemplateResponse("result.html", {
            "request": request,
            "filename": saved_path.name,           # use 'filename' for the template
            "prediction": pred["class"],
            "confidence": pred["confidence"],
            "crop": crop_name,
            "stats": stats,
            "top3": top3,
            "lime_image": lime_file,
            "occlusion_image": occ_file,
            "ig_image": ig_file,                   # new: Integrated Gradients
            "is_ood": False,                       # keep flag for template compatibility
            "scores": pred.get("scores", {}),
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "crop_types": list(modelmod.CLASS_NAMES.keys())
        }, status_code=400)

@app.get("/index")
async def go_index():
    return RedirectResponse(url="/")
