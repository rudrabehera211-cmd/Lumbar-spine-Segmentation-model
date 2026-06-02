import io
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from utils.mha_utils import load_mha, save_mha
from inference import Predictor
from models import create_model
from evaluation.metrics import SegmentationMetrics

app = FastAPI(title="Lumbar Spine MRI Segmentation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path: str, config: Dict[str, Any]):
    global predictor
    model = create_model(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    predictor = Predictor(model, config, device)


@app.on_event("startup")
async def startup():
    pass


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device), "model_loaded": predictor is not None}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_probs: bool = Form(False),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /load-model first.")

    contents = await file.read()
    suffix = Path(file.filename).suffix if file.filename else '.mha'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        mask, probs = predictor.predict_volume(tmp_path, return_probabilities=True)

        with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as out_tmp:
            sitk_img, _ = load_mha(tmp_path)
            save_mha(mask, out_tmp.name, ref_image=sitk_img)
            out_path = out_tmp.name

        response_data = {
            "shape": list(mask.shape),
            "unique_labels": sorted(np.unique(mask).tolist()),
            "success": True,
        }

        if return_probs:
            probs_path = tmp_path + '_probs.npy'
            np.save(probs_path, probs)

        return FileResponse(out_path, media_type="application/octet-stream",
                            filename=f"segmentation_{file.filename}",
                            headers={"X-Metadata": json.dumps(response_data)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/load-model")
async def load_model_endpoint(
    checkpoint_path: str = Form(...),
    config_json: str = Form(...),
):
    try:
        config = json.loads(config_json)
        load_model(checkpoint_path, config)
        return {"status": "loaded", "device": str(device)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
