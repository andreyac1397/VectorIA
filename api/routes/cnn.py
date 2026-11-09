"""Endpoint de clasificación de mosquitos con MobileNetV2."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from PIL import Image

from src.predict_mobilenet import predecir_mobilenet

router = APIRouter(
    prefix="/cnn",
    tags=["CNN - Identificación de mosquitos"],
)

MAX_BYTES = 10 * 1024 * 1024
FORMATOS = {"image/jpeg", "image/png"}


@router.post("/identificar-mosquito")
async def identificar_mosquito(imagen: UploadFile = File(...)):
    if imagen.content_type not in FORMATOS:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes JPG, JPEG o PNG.",
        )

    contenido = await imagen.read(MAX_BYTES + 1)
    if not contenido:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")
    if len(contenido) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Máximo 10 MB por imagen.")

    try:
        # La inferencia TensorFlow es sincrónica; evitar bloquear el event loop.
        return await run_in_threadpool(predecir_mobilenet, contenido)
    except (ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo ejecutar MobileNetV2: {type(error).__name__}: {error}",
        ) from error
