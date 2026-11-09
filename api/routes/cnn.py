from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(
    prefix="/cnn",
    tags=["CNN - Identificación de mosquitos"]
)


@router.post("/identificar-mosquito")
async def predecir_mosquito(imagen: UploadFile = File(...)):

    # Validar que el archivo sea una imagen
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes JPG o PNG."
        )

    # Leer la imagen
    contenido = await imagen.read()

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail="La imagen está vacía."
        )

    # El modelo CNN se integrará posteriormente
    return {
        "estado": "pendiente",
        "mensaje": "El modelo CNN todavía no está conectado.",
        "archivo": imagen.filename
    }