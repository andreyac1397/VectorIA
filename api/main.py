from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import cnn, lstm, ann

# Crear la API
app = FastAPI(
    title="VectorIA API",
    description="API para la vigilancia inteligente del dengue en Costa Rica",
    version="1.0.0"
)

# Permitir la comunicación con Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Registrar las rutas de los modelos
app.include_router(cnn.router)
app.include_router(lstm.router)
app.include_router(ann.router)

# Ruta principal
@app.get("/")
def inicio():
    return {
        "proyecto": "VectorIA",
        "estado": "API funcionando",
        "modelos": ["CNN", "LSTM", "ANN"]
    }

# Ruta para comprobar el estado de la API
@app.get("/health")
def estado():
    return {"status": "ok"}