
import os
import requests

from fastapi import FastAPI, Request, HTTPException

app = FastAPI(title="Detector de Placas Integrado")

API_URL = "https://api.platerecognizer.com/v1/plate-reader/"
API_IA = "https://detector-placas-ia.onrender.com/detectar"


@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "mensaje": "API Detector de Placas funcionando",
        "integracion": "PictoBlox + Plate Recognizer"
    }


@app.post("/detectar")
async def detectar_placa(request: Request):

    token = os.getenv("PLATE_RECOGNIZER_TOKEN")

    if not token:
        raise HTTPException(
            status_code=503,
            detail="Servicio de reconocimiento no configurado"
        )

    imagen = await request.body()

    if not imagen:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ninguna imagen"
        )

    if len(imagen) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera los 10 MB"
        )

    # 1. Detectar la placa con el modelo de PictoBlox
    deteccion_ia = None

    try:
        respuesta_ia = requests.post(
            API_IA,
            data=imagen,
            headers={"Content-Type": "image/jpeg"},
            timeout=40
        )

        respuesta_ia.raise_for_status()
        deteccion_ia = respuesta_ia.json()

    except (requests.RequestException, ValueError):
        # Si falla PictoBlox, conservar el OCR original
        deteccion_ia = None

    # 2. Reconocer el número con Plate Recognizer
    headers = {"Authorization": f"Token {token}"}

    try:
        respuesta = requests.post(
            API_URL,
            headers=headers,
            files={
                "upload": ("vehiculo.jpg", imagen, "image/jpeg")
            },
            data={"regions": "co"},
            timeout=40
        )

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="No fue posible conectar con Plate Recognizer"
        )

    if respuesta.status_code != 201:
        raise HTTPException(
            status_code=502,
            detail="El servicio de reconocimiento devolvió un error"
        )

    datos = respuesta.json()
    resultados = datos.get("results", [])

    if not resultados:
        return {
            "detectada": False,
            "mensaje": "No se detectó ninguna placa",
            "placa": "",
            "coordenadas": None,
            "deteccion_ia": deteccion_ia
        }

    primera = resultados[0]

    placa = primera.get("plate", "")
    confianza = primera.get("score", 0)

    # Coordenadas de PictoBlox cuando la IA detecta
    if deteccion_ia and deteccion_ia.get("detectada"):
        coordenadas = deteccion_ia.get("coordenadas")
    else:
        coordenadas = primera.get("box", {})

    return {
        "detectada": True,
        "mensaje": "Placa detectada",
        "placa": placa.upper(),
        "confianza": confianza,
        "coordenadas": coordenadas,
        "deteccion_ia": deteccion_ia
    }
