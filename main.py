
import os
import requests

from fastapi import FastAPI, Request, HTTPException

app = FastAPI(title="Detector de Placas")

API_URL = "https://api.platerecognizer.com/v1/plate-reader/"


@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "mensaje": "API Detector de Placas funcionando"
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
            "coordenadas": None
        }

    primera = resultados[0]

    placa = primera.get("plate", "")
    coordenadas = primera.get("box", {})
    confianza = primera.get("score", 0)

    return {
        "detectada": True,
        "mensaje": "Placa detectada",
        "placa": placa.upper(),
        "confianza": confianza,
        "coordenadas": coordenadas
    }
    
@app.get("/modelo/info")
def informacion_modelo():
    from tflite_runtime.interpreter import Interpreter

    ruta = os.path.join(
        os.path.dirname(__file__),
        "model.tflite"
    )

    if not os.path.isfile(ruta):
        raise HTTPException(
            status_code=404,
            detail="No se encontró model.tflite"
        )

    try:
        interprete = Interpreter(model_path=ruta)
        interprete.allocate_tensors()

        entradas = interprete.get_input_details()
        salidas = interprete.get_output_details()

        return {
            "modelo": "DetectorPlacas_V2",
            "estado": "cargado",
            "entradas": [
                {
                    "nombre": x["name"],
                    "forma": x["shape"].tolist(),
                    "tipo": str(x["dtype"])
                }
                for x in entradas
            ],
            "salidas": [
                {
                    "nombre": x["name"],
                    "forma": x["shape"].tolist(),
                    "tipo": str(x["dtype"])
                }
                for x in salidas
            ]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo cargar el modelo: {type(error).__name__}"
        )

